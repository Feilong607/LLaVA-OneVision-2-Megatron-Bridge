"""CPU-only long-context topology, restart-contract and multi-pod controller tests."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


REPO = Path(__file__).resolve().parents[4]
FOLDER = REPO / "examples/models/qwen/qwen35_vl_ov2/gb200"
sys.path.insert(0, str(FOLDER))
import run_long_context_matrix as lc  # noqa: E402


pytestmark = pytest.mark.unit


def experiment(tmp_path, *, world=48, stage=2, tps=(4, 2)):
    return lc.Experiment(
        root=str(tmp_path / "result"),
        init=str(tmp_path / "iter_0001000"),
        world=world,
        stage=stage,
        tps=tps,
        steps=8,
        split=5,
        discard=1,
    )


@pytest.mark.parametrize(
    "world,tp,etp,dp,edp", [(48, 2, 2, 24, 3), (48, 4, 2, 12, 3), (64, 2, 2, 32, 4), (64, 4, 4, 16, 2)]
)
def test_rank_groups_from_actual_mcore(tmp_path, world, tp, etp, dp, edp):
    e = experiment(tmp_path, world=world)
    e.validate()
    assert e.case(tp) == dict(world=world, tp=tp, etp=etp, ep=8, dp=dp, expert_dp=edp)
    assert e.gbs // dp == tp // 2
    source = REPO / "3rdparty/Megatron-LM/megatron/core/parallel_state.py"
    tree = ast.parse(source.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and n.name in ("RankGenerator", "generate_masked_orthogonal_rank_groups")
    ]
    namespace = {"List": list, "Optional": __import__("typing").Optional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    generator = namespace["RankGenerator"](tp=etp, ep=8, dp=edp, pp=1, cp=1, order="tp-cp-ep-dp-pp")
    groups = generator.get_ranks("tp-ep")
    assert len(groups) == edp
    assert sorted(r for group in groups for r in group) == list(range(world))
    assert all(len(group) == etp * 8 for group in groups)


@pytest.mark.parametrize("world", [48, 64])
@pytest.mark.parametrize("stage", [2, 3])
def test_phase_contracts_and_environment_isolation(tmp_path, monkeypatch, world, stage):
    e = experiment(tmp_path, world=world, stage=stage)
    monkeypatch.setenv("OV2_FREEZE_LLM", "1")
    monkeypatch.setenv("OV2_MTP_LOSS_SCALE", "0")
    monkeypatch.setenv("OV2_AB_INPUT_DIR", "/old/evidence")
    monkeypatch.setenv("SAVE", "/production")
    monkeypatch.setenv("EXTRA_ARGS", "checkpoint.load=/production")
    for tp in (4, 2):
        for phase in lc.PHASES:
            env = lc.phase_environment(e, tp=tp, phase=phase, assets={})
            assert env["SAVE"].startswith(e.root)
            assert env["OV2_ETP"] == str(e.case(tp)["etp"])
            assert env["OV2_SEQ_LEN"] == "73728"
            assert env["OV2_MIDTRAIN_N_SAMPLES"] == str(lc.BUDGETS[stage])
            assert env["ITERS"] == "8"
            assert not {"OV2_FREEZE_LLM", "OV2_MTP_LOSS_SCALE", "OV2_AB_INPUT_DIR"} & env.keys()
            overrides = dict(x.split("=", 1) for x in env["EXTRA_ARGS"].split())
            assert overrides["scheduler.lr_decay_iters"] == str(e.schedule)
            assert overrides["optimizer.adam_beta2"] == ("0.95" if stage == 2 else "0.99")
            if phase == "resume":
                assert overrides["checkpoint.load"] == str(e.folder(tp, "split"))
                assert overrides["dataset.dataloader_load"] == str(e.folder(tp, "split"))
                assert env["INIT_CKPT"].endswith("split/iter_0000005")
            if phase != "split":
                assert overrides["checkpoint.save"] == "null"
            else:
                assert overrides["train.exit_interval"] == "5"


@pytest.fixture
def base_launcher(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src/megatron/bridge").mkdir(parents=True)
    patcher = repo / "3rdparty/apply_megatron_patch.sh"
    patcher.parent.mkdir(parents=True)
    patcher.write_text("exit 0\n")
    marker = repo / "3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("_HYBRID_EP_PAD_INFO\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    capture = tmp_path / "capture.json"
    stub = binaries / "python"
    stub.write_text(
        f"#!{sys.executable}\nimport os,sys,json\nfrom pathlib import Path\n"
        f"if '-m' in sys.argv: Path({str(capture)!r}).write_text(json.dumps(dict(argv=sys.argv,env=dict(os.environ))))\n"
    )
    stub.chmod(0o755)

    def run(e, *, tp, phase="reference", overrides=None):
        env = lc.phase_environment(e, tp=tp, phase=phase, assets={})
        env.update(
            REPO=str(repo),
            PET_NNODES=str(e.nodes),
            PET_NODE_RANK="0",
            MASTER_ADDR="127.0.0.1",
            PATH=f"{binaries}:/usr/bin:/bin",
            TRITON_CACHE_DIR=str(tmp_path / "triton"),
            TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
        )
        if overrides:
            env.update(overrides)
        if capture.exists():
            capture.unlink()
        result = subprocess.run(
            ["bash", str(FOLDER / "ax_ov2_qwen35_35b_a3b_gb200.sh")],
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        return result, json.loads(capture.read_text()) if capture.exists() else None

    return run


@pytest.mark.parametrize("world", [48, 64])
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("stage", [2, 3])
@pytest.mark.parametrize("phase", lc.PHASES)
def test_real_launcher_hydra_and_cap(tmp_path, base_launcher, world, tp, stage, phase):
    e = experiment(tmp_path, world=world, stage=stage)
    result, captured = base_launcher(e, tp=tp, phase=phase)
    assert result.returncode == 0, result.stdout + result.stderr
    args = dict(x.split("=", 1) for x in captured["argv"] if "=" in x)
    env = captured["env"]
    assert args["model.tensor_model_parallel_size"] == str(tp)
    assert args["model.expert_tensor_parallel_size"] == str(e.case(tp)["etp"])
    assert args["train.train_iters"] == str(e.steps)
    assert args["scheduler.lr_decay_iters"] == str(e.schedule)
    assert args["checkpoint.load"] == str(e.folder(tp, "split" if phase == "resume" else phase))
    assert env["OV2_FLEX_BACKEND"] == ("hybridep" if tp == 4 else "")
    if tp == 4:
        assert env["HYBRID_EP_MAX_TOKENS_PER_RANK"] == "18432"


def test_reject_48_tp4_without_expert_folding(tmp_path, base_launcher):
    result, captured = base_launcher(experiment(tmp_path), tp=4, overrides={"OV2_ETP": ""})
    assert result.returncode != 0 and captured is None
    assert "ETP=4 * EP=8" in result.stderr


def test_hybrid_domain_uses_expert_tp(tmp_path, base_launcher):
    e = experiment(tmp_path)
    result, _ = base_launcher(e, tp=4, overrides={"OV2_HYBRIDEP_NVLINK_DOMAIN_RANKS": "32"})
    assert result.returncode != 0 and "ETPxEP=16" in result.stderr
    result, _ = base_launcher(e, tp=4, overrides={"OV2_HYBRIDEP_NVLINK_DOMAIN_RANKS": "16"})
    assert result.returncode == 0, result.stderr


def make_checkpoint(e, tp):
    root = e.folder(tp, "split")
    folder = root / f"iter_{e.split:07d}"
    folder.mkdir(parents=True, exist_ok=True)
    for name in (".metadata", "metadata.json"):
        (folder / name).touch()
    state = dict(step=torch.tensor(e.split), consumed_train_samples=torch.tensor(e.split * e.gbs))
    torch.save(state, folder / "train_state.pt")
    torch.save(state, root / "latest_train_state.pt")
    (root / "latest_checkpointed_iteration.txt").write_text(str(e.split))
    for rank in range(e.case(tp)["dp"]):
        (folder / f"train_dataloader_dprank{rank:03d}.pt").write_bytes(b"cursor")
    config = dict(
        model=dict(
            tensor_model_parallel_size=tp,
            expert_tensor_parallel_size=e.case(tp)["etp"],
            expert_model_parallel_size=8,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        ),
        train=dict(global_batch_size=e.gbs, micro_batch_size=1, train_iters=e.steps),
        checkpoint=dict(save_optim=True, save_rng=True, ckpt_format="torch_dist"),
        optimizer=dict(optimizer="dist_muon", lr=1e-5 if e.stage == 2 else 2e-5),
        scheduler=dict(lr_decay_iters=e.schedule, lr_warmup_iters=e.schedule * 2 // 1000),
    )
    (folder / "run_config.yaml").write_text(yaml.safe_dump(config))
    return folder


@pytest.mark.parametrize("world,tp", [(48, 4), (48, 2), (64, 4), (64, 2)])
def test_checkpoint_gate(tmp_path, world, tp):
    e = experiment(tmp_path, world=world)
    folder = make_checkpoint(e, tp)
    assert lc.validate_checkpoint(e, tp)["step"] == e.split
    (folder / "train_dataloader_dprank000.pt").unlink()
    with pytest.raises(ValueError, match="cursors"):
        lc.validate_checkpoint(e, tp)


def test_checkpoint_rejects_tracker_disagreement(tmp_path):
    e = experiment(tmp_path)
    folder = make_checkpoint(e, 4)
    (folder.parent / "latest_checkpointed_iteration.txt").write_text("9")
    with pytest.raises(ValueError, match="trackers disagree"):
        lc.validate_checkpoint(e, 4)


def write_evidence(e, *, tp, phase, node):
    folder = e.folder(tp, phase)
    (folder / "evidence").mkdir(parents=True, exist_ok=True)
    first = e.split + 1 if phase == "resume" else 1
    last = e.split if phase == "split" else e.steps
    mb = e.gbs // e.case(tp)["dp"]
    for rank in range(node * 4, node * 4 + 4):
        runtime = lc.expected_runtime(e, tp=tp, phase=phase)
        runtime["mismatches"] = {}
        (folder / "evidence" / f"runtime-rank{rank:03d}.json").write_text(json.dumps(runtime))
        rows = [
            dict(
                index=i + 1,
                digest=f"dp{rank // tp}-batch{(first - 1) * mb + i}",
                tokens=64000,
                payload_tokens=64000,
                loss_tokens=1000,
                patches=20000,
                temporal_max=10,
            )
            for i in range((last - first + 1) * mb)
        ]
        (folder / "evidence" / f"inputs-rank{rank:03d}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    with (folder / f"train_node{node}.log").open("w") as stream:
        if tp == 4:
            for rank in range(node * 4, node * 4 + 4):
                stream.write(f"[OV2-HYBRID-AG] rank={rank} custom=1 domains=1\n")
        if node == e.nodes - 1:
            for i in range(first, last + 1):
                stream.write(
                    f"[2026-09-07 00:{i:02d}:00] iteration {i}/{e.steps} | consumed samples: {i * e.gbs} | "
                    f"elapsed time per iteration (ms): 1000 | global batch size: {e.gbs} | "
                    "lm loss: 1.25 | grad norm: 12.0 | learning rate: 1e-5 | "
                    "number of skipped iterations: 0 | number of nan iterations: 0 |\n"
                )


def test_resume_report_rejects_data_restart_and_numeric_drift(tmp_path):
    e = experiment(tmp_path)
    for phase in lc.PHASES:
        for node in range(e.nodes):
            write_evidence(e, tp=4, phase=phase, node=node)
    result = lc.compare(e, 4)
    assert result["resume_screen_passed"] and not result["full_gradient_or_routing_parity_proven"]
    for rank in range(4):
        path = e.folder(4, "resume") / f"evidence/inputs-rank{rank:03d}.jsonl"
        path.write_text(path.read_text().replace("batch10", "batch0"))
    result = lc.compare(e, 4)
    assert result["input_metadata_mismatch_ranks"] == [0, 1, 2, 3] and not result["resume_screen_passed"]
    path = e.folder(4, "resume") / f"train_node{e.nodes - 1}.log"
    path.write_text(path.read_text().replace("learning rate: 1e-5", "learning rate: 2e-5"))
    assert not lc.compare(e, 4)["paired_metrics"]["learning rate"]["within_screening_tolerance"]


def test_runtime_propagation_reads_inner_model(tmp_path, monkeypatch):
    name = "megatron.bridge.models.qwen_vl_ov2.ablation_inputs"
    path = REPO / "src/megatron/bridge/models/qwen_vl_ov2"
    spec = importlib.util.spec_from_file_location(name, path / "ablation_inputs.py")
    inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inputs)
    monkeypatch.setitem(sys.modules, name, inputs)
    spec = importlib.util.spec_from_file_location("long_context_probe_test", path / "long_context_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    monkeypatch.setenv("OV2_LONG_CONTEXT_PROBE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    batch = {
        "tokens": torch.zeros(1, 64000, dtype=torch.int64),
        "cu_seqlens": torch.tensor([0, 32000, 64000]),
        "pixel_values": torch.zeros(10, 4),
    }
    probe.record_batch(batch)
    row = json.loads((tmp_path / "inputs-rank000.jsonl").read_text())
    assert row["payload_tokens"] == 64000 and row["patches"] == 10
    probe._COUNTS.clear()
    with pytest.raises(FileExistsError):
        probe.record_batch(batch)
    inner = SimpleNamespace(
        tensor_model_parallel_size=4,
        expert_tensor_parallel_size=4,
        expert_model_parallel_size=8,
        moe_token_dispatcher_type="flex",
        recompute_granularity="selective",
    )
    component = SimpleNamespace(config=inner, parameters=lambda: [torch.nn.Parameter(torch.ones(1))])
    model = SimpleNamespace(
        module=SimpleNamespace(language_model=component, vision_model=component, adapter=component)
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(seq_length=73728),
        train=SimpleNamespace(global_batch_size=24, micro_batch_size=1),
        checkpoint=SimpleNamespace(finetune=False, load_optim=True, load_rng=True, load="/test/split"),
        optimizer=SimpleNamespace(optimizer="dist_muon"),
        scheduler=SimpleNamespace(use_checkpoint_opt_param_scheduler=True),
    )
    state = SimpleNamespace(cfg=cfg, train_state=SimpleNamespace(step=60, consumed_train_samples=1440))
    parallel = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 4,
        get_expert_tensor_parallel_world_size=lambda: 2,
        get_expert_model_parallel_world_size=lambda: 8,
        get_data_parallel_world_size=lambda: 12,
        get_expert_data_parallel_world_size=lambda: 3,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 48)
    snapshot = probe.runtime_snapshot(state, model, parallel)
    assert snapshot["etp"] == 2 and snapshot["inner_etp"] == 4  # Detect a stale inner builder independently.
    assert snapshot["step"] == 60 and snapshot["samples"] == 1440


@pytest.mark.parametrize("field,value", [("payload_tokens", 8000), ("temporal_max", 0)])
def test_report_rejects_short_or_image_only_work(tmp_path, field, value):
    e = experiment(tmp_path)
    for node in range(e.nodes):
        write_evidence(e, tp=4, phase="reference", node=node)
    for path in (e.folder(4, "reference") / "evidence").glob("inputs-*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row[field] = value
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="temporal patches"):
        lc.summarize_phase(e, tp=4, phase="reference")


@pytest.mark.parametrize("world,tp", [(32, 1), (32, 2), (64, 1), (64, 2)])
def test_existing_midtrain_default_etp_unchanged(tmp_path, base_launcher, world, tp):
    e = experiment(tmp_path)
    result, captured = base_launcher(
        e,
        tp=tp,
        overrides={"PET_NNODES": str(world // 4), "OV2_ETP": "", "OV2_MIDTRAIN_GBS": "256", "OV2_SEQ_LEN": "10192"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(x.startswith("model.expert_tensor_parallel_size=") for x in captured["argv"])


@pytest.fixture
def cluster(tmp_path):
    e = experiment(tmp_path, tps=(4,))
    repo = tmp_path / "repo"
    patcher = repo / "3rdparty/apply_megatron_patch.sh"
    patcher.parent.mkdir(parents=True)
    patcher.write_text("exit 0\n")
    fake = tmp_path / "launchers"
    fake.mkdir()
    worker = fake / "worker.py"
    worker.write_text(
        f"import sys,os,json\nsys.path.insert(0,{str(Path(__file__).parent)!r})\n"
        "from test_long_context_matrix import lc,write_evidence,make_checkpoint\n"
        "from pathlib import Path\n"
        "folder=Path(os.environ['SAVE']); root=folder.parent.parent\n"
        "v=json.loads((root/'experiment.json').read_text());v['tps']=tuple(v['tps']);e=lc.Experiment(**v)\n"
        "phase=folder.name;node=int(os.environ['PET_NODE_RANK']);tp=int(os.environ['TP'])\n"
        "if os.environ.get('FAIL_NODE') == str(node) and phase=='split': sys.exit(7)\n"
        "write_evidence(e,tp=tp,phase=phase,node=node)\n"
        "if phase=='split' and node==0: make_checkpoint(e,tp)\n"
    )
    (fake / "ax_ov2_qwen35_35b_a3b_gb200.sh").write_text(f'exec "{sys.executable}" "{worker}"\n')
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"import sys,json\nsys.path.insert(0,{str(FOLDER)!r})\nimport run_long_context_matrix as lc\n"
        f"lc.REPO=lc.Path({str(repo)!r});lc.HERE=lc.Path({str(fake)!r})\n"
        f"e=lc.Experiment(**{as_contract(e)!r})\n"
        "node=int(lc.os.environ['PET_NODE_RANK'])\n"
        "try:\n"
        " lc.join(e,node=node,manifest={})\n"
        " for p in lc.PHASES: lc.run_phase(e,tp=4,phase=p,node=node,assets={})\n"
        " if node==0: lc._write(lc.Path(e.root)/'comparison.json',lc.compare(e,4))\n"
        " lc._wait([lc.Path(e.root)/'comparison.json'],root=lc.Path(e.root),seconds=20)\n"
        "except BaseException as exc:\n"
        " lc._write(lc.Path(e.root)/'control'/f'failed-{node}.json',{'error':str(exc)})\n"
        " raise\n"
    )

    def launch(*, fail=False):
        processes = []
        outputs = []
        for node in range(e.nodes):
            log = tmp_path / f"node{node}.log"
            output = log.open("w")
            outputs.append((log, output))
            env = {**os.environ, "PET_NODE_RANK": str(node)}
            if fail:
                env["FAIL_NODE"] = "2"
            processes.append(
                subprocess.Popen([sys.executable, str(driver)], env=env, stdout=output, stderr=subprocess.STDOUT)
            )
        try:
            codes = [p.wait(timeout=60) for p in processes]
        finally:
            for p in processes:
                if p.poll() is None:
                    p.kill()
                p.wait()
            for _, output in outputs:
                output.close()
        return codes, "\n".join(log.read_text() for log, _ in outputs)

    return e, launch


def as_contract(e):
    return lc.asdict(e)


def test_twelve_pod_save_relaunch_report(cluster):
    e, launch = cluster
    codes, logs = launch()
    assert codes == [0] * e.nodes, logs
    assert json.loads((Path(e.root) / "comparison.json").read_text())["resume_screen_passed"]


def test_worker_failure_never_releases_resume(cluster):
    e, launch = cluster
    codes, logs = launch(fail=True)
    assert all(c != 0 for c in codes), logs
    assert not (e.folder(4, "resume") / "train_node0.log").exists()


def test_wrapper_rejects_production_overrides():
    result = subprocess.run(
        ["bash", str(FOLDER / "ax_ov2_qwen35_long_context_matrix.sh"), "SAVE=/production"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "unsupported long-context argument" in result.stderr
