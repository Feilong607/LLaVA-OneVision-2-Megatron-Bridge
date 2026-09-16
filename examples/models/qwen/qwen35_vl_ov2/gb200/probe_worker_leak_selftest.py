# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local dry run of probe_worker_leak.main() with fake torch/energon/bridge and a synthetic retention (a suspended
generator per sample). No GPU, no cluster, no torch needed -- run before every push of the probe:

  python3 examples/models/qwen/qwen35_vl_ov2/gb200/probe_worker_leak_selftest.py

It must print the holder chain naming generator[_hold ...] and end with SELFTEST OK."""
import importlib.util
import os
import sys
import tempfile
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))

def fake(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); m.__path__ = []; sys.modules[name] = m; return m

class FakeTensor:
    def __init__(self, n): self.n = n; self.device = types.SimpleNamespace(type="cpu")
    def numel(self): return self.n
    def element_size(self): return 4
torch = fake("torch", __version__="0.0-fake", is_tensor=lambda o: isinstance(o, FakeTensor), get_num_threads=lambda: 8)
fake("transformers", __version__="5.3.0-fake")
try:
    import PIL  # real if present
except ImportError:
    fake("PIL", __version__="fake"); fake("PIL.Image")
try:
    import numpy
except ImportError:
    fake("numpy", __version__="fake")

class Image:                       # PIL-like retained object
    __module__ = "PIL.Image"
    def __init__(self): self.mode, self.size = "RGB", (640, 480)
    def getbands(self): return ("R", "G", "B")
class PackedCaptioningSample:
    def __init__(self, imgs): self.images = imgs

# Run 6 died on an object whose type's __module__ is not a str (metaclass descriptor) -- keep such objects alive.
class Meta(type):
    __module__ = property(lambda cls: object())
class Weird(metaclass=Meta):
    pass
WEIRD = [Weird() for _ in range(3)]

# More zoo: things the census / walks touch on a real torch+transformers heap.
class BadRepr:                      # dict key whose repr raises (tensor-like keys, broken __repr__)
    def __repr__(self): raise RuntimeError("no repr")
class BadGetattr:                   # hasattr() raising something other than AttributeError
    def __getattr__(self, name): raise RuntimeError("no attrs")
class PILish:                       # PIL-module object with mode + unhashable size
    __module__ = "PIL.ImageDraw"
    def __init__(self): self.mode, self.size = "RGB", [1, 2]
class BadDict:                      # __dict__ that is not a mapping
    @property
    def __dict__(self): raise RuntimeError("no dict")
ZOO = [BadGetattr(), PILish(), BadDict()]

LEAK = []                          # simulate the production retention: a suspended generator per sample
def _hold(sample):
    for img in sample.images:
        yield img

BADKEYED = {}                       # a retained image reachable only through a dict with an unprintable key

class FakeLoader:
    def __iter__(self):
        while True:
            s = PackedCaptioningSample([Image() for _ in range(3)])
            g = _hold(s); next(g); LEAK.append(g)
            BADKEYED[BadRepr()] = Image()
            yield {"tokens": FakeTensor(100), "pixel_values": FakeTensor(1000), "cu_seqlens": None, "list": [FakeTensor(5)]}

class WorkerConfig:
    @staticmethod
    def default_worker_config(n): return "wc"
fake("megatron"); fake("megatron.bridge"); fake("megatron.bridge.recipes"); fake("megatron.bridge.recipes.ov2")
fake("megatron.bridge.recipes.ov2.data")
fake("megatron.bridge.recipes.ov2.data.energon").__path__ = [os.path.join(REPO, "src/megatron/bridge/recipes/ov2/data/energon")]  # real energon_patches
fake("megatron.energon", __version__="7.4.1-fake", WorkerConfig=WorkerConfig,
     get_train_dataset=lambda *a, **k: "ds", get_savable_loader=lambda ds, worker_config=None: FakeLoader())
fake("megatron.bridge.recipes.ov2.ov2_qwen35", _QWEN35_BACKBONE="qwen3.5-35b-a3b")
fake("megatron.bridge.recipes.ov2.ov2", _OV2_BACKBONES={"qwen3.5-35b-a3b": {"hf_proc": "/nonexistent"}})
class OV2TaskEncoder:
    def __init__(self, hf_processor_path, seq_length, spatial_merge_size=None):
        ip = type("Qwen2VLImageProcessorFast", (), {})()
        self.proc = types.SimpleNamespace(image_processor=ip, tokenizer=types.SimpleNamespace())
fake("megatron.bridge.recipes.ov2.data.energon.task_encoder", OV2TaskEncoder=OV2TaskEncoder)

spec = importlib.util.spec_from_file_location("probe", f"{REPO}/examples/models/qwen/qwen35_vl_ov2/gb200/probe_worker_leak.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
_tmp = tempfile.mkdtemp(); open(os.path.join(_tmp, "preprocessor_config.json"), "w").write("{}")
sys.argv = ["probe", "--n", "6", "--every", "3", "--proc", _tmp, "--data", "fake.yaml"]
m.main()

# ---- --fix path, phase A: fake energon has no wrappers -> patch must report NOT applied and the run must continue
sys.argv = ["probe", "--n", "3", "--every", "3", "--proc", _tmp, "--data", "fake.yaml", "--fix", "--no-thp", "--trim-every", "2"]
m.main()

# ---- --fix path, phase B: fake wrappers shaped exactly like energon 7.4.1 -> patch applies, frames stop pinning
_wr = tempfile.mkdtemp()
open(os.path.join(_wr, "repeat_dataset.py"), "w").write(
    "import math\n"
    "class RepeatDataset:\n"
    "    def __init__(self, ds): self.dataset = ds; self._index = 0; self.repeats = 1; self._repetition = 0\n"
    "    def __iter__(self):\n"
    "        while self._repetition < self.repeats:\n"
    "            for sample in self.dataset:\n"
    "                self._index += 1\n"
    "                yield sample\n"
    "            self._repetition += 1\n")
open(os.path.join(_wr, "map_dataset.py"), "w").write(
    "def add_sample_restore_key(sample, *a, **k): return sample\n"
    "class _H:\n"
    "    def reset(self): pass\n"
    "class MapDataset:\n"
    "    def __init__(self, ds, fn): self.dataset = ds; self.map_fn = fn; self._map_failure_handler = _H()\n"
    "    def __iter__(self):\n"
    "        for sample in self.dataset:\n"
    "            if True:\n"
    "                sample_idx = 0\n"
    "                mapped_sample = self.map_fn(sample)\n"
    "                if False:\n"
    "                    pass\n"
    "                else:\n"
    "                    self._map_failure_handler.reset()\n"
    "                    yield add_sample_restore_key(\n"
    "                        mapped_sample,\n"
    "                        sample_idx,\n"
    "                        src=self,\n"
    "                    )\n")
fake("megatron.energon.wrappers").__path__ = [_wr]
from megatron.bridge.recipes.ov2.data.energon.energon_patches import apply_drop_yielded_patch
assert apply_drop_yielded_patch() is True, "patch did not apply on 7.4.1-shaped fakes"
import gc
from megatron.energon.wrappers.repeat_dataset import RepeatDataset
from megatron.energon.wrappers.map_dataset import MapDataset
class Raw:            # the undecoded tar sample (bytes) -- stays bound in MapDataset's frame, accepted
    pass
class Decoded:        # what map_fn (decode) produces -- the heavy object the patch must release
    def __init__(self, raw): self.raw = raw
raws = [Raw() for _ in range(3)]
decoded_seen = []
def _decode(r):
    d = Decoded(r); decoded_seen.append(d); return d
chain = RepeatDataset(MapDataset(raws, _decode))             # RepeatDataset -> MapDataset(decode) -> shard list
it = iter(chain); first = next(it)
assert first is decoded_seen[0]
pinned = [r for r in gc.get_referrers(decoded_seen[0]) if type(r).__name__ == "generator"]
assert not pinned, f"patched generators still pin the DECODED sample: {pinned}"
assert [x.raw is y for x, y in zip(RepeatDataset(MapDataset(raws, _decode)), raws)] == [True, True, True]
print("[selftest] drop_yielded: applied on 7.4.1-shaped fakes, no generator pins the yielded sample, order intact")
# ---- smaps parser on a synthetic Linux smaps (macOS has no /proc)
_smaps = (
    "55d0c0000000-55d0c8000000 rw-p 00000000 00:00 0                          [heap]\n"
    "Rss:              131072 kB\nAnonHugePages:      2048 kB\n"
    "7f0000000000-7f0010000000 rw-p 00000000 00:00 0 \n"
    "Rss:               98304 kB\nAnonHugePages:         0 kB\n"
    "7f1000000000-7f1000001000 rw-s 00000000 00:05 12345                      /memfd:torch (deleted)\n"
    "Rss:                4096 kB\n"
    "7f2000000000-7f2000100000 r-xp 00000000 08:01 999                        /usr/lib/libc.so.6\n"
    "Rss:                1024 kB\n"
)
_ps = m._parse_smaps(_smaps.splitlines(keepends=True))
assert _ps == {"heap": 128, "anon": 96, "shm": 4, "file": 1, "thp": 2, "anon_ge64M": 1}, _ps
print("[selftest] smaps parser OK", _ps)
print("SELFTEST OK")
