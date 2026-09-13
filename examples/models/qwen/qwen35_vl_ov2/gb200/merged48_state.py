# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic launch fingerprints and checkpoint archives for the merged48 wrapper."""

import argparse
import fcntl
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path


logger = logging.getLogger(__name__)


def publish_fingerprint(path: Path, stream: str, memory: str) -> None:
    """Publish once without truncating an existing fingerprint; verify every caller."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".fingerprint-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump({"stream": stream, "memory": memory}, output)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        saved = json.loads(path.read_text())
        if saved.get("stream") != stream or saved.get("memory") != memory:
            raise ValueError(f"launch fingerprint conflict: {path}; all pods must use the same settings")
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == ".archived":
            continue
        if path.is_symlink():
            raise ValueError(f"checkpoint symlinks are not supported: {path}")
        stat = path.stat()
        if path.is_file():
            result[relative] = ["file", stat.st_size, stat.st_mtime_ns]
        elif path.is_dir():
            result[relative] = ["directory"]
        else:
            raise ValueError(f"unsupported checkpoint entry: {path}")
    return result


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def archive_checkpoint(source: Path, destination: Path, *, dp: int) -> None:
    """Verify and atomically publish a complete archive, without replacing prior output."""
    source, destination = source.resolve(), destination.resolve()
    if destination == source.parent or source.parent in destination.parents:
        raise ValueError("archive destination must be outside the training SAVE directory")
    if destination.exists():
        marker = destination / ".archived"
        if not marker.is_file() or json.loads(marker.read_text()).get("manifest") != _manifest(destination):
            raise ValueError(f"existing archive is unmarked or inconsistent: {destination}")
        return
    required = {".metadata", "metadata.json", "train_state.pt", "run_config.yaml"}
    required.update(f"train_dataloader_dprank{rank:03d}.pt" for rank in range(dp))
    before = _manifest(source)
    if any(name not in before or before[name][0] != "file" for name in required):
        raise ValueError(f"checkpoint is not complete: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # flock is released by the OS on process death; a restart never inherits a stale lock.
    lock = (destination.parent / f".{destination.name}.archive-lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    temporary = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        shutil.copytree(source, temporary, dirs_exist_ok=True, copy_function=_link_or_copy)
        if _manifest(source) != before or _manifest(temporary) != before:
            raise ValueError(f"checkpoint changed or archive copy is incomplete: {source}")
        marker = temporary / ".archived"
        with marker.open("w") as output:
            json.dump({"source": str(source), "manifest": before}, output)
            output.flush()
            os.fsync(output.fileno())
        if destination.exists():
            raise FileExistsError(f"archive destination appeared during copy: {destination}")
        temporary.rename(destination)
        temporary = None
        logger.info("archived %s -> %s", source, destination)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        lock.close()


def main() -> None:
    """Run one fingerprint or archive operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fingerprint = commands.add_parser("fingerprint")
    fingerprint.add_argument("path", type=Path)
    fingerprint.add_argument("stream")
    fingerprint.add_argument("memory")
    archive = commands.add_parser("archive")
    archive.add_argument("source", type=Path)
    archive.add_argument("destination", type=Path)
    archive.add_argument("--dp", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "fingerprint":
        publish_fingerprint(args.path, args.stream, args.memory)
    else:
        if args.dp < 1:
            parser.error("--dp must be positive")
        archive_checkpoint(args.source, args.destination, dp=args.dp)


if __name__ == "__main__":
    main()
