#!/usr/bin/env python3
"""Answer "does this torch_dist iter dir carry tensor KEY?" from its metadata alone (no GPU, no weights read).

    python3 ckpt_has_key.py <iter_dir> <key> [<key> ...]

Prints one line per key: ``<key> 1`` or ``<key> 0``; exit 0 when every key is present, 1 when any is absent,
2 on a read error. Export pipelines use it to derive the HF skeleton from what the SAVE actually contains
instead of from assumptions -- e.g. ``vision_model.decoder.final_layernorm.weight``: mcore only builds the
vision tower's final LayerNorm when ``config.mtp_num_layers`` is not None (TransformerBlock
``has_final_layernorm_in_this_stage``), so the s1.5 SAVE (MTP head) has it and an ``OV2_MTP_LAYERS=0`` SAVE
does not; the HF ``vision_config.use_post_layernorm`` flag must match or the export fails its structure check
(or, worse, applies an untrained LayerNorm at inference).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    iter_dir = Path(sys.argv[1])
    if not (iter_dir / ".metadata").is_file():
        print(f"FATAL: {iter_dir} has no .metadata (not a torch_dist checkpoint dir)", file=sys.stderr)
        return 2
    try:
        from torch.distributed.checkpoint import FileSystemReader

        keys = set(FileSystemReader(str(iter_dir)).read_metadata().state_dict_metadata.keys())
    except Exception as exc:  # noqa: BLE001 -- any reader failure is a hard error for the caller
        print(f"FATAL: cannot read {iter_dir}/.metadata: {exc!r}", file=sys.stderr)
        return 2
    rc = 0
    for key in sys.argv[2:]:
        present = key in keys
        print(f"{key} {int(present)}")
        if not present:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
