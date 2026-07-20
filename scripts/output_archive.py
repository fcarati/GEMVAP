"""Archive/verification helpers for the notebook's OUTPUT_DIR.

Backs the "archive & regenerate" cell in GEMVAP_Pipeline_gnomAD4.ipynb: either
moves every existing output file out of the way so the pipeline regenerates
everything from scratch, or -- if the user opts to keep the cached outputs --
hashes them against the manifest saved at the end of the last run, so silent
drift (a file edited or deleted outside the notebook) is caught instead of
trusted blindly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

MANIFEST_NAME = ".output_manifest.json"


def _iter_files(output_dir):
    output_dir = Path(output_dir)
    for p in sorted(output_dir.rglob("*")):
        if p.is_file() and p.name != MANIFEST_NAME:
            yield p


def compute_manifest(output_dir) -> dict:
    """Map each file under output_dir (relative posix path) to its sha256 hash."""
    output_dir = Path(output_dir)
    return {
        p.relative_to(output_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in _iter_files(output_dir)
    }


def load_manifest(output_dir):
    path = Path(output_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_manifest(output_dir, manifest: dict) -> None:
    path = Path(output_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def diff_manifest(old: dict, new: dict) -> dict:
    old_keys, new_keys = set(old), set(new)
    return {
        "added": sorted(new_keys - old_keys),
        "removed": sorted(old_keys - new_keys),
        "changed": sorted(k for k in old_keys & new_keys if old[k] != new[k]),
    }


def archive_output_dir(output_dir, archive_root):
    """Move every file currently in output_dir into a timestamped folder
    under archive_root, leaving output_dir empty so the pipeline regenerates
    everything from scratch. Returns the archive path, or None if output_dir
    had nothing to move.
    """
    output_dir = Path(output_dir)
    files = list(_iter_files(output_dir))
    old_manifest = output_dir / MANIFEST_NAME
    if not files and not old_manifest.exists():
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(archive_root) / f"{output_dir.name}_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)

    for p in files:
        rel = p.relative_to(output_dir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(target))

    if old_manifest.exists():
        shutil.move(str(old_manifest), str(dest / MANIFEST_NAME))

    # Drop now-empty subdirectories left behind by the move.
    for p in sorted(output_dir.rglob("*"), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()

    return dest
