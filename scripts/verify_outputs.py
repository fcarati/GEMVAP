"""Check whether any file in OUTPUT_DIR has changed since the last notebook run.

The notebook writes a manifest (sha256 per file) to OUTPUT_DIR/.output_manifest.json
at the end of every run. This script recomputes the current hashes and diffs them
against that manifest, so cached outputs that are being reused (rather than
regenerated) can be trusted -- or flagged if something changed outside the notebook.

Usage (from gemvap_clean_pipeline/):
    python scripts/verify_outputs.py [output_dir]

Exit code 0 = nothing changed (or no baseline existed yet, one was just created).
Exit code 1 = added / removed / changed files were found.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from output_archive import compute_manifest, diff_manifest, load_manifest, save_manifest

DEFAULT_OUTPUT_DIR = "output/gemvap_notebook_gnomad4"


def verify(output_dir) -> bool:
    """Print a diff report against the saved manifest. Returns True if clean."""
    output_dir = Path(output_dir)
    current = compute_manifest(output_dir)
    baseline = load_manifest(output_dir)

    if baseline is None:
        print(f"No manifest found in {output_dir} -- treating current state as the baseline.")
        save_manifest(output_dir, current)
        print(f"Saved baseline manifest for {len(current)} file(s).")
        return True

    diff = diff_manifest(baseline, current)
    if not any(diff.values()):
        print(f"OK -- all {len(current)} file(s) in {output_dir} match the saved manifest.")
        return True

    print(f"CHANGES DETECTED in {output_dir}:")
    for label in ("added", "removed", "changed"):
        for rel in diff[label]:
            print(f"  [{label}] {rel}")
    return False


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    ok = verify(target)
    sys.exit(0 if ok else 1)
