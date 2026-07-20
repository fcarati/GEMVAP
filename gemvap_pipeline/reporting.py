"""Simple inventory helpers for reporting on pipeline output."""
from pathlib import Path

from .verbose import result, step


def list_output_files(output_dir) -> int:
    """Print every file under output_dir (relative path + size in KB) and
    return the file count."""
    output_dir = Path(output_dir)
    step("Listing every file under OUTPUT_DIR")
    n_files = 0
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(output_dir)
            size_kb = p.stat().st_size / 1024
            print(f"  {rel}  ({size_kb:.1f} KB)")
            n_files += 1
    result(f"{n_files} file(s) under {output_dir}")
    return n_files
