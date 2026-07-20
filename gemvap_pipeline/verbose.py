"""
Shared formatting helpers for pipeline-wide verbose output.

Usage:
    from .verbose import section, step, result, info, note

All output goes to stdout via plain print() so it appears
inline in Jupyter notebook cells.
"""

_W = 68  # column width for section dividers


def section(title: str) -> None:
    """Top-level section header — marks the start of a major pipeline stage."""
    print(f"\n{'=' * _W}")
    print(f"  {title}")
    print(f"{'=' * _W}")


def substep(title: str) -> None:
    """Sub-section divider within a stage."""
    print(f"\n  {'-' * (_W - 2)}")
    print(f"  {title}")
    print(f"  {'-' * (_W - 2)}")


def step(msg: str, indent: int = 1) -> None:
    """An action the pipeline is about to take."""
    print(f"{'  ' * indent}>> {msg}")


def result(msg: str, indent: int = 1) -> None:
    """Outcome or count produced by the previous step."""
    print(f"{'  ' * indent}-> {msg}")


def info(msg: str, indent: int = 2) -> None:
    """Contextual explanation — the 'why' behind a step."""
    print(f"{'  ' * indent}[i] {msg}")


def note(msg: str, indent: int = 2) -> None:
    """A caution or edge-case remark."""
    print(f"{'  ' * indent}[!] {msg}")
