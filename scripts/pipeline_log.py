"""Console-narration helpers for the GEMVAP notebook.

Every pipeline step prints through these functions instead of hand-rolled
``print("=" * 70)`` calls, so the shape of the output (banner width, prefixes,
table formatting) is identical everywhere: ``>>`` actions, ``[i]`` reasoning,
``->`` results, ``[!]`` warnings.
"""

_WIDTH = 70


def step(n, title):
    """Print the banner that opens step ``n``."""
    line = "=" * _WIDTH
    print(line)
    print(f"  Step {n} — {title}")
    print(line)


def inputs(items):
    """List every file/table this step reads.

    Each item is either a plain path string, or a ``(path, note)`` tuple
    where ``note`` explains what the file is / which earlier step wrote it.
    """
    print("  Inputs:")
    for item in items:
        if isinstance(item, tuple):
            path, note = item
            print(f"    - {path}  ({note})")
        else:
            print(f"    - {item}")


def subsection(title):
    """Print a named sub-part header within the current step."""
    line = "-" * _WIDTH
    print()
    print(f"  {line}")
    print(f"  {title}")
    print(f"  {line}")


def action(text):
    """State an operation right before it happens."""
    print(f"  >> {text}")


def info(text):
    """Explain the *why* behind the step, for connective context."""
    for line in str(text).splitlines():
        print(f"    [i] {line}")


def result(text):
    """Report the outcome of an action (counts, values, decisions)."""
    print(f"  -> {text}")


def warning(text):
    """Flag anything the reader must not miss (exclusions, caveats)."""
    for line in str(text).splitlines():
        print(f"    [!] {line}")


def summary_table(title, columns, rows):
    """Print a recap table.

    ``columns`` is a list of header strings; ``rows`` is a list of sequences
    (one per row) with the same length as ``columns``. The first column is
    left-aligned (labels), the rest are right-aligned (numbers).
    """
    str_rows = [[str(v) for v in row] for row in rows]
    widths = [
        max(len(columns[i]), *(len(r[i]) for r in str_rows)) if str_rows else len(columns[i])
        for i in range(len(columns))
    ]
    total_width = max(_WIDTH + 15, sum(widths) + 2 * (len(columns) - 1))

    def _fmt(row):
        cells = []
        for i, v in enumerate(row):
            cells.append(v.ljust(widths[i]) if i == 0 else v.rjust(widths[i]))
        return "  ".join(cells)

    print()
    print("=" * total_width)
    print(title.center(total_width))
    print("=" * total_width)
    print(_fmt(columns))
    print("-" * total_width)
    for row in str_rows:
        print(_fmt(row))
    print("=" * total_width)
