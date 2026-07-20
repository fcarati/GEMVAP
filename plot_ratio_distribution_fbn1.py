"""
Reproduce the per-reference-amino-acid gnomAD4 substitution-ratio histogram
figure for FBN1 from `data/raw/ratio_list_gnomad4_000005.csv`.

Each cell of the raw CSV is a stringified {substituted_aa: pct} dict (percent
of substitutions from a given reference amino acid, in a given gene, that
were observed in gnomAD4 at the 0.00005 frequency threshold). This script
collapses each cell to its mean to get one scalar ratio per (reference aa,
gene), then -- for each reference amino acid -- plots the distribution of
that ratio across genes, with FBN1's value and the cross-gene mean marked.
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "ratio_list_gnomad4_000005.csv"
OUTPUT_PATH = ROOT / "output" / "ratio_distribution_fbn1_gnomad4_000005.png"

GENE = "FBN1"
AA_ORDER = "MKQAVIHTEPLSRDNFYGWC"
BINS = list(range(0, 50, 5))


def _parse_ratio_cell(cell):
    d = ast.literal_eval(cell) if isinstance(cell, str) else cell
    if not d:
        return np.nan
    return float(np.mean(list(d.values())))


def load_flat_ratio_matrix(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path).set_index("Unnamed: 0")
    return raw.map(_parse_ratio_cell)


def _nearest_bin_height(counts, edges, value):
    """Height of the histogram bin whose left edge is the closest edge <= value."""
    if value is None or pd.isna(value):
        return None
    candidates = [(value - e, i) for i, e in enumerate(edges) if value - e >= 0]
    if not candidates:
        return None
    _, idx = min(candidates, key=lambda t: t[0])
    return counts[min(idx, len(counts) - 1)]


def main() -> None:
    flat_df = load_flat_ratio_matrix(DATA_PATH)

    if flat_df[GENE].isna().all():
        print(
            f"Warning: every '{GENE}' ratio in {DATA_PATH.name} is empty "
            "({}) -- no substitutions passed the 0.00005 threshold for this "
            "gene, so no FBN1 marker line can be drawn on any panel."
        )

    fig, axis = plt.subplots(7, 3, figsize=(20, 30))
    num_1, num_2 = 0, 0

    for aa in AA_ORDER:
        res = flat_df.loc[aa].dropna().tolist()
        gene_value = flat_df.loc[aa, GENE]
        bins = None if aa == "R" else BINS

        ax = axis[num_2, num_1]
        counts, edges, _ = ax.hist(
            res, bins=bins, color=(0.1, 0.1, 0.9, 0.1), edgecolor="lightgrey"
        )

        gene_y = _nearest_bin_height(counts, edges, gene_value)
        if gene_y is not None:
            ax.plot([gene_value, gene_value], [0, gene_y], color="black", marker="o")

        mean_value = float(np.mean(res)) if res else np.nan
        mean_y = _nearest_bin_height(counts, edges, mean_value)
        if mean_y is not None:
            ax.plot([mean_value, mean_value], [0, mean_y], color="goldenrod", marker="o")

        ax.set_title(aa)

        num_1 += 1
        if num_1 > 2:
            num_1 = 0
            num_2 += 1

    for j in range(num_1, 3):
        axis[num_2, j].axis("off")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
