"""
GEMVAP vs AlphaMissense vs REVEL Venn-diagram comparison on an independently
annotated variant cohort (Step 11 of the notebook).
"""
import pickle
from pathlib import Path

import pandas as pd
from matplotlib_venn import venn3

from .pejaver_tools import parse_pejaver_thresholds
from .verbose import note, result, step

MODEL_LABELS = {"GEMVAP_1": "GEMVAP 1", "GEMVAP_2": "GEMVAP 2", "GEMVAP_3": "GEMVAP 3"}
MODEL_COLORS = {"GEMVAP_1": "#E07B39", "GEMVAP_2": "#9B59B6", "GEMVAP_3": "#E74C3C"}
AM_COLOR = "#4C72B0"
REVEL_COLOR = "#55A868"
# AlphaMissense score thresholds (Cheng et al. 2023): >0.5642 = Pathogenic,
# <0.34 = Benign, the closed interval in between = Ambiguous. Applied to the
# raw 'pathogenicity score' column directly rather than trusted from
# AlphaMissense's own precomputed 'am_class' label.
AM_PATHOGENIC_THRESHOLD = 0.5642
AM_BENIGN_THRESHOLD = 0.34
# REVEL / GEMVAP / AlphaMissense palette matching the reference 3-way figure
# (salmon, soft green, lavender), used for the single "Any GEMVAP" summary
# Venn and the all-missense-vs-cohort comparison.
REVEL_GEMVAP_AM_COLORS = ("#F2A6A0", "#8FCB8F", "#B0A3E0")
# Dedicated triple-overlap (REVEL & GEMVAP & AlphaMissense) fill color for
# draw_true_label_venn_grid, used in place of matplotlib_venn's default
# auto-blended tan (0.4*(col1+col2+col3), which stays pale even at full
# alpha) so the "all three agree" region reads as a distinctly stronger grey.
_TRIPLE_OVERLAP_COLOR = "#5A5A5A"
PANEL_SPECS = [
    ("GEMVAP_1", "GEMVAP 1"),
    ("GEMVAP_2", "GEMVAP 2"),
    ("GEMVAP_3", "GEMVAP 3"),
    ("ANY", "Any GEMVAP (G1 OR G2 OR G3)"),
]


def load_alphamissense_hg38(alphamissense_path) -> pd.DataFrame:
    """
    Load AlphaMissense's genome-wide hg38 predictions, pre-filtered to the
    FBN1 region (chr15:48,400,000-48,650,000) from the full ~71M-row/5.5GB
    release -- see data/raw/AlphaMissense_hg38_FBN1.tsv, extracted with:
        awk -F'\\t' '/^#CHROM/{print;next} /^#/{next}
                     $1=="chr15" && $2>=48400000 && $2<=48650000{print}'
    Reading the full genome-wide file on every pipeline run would mean
    parsing 5.5 GB for a single gene, so the pipeline only ever reads this
    cached FBN1 extract, never AlphaMissense_hg38.tsv directly.

    Returns a frame with columns CHROM/POS/REF/ALT (genomic join key,
    unambiguous unlike the old protein-position-based join) and
    'pathogenicity score' (renamed from 'am_pathogenicity' so every
    downstream consumer -- build_full_predictor_table, run_part3_venn.py's
    ClinVar/gnomAD annotation step, venn_variant_breakdown.py -- keeps
    working against the same column name regardless of AlphaMissense
    source). Only FBN1's canonical transcript (ENST00000316623) appears in
    this region, so no transcript-based filtering or dedup is needed.
    """
    am = pd.read_csv(alphamissense_path, sep="\t")
    am.columns = am.columns.str.lstrip("#").str.strip()
    am["CHROM"] = pd.to_numeric(am["CHROM"].str.replace("chr", "", regex=False), errors="coerce")
    am = am.rename(columns={"am_pathogenicity": "pathogenicity score"})
    return am[["CHROM", "POS", "REF", "ALT", "pathogenicity score"]]


def load_gemvap_fits(output_dir, model_names=("GEMVAP_1", "GEMVAP_2", "GEMVAP_3")) -> dict:
    """Load each GEMVAP fit_gemvap*() result pickled by Steps 3-5."""
    fits = {}
    for name in model_names:
        fp = Path(output_dir) / f"{name}_fit.pkl"
        if not fp.exists():
            note(f"{fp} not found -- run Steps 3-5 first.")
            continue
        with open(fp, "rb") as fh:
            fits[name] = pickle.load(fh)
        result(f"{name}: {len(fits[name]['top_predictors'])} predictors")
    return fits


def load_venn_cohort(excel_path, data_path: str, fits: dict, alphamissense_path):
    """
    Load the independently annotated Excel cohort (pre-called REVEL_Prediction
    column -- confirmed equivalent to REVEL_score >= Pejaver PP3_Supporting)
    and merge in the predictor score columns needed by every loaded GEMVAP
    fit, joined by cDNA.

    AlphaMissense's pathogenic-call set is NOT taken from the Excel's own
    AM_SCORE column: cross-checking against AlphaMissense's own genomic
    predictions previously found 62 of 1346 rows where AM_SCORE disagreed
    with AlphaMissense's own score (53 of them AM_SCORE=True on a
    likely_benign score, some as low as 0.06). Instead, the raw AlphaMissense
    score is joined in from alphamissense_path (see load_alphamissense_hg38)
    by genomic coordinates (chr, pos, ref, alt) -- the same source and join
    keys load_all_missense_variants() uses for the all-missense population --
    and set_am is thresholded directly against that score at
    AM_PATHOGENIC_THRESHOLD (score > 0.5642), so the cohort and all-missense
    populations are classified identically and the displayed score always
    matches the call.

    Returns (merged_df, set_am, set_revel, n).
    """
    excel_path = Path(excel_path)
    step(f"Reading {excel_path.name}")
    df = pd.read_excel(excel_path)
    n = len(df)
    result(f"{n} variants loaded")

    step(f"Reading predictor table {data_path}")
    tsv = pd.read_csv(data_path, sep="\t", low_memory=False, na_values=["."])
    tsv.columns = tsv.columns.str.lstrip("#").str.strip()

    all_preds = set()
    for fit in fits.values():
        all_preds.update(fit["top_predictors"])
    for c in all_preds:
        if c in tsv.columns:
            tsv[c] = pd.to_numeric(tsv[c], errors="coerce")
    tsv_dd = tsv.drop_duplicates(subset="cDNA")
    jcols = [c for c in all_preds if c in tsv_dd.columns]
    merged = df.merge(tsv_dd[["cDNA"] + jcols], on="cDNA", how="left")
    if jcols:
        result(f"{merged[jcols[0]].notna().sum()}/{n} variants matched to the predictor table by cDNA")

    step(f"Reading {Path(alphamissense_path).name} for AlphaMissense score + class")
    am_small = load_alphamissense_hg38(alphamissense_path)
    merged = merged.merge(
        am_small, how="left",
        left_on=["#chr", "pos(1-based)", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"],
    )
    am_scores = pd.to_numeric(merged["pathogenicity score"], errors="coerce")
    n_am_matched = int(am_scores.notna().sum())
    result(f"{n_am_matched}/{n} variants matched to AlphaMissense by (chr, pos, ref, alt)")

    set_am = set(merged.index[am_scores > AM_PATHOGENIC_THRESHOLD])
    result(f"{len(set_am)}/{n} variants called pathogenic by AlphaMissense (score > {AM_PATHOGENIC_THRESHOLD})")
    set_revel = set(merged.index[merged["REVEL_Prediction"] == True])
    return merged, set_am, set_revel, n


def alphamissense_three_way(scores: pd.Series) -> pd.Series:
    """Bucket raw AlphaMissense 'pathogenicity score' into Pathogenic (>0.5642)
    / Benign (<0.34) / Ambiguous (the closed interval in between, including
    NaN)."""
    out = pd.Series("Ambiguous", index=scores.index, dtype=object)
    out[scores > AM_PATHOGENIC_THRESHOLD] = "Pathogenic"
    out[scores < AM_BENIGN_THRESHOLD] = "Benign"
    return out


def load_all_missense_variants(data_path: str, alphamissense_path, pejaver_csv_path):
    """
    Load every FBN1 missense variant in data_path (not just the independently
    annotated 1346-variant cohort) and attach AlphaMissense + REVEL
    pathogenic-call sets, for a Venn comparison against GEMVAP's own calls on
    the full missense population.

    AlphaMissense calls are joined in from alphamissense_path (the FBN1-region
    extract of AlphaMissense's genome-wide hg38 predictions, see
    load_alphamissense_hg38) by genomic coordinates (chr, pos, ref, alt) <->
    (CHROM, POS, REF, ALT); "pathogenic" is thresholded off its own raw
    'pathogenicity score' column (score > 0.5642), not trusted from its
    precomputed 'am_class' label.

    REVEL calls are read directly from data_path's own REVEL_score column
    (far broader coverage than alphamissense_path's sparse REVEL columns,
    which are only populated for a small pre-selected subset) using the
    Pejaver et al. (2022) PP3_Supporting threshold -- the same convention
    used everywhere else in this notebook for individual predictors.

    Returns (merged_df, set_am, set_revel, n). merged_df retains every
    predictor score column from data_path, so pathogenic_set() can be called
    on it directly with each loaded GEMVAP fit.
    """
    data_path = Path(data_path)
    step(f"Reading {data_path.name}")
    tsv = pd.read_csv(data_path, sep="\t", low_memory=False, na_values=["."])
    tsv.columns = tsv.columns.str.lstrip("#").str.strip()
    # Some missense variants carry a compound VEP consequence (e.g.
    # "missense_variant,splice_region_variant"); include those too rather
    # than only exact "missense_variant" matches.
    is_missense = tsv["Consequence"].fillna("").str.split(",").apply(lambda cs: "missense_variant" in cs)
    missense = tsv[is_missense].reset_index(drop=True)
    n = len(missense)
    result(f"{n} missense variants loaded")

    alphamissense_path = Path(alphamissense_path)
    step(f"Reading {alphamissense_path.name}")
    am_small = load_alphamissense_hg38(alphamissense_path)
    merged = missense.merge(
        am_small, how="left",
        left_on=["chr", "pos(1-based)", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"],
    )
    am_scores = pd.to_numeric(merged["pathogenicity score"], errors="coerce")
    n_matched = int(am_scores.notna().sum())
    result(f"{n_matched}/{n} variants matched to AlphaMissense by (chr, pos, ref, alt)")
    set_am = set(merged.index[am_scores > AM_PATHOGENIC_THRESHOLD])
    result(f"{len(set_am)}/{n} variants called pathogenic by AlphaMissense (score > {AM_PATHOGENIC_THRESHOLD})")

    revel_supporting = parse_pejaver_thresholds(pejaver_csv_path)["REVEL"]["PP3_Supporting"]
    revel_scores = pd.to_numeric(merged["REVEL_score"], errors="coerce")
    n_revel_scored = int(revel_scores.notna().sum())
    set_revel = set(merged.index[revel_scores >= revel_supporting])
    result(f"{n_revel_scored}/{n} variants have a REVEL score; "
           f"{len(set_revel)}/{n} called pathogenic (REVEL >= {revel_supporting}, Pejaver PP3_Supporting)")

    return merged, set_am, set_revel, n


def _draw_revel_gemvap_am_panel(ax, gemvap_set, am_set, revel_set,
                                 subset_fontsize=13, label_fontsize=15):
    """
    Draw one REVEL / GEMVAP / AlphaMissense circle into `ax`, styled to match
    the reference figure: salmon/green/lavender fills, bold colored set
    labels, bold black subset counts, no legend box. Shared by the single
    "Any GEMVAP" panel and the two-panel population-comparison figure.
    """
    colors = REVEL_GEMVAP_AM_COLORS  # REVEL, GEMVAP, AlphaMissense
    v = venn3(
        [revel_set, gemvap_set, am_set],
        set_labels=("REVEL", "GEMVAP", "AlphaMissense"),
        set_colors=colors, alpha=0.6, ax=ax,
    )
    for lbl in (v.subset_labels or []):
        if lbl:
            lbl.set_fontsize(subset_fontsize)
            lbl.set_fontweight("bold")
    for lbl, c in zip(v.set_labels or [], colors):
        if lbl:
            lbl.set_fontsize(label_fontsize)
            lbl.set_fontweight("bold")
            lbl.set_color(c)

    # matplotlib_venn's default position for the GEMVAP label (set_labels[1])
    # sits at the top, crowded against REVEL's label (and, when the GEMVAP
    # circle is nearly fully engulfed, can land on top of subset-count text).
    # Anchor it off the actual circle geometry instead of a fixed fraction of
    # the axes, so it always clears every circle: just to the right of the
    # rightmost circle boundary, vertically centered on the three circle
    # centers. This adapts to each panel's own layout instead of assuming a
    # fixed position that only happens to work for one panel's proportions.
    gemvap_label = v.set_labels[1] if v.set_labels else None
    if gemvap_label and v.centers:
        rightmost_edge = max(c.x + r for c, r in zip(v.centers, v.radii))
        y_center = sum(c.y for c in v.centers) / len(v.centers)
        margin = 0.12 * max(v.radii)
        gemvap_label.set_position((rightmost_edge + margin, y_center))
        gemvap_label.set_horizontalalignment("left")
    return v


def draw_any_gemvap_venn(
    gemvap_set: set, am_set: set, revel_set: set, n: int,
    output_path, title: str = None,
) -> None:
    """
    Single 3-circle Venn: "Any GEMVAP" (union of G1|G2|G3) vs AlphaMissense
    vs REVEL, styled to match the reference REVEL/GEMVAP/AlphaMissense
    figure (salmon/green/lavender circles, bold colored set labels, bold
    black subset counts, no legend box).
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    v = _draw_revel_gemvap_am_panel(ax, gemvap_set, am_set, revel_set,
                                     subset_fontsize=16, label_fontsize=20)

    # When REVEL/GEMVAP/AlphaMissense overlap this heavily, the three circles
    # nearly coincide and matplotlib_venn's default label positions can end up
    # sitting on top of a neighbouring region's color. Nudge the ones that
    # collide: REVEL-only up (away from the tri-overlap fill below it),
    # AlphaMissense-only down (away from the REVEL-only crescent above it),
    # REVEL&AlphaMissense-only left (into the REVEL-tinted crescent it
    # belongs to, rather than the neutral tri-overlap fill).
    for sid, (dx, dy) in {"100": (0.0, 0.07), "001": (0.0, -0.07), "101": (-0.07, 0.0)}.items():
        lbl = v.get_label_by_id(sid)
        if lbl:
            x, y = lbl.get_position()
            lbl.set_position((x + dx, y + dy))

    if title:
        ax.set_title(title, fontsize=12, pad=16)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=600, bbox_inches="tight")
    plt.close(fig)
    result(f"Saved -> {output_path}")


# Region opacity range for draw_true_label_venn_grid's count-driven shading:
# the emptiest region in a panel is barely tinted, the panel's own busiest
# region is shown at near-full saturation, scaled independently per panel
# (so each panel's own standout region reads as "the strong one" even though
# panels differ wildly in scale, e.g. a max of 62 vs a max of 22).
_REGION_ALPHA_MIN = 0.08
_REGION_ALPHA_MAX = 0.95
_REGION_SUBSET_IDS = ("100", "010", "001", "110", "101", "011", "111")


def _region_counts(set_a: set, set_b: set, set_c: set) -> dict:
    """Exclusive counts for all 7 venn3 regions of (A, B, C), keyed by
    matplotlib_venn's subset id convention ("100" = A only, "111" = all
    three, etc.)."""
    return {
        "100": len(set_a - set_b - set_c),
        "010": len(set_b - set_a - set_c),
        "001": len(set_c - set_a - set_b),
        "110": len((set_a & set_b) - set_c),
        "101": len((set_a & set_c) - set_b),
        "011": len((set_b & set_c) - set_a),
        "111": len(set_a & set_b & set_c),
    }


def draw_true_label_venn_grid(panels: list, output_path, ncols: int = 2, row_titles: list = None) -> None:
    """
    Grid of REVEL/GEMVAP/AlphaMissense Venn diagrams (row-major, `ncols` per
    row) sharing one fixed, non-proportional circle layout (matplotlib_venn's
    DefaultLayoutAlgorithm(fixed_subset_sizes=...)) so each predictor's
    circle sits in the same position and size in every panel, regardless of
    that panel's actual subset sizes -- panels are then only distinguishable
    by their subset-count labels/shading, not circle geometry. No per-panel
    titles or set-name labels; a single shared legend at the bottom maps
    circle color to predictor name instead.

    Each of the 7 regions in a panel is shaded by its own count, normalized
    against that panel's own busiest region (_REGION_ALPHA_MIN..MAX): the
    highest-count region in a panel renders at near-full color strength,
    emptier regions fade toward it, and a region with a count of exactly 0 is
    plain white (not just faint). A thin edge is kept on every region so
    white/faint regions stay outlined rather than disappearing. Each circle
    is labelled directly (REVEL/GEMVAP/AlphaMissense) in its own color; the
    shared legend below only explains the 4 overlap-region colors (pairwise
    and triple intersections), since the single-set colors are already named
    on the circles themselves.

    panels is a list of (gemvap_set, am_set, revel_set, n) tuples, laid out
    row-major with `ncols` panels per row (any unused trailing axes in the
    last row are hidden). If given, row_titles must have one entry per row
    (nrows = ceil(len(panels) / ncols)); each is drawn centered above its row
    (e.g. identifying which population that row covers).
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3
    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm

    n_panels = len(panels)
    nrows = -(-n_panels // ncols)
    if row_titles is not None and len(row_titles) != nrows:
        raise ValueError(f"row_titles must have {nrows} entries (one per row), got {len(row_titles)}")

    output_path = Path(output_path)
    colors = REVEL_GEMVAP_AM_COLORS  # REVEL, GEMVAP, AlphaMissense
    fixed_layout = DefaultLayoutAlgorithm(fixed_subset_sizes=(1, 1, 1, 1, 1, 1, 1))

    row_height = 6.3 if row_titles is not None else 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, row_height * nrows), squeeze=False)
    flat_axes = axes.flatten()
    overlap_swatch_colors = {}
    for ax, (gemvap_set, am_set, revel_set, n) in zip(flat_axes, panels):
        v = venn3(
            [revel_set, gemvap_set, am_set],
            set_labels=("REVEL", "GEMVAP", "AlphaMissense"),
            set_colors=colors, alpha=1.0, ax=ax,
            layout_algorithm=fixed_layout,
        )
        triple_patch = v.get_patch_by_id("111")
        if triple_patch is not None:
            triple_patch.set_facecolor(_TRIPLE_OVERLAP_COLOR)
        if not overlap_swatch_colors:
            for sid in ("110", "101", "011"):
                patch = v.get_patch_by_id(sid)
                if patch is not None:
                    overlap_swatch_colors[sid] = patch.get_facecolor()
            overlap_swatch_colors["111"] = _TRIPLE_OVERLAP_COLOR

        counts = _region_counts(revel_set, gemvap_set, am_set)
        local_max = max(counts.values()) or 1
        for sid in _REGION_SUBSET_IDS:
            patch = v.get_patch_by_id(sid)
            if patch is None:
                continue
            if counts[sid] == 0:
                patch.set_facecolor("white")
                patch.set_alpha(1.0)
            else:
                strength = counts[sid] / local_max
                patch.set_alpha(_REGION_ALPHA_MIN + (_REGION_ALPHA_MAX - _REGION_ALPHA_MIN) * strength)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)

            lbl = v.get_label_by_id(sid)
            if lbl is None:
                continue
            lbl.set_fontsize(15)
            lbl.set_fontweight("bold")
            face_r, face_g, face_b = patch.get_facecolor()[:3]
            a = patch.get_alpha() if patch.get_alpha() is not None else 1.0
            apparent_r = a * face_r + (1 - a) * 1.0
            apparent_g = a * face_g + (1 - a) * 1.0
            apparent_b = a * face_b + (1 - a) * 1.0
            brightness = 0.299 * apparent_r + 0.587 * apparent_g + 0.114 * apparent_b
            lbl.set_color("white" if brightness < 0.5 else "black")
        for lbl, c in zip(v.set_labels or [], colors):
            if lbl:
                lbl.set_fontsize(13)
                lbl.set_fontweight("bold")
                lbl.set_color(c)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(color=overlap_swatch_colors.get("110", colors[0]), label="REVEL ∩ GEMVAP"),
        mpatches.Patch(color=overlap_swatch_colors.get("101", colors[0]), label="REVEL ∩ AlphaMissense"),
        mpatches.Patch(color=overlap_swatch_colors.get("011", colors[0]), label="GEMVAP ∩ AlphaMissense"),
        mpatches.Patch(color=overlap_swatch_colors.get("111", colors[0]),
                        label="REVEL ∩ GEMVAP ∩ AlphaMissense"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.03))

    if row_titles is not None:
        # Extra headroom between rows (and above row 0) for the row-title
        # text, in place of tight_layout's auto-computed spacing.
        fig.subplots_adjust(hspace=0.12, top=0.96, bottom=0.05)
        fig.canvas.draw()
        for row_idx, title in enumerate(row_titles):
            row_axes = axes[row_idx]
            x_center = (row_axes[0].get_position().x0 + row_axes[-1].get_position().x1) / 2
            y_top = max(a.get_position().y1 for a in row_axes)
            fig.text(x_center, y_top + 0.012, title, ha="center", va="bottom",
                     fontsize=17, fontweight="bold")
    else:
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    fig.savefig(str(output_path), dpi=600, bbox_inches="tight")
    plt.close(fig)
    result(f"Saved -> {output_path}")


def draw_confusion_matrix_grid(matrices: dict, output_path, dpi: int = 600) -> None:
    """
    One 2x2 TP/FN/FP/TN heatmap per entry in `matrices` (predictor name ->
    (tp, fp, fn, tn) counts against a fixed true-label population), arranged
    in a grid with up to 3 columns per row.

    Each row of a panel is shaded by its own true class, matching this
    pipeline's Pathogenic/Control palette used elsewhere (visualization.py's
    "#FAA71A" orange / "#1876BD" blue): the "True: Pathogenic" row (TP, FN)
    is an orange gradient, the "True: Control" row (FP, TN) a blue gradient.
    Within each hue, intensity scales with the cell's own absolute count on a
    single scale shared across every panel in the grid (not renormalized per
    panel), so e.g. a TP=66 cell reads visibly stronger orange than a TP=49
    cell in another predictor's panel.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    output_path = Path(output_path)
    names = list(matrices.keys())
    n = len(names)
    ncols = 3
    nrows = -(-n // ncols)

    pathogenic_max = max((matrices[name][0] for name in names), default=0)
    pathogenic_max = max(pathogenic_max, max((matrices[name][2] for name in names), default=0)) or 1
    control_max = max((matrices[name][1] for name in names), default=0)
    control_max = max(control_max, max((matrices[name][3] for name in names), default=0)) or 1
    orange_cmap = LinearSegmentedColormap.from_list("pathogenic_orange", ["#FFFFFF", "#FAA71A"])
    blue_cmap = LinearSegmentedColormap.from_list("control_blue", ["#FFFFFF", "#1876BD"])
    norm_pathogenic = Normalize(vmin=0, vmax=pathogenic_max)
    norm_control = Normalize(vmin=0, vmax=control_max)

    def _text_color(rgba):
        brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        return "white" if brightness < 0.6 else "black"

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for idx, name in enumerate(names):
        ax = axes[idx // ncols][idx % ncols]
        tp, fp, fn, tn = matrices[name]
        counts = [[tp, fn], [fp, tn]]  # rows: True Pathogenic, True Control; cols: Pred Pathogenic, Pred Control
        cell_colors = [
            [orange_cmap(norm_pathogenic(tp)), orange_cmap(norm_pathogenic(fn))],
            [blue_cmap(norm_control(fp)), blue_cmap(norm_control(tn))],
        ]
        rgba_grid = np.array(cell_colors)

        ax.imshow(rgba_grid)
        for i in range(2):
            for j in range(2):
                label = f"{int(counts[i][j])}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color=_text_color(cell_colors[i][j]))

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Pathogenic", "Pred: Control"])
        ax.set_yticklabels(["True: Pathogenic", "True: Control"])
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Confusion matrices -- held-out test set", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    result(f"Saved -> {output_path}")


def draw_two_set_venn(
    set_a: set, set_b: set, n: int,
    labels: tuple, colors: tuple, title: str, output_path,
) -> None:
    """
    Plain 2-circle Venn (e.g. AlphaMissense vs REVEL) for a single subset of
    variants -- used where a third circle (typically GEMVAP) already defines
    the subset itself and would only ever render as trivially full or empty,
    e.g. one confusion-matrix quadrant of the held-out test set.
    """
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 7))
    v = venn2([set_a, set_b], set_labels=labels, set_colors=colors, alpha=0.6, ax=ax)
    for lbl in (v.subset_labels or []):
        if lbl:
            lbl.set_fontsize(14)
            lbl.set_fontweight("bold")
    for lbl, c in zip(v.set_labels or [], colors):
        if lbl:
            lbl.set_fontsize(16)
            lbl.set_fontweight("bold")
            lbl.set_color(c)
    ax.set_title(f"{title}\n(n={n})", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=600, bbox_inches="tight")
    plt.close(fig)
    result(f"Saved -> {output_path}")


# Manual subset-label position overrides for draw_missense_vs_cohort_venn,
# keyed by panel letter then subset id ("100"/"010"/"001"/"110"/"101"/"011"/
# "111"). Coordinates are absolute axes-fraction positions (same frame as
# Axes.transAxes); None for a coordinate means "leave matplotlib_venn's
# computed value alone" for that axis. Needed because the default layout put
# a few labels for these two panels too close to a region boundary.
_MISSENSE_VS_COHORT_LABEL_OVERRIDES = {
    "A": {
        "100": (-0.22, 0.42),        # REVEL only
        "101": (-0.29, -0.2257),     # REVEL & AlphaMissense
    },
    "B": {
        "100": (0.0, 0.55),          # REVEL only
        "001": (-0.0655, -0.585),    # AlphaMissense only
        "101": (-0.46, -0.27),       # REVEL & AlphaMissense
    },
}


def draw_missense_vs_cohort_venn(
    panel_a: tuple, panel_b: tuple,
    output_path,
    label_a: str = "All FBN1 missense variants",
    label_b: str = "1346-variant annotated cohort",
) -> None:
    """
    Side-by-side single-panel Venn comparison: GEMVAP (Any G1|G2|G3) vs
    AlphaMissense vs REVEL, panel A = one variant population, panel B =
    another (e.g. all missense variants vs. the independently annotated
    cohort). Each panel is (gemvap_set, am_set, revel_set, n).
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"wspace": 0.5})
    for ax, letter, label, (gemvap_set, am_set, revel_set, n) in zip(
        axes, ("A", "B"), (label_a, label_b), (panel_a, panel_b)
    ):
        v = _draw_revel_gemvap_am_panel(ax, gemvap_set, am_set, revel_set)
        for sid, (ox, oy) in _MISSENSE_VS_COHORT_LABEL_OVERRIDES.get(letter, {}).items():
            lbl = v.get_label_by_id(sid)
            if lbl:
                x, y = lbl.get_position()
                lbl.set_position((x if ox is None else ox, y if oy is None else oy))
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="top", ha="left")
        ax.set_title(f"{label}\n(n={n})", fontsize=10, pad=8)

    # subplots_adjust (not tight_layout, which would recompute wspace back
    # down) to keep the panels apart now that the GEMVAP label sits outside
    # each panel's rightmost circle and needs clearance from its neighbor.
    fig.subplots_adjust(wspace=0.5)
    fig.savefig(str(output_path), dpi=600, bbox_inches="tight")
    plt.close(fig)
    result(f"Saved -> {output_path}")


def pathogenic_set(merged_df: pd.DataFrame, fit: dict, threshold: int) -> set:
    """Indices of merged_df whose consensus score under `fit` is >= threshold."""
    top = fit["top_predictors"]
    thrs = fit["rbc"]["threshold"]["case"]
    scores = merged_df.apply(
        lambda row: sum(
            1 if pd.notna(row[tp]) and row[tp] >= thrs[tp] else 0
            for tp in top
        ),
        axis=1,
    )
    return set(merged_df.index[scores >= threshold])


def benign_set(merged_df: pd.DataFrame, fit: dict, threshold: int) -> set:
    """Indices of merged_df whose consensus score under `fit` is <= threshold
    (the BP4_Supporting-style "confidently benign" call, mirroring
    pathogenic_set()'s PP3_Supporting-style ">= threshold"). A threshold of
    -1 (bp4_supporting_thresholds()'s fallback for "never satisfied") never
    matches, since consensus scores are non-negative."""
    top = fit["top_predictors"]
    thrs = fit["rbc"]["threshold"]["case"]
    scores = merged_df.apply(
        lambda row: sum(
            1 if pd.notna(row[tp]) and row[tp] >= thrs[tp] else 0
            for tp in top
        ),
        axis=1,
    )
    return set(merged_df.index[scores <= threshold])


def consensus_scores(df: pd.DataFrame, fit: dict) -> pd.Series:
    """Raw consensus score per row (count of top_predictors meeting their
    individual case-threshold) under `fit` -- the same computation
    pathogenic_set() thresholds against, exposed here as a Series."""
    top = fit["top_predictors"]
    thrs = fit["rbc"]["threshold"]["case"]
    return df.apply(
        lambda row: sum(1 if pd.notna(row[tp]) and row[tp] >= thrs[tp] else 0 for tp in top),
        axis=1,
    )


def build_missed_by_all_table(
    pool: pd.DataFrame,
    set_am: set, set_revel: set, any_gemvap: set,
    fits: dict,
    revel_scores: pd.Series, am_scores: pd.Series,
    gnomad4_af: pd.Series, clinvar: pd.Series,
) -> pd.DataFrame:
    """
    Variants in pool called benign by all three (AlphaMissense, REVEL, and
    every GEMVAP model -- none in set_am/set_revel/any_gemvap), as a table
    with columns hg38, cDNA, Protein, gnomAD 4 (joint AF, 0 when absent),
    ClinVar (clinvar_clnsig, blank when unreported), Domain (left blank --
    no clean FBN1 domain-name table is available in this repo), G1/G2/G3
    (raw consensus score as "x/n" of that model's top_predictors), REVEL and
    Alpha Missense (raw scores). Sorted by cDNA.
    """
    missed_idx = [i for i in pool.index if i not in set_am and i not in set_revel and i not in any_gemvap]
    sub = pool.loc[sorted(missed_idx)].copy()

    npreds = {name: len(fits[name]["top_predictors"]) for name in fits}
    cons = {name: consensus_scores(sub, fits[name]) for name in fits}

    out = pd.DataFrame({
        "hg38": sub["#chr"].astype(str) + ":" + sub["pos(1-based)"].astype(str),
        "cDNA": sub["cDNA"],
        "Protein": sub.apply(protein_notation, axis=1),
        "gnomAD 4": gnomad4_af.reindex(sub.index).fillna(0.0),
        "ClinVar": clinvar.reindex(sub.index).fillna(""),
        "Domain": "",
        "G1": cons["GEMVAP_1"].map(lambda v: f"{v}/{npreds['GEMVAP_1']}") if "GEMVAP_1" in fits else "",
        "G2": cons["GEMVAP_2"].map(lambda v: f"{v}/{npreds['GEMVAP_2']}") if "GEMVAP_2" in fits else "",
        "G3": cons["GEMVAP_3"].map(lambda v: f"{v}/{npreds['GEMVAP_3']}") if "GEMVAP_3" in fits else "",
        "REVEL": revel_scores.reindex(sub.index).round(3),
        "Alpha Missense": am_scores.reindex(sub.index).round(3),
    }, index=sub.index)
    out["gnomAD 4"] = out["gnomAD 4"].round(6)
    return out.sort_values("cDNA").reset_index(drop=True)


def pejaver_supporting_thresholds(fits: dict, calib_dir) -> dict:
    """
    Each model's PP3_Supporting consensus-score threshold from its Step 6
    calibration output, falling back to "unanimous" (every top predictor must
    fire) if no calibrated PP3_Supporting threshold was reached.
    """
    calib_dir = Path(calib_dir)
    thresholds = {}
    for name in fits:
        tp = calib_dir / f"{name}_thresholds.csv"
        npreds = len(fits[name]["top_predictors"])
        if tp.exists():
            td = pd.read_csv(tp)
            row = td[td["evidence_level"] == "PP3_Supporting"]
            if not row.empty and pd.notna(row["score_threshold"].values[0]):
                thresholds[name] = int(row["score_threshold"].values[0])
            else:
                thresholds[name] = npreds  # fallback: unanimous
        else:
            thresholds[name] = npreds  # fallback: unanimous

    for name, t in thresholds.items():
        result(f"{name}: >= {t}/{len(fits[name]['top_predictors'])}")
    return thresholds


def bp4_supporting_thresholds(fits: dict, calib_dir) -> dict:
    """
    Each model's BP4_Supporting consensus-score threshold from its Step 6
    calibration output (calibration/{name}_thresholds.csv). Falls back to -1
    (never satisfied -- no variant qualifies as confidently benign) when no
    BP4_Supporting row was reached, mirroring the pejaver_supporting_thresholds
    fallback used for the pathogenic side.
    """
    calib_dir = Path(calib_dir)
    thresholds = {}
    for name in fits:
        tp = calib_dir / f"{name}_thresholds.csv"
        if tp.exists():
            td = pd.read_csv(tp)
            row = td[td["evidence_level"] == "BP4_Supporting"]
            if not row.empty and pd.notna(row["score_threshold"].values[0]):
                thresholds[name] = int(row["score_threshold"].values[0])
            else:
                thresholds[name] = -1
        else:
            thresholds[name] = -1

    for name, t in thresholds.items():
        result(f"{name}: <= {t}")
    return thresholds


def three_way(scores: pd.Series, pp3_supporting: float, bp4_supporting: float) -> pd.Series:
    """Bucket a continuous score series into Pathogenic/Ambiguous/Benign using
    Pejaver PP3_Supporting (>=) / BP4_Supporting (<=) thresholds. NaN -> Benign
    bucket (unscored / not called), matching the existing binary convention
    used elsewhere in this pipeline for unmatched variants."""
    out = pd.Series("Benign", index=scores.index, dtype=object)
    out[scores >= pp3_supporting] = "Pathogenic"
    out[(scores > bp4_supporting) & (scores < pp3_supporting)] = "Ambiguous"
    return out


def build_full_predictor_table(
    df: pd.DataFrame, fits: dict,
    pej_thr: dict, bp4_thr: dict,
    revel_pp3: float, revel_bp4: float,
) -> pd.DataFrame:
    """
    Per-variant table with each predictor's raw score and 3-way
    Pathogenic/Ambiguous/Benign call side by side: AlphaMissense, REVEL, and
    every loaded GEMVAP model.

    df must already carry 'pathogenicity score' (AlphaMissense, joined by
    load_venn_cohort / load_all_missense_variants), 'REVEL_score', and each
    model's top_predictor score columns. Handles both Excel-derived cohort
    frames (chromosome column '#chr') and tsv-derived missense frames
    (stripped to 'chr') for the hg38 column.
    """
    chr_col = "#chr" if "#chr" in df.columns else "chr"
    am_scores = pd.to_numeric(df["pathogenicity score"], errors="coerce")
    revel_scores = pd.to_numeric(df["REVEL_score"], errors="coerce")

    out = pd.DataFrame({
        "hg38": df[chr_col].astype(str) + ":" + df["pos(1-based)"].astype(str),
        "cDNA": df["cDNA"],
        "Protein": df.apply(protein_notation, axis=1),
        "AM score": am_scores.round(3),
        "AM prediction": alphamissense_three_way(am_scores),
        "REVEL score": revel_scores.round(3),
        "REVEL prediction": three_way(revel_scores, revel_pp3, revel_bp4),
    }, index=df.index)

    for name in fits:
        npreds = len(fits[name]["top_predictors"])
        cons = consensus_scores(df, fits[name])
        label = MODEL_LABELS.get(name, name)
        out[f"{label} score"] = cons.map(lambda v, n=npreds: f"{v}/{n}")
        out[f"{label} prediction"] = three_way(cons, pej_thr[name], bp4_thr[name])

    return out


def predictor_tier_counts(full_table: pd.DataFrame) -> pd.DataFrame:
    """Pathogenic/Ambiguous/Benign counts for every '<predictor> prediction'
    column in a build_full_predictor_table() output (AlphaMissense, REVEL,
    and every loaded GEMVAP model), plus a GEMVAP (Union) and GEMVAP
    (Intersection) row derived from the GEMVAP models' own prediction
    columns -- one row per predictor/combination."""
    pred_cols = [c for c in full_table.columns if c.endswith(" prediction")]
    rows = []
    for col in pred_cols:
        counts = full_table[col].value_counts()
        predictor = col[: -len(" prediction")]
        rows.append((
            predictor,
            int(counts.get("Pathogenic", 0)),
            int(counts.get("Ambiguous", 0)),
            int(counts.get("Benign", 0)),
        ))
    out = pd.DataFrame(rows, columns=["Predictor", "Pathogenic", "Ambiguous", "Benign"])
    union_inter = gemvap_union_intersection_counts(full_table)
    return pd.concat([out, union_inter], ignore_index=True)


def gemvap_union_intersection_counts(full_table: pd.DataFrame) -> pd.DataFrame:
    """
    Pathogenic/Ambiguous/Benign counts for two GEMVAP combinations derived
    from the individual "GEMVAP {1,2,3} prediction" columns of a
    build_full_predictor_table() output:

    - GEMVAP (Union): Pathogenic if ANY loaded model calls it Pathogenic
      (matching this script's "Any GEMVAP" OR convention elsewhere); Benign
      if none call it Pathogenic but at least one calls it Benign (pathogenic
      call takes priority over a benign call from a different model);
      Ambiguous otherwise.
    - GEMVAP (Intersection): Pathogenic only if ALL loaded models call it
      Pathogenic; Benign only if ALL loaded models call it Benign; Ambiguous
      otherwise (any disagreement, or any model itself Ambiguous).

    Returns an empty-rowed frame (right columns, no rows) if no "GEMVAP {n}
    prediction" column is present.
    """
    gemvap_cols = [
        f"{MODEL_LABELS[name]} prediction" for name in ("GEMVAP_1", "GEMVAP_2", "GEMVAP_3")
        if f"{MODEL_LABELS[name]} prediction" in full_table.columns
    ]
    if not gemvap_cols:
        return pd.DataFrame(columns=["Predictor", "Pathogenic", "Ambiguous", "Benign"])

    preds = full_table[gemvap_cols]
    is_pathogenic = preds.eq("Pathogenic")
    is_benign = preds.eq("Benign")

    union_pathogenic = is_pathogenic.any(axis=1)
    union_benign = is_benign.any(axis=1) & ~union_pathogenic
    union_ambiguous = ~union_pathogenic & ~union_benign

    inter_pathogenic = is_pathogenic.all(axis=1)
    inter_benign = is_benign.all(axis=1)
    inter_ambiguous = ~inter_pathogenic & ~inter_benign

    return pd.DataFrame([
        ("GEMVAP (Union)", int(union_pathogenic.sum()), int(union_ambiguous.sum()), int(union_benign.sum())),
        ("GEMVAP (Intersection)", int(inter_pathogenic.sum()), int(inter_ambiguous.sum()), int(inter_benign.sum())),
    ], columns=["Predictor", "Pathogenic", "Ambiguous", "Benign"])


def basic_f1_thresholds(fits: dict) -> dict:
    """Each model's F1-optimal training-set consensus threshold (fit['ci_data_ks']['ks']['cons'])."""
    thresholds = {name: int(fits[name]["ci_data_ks"]["ks"]["cons"]) for name in fits}
    for name, t in thresholds.items():
        result(f"{name}: >= {t}/{len(fits[name]['top_predictors'])}")
    return thresholds


def _draw_venn(ax, sa, sb, sc, labels, colors, title, n):
    v = venn3([sa, sb, sc], set_labels=labels, set_colors=colors, alpha=0.55, ax=ax)
    for lbl in (v.subset_labels or []):
        if lbl:
            lbl.set_fontsize(11)
            lbl.set_fontweight("bold")
    for lbl in (v.set_labels or []):
        if lbl:
            lbl.set_fontsize(12)
            lbl.set_fontweight("bold")
    ax.set_title(title, fontsize=10, pad=10)
    ax.text(
        0.98, 0.02,
        f"{labels[0]}: {len(sa)}\n{labels[1]}: {len(sb)}\n{labels[2]}: {len(sc)}\n(of {n} variants)",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="grey"),
    )


def draw_venn_grid(
    merged_df: pd.DataFrame,
    fits: dict,
    thresholds: dict,
    set_am: set,
    set_revel: set,
    n: int,
    suptitle: str,
    output_path,
    set_builder=pathogenic_set,
    comparator: str = ">=",
) -> dict:
    """
    Draw the 2x2 Venn-diagram grid (GEMVAP_1, GEMVAP_2, GEMVAP_3, ANY) against
    AlphaMissense and REVEL, at the given per-model thresholds. Saves to
    output_path and returns the per-panel call sets (including "ANY" = the
    union of the three).

    set_builder/comparator let the same grid layout serve either the
    pathogenic side (pathogenic_set, ">=", the default) or the benign side
    (benign_set, "<="), so calibrated PP3_Supporting and BP4_Supporting
    panels share one implementation.
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    ps = {name: set_builder(merged_df, fits[name], thresholds[name]) for name in fits}
    ps["ANY"] = ps.get("GEMVAP_1", set()) | ps.get("GEMVAP_2", set()) | ps.get("GEMVAP_3", set())

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax, (key, disp) in zip(axes.flatten(), PANEL_SPECS):
        t = thresholds.get(key, "")
        npreds = len(fits[key]["top_predictors"]) if key in fits else ""
        ts = f" ({comparator}{t}/{npreds})" if key != "ANY" else ""
        _draw_venn(
            ax, ps[key], set_am, set_revel,
            labels=(disp, "AlphaMissense", "REVEL"),
            colors=(MODEL_COLORS.get(key, "#C0392B"), AM_COLOR, REVEL_COLOR),
            title=f"{disp}{ts}", n=n,
        )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=600, bbox_inches="tight")
    plt.close(fig)
    result(f"Saved -> {output_path}")
    return ps


_AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln",
    "E": "Glu", "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys",
    "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
    "Y": "Tyr", "V": "Val", "X": "Ter", "*": "Ter",
}


def aa3(code: str) -> str:
    """Three-letter amino-acid code for a one-letter (or already-3-letter) input."""
    return _AA3.get(str(code).upper(), str(code))


def protein_notation(row) -> str:
    """HGVS-style p. notation (e.g. 'p.Cys628Tyr') from a row's aapos/aaref/aaalt."""
    pos = str(row["aapos"]).split(";")[0]
    return f"p.{aa3(row['aaref'])}{int(float(pos))}{aa3(row['aaalt'])}"


def build_variant_rows(df: pd.DataFrame, idx, pathogenic_by: dict = None) -> pd.DataFrame:
    """
    Variant-identity columns (hg38, Ref, Alt, cDNA, Ref aa, Alt aa, Protein) for
    the given index labels of df, sorted by nothing (caller sorts as needed).
    If pathogenic_by is given (a dict mapping index label -> caller name), it is
    inserted as the first column, "Pathogenic by".
    """
    sub = df.loc[sorted(idx)].copy()
    sub["hg38"] = sub["#chr"].astype(str) + ":" + sub["pos(1-based)"].astype(str)
    sub["Protein"] = sub.apply(protein_notation, axis=1)
    out = pd.DataFrame({
        "hg38": sub["hg38"],
        "Ref": sub["ref"],
        "Alt": sub["alt"],
        "cDNA": sub["cDNA"],
        "Ref aa": sub["aaref"],
        "Alt aa": sub["aaalt"],
        "Protein": sub["Protein"],
    })
    if pathogenic_by is not None:
        out.insert(0, "Pathogenic by", [pathogenic_by[i] for i in sorted(idx)])
    return out


def exclusive_single_predictor_table(pool: pd.DataFrame, caller_sets: dict) -> pd.DataFrame:
    """
    Variants in pool called pathogenic by exactly one of the named caller_sets
    (e.g. {"AlphaMissense": set_am, "REVEL": set_revel, "GEMVAP": any_gemvap}),
    as a table with columns Pathogenic by, hg38, Ref, Alt, cDNA, Ref aa, Alt aa,
    Protein, sorted by (Pathogenic by, cDNA).
    """
    exclusive = {}
    for i in pool.index:
        callers = [name for name, s in caller_sets.items() if i in s]
        if len(callers) == 1:
            exclusive[i] = callers[0]
    out = build_variant_rows(pool, exclusive.keys(), pathogenic_by=exclusive)
    return out.sort_values(["Pathogenic by", "cDNA"]).reset_index(drop=True)


def build_comparison_table(
    fits: dict, pej_thr: dict, basic_thr: dict,
    ps_pej: dict, ps_basic: dict,
    set_am: set, set_revel: set, n: int,
) -> pd.DataFrame:
    """Tabulate pathogenic-call counts for every model/strategy on the venn cohort."""
    rows = []
    for name in fits:
        npreds = len(fits[name]["top_predictors"])
        rows.append({
            "Model": MODEL_LABELS[name],
            "Predictors": npreds,
            "Pejaver thr": f">={pej_thr[name]}/{npreds}",
            "Pejaver path. (n)": len(ps_pej[name]),
            "Pejaver path. (%)": f"{100 * len(ps_pej[name]) / n:.1f}%",
            "Basic thr": f">={basic_thr[name]}/{npreds}",
            "Basic path. (n)": len(ps_basic[name]),
            "Basic path. (%)": f"{100 * len(ps_basic[name]) / n:.1f}%",
        })
    for lbl, s in [("AlphaMissense", set_am), ("REVEL", set_revel)]:
        rows.append({
            "Model": lbl, "Predictors": "—",
            "Pejaver thr": "pre-called", "Pejaver path. (n)": len(s),
            "Pejaver path. (%)": f"{100 * len(s) / n:.1f}%",
            "Basic thr": "pre-called", "Basic path. (n)": len(s),
            "Basic path. (%)": f"{100 * len(s) / n:.1f}%",
        })
    rows.append({
        "Model": "Any GEMVAP", "Predictors": "—",
        "Pejaver thr": "union G1|G2|G3", "Pejaver path. (n)": len(ps_pej["ANY"]),
        "Pejaver path. (%)": f"{100 * len(ps_pej['ANY']) / n:.1f}%",
        "Basic thr": "union G1|G2|G3", "Basic path. (n)": len(ps_basic["ANY"]),
        "Basic path. (%)": f"{100 * len(ps_basic['ANY']) / n:.1f}%",
    })
    return pd.DataFrame(rows).set_index("Model")
