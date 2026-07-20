from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


def plot_ks_f1_curve(
    ks_values, ci_stats, output_path: str, dpi: int = 600,
    f1_ylim: tuple[float, float] | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    raw_names = [name.replace("rankscore", "").strip("_") for name in ks_values.keys()]
    raw_scores = list(ks_values.values())
    sorted_pairs = sorted(zip(raw_names, raw_scores), key=lambda x: x[1], reverse=True)
    model_names, model_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])

    models = ci_stats["npreds"]
    f1_scores = ci_stats["F1_score"]
    best_index = int(np.nanargmax(f1_scores)) if len(f1_scores) > 0 else 0

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        figsize=(16, 10),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1, 2]},
    )

    shifted_models = [x - 1 for x in models]
    sns.lineplot(ax=ax1, x=shifted_models, y=f1_scores, marker="o", color="#1876BD", label="F1 Score")

    # Round the F1 axis limits out to the nearest 0.05 tick, then add one
    # extra tick of headroom above the max so the best-score label (which is
    # placed with a fixed pixel offset, not a data offset) never overlaps the
    # line or gets clipped by the axes frame.
    if f1_ylim is not None:
        ymin, ymax = f1_ylim
    else:
        f1_min = min(f1_scores) if len(f1_scores) > 0 else 0.0
        f1_max = max(f1_scores) if len(f1_scores) > 0 else 1.0
        ymin = max(0.0, np.floor(f1_min / 0.05) * 0.05 - 0.05)
        ymax = min(1.05, np.ceil(f1_max / 0.05) * 0.05 + 0.05)
    ax1.set_ylim(ymin, ymax)
    ax1.yaxis.set_major_locator(MultipleLocator(0.05))

    if len(f1_scores) > 0:
        ax1.scatter(
            shifted_models[best_index],
            f1_scores[best_index],
            color="#FAA71A",
            s=120,
            label="Best",
        )
        ax1.annotate(
            f"{f1_scores[best_index]:.3f}",
            xy=(shifted_models[best_index], f1_scores[best_index]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    ax1.set_ylabel("F1 Score", fontsize=14)
    ax1.set_title("F1 Score of each ensemble of models", fontsize=15)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.legend(fontsize=12)

    best_n = int(models.iloc[best_index]) if len(models) > 0 else 0
    colors = ["#FAA71A" if i < best_n else "#8b8a87" for i in range(len(model_names))]
    ax2.bar(model_names, model_scores, color=colors)
    ks_max = max(model_scores) if len(model_scores) > 0 else 1.0
    ax2.set_ylim(0, np.ceil(ks_max / 0.1) * 0.1)
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.set_ylabel("KS Test Value", fontsize=14)
    ax2.set_title("Kolmogorov-Smirnov for each individual model", fontsize=15)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=90, fontsize=12)
    ax2.tick_params(axis="y", labelsize=12)

    legend_elements = [
        Patch(facecolor="#FAA71A", label="Selected models"),
        Patch(facecolor="#8b8a87", label="Non-selected models"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=12)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _aa_consensus(training_data, df, top_predictors, thresholds):
    """Compute mean consensus score per reference amino acid."""
    result = {}
    for aacid in training_data["aaref"].unique():
        cond = df.is_mis & (training_data["aaref"] == aacid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from packages.package1.predictor_selection import build_trace
            trace = build_trace(
                training_data[~df.training_sets & cond][top_predictors],
                thresholds, name="aa", text=False, y_percentage=True,
            )
        result[aacid] = sum(l * v for l, v in enumerate(trace))
    sorted_result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}
    return {k: v for k, v in sorted_result.items() if pd.Series(k).notna().all()}


def plot_horizontal_gemvap1(base, df, output_path: str, dpi: int = 600) -> None:
    from packages.package1.predictor_selection import build_trace

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    top_predictors = base["top_predictors"]
    thresholds = {k: base["rbc"]["threshold"]["case"][k] for k in top_predictors}
    training_data = base["training_data"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace_1 = np.array(build_trace(
            training_data[(~df.is_case & df.is_ctrl) & ~df.is_cys][top_predictors],
            thresholds, name="1", text=False, y_percentage=False))
        trace_2 = np.array(build_trace(
            training_data[(df.is_case & ~df.is_ctrl) & ~df.is_cys][top_predictors],
            thresholds, name="2", text=False, y_percentage=False))
        trace_3 = np.array(build_trace(
            training_data[(df.is_case | df.is_ctrl) & df.is_cys][top_predictors],
            thresholds, name="3", text=False, y_percentage=False))

    trace_12 = trace_1 + trace_2
    test_bar = list(range(len(top_predictors) + 1))
    sorted_dict = _aa_consensus(training_data, df, top_predictors, thresholds)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10), layout="constrained")

    ax1.bar(test_bar, list(trace_1), color="#1876BD")
    ax1.bar(test_bar, list(trace_2), bottom=list(trace_1), color="#FAA71A")
    ax1.bar(test_bar, list(trace_3), bottom=list(trace_12), color="#FAA71A", hatch="//")
    max_height = max(trace_1 + trace_2 + trace_3)
    ax1.vlines([base["ci_data_ks"]["ks"]["cons"] - 0.5], 0, max_height, transform=ax1.get_xaxis_transform(), colors="r")
    ax1.set_ylim(0, 320)
    ax1.set_xlabel("Consensus Score (GEMVAP 1)")
    ax1.set_ylabel("Number of Variants")
    ax1.set_xticks(test_bar)
    ax1.set_xticklabels([str(i) for i in test_bar])
    ax1.legend(handles=[
        Patch(facecolor="#1876BD", label="Control"),
        Patch(facecolor="#FAA71A", label="Pathogenic Non-Cysteines"),
        Patch(facecolor="#FAA71A", hatch="//", label="Pathogenic Cysteines"),
        Patch(facecolor="none", hatch="//", label="Filtered out in the GEMVAP 2 training dataset"),
    ], loc="best")

    ax2.bar(sorted_dict.keys(), sorted_dict.values(), width=0.6, align="center", color="#1876BD")
    ax2.set_xlabel("Amino Acid")
    ax2.set_ylabel("Level of Consensus (GEMVAP 1)")
    ax2.set_xticks(range(len(sorted_dict)))
    ax2.set_xticklabels(sorted_dict.keys())
    ax2.legend(handles=[
        Patch(facecolor="#1876BD", label=r"Average number of predictors predicting $\bf{\mathit{Pathogenic}}$"),
    ], loc="upper left")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _center_legend_pair(ax, first_handles, first_kwargs, second_handles, second_kwargs, gap: float = 0.03):
    """
    Place two 'upper center' legends side by side so that, as a pair, they
    are horizontally centered on the axes -- using each legend's actual
    rendered width rather than a guessed fraction, so the pair stays
    centered regardless of how wide the entries are.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    leg1 = ax.legend(handles=first_handles, loc="upper center", bbox_to_anchor=(0.25, 0.99), **first_kwargs)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=second_handles, loc="upper center", bbox_to_anchor=(0.75, 0.99), **second_kwargs)
    fig.canvas.draw()

    ax_bbox = ax.get_window_extent(renderer)
    w1 = leg1.get_window_extent(renderer).width / ax_bbox.width
    w2 = leg2.get_window_extent(renderer).width / ax_bbox.width

    total = w1 + gap + w2
    x1 = 0.5 - total / 2 + w1 / 2
    x2 = 0.5 + total / 2 - w2 / 2

    leg1.remove()
    leg2.remove()

    leg1 = ax.legend(handles=first_handles, loc="upper center", bbox_to_anchor=(x1, 0.99), **first_kwargs)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=second_handles, loc="upper center", bbox_to_anchor=(x2, 0.99), **second_kwargs)
    return leg1, leg2


def plot_horizontal_gemvap1_calibrated(
    base, df, calib_path, calib_benign, calib_thresholds: dict,
    output_path: str, dpi: int = 600, model_name: str = "GEMVAP 1",
    center_legends: bool = False, aa_training_data=None,
) -> None:
    from gemvap_pejaver_calibration import (
        EVIDENCE_LEVELS_PP3, EVIDENCE_LEVELS_BP4, LEVEL_COLORS, INDETERMINATE,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    top_predictors = base["top_predictors"]
    thresholds = {k: base["rbc"]["threshold"]["case"][k] for k in top_predictors}
    training_data = base["training_data"]
    # The amino-acid consensus panel (ax2) can be computed over a different
    # (typically less-restricted) training set than the one this model was
    # fitted on -- e.g. GEMVAP 2/3 are fitted on cysteine- and/or conserved-
    # domain-excluded subsets, but their amino-acid average-score panel should
    # reflect the full training set. Falls back to this model's own training
    # data (unchanged behaviour) when no override is given.
    aa_source = aa_training_data if aa_training_data is not None else training_data
    sorted_dict = _aa_consensus(aa_source, df, top_predictors, thresholds)

    all_scores = np.concatenate([calib_path, calib_benign])
    score_values = np.arange(int(all_scores.min()), int(all_scores.max()) + 1)
    width = 0.4

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10), layout="constrained")

    ax1.bar(
        score_values - width / 2,
        [np.sum(calib_path == sv) for sv in score_values],
        width=width, color="#FAA71A", alpha=0.9, label="Pathogenic", zorder=3,
    )
    ax1.bar(
        score_values + width / 2,
        [np.sum(calib_benign == sv) for sv in score_values],
        width=width, color="#1876BD", alpha=0.9, label="Control", zorder=3,
    )

    x_min = score_values.min() - 0.5
    x_max = score_values.max() + 0.5
    bp4_vals = [calib_thresholds.get(l) for l in EVIDENCE_LEVELS_BP4]
    pp3_vals = [calib_thresholds.get(l) for l in EVIDENCE_LEVELS_PP3]

    for tval, level in zip(bp4_vals, EVIDENCE_LEVELS_BP4):
        if tval is not None:
            ax1.axvspan(x_min, tval + 0.5, color=LEVEL_COLORS[level], alpha=0.18, zorder=1)
    for tval, level in zip(pp3_vals, EVIDENCE_LEVELS_PP3):
        if tval is not None:
            ax1.axvspan(tval - 0.5, x_max, color=LEVEL_COLORS[level], alpha=0.18, zorder=1)

    ax1.set_xlabel(f"{model_name} Consensus Score", fontsize=11)
    ax1.set_ylabel("Variant Count", fontsize=11)
    ax1.set_title(f"Calibration Set Score Distribution with PP3/BP4 Evidence Zones ({model_name})", fontsize=11)
    ax1.set_xticks(score_values)

    bar_handles = [
        Patch(facecolor="#FAA71A", alpha=0.9, label="Pathogenic"),
        Patch(facecolor="#1876BD", alpha=0.9, label="Control"),
    ]
    zone_handles = []
    active_bp4 = [(l, t) for l, t in zip(EVIDENCE_LEVELS_BP4, bp4_vals) if t is not None]
    active_pp3 = [(l, t) for l, t in zip(EVIDENCE_LEVELS_PP3, pp3_vals) if t is not None]
    for level, _ in active_bp4:
        label = level.replace("BP4_", "BP4 ").replace("VeryStrong", "Very Strong")
        zone_handles.append(Patch(facecolor=LEVEL_COLORS[level], alpha=0.4, label=label))
    zone_handles.append(Patch(facecolor=LEVEL_COLORS[INDETERMINATE], alpha=0.4, label="Indeterminate"))
    for level, _ in active_pp3:
        label = level.replace("PP3_", "PP3 ").replace("VeryStrong", "Very Strong")
        zone_handles.append(Patch(facecolor=LEVEL_COLORS[level], alpha=0.4, label=label))

    if center_legends:
        _center_legend_pair(
            ax1,
            bar_handles, {"fontsize": 9, "title": "Variants"},
            zone_handles, {"fontsize": 9, "title": "ACMG/AMP evidence zone"},
        )
    else:
        first_legend = ax1.legend(
            handles=bar_handles, loc="upper center", bbox_to_anchor=(0.28, 0.99),
            fontsize=9, title="Variants",
        )
        ax1.add_artist(first_legend)
        ax1.legend(
            handles=zone_handles, loc="upper center", bbox_to_anchor=(0.72, 0.99),
            fontsize=9, title="ACMG/AMP evidence zone",
        )

    ax2.bar(sorted_dict.keys(), sorted_dict.values(), width=0.6, align="center", color="#1876BD")
    ax2.set_xlabel("Amino Acid")
    ax2.set_ylabel(f"Level of Consensus ({model_name})")
    ax2.set_xticks(range(len(sorted_dict)))
    ax2.set_xticklabels(sorted_dict.keys())
    ax2.legend(handles=[
        Patch(facecolor="#1876BD", label=r"Average number of predictors predicting $\bf{\mathit{Pathogenic}}$"),
    ], loc="upper left")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_horizontal_gemvap2(base, cyst, df, conserved_data_path: str, output_path: str, dpi: int = 600) -> None:
    from packages.package1.predictor_selection import build_trace

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tableS1_cons = pd.read_csv(conserved_data_path, sep="\t", low_memory=False, na_values=["."])
    in_dom = df.data.join(tableS1_cons.set_index("variantvcf"), on="variantvcf")["in_dom_conserved"]

    top_predictors = cyst["top_predictors"]
    thresholds = {k: cyst["rbc"]["threshold"]["case"][k] for k in top_predictors}
    training_data = base["training_data"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace_1 = np.array(build_trace(
            training_data[(~df.is_case & df.is_ctrl) & (in_dom == 0)][top_predictors],
            thresholds, name="1", text=False, y_percentage=False))
        trace_2 = np.array(build_trace(
            training_data[(~df.is_case & df.is_ctrl) & (in_dom == 1)][top_predictors],
            thresholds, name="2", text=False, y_percentage=False))
        trace_3 = np.array(build_trace(
            training_data[(df.is_case & ~df.is_ctrl) & (in_dom == 0)][top_predictors],
            thresholds, name="3", text=False, y_percentage=False))
        trace_4 = np.array(build_trace(
            training_data[((df.is_case & ~df.is_ctrl) & ~df.is_cys) & (in_dom == 1)][top_predictors],
            thresholds, name="4", text=False, y_percentage=False))
        trace_5 = np.array(build_trace(
            training_data[((df.is_case & ~df.is_ctrl) & df.is_cys) & (in_dom == 1)][top_predictors],
            thresholds, name="5", text=False, y_percentage=False))

    trace_12 = trace_1 + trace_2
    trace_123 = trace_12 + trace_3
    trace_1234 = trace_123 + trace_4
    test_bar = list(range(len(top_predictors) + 1))
    sorted_dict = _aa_consensus(training_data, df, top_predictors, thresholds)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10), layout="constrained")

    ax1.bar(test_bar, list(trace_1), color="#1876BD")
    ax1.bar(test_bar, list(trace_2), bottom=list(trace_1), color="#1876BD", hatch="//")
    ax1.bar(test_bar, list(trace_3), bottom=list(trace_12), color="#F4D091")
    ax1.bar(test_bar, list(trace_4), bottom=list(trace_123), color="#F4D091", hatch="//")
    ax1.bar(test_bar, list(trace_5), bottom=list(trace_1234), color="#FAA71A", hatch="//")
    max_height = max(trace_1 + trace_2 + trace_3 + trace_4 + trace_5)
    ax1.vlines([cyst["ci_data_ks"]["ks"]["cons"] - 0.5], 0, max_height, transform=ax1.get_xaxis_transform(), colors="r")
    ax1.set_xlabel("Consensus Score (GEMVAP 2)")
    ax1.set_ylabel("Number of variants")
    ax1.set_xticks(test_bar)
    ax1.set_xticklabels([str(i) for i in test_bar])
    ax1.legend(handles=[
        Patch(facecolor="#1876BD", label="Control Non-Conserved, Non-Cysteines"),
        Patch(facecolor="#1876BD", hatch="//", label="Control Conserved, Non-Cysteines"),
        Patch(facecolor="#F4D091", label="Pathogenic Non-Conserved, Non-Cysteines"),
        Patch(facecolor="#F4D091", hatch="//", label="Pathogenic Conserved, Non-Cysteines"),
        Patch(facecolor="#FAA71A", hatch="//", label="Pathogenic Conserved, Cysteines"),
        Patch(facecolor="none", hatch="//", label="Filtered out in the GEMVAP 3 dataset"),
    ], loc="best")

    ax2.bar(sorted_dict.keys(), sorted_dict.values(), width=0.6, align="center", color="#1876BD")
    ax2.set_xlabel("Amino Acid")
    ax2.set_ylabel("Level of Consensus (GEMVAP 2)")
    ax2.set_xticks(range(len(sorted_dict)))
    ax2.set_xticklabels(sorted_dict.keys())
    ax2.legend(handles=[
        Patch(facecolor="#1876BD", label=r"Average number of predictors predicting $\bf{\mathit{Pathogenic}}$"),
    ], loc="upper left")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(
    acc_df: pd.DataFrame,
    output_path: str,
    dpi: int = 600,
    title: str = None,
) -> None:
    """
    Horizontal bar chart comparing calibrated accuracy across GEMVAP models and
    individual Pejaver et al. (2022) tools on the performance test set.

    GEMVAP models are highlighted in orange; individual tools in blue.
    Each bar is annotated with the accuracy value and the number of variants
    that were classified (i.e. not assigned Indeterminate / VUS).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = acc_df.dropna(subset=["accuracy"]).sort_values("accuracy").reset_index(drop=True)

    gemvap_names = {n for n in df["model"] if n.upper().startswith("GEMVAP")}
    colors = [
        "#FAA71A" if row["model"] in gemvap_names else "#1876BD"
        for _, row in df.iterrows()
    ]

    fig_height = max(4, 0.45 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bars = ax.barh(df["model"], df["accuracy"], color=colors, edgecolor="white", height=0.65)

    # Dashed chance-level reference
    ax.axvline(0.5, color="#8b8a87", linestyle="--", linewidth=1.2, label="Chance (0.5)")

    # Annotate each bar: accuracy + classified count
    for bar, (_, row) in zip(bars, df.iterrows()):
        x_end = bar.get_width()
        n_text = f"{row['n_classified']}/{int(row['n_total_perf'])} classified"
        acc_text = f" {x_end:.3f}  ({n_text})"
        ax.text(
            min(x_end + 0.01, 0.99),
            bar.get_y() + bar.get_height() / 2,
            acc_text,
            va="center", ha="left", fontsize=8.5, color="#222222",
        )

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy (VUS excluded)", fontsize=11)
    ax.set_title(
        title or (
            "Calibrated Accuracy on Performance Test Set\n"
            "GEMVAP models vs individual Pejaver et al. (2022) tools"
        ),
        fontsize=12, fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor="#FAA71A", label="GEMVAP (consensus)"),
        Patch(facecolor="#1876BD", label="Individual tool (Pejaver thresholds)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Accuracy comparison plot saved → {output_path}")


def plot_metrics_comparison(
    acc_df: pd.DataFrame,
    output_path: str,
    dpi: int = 600,
    title: str = None,
) -> None:
    """
    Four-panel horizontal bar chart: F1, Accuracy, MCC, VUS rate.
    Models share the Y axis, sorted by F1 descending (best at top).
    GEMVAP models in orange (#FAA71A); individual tools in blue (#1876BD).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = acc_df.copy()
    df = df.sort_values("f1", ascending=True).reset_index(drop=True)

    gemvap_names = {n for n in df["model"] if n.upper().startswith("GEMVAP")}
    colors = ["#FAA71A" if row["model"] in gemvap_names else "#1876BD" for _, row in df.iterrows()]

    fig_height = max(5, 0.45 * len(df) + 2.0)
    fig, axes = plt.subplots(1, 4, figsize=(20, fig_height), sharey=True)

    panels = [
        ("f1",        "F1 (pathogenic class)",         None),
        ("accuracy",  "Accuracy (VUS excluded)",        None),
        ("mcc",       "MCC",                            0.0),
        ("vus_rate",  "VUS rate",                       None),
    ]

    for ax, (metric, xlabel, ref_line) in zip(axes, panels):
        vals = df[metric]
        ax.barh(df["model"], vals, color=colors, edgecolor="white", height=0.65)
        if ref_line is not None:
            ax.axvline(ref_line, color="#8b8a87", linestyle="--", linewidth=1.2)
        for i, (val, (_, row)) in enumerate(zip(vals, df.iterrows())):
            if pd.notna(val):
                ax.text(
                    min(val + 0.02, 0.99) if metric != "mcc" else val + 0.02,
                    i,
                    f"{val:.3f}",
                    va="center", ha="left", fontsize=8, color="#222222",
                )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_xlim((-1.1 if metric == "mcc" else -0.02), 1.15)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_title(xlabel, fontsize=10, fontweight="bold")

    axes[0].set_yticks(range(len(df)))
    axes[0].set_yticklabels(df["model"], fontsize=9)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    legend_handles = [
        Patch(facecolor="#FAA71A", label="GEMVAP (consensus)"),
        Patch(facecolor="#1876BD", label="Individual tool (Pejaver thresholds)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        title or (
            "Calibrated Performance Metrics on Performance Test Set\n"
            "GEMVAP models vs individual Pejaver et al. (2022) tools"
        ),
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Metrics comparison plot saved → {output_path}")


def plot_multi_seed_accuracy(
    df: pd.DataFrame,
    output_path: str,
    dpi: int = 600,
    title: str = None,
) -> None:
    """
    Four-panel bar chart showing mean F1, Accuracy, MCC and VUS rate across seeds.
    Delegates to plot_metrics_comparison after averaging over the seed column.
    """
    mean_df = (
        df.groupby("model")[["f1", "accuracy", "mcc", "vus_rate", "n_classified", "n_total_perf"]]
        .mean()
        .reset_index()
    )
    plot_metrics_comparison(
        mean_df,
        output_path,
        dpi=dpi,
        title=title or (
            "Mean F1 / Accuracy / MCC / VUS Rate across Seeds\n"
            "GEMVAP models vs individual Pejaver et al. (2022) tools"
        ),
    )


def plot_pathogenic_vus_rate(
    df: pd.DataFrame,
    output_path: str,
    dpi: int = 600,
    title: str = None,
) -> None:
    """
    Horizontal strip+mean plot of pathogenic VUS rate across seeds.

    Each model is a row; each seed is a translucent dot; a filled diamond marks
    the mean. GEMVAP models in orange (#FAA71A); individual tools in blue (#1876BD).
    Models ordered by mean vus_rate_pathogenic (ascending = fewest VUS at top).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model_order = (
        df.groupby("model")["vus_rate_pathogenic"].mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    gemvap_models = {m for m in model_order if m.upper().startswith("GEMVAP")}
    color_map = {
        m: "#FAA71A" if m in gemvap_models else "#1876BD"
        for m in model_order
    }

    fig_height = max(4, 0.45 * len(model_order) + 2.0)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    for model in model_order:
        subset = df.loc[df["model"] == model, "vus_rate_pathogenic"].dropna()
        if subset.empty:
            continue
        y = model_order.index(model)
        color = color_map[model]
        ax.hlines(y, subset.min(), subset.max(), color=color, alpha=0.35, linewidth=1.5)
        ax.scatter(subset, [y] * len(subset), color=color, alpha=0.65, s=35, zorder=3)
        ax.scatter(
            [subset.mean()], [y],
            color=color, s=90, marker="D", zorder=4,
            edgecolors="white", linewidths=0.7,
        )

    ax.set_xlabel("VUS rate — pathogenic variants only (fraction not classified)", fontsize=10)
    ax.set_xlim(-0.02, 1.05)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=9)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_title(
        title or (
            "Pathogenic-only VUS Rate across Seeds\n"
            "GEMVAP models vs individual Pejaver et al. (2022) tools"
        ),
        fontsize=12, fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor="#FAA71A", label="GEMVAP (consensus)"),
        Patch(facecolor="#1876BD", label="Individual tool (Pejaver thresholds)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pathogenic VUS rate plot saved → {output_path}")


def plot_metric_vs_train_fraction(
    df: pd.DataFrame,
    output_path: str,
    dpi: int = 600,
    title: str = None,
) -> None:
    """
    Four-panel line chart: F1, Accuracy, MCC, VUS rate as a function of the
    training-set fraction used to build each batch of calibration sets (see
    run_calibration_sets_eval.py --train-fraction). One line per GEMVAP
    variant; error bars show the std across calibration sets at that fraction.

    df must have columns: train_fraction, model, f1_mean, f1_std,
    accuracy_mean, accuracy_std, mcc_mean, mcc_std, vus_rate_mean, vus_rate_std.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model_colors = {
        "GEMVAP_1": "#FAA71A",
        "GEMVAP_2": "#1876BD",
        "GEMVAP_3": "#2CA02C",
    }
    models = [m for m in model_colors if m in df["model"].unique()]

    panels = [
        ("f1", "F1 (pathogenic class)"),
        ("accuracy", "Accuracy (VUS excluded)"),
        ("mcc", "MCC"),
        ("vus_rate", "VUS rate"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)

    for ax, (metric, xlabel) in zip(axes, panels):
        for model in models:
            sub = df[df["model"] == model].sort_values("train_fraction")
            ax.errorbar(
                sub["train_fraction"] * 100,
                sub[f"{metric}_mean"],
                yerr=sub[f"{metric}_std"],
                marker="o", color=model_colors[model], label=model,
                capsize=3, linewidth=1.5,
            )
        if metric == "mcc":
            ax.axhline(0.0, color="#8b8a87", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Training-set fraction (%)", fontsize=10)
        ax.set_title(xlabel, fontsize=10, fontweight="bold")
        ax.grid(linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_xticks(sorted(df["train_fraction"].unique() * 100))

    axes[0].set_ylabel("Score", fontsize=10)
    axes[0].legend(fontsize=9, loc="best")

    fig.suptitle(
        title or (
            "GEMVAP Performance vs. Calibration-Set Training Fraction\n"
            "(mean ± std across calibration sets at each fraction)"
        ),
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Metric-vs-train-fraction plot saved → {output_path}")


def plot_figure4(
    dfgrouped_by_proteicpos: pd.DataFrame,
    output_path: str,
    protein_length: int = 2871,
    dpi: int = 600,
    xlim: tuple[int, int] | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dfgrouped = dfgrouped_by_proteicpos[
        (dfgrouped_by_proteicpos["proteic_pos"] > 0) &
        (dfgrouped_by_proteicpos["proteic_pos"] < protein_length)
    ]

    fig, ax = plt.subplots(figsize=(20, 10), dpi=dpi)

    percentages = dfgrouped["Pathogenic"] * 100
    ax.bar(dfgrouped["proteic_pos"], percentages)

    if xlim is not None:
        ax.set_xlim(*xlim)
        xticks = np.linspace(xlim[0], xlim[1], num=12, dtype=int)
    else:
        ax.set_xlim(0, protein_length)
        xticks = np.linspace(0, 2750, num=12, dtype=int)
    ax.set_xticks(sorted(set(xticks)))
    ax.tick_params(axis="x", labelsize=16)
    ax.set_xlabel("Residue Position", fontsize=16)
    ax.set_ylabel("Percentage of missenses predicted as pathogenic", fontsize=16)

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    legend_labels = [
        "Signal peptide (position 1 to 27)",
        "Proline-rich region (position 392 to 446)",
        "C-terminal region (position 2687 to 2871)",
    ]
    legend_colors = ["purple", "grey", "seagreen"]

    # The axes box's pixel height is fixed by the subplot layout regardless
    # of ylim, so the only way to give the legend exactly a 1-pixel gap to
    # the tallest bar AND a 1-pixel gap to the top spine is to solve for the
    # y-axis upper bound that leaves precisely enough headroom above the
    # highest bar to fit "1px + legend height + 1px" in that fixed pixel
    # space. Measure the legend's rendered height first via proxy patches
    # (the real axvspans need span_frac, which itself needs y_top).
    proxy_legend = ax.legend(
        handles=[Patch(color=c, alpha=0.2) for c in legend_colors],
        labels=legend_labels, loc="upper right", bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_height_px = proxy_legend.get_window_extent(renderer).height
    axes_height_px = ax.get_window_extent(renderer).height
    proxy_legend.remove()

    bar_max = percentages.max()
    headroom_px = legend_height_px + 2
    y_top = bar_max / (1 - headroom_px / axes_height_px)
    ax.set_ylim(0, y_top)

    # axvspan's ymin/ymax are always in axes-fraction (0-1): axvspan()
    # unconditionally overwrites any transform passed in via kwargs with its
    # own blended x-data/y-axes-fraction transform, so passing transData and
    # a literal ymax=100 does NOT mean "stop at the data value 100" -- it
    # stretches the box to 100x the axes height. Convert the 100 data value
    # to the matching fraction of y_top instead.
    span_frac = 100 / y_top
    ax.axvspan(1, 27, ymin=0, ymax=span_frac, color=legend_colors[0], alpha=0.2,
               label=legend_labels[0])
    ax.axvspan(392, 446, ymin=0, ymax=span_frac, color=legend_colors[1], alpha=0.2,
               label=legend_labels[1])
    ax.axvspan(2687, 2871, ymin=0, ymax=span_frac, color=legend_colors[2], alpha=0.2,
               label=legend_labels[2])

    # Note: ax.legend() applies a small inward "borderaxespad" offset from
    # bbox_to_anchor by default even when an explicit anchor point is given
    # -- borderaxespad=0 is required here, otherwise the legend lands ~40px
    # off from the pixel-exact position computed below and overlaps the bars.
    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bar_top_px = ax.transData.transform((0, bar_max))[1]
    target_top_px = bar_top_px + 1 + legend.get_window_extent(renderer).height
    _, target_top_frac = ax.transAxes.inverted().transform((0, target_top_px))

    legend.remove()
    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(1.0, target_top_frac), borderaxespad=0,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def _gemvap1_score_by_aa(base: dict, data_path: str) -> pd.DataFrame:
    """
    Per-variant GEMVAP 1 consensus score (count of top predictors voting
    pathogenic at their calibrated threshold), summarized by wild-type amino
    acid across all missense variants. data_path must have aaref/aaalt/
    Consequence columns (the same full predictor table used to train GEMVAP 1).

    Shared by plot_ddgun_vs_gemvap1 and classify_aa_by_consensus so both
    report on the exact same per-variant scores.
    """
    top_predictors = base["top_predictors"]
    thresholds = base["rbc"]["threshold"]["case"]

    tsv = pd.read_csv(data_path, sep="\t", low_memory=False, na_values=["."])
    tsv.columns = tsv.columns.str.lstrip("#").str.strip()

    def gemvap1_score(row):
        return sum(
            1 if pd.notna(row[tp]) and row[tp] >= thresholds[tp] else 0
            for tp in top_predictors
        )
    tsv["gemvap1"] = tsv.apply(gemvap1_score, axis=1)

    missense = tsv[
        tsv["Consequence"].str.contains("missense_variant", na=False) &
        tsv["aaref"].isin(AA_ORDER) &
        tsv["aaalt"].isin(AA_ORDER) &
        (tsv["aaref"] != tsv["aaalt"])
    ]
    return missense.groupby("aaref")["gemvap1"].agg(["mean", "sem", "count"]).reindex(AA_ORDER)


def classify_aa_by_consensus(
    base: dict, data_path: str, output_path: str, n_tiers: int = 6,
) -> pd.DataFrame:
    """
    Rank wild-type amino acids by mean GEMVAP 1 consensus score and split
    them into n_tiers groups at the (n_tiers - 1) largest gaps between
    consecutive sorted means -- a simple natural-breaks clustering, so tier
    boundaries fall in actual gaps in the data rather than at arbitrary fixed
    cutoffs. Saves (aaref, mean, sem, count, tier) as a CSV sorted by
    descending mean, and returns it as a DataFrame indexed by aaref.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    summary = _gemvap1_score_by_aa(base, data_path).dropna().sort_values("mean", ascending=False)

    default_labels = ["Very High", "High", "Moderate-High", "Moderate", "Low-Moderate", "Low"]
    tier_labels = default_labels[:n_tiers] if n_tiers <= len(default_labels) \
        else [f"Tier {i + 1}" for i in range(n_tiers)]

    means = summary["mean"].to_numpy()
    n_cuts = n_tiers - 1
    if n_cuts > 0:
        gaps = -np.diff(means)  # sorted descending, so consecutive diffs are >= 0
        cut_after_row = set(np.argsort(gaps)[-n_cuts:].tolist())
    else:
        cut_after_row = set()

    tiers = []
    label_idx = 0
    for i in range(len(summary)):
        tiers.append(tier_labels[label_idx])
        if i in cut_after_row:
            label_idx += 1
    summary["tier"] = tiers

    summary.to_csv(output_path, index_label="aaref")
    return summary


def plot_ddgun_vs_gemvap1(
    base: dict, data_path: str, ddgun_path: str, output_path: str, dpi: int = 600,
) -> None:
    """
    Two-panel bar chart comparing mean GEMVAP 1 consensus score (top) to mean
    DDGun-Seq S_DDG[SEQ] stability score (bottom), both grouped by wild-type
    amino acid, across all FBN1 missense variants.

    data_path is the same full predictor table used to train GEMVAP 1 (must
    have aaref/aaalt/Consequence columns). ddgun_path is DDGun-Seq's per-variant
    output TSV (columns: seqfile, variant, s_ddg, t_ddg, stability), keyed by
    variant strings like "M1A" (wild-type letter + position + mutant letter).
    Amino acids are ordered along the x-axis by decreasing mean GEMVAP 1
    score, so both panels share the same order and the DDGun-Seq panel reads
    as "stability score at each GEMVAP-1-pathogenicity rank."

    Ported from archive/plot_sddg_vs_gemvap1.py.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gemvap_summary = _gemvap1_score_by_aa(base, data_path).dropna().sort_values(
        "mean", ascending=False,
    )
    aa_order = gemvap_summary.index.tolist()

    ddg = pd.read_csv(
        ddgun_path, sep="\t", comment="#",
        names=["seqfile", "variant", "s_ddg", "t_ddg", "stability"],
    )
    ddg["wt_aa"] = ddg["variant"].str[0]
    ddg_summary = ddg.groupby("wt_aa")["s_ddg"].agg(["mean", "sem"]).reindex(aa_order)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    x = np.arange(len(aa_order))
    bar_kw = dict(width=0.65, edgecolor="white", linewidth=0.4,
                  error_kw=dict(elinewidth=0.8, capsize=3, ecolor="grey"))

    g_vals = gemvap_summary["mean"].values
    g_norm = (g_vals - g_vals.min()) / (g_vals.max() - g_vals.min())
    ax1.bar(x, gemvap_summary["mean"], yerr=gemvap_summary["sem"],
            color=plt.cm.Oranges(0.35 + 0.55 * g_norm), **bar_kw)
    ax1.set_ylabel("Mean GEMVAP 1 score", fontsize=11)
    ax1.set_title(
        "Average GEMVAP 1 score vs. DDGun-Seq S_DDG per wild-type amino acid\n"
        "P35555 (FBN1) — all missense variants",
        fontsize=12,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(bottom=False)

    d_vals = ddg_summary["mean"].values
    d_norm = np.clip((d_vals - d_vals.min()) / (d_vals.max() - d_vals.min()), 0, 1)
    ax2.bar(x, ddg_summary["mean"], yerr=ddg_summary["sem"],
            color=plt.cm.Blues_r(0.25 + 0.65 * d_norm), **bar_kw)
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Mean S_DDG[SEQ]", fontsize=11)
    ax2.set_xlabel("Wild-type amino acid", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(aa_order, fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def display_figures(output_dir, filenames, width: int = 900) -> None:
    """Display each named PNG under output_dir that exists (skips missing files)."""
    output_dir = Path(output_dir)
    for fname in filenames:
        p = output_dir / fname
        if p.exists():
            print(fname)
            display(Image(str(p), width=width))


def display_all_figures(dir_path, pattern: str = "*.png", width: int = 900) -> None:
    """Display every file matching pattern under dir_path, sorted by name."""
    for p in sorted(Path(dir_path).glob(pattern)):
        print(p.name)
        display(Image(str(p), width=width))


def build_thresholds_table(models, calib_dir) -> pd.DataFrame:
    """
    Read each model's {name}_thresholds.csv from calib_dir and pivot into a
    Model x evidence_level table of calibrated consensus-score thresholds.
    Returns an empty DataFrame if no threshold CSVs are found.
    """
    calib_dir = Path(calib_dir)
    thr_rows = []
    for m in models:
        p = calib_dir / f"{m['name']}_thresholds.csv"
        if p.exists():
            t = pd.read_csv(p)
            t.insert(0, "Model", m["name"])
            thr_rows.append(t)

    if not thr_rows:
        return pd.DataFrame()

    thr_all = pd.concat(thr_rows, ignore_index=True)
    pivot = thr_all.pivot(index="Model", columns="evidence_level", values="score_threshold")
    ordered_cols = [c for c in [
        "PP3_VeryStrong", "PP3_Strong", "PP3_Moderate", "PP3_Supporting",
        "BP4_Supporting", "BP4_Moderate", "BP4_Strong", "BP4_VeryStrong",
    ] if c in pivot.columns]
    return pivot[ordered_cols]


def plot_test_score_distributions(models, output_dir, dpi: int = 600) -> str | None:
    """
    Per-model consensus-score histograms (pathogenic vs. control) on the test
    set, with calibrated PP3/BP4 threshold lines overlaid. Requires
    test_scores.csv and calibration/{name}_thresholds.csv under output_dir.
    Returns the saved plot path, or None if test_scores.csv is missing.
    """
    output_dir = Path(output_dir)
    test_scores_path = output_dir / "test_scores.csv"
    calib_dir = output_dir / "calibration"
    if not test_scores_path.exists():
        return None

    tsc = pd.read_csv(test_scores_path)
    n_m = len(models)
    fig, axes = plt.subplots(1, n_m, figsize=(6 * n_m, 5), squeeze=False)
    fig.suptitle("Test-set score distributions with PP3/BP4 thresholds", fontsize=12, fontweight="bold")

    for ax, m in zip(axes[0], models):
        col = m["col"]
        if col not in tsc.columns:
            ax.set_visible(False)
            continue

        path_s = tsc.loc[tsc["label"] == "pathogenic", col].dropna()
        ctrl_s = tsc.loc[tsc["label"] == "benign", col].dropna()
        max_s = int(max(tsc[col].max(), 1)) + 1
        bins = list(range(0, max_s + 2))
        ax.hist(path_s, bins=bins, alpha=0.65, density=True,
                label=f"Pathogenic (n={len(path_s)})", color="#d62728")
        ax.hist(ctrl_s, bins=bins, alpha=0.65, density=True,
                label=f"Control (n={len(ctrl_s)})", color="#1f77b4")

        thr_path = calib_dir / f"{m['name']}_thresholds.csv"
        if thr_path.exists():
            thr = (pd.read_csv(thr_path)
                   .set_index("evidence_level")["score_threshold"]
                   .dropna())
            for lvl, val in thr.items():
                if lvl.startswith("PP3"):
                    ax.axvline(val - 0.5, color="#c0392b", linestyle=":", linewidth=1.2,
                               alpha=0.85, label=f"{lvl}>={val:.0f}")
                elif lvl.startswith("BP4"):
                    ax.axvline(val - 0.5, color="#2980b9", linestyle=":", linewidth=1.2,
                               alpha=0.85, label=f"{lvl}<={val:.0f}")

        ax.set_xlabel(f"{m['name']} consensus score")
        ax.set_ylabel("Density")
        ax.set_title(m["name"])
        ax.legend(fontsize=6, loc="upper left")

    plt.tight_layout()
    out_path = output_dir / "intermediate_score_distributions_thresholds.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _prep_boxplot_data(acc_df: pd.DataFrame, metrics: list, metrics_cols: list):
    """Split a performance table into individual-predictor distributions
    (boxplot) and GEMVAP points (scatter) for plot_publication_boxplot."""
    tool_df = acc_df[~acc_df["model"].str.startswith("GEMVAP")]
    gemvap = {
        row["model"]: [float(row[c]) if not isinstance(row[c], str) else None for c in metrics_cols]
        for _, row in acc_df[acc_df["model"].str.startswith("GEMVAP")].iterrows()
    }
    box_data = {m: tool_df[c].dropna().tolist() for m, c in zip(metrics, metrics_cols)}
    return box_data, gemvap


def _ordered_gemvap_points(gemvap_dict: dict):
    return [gemvap_dict.get(f"GEMVAP_{i}", [None, None, None]) for i in range(1, 4)]


def _peek_model_point(acc_df: pd.DataFrame, model_name: str, metrics_cols: list):
    """
    Read a single model's row out of acc_df by exact name match (if present)
    and return [f1, mcc, accuracy] (None for any missing/non-numeric metric,
    or all-None if model_name isn't in acc_df -- e.g. filtered out earlier
    for exceeding max_vus_rate). Unlike GEMVAP, a real individual predictor
    like REVEL stays IN the box's IQR population -- it's a genuine member of
    "the predictors" the min/max whiskers describe -- this only reads its
    value to additionally overlay it as its own highlighted scatter point.
    """
    match = acc_df[acc_df["model"] == model_name]
    if match.empty:
        return [None, None, None]
    row = match.iloc[0]
    return [float(row[c]) if not isinstance(row[c], str) else None for c in metrics_cols]


def _lower_legends_clear_of_data(ax, legend_specs, margin_frac: float = 0.05, max_iter: int = 10):
    """
    Place one or more corner legends (e.g. loc='lower left' / 'lower right')
    and, if any of them overlaps the actual plotted data (boxes, whiskers,
    scatter points, value annotations), push them down by extending the
    y-axis lower limit and retrying -- so the legend ends up lower, clear of
    the data, instead of floating on top of it.
    """
    fig = ax.figure

    def _place():
        placed = []
        for handles, kwargs in legend_specs:
            leg = ax.legend(handles=handles, **kwargs)
            ax.add_artist(leg)
            placed.append(leg)
        return placed

    def _data_bboxes(renderer):
        artists = list(ax.patches) + list(ax.lines) + list(ax.collections) + list(ax.texts)
        return [a.get_window_extent(renderer) for a in artists if a.get_visible()]

    legends = _place()
    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        data_bboxes = _data_bboxes(renderer)
        legend_bboxes = [leg.get_window_extent(renderer) for leg in legends]
        overlap = any(lb.overlaps(db) for lb in legend_bboxes for db in data_bboxes)
        if not overlap:
            break
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo - margin_frac * (yhi - ylo), yhi)
        for leg in legends:
            leg.remove()
        legends = _place()
    return legends


def _finalize_boxplot_layout(ax, annotations_by_column, min_gap_px: float = 12,
                              top_margin_frac: float = 0.04, max_iter: int = 12):
    """
    Two passes, repeated until stable:
    1. Within each x-column, push overlapping value-annotations apart
       vertically (nudging the lower-valued one further down) so labels for
       close-scoring models (e.g. GEMVAP 1/2/3 on the same metric) don't
       overlap each other.
    2. If any scatter point or label still pokes above the top of the axes
       (e.g. the highest-scoring model's label getting clipped by the
       frame), extend the y-axis upper limit so it renders lower, clear of
       the top edge.
    """
    fig = ax.figure
    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        changed = False

        for anns in annotations_by_column.values():
            if len(anns) < 2:
                continue
            ordered = sorted(anns, key=lambda a: -a.get_window_extent(renderer).y0)
            for upper, lower in zip(ordered, ordered[1:]):
                gap = upper.get_window_extent(renderer).y0 - lower.get_window_extent(renderer).y1
                if gap < min_gap_px:
                    ox, oy = lower.xyann
                    lower.xyann = (ox, oy - (min_gap_px - gap))
                    changed = True

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer)
        artists = list(ax.collections) + list(ax.texts)
        overflow = max(
            (a.get_window_extent(renderer).y1 - ax_bbox.y1 for a in artists if a.get_visible()),
            default=0,
        )
        if overflow > 0:
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(ylo, yhi + top_margin_frac * (yhi - ylo))
            changed = True

        if not changed:
            break


def _uniform_offsets(offset):
    """Broadcast a single (dx, dy) annotation offset to all 3 metric columns
    -- for models whose label should sit the same way relative to the point
    regardless of which column (F1/MCC/Accuracy) it's in."""
    return [offset, offset, offset]


def _draw_perf_boxplot(ax, data_dict, model_perfs, metrics, title, annotation_offsets,
                        box_color, median_color, point_colors, point_labels, auto_nudge=True):
    """
    annotation_offsets: one entry per model, each either a single (dx, dy)
    tuple (applied to all 3 metric columns -- see _uniform_offsets) or a
    list of 3 (dx, dy) tuples, one per column, for labels that need to sit
    differently per column (e.g. hugging a point near the left/right plot
    edge from a different side than one in the middle column).

    auto_nudge: when True (default), overlapping labels get pushed apart by
    _finalize_boxplot_layout. Set False when annotation_offsets already
    encode exact, deliberately-chosen label positions (e.g. solved from
    target coordinates) -- the collision-avoidance pass only pushes labels
    DOWN and can cascade badly when several targets are intentionally close
    together, undoing the precise placement it was given.
    """
    data = [data_dict[m] for m in metrics]
    ax.boxplot(
        data, patch_artist=True, labels=metrics,
        # whis=(0, 100): whiskers span the true min-max, matching the legend's
        # "Minimum and Maximum performance of predictors" label. matplotlib's
        # default whis=1.5 draws Tukey fences instead (1.5x IQR beyond
        # Q1/Q3) and, combined with showfliers=False, would silently drop any
        # predictor beyond that fence from the plot entirely.
        whis=(0, 100),
        boxprops=dict(facecolor=box_color, color="gray", linewidth=1.5),
        medianprops=dict(color=median_color, linewidth=2),
        whiskerprops=dict(color="gray", linewidth=1.5),
        capprops=dict(color="gray", linewidth=1.5),
        flierprops=dict(markerfacecolor="gray", marker="o", markersize=5, linestyle="none"),
        showfliers=False,
    )
    ax.tick_params(axis="both", which="major", length=5, width=1, direction="out",
                    bottom=True, left=True, top=False, right=False)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(metrics)
    ax.grid(False)
    ax.set_title(title, fontsize=14, pad=12)

    def _is_valid(v):
        # Excludes both None (metric not applicable, e.g. GEMVAP models
        # aren't scored on a metric) and NaN (metric computed but undefined,
        # e.g. MCC when the confusion-matrix denominator is 0) -- `v is not
        # None` alone lets NaN through, which matplotlib silently drops from
        # scatter but still turns into a stray "nan" annotate() call at an
        # undefined position.
        return v is not None and not math.isnan(v)

    for perfs, color, lbl in zip(model_perfs, point_colors, point_labels):
        valid_x = [j + 1 for j, v in enumerate(perfs) if _is_valid(v)]
        valid_y = [v for v in perfs if _is_valid(v)]
        if valid_x:
            ax.scatter(valid_x, valid_y, color=color, s=100, zorder=3, edgecolor="grey", label=lbl)

    annotations_by_column = {1: [], 2: [], 3: []}
    for perfs, color, offsets in zip(model_perfs, point_colors, annotation_offsets):
        per_column_offsets = offsets if isinstance(offsets[0], (tuple, list)) else _uniform_offsets(offsets)
        for j, (val, offset) in enumerate(zip(perfs, per_column_offsets), start=1):
            if _is_valid(val):
                ann = ax.annotate(f"{val:.2f}", (j, val), textcoords="offset points", xytext=offset,
                                   ha="center", color=color, fontsize=12)
                annotations_by_column[j].append(ann)

    if auto_nudge:
        _finalize_boxplot_layout(ax, annotations_by_column)


def plot_publication_boxplot(
    all_acc_df: pd.DataFrame, all_acc_dom_df: pd.DataFrame, output_dir,
    max_vus_rate: float = 0.5,
) -> tuple[str, str]:
    """
    Publication-style boxplots: interquartile range of individual predictor
    performance (F1 / MCC / Accuracy) with GEMVAP 1/2/3 overlaid as points,
    saved as two standalone figures -- one for the full test set, one for the
    non-cysteine/non-conserved-domain subset -- under output_dir. Returns
    (path_full, path_noncys_nondom).

    Models with vus_rate > max_vus_rate are dropped from each panel before
    plotting (independently per panel, since a model's VUS rate can differ
    between the full test set and the harder subset) -- a model that rarely
    commits to a call isn't representative of "typical" predictor performance.
    """
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 14,
        "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 14, "figure.titlesize": 18,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.edgecolor": "black", "axes.linewidth": 1.2,
    })

    metrics = ["F1 Score", "Matthews Correlation\nCoefficient", "Accuracy"]
    metrics_cols = ["f1", "mcc", "accuracy"]

    box_color = "lightsteelblue"
    median_color = "midnightblue"
    point_colors = ["#1876BD", "#E41175", "#000000"]
    point_labels = ["GEMVAP 1", "GEMVAP 2", "GEMVAP 3"]

    all_acc_df = all_acc_df[all_acc_df["vus_rate"] <= max_vus_rate]
    all_acc_dom_df = all_acc_dom_df[all_acc_dom_df["vus_rate"] <= max_vus_rate]

    box_base, gemvap_base = _prep_boxplot_data(all_acc_df, metrics, metrics_cols)
    box_dom, gemvap_dom = _prep_boxplot_data(all_acc_dom_df, metrics, metrics_cols)
    pts_base = _ordered_gemvap_points(gemvap_base)
    pts_dom = _ordered_gemvap_points(gemvap_dom)

    legend_models = [
        Line2D([], [], marker="o", color=c, label=lbl, linestyle="None",
               markersize=10, markeredgecolor="grey")
        for c, lbl in zip(point_colors, point_labels)
    ]
    custom_handles = [
        Patch(facecolor=box_color, edgecolor="gray",
              label="Interquartile range for the\nperformance of predictors\n(25th-75th percentile)"),
        Line2D([], [], color=median_color, linewidth=2, label="Median performance\nof predictors"),
        Line2D([], [], color="gray", linewidth=1.5, linestyle="-",
               label="Minimum and Maximum\nperformance of predictors"),
    ]

    def _save_panel(box_data, pts, title, annotation_offsets, filename) -> str:
        fig, ax = plt.subplots(figsize=(9, 8))
        _draw_perf_boxplot(ax, box_data, pts, metrics, title, annotation_offsets,
                            box_color, median_color, point_colors, point_labels)
        ax.set_ylabel("Performance Value")

        _lower_legends_clear_of_data(ax, [
            (legend_models, dict(frameon=True, framealpha=0.9, facecolor="white",
                                  edgecolor="gray", loc="lower left", title="Selected Models")),
            (custom_handles, dict(frameon=True, framealpha=0.9, fontsize=12, facecolor="white",
                                   edgecolor="gray", loc="lower right", title="Boxplot Interpretation")),
        ])

        plt.tight_layout()
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    path_full = _save_panel(box_base, pts_base, "Full Test Set",
                             [(-20, -3), (25, -5), (-25, -10)],
                             "model_performance_boxplot_full.png")
    path_dom = _save_panel(box_dom, pts_dom,
                            "Without Cysteines nor\nUltra-conserved Positions",
                            [(-25, 0), (25, -4), (20, 5)],
                            "model_performance_boxplot_noncys_nondom.png")
    return path_full, path_dom


def plot_publication_boxplot_with_alphamissense(
    all_acc_df: pd.DataFrame, all_acc_dom_df: pd.DataFrame,
    am_metrics_full: dict | None, am_metrics_dom: dict | None,
    output_dir, max_vus_rate: float = 0.5,
) -> tuple[str, str]:
    """
    Variant of plot_publication_boxplot that also overlays AlphaMissense
    (Cheng et al. 2023) and REVEL as their own highlighted points alongside
    GEMVAP 1/2/3.

    GEMVAP is excluded from the box on purpose -- it's the novel method being
    benchmarked against the pool of prior predictors, not a member of that
    pool. AlphaMissense and REVEL are different: they ARE individual
    predictors, so unlike GEMVAP they stay IN the box's IQR/min-max
    population -- the whiskers must still reflect their values, or "minimum
    performance of predictors" would be a lie whenever one of them happens to
    be the worst performer. AlphaMissense additionally predates Pejaver et
    al. (2022) and has no entry in pejaver_thresholds.csv, so it's scored
    separately (see pejaver_tools.compute_alphamissense_metrics) and passed
    in rather than read from all_acc_df/all_acc_dom_df -- but it is *not*
    added into the box population either, since it was never part of it to
    begin with (there's no "un-highlighted" version of this comparison that
    included it). REVEL already has a row in all_acc_df/all_acc_dom_df (it's
    one of the Pejaver-calibrated tools), so its box contribution is
    untouched -- it's simply read a second time to draw its point.

    am_metrics_full/am_metrics_dom: dict with 'f1'/'mcc'/'accuracy'/'vus_rate'
    keys (e.g. compute_alphamissense_metrics(...).iloc[0].to_dict()), or None
    if AlphaMissense couldn't be scored (missing data file). Its point (only
    -- AlphaMissense was never part of the box) is dropped from a panel if
    its vus_rate exceeds max_vus_rate, same rule as every other model. Saved
    as two standalone figures under output_dir. Returns (path_full,
    path_noncys_nondom).
    """
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 14,
        "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 14, "figure.titlesize": 18,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.edgecolor": "black", "axes.linewidth": 1.2,
    })

    metrics = ["F1 Score", "Matthews Correlation\nCoefficient", "Accuracy"]
    metrics_cols = ["f1", "mcc", "accuracy"]

    box_color = "lightsteelblue"
    median_color = "midnightblue"
    point_colors = ["#1876BD", "#E41175", "#000000", "#DFA829", "#2CA02C"]
    point_labels = ["GEMVAP 1", "GEMVAP 2", "GEMVAP 3", "AlphaMissense", "REVEL"]

    all_acc_df = all_acc_df[all_acc_df["vus_rate"] <= max_vus_rate]
    all_acc_dom_df = all_acc_dom_df[all_acc_dom_df["vus_rate"] <= max_vus_rate]

    # REVEL stays in all_acc_df/all_acc_dom_df below -- it's read here only to
    # additionally draw its point, not removed from the box population.
    revel_point_full = _peek_model_point(all_acc_df, "REVEL", metrics_cols)
    revel_point_dom = _peek_model_point(all_acc_dom_df, "REVEL", metrics_cols)

    def _am_point(am_metrics):
        if am_metrics is None or am_metrics.get("vus_rate", 1.0) > max_vus_rate:
            return [None, None, None]
        return [am_metrics.get(c) for c in metrics_cols]

    box_base, gemvap_base = _prep_boxplot_data(all_acc_df, metrics, metrics_cols)
    box_dom, gemvap_dom = _prep_boxplot_data(all_acc_dom_df, metrics, metrics_cols)
    pts_base = _ordered_gemvap_points(gemvap_base) + [_am_point(am_metrics_full), revel_point_full]
    pts_dom = _ordered_gemvap_points(gemvap_dom) + [_am_point(am_metrics_dom), revel_point_dom]

    legend_models = [
        Line2D([], [], marker="o", color=c, label=lbl, linestyle="None",
               markersize=10, markeredgecolor="grey")
        for c, lbl in zip(point_colors, point_labels)
    ]
    custom_handles = [
        Patch(facecolor=box_color, edgecolor="gray",
              label="Interquartile range for the\nperformance of predictors\n(25th-75th percentile)"),
        Line2D([], [], color=median_color, linewidth=2, label="Median performance\nof predictors"),
        Line2D([], [], color="gray", linewidth=1.5, linestyle="-",
               label="Minimum and Maximum\nperformance of predictors"),
    ]

    def _save_panel(box_data, pts, title, annotation_offsets, filename, auto_nudge=True) -> str:
        fig, ax = plt.subplots(figsize=(9, 8))
        _draw_perf_boxplot(ax, box_data, pts, metrics, title, annotation_offsets,
                            box_color, median_color, point_colors, point_labels,
                            auto_nudge=auto_nudge)
        ax.set_ylabel("Performance Value")

        _lower_legends_clear_of_data(ax, [
            (legend_models, dict(frameon=True, framealpha=0.9, facecolor="white",
                                  edgecolor="gray", loc="lower left", title="Selected Models")),
            (custom_handles, dict(frameon=True, framealpha=0.9, fontsize=12, facecolor="white",
                                   edgecolor="gray", loc="lower right", title="Boxplot Interpretation")),
        ])

        plt.tight_layout()
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    # REVEL's label sits close to its own point rather than far off to one
    # side, and hugs the side away from the nearest plot edge (right, for the
    # leftmost F1/MCC columns; left, for the rightmost Accuracy column) so it
    # never runs past the axes into the tick-label margin.
    revel_offsets_dom = [(22, -8), (22, -8), (-22, -8)]

    # Full Test Set: offsets solved to hit exact target label-center
    # coordinates (data units). auto_nudge=False because those targets are
    # deliberately close together in places (e.g. REVEL and GEMVAP 2's F1
    # labels), and the collision-avoidance pass -- which only ever pushes
    # labels DOWN -- cascades badly on tight clusters like that, undoing the
    # precise placement instead of gently nudging it.
    path_full = _save_panel(box_base, pts_base, "", [
        (-22.6, -0.8),                                 # GEMVAP 1
        [(-22.6, -5), (-22.6, -3.3), (-22.6, 0.4)],    # GEMVAP 2
        [(22.6, -2.3), (22.3, 0.0), (22.3, -0.3)],     # GEMVAP 3
        [(22.6, -1.1), (22.6, -2.9), (22.6, -1.9)],    # AlphaMissense
        [(-22.6, 4.3), (-22.6, 3), (-43, -4.1)],       # REVEL
    ], "model_performance_boxplot_full_with_AM_REVEL.png", auto_nudge=False)
    path_dom = _save_panel(box_dom, pts_dom, "Without Cysteines nor\nUltra-conserved Positions", [
        (-25, 0),                           # GEMVAP 1
        [(25, 6), (25, 6), (25, -4)],       # GEMVAP 2 -- higher, except Accuracy
        (20, 15),                           # GEMVAP 3 -- higher on every metric
        (-20, 12),                          # AlphaMissense
        revel_offsets_dom,                  # REVEL
    ], "model_performance_boxplot_noncys_nondom_with_AM_REVEL.png", auto_nudge=False)
    return path_full, path_dom


def plot_confusion_matrices(full_test_df: pd.DataFrame, output_dir, dpi: int = 600) -> str | None:
    """
    Per-GEMVAP-model confusion matrix (TP/FN/FP/TN heatmap) from the Step 7
    full-test-set results table. Returns the saved plot path, or None if
    full_test_df has no GEMVAP rows.
    """
    output_dir = Path(output_dir)
    gemvap_rows = full_test_df[full_test_df["model"].str.startswith("GEMVAP")].reset_index(drop=True)
    if gemvap_rows.empty:
        return None

    n_m = len(gemvap_rows)
    fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 5), squeeze=False)
    fig.suptitle("Confusion matrices -- full held-out test set", fontsize=12, fontweight="bold")

    for ax, (_, row) in zip(axes[0], gemvap_rows.iterrows()):
        tp = int(row["tp"]); tn = int(row["tn"])
        fp = int(row["fp"]); fn = int(row["fn"])
        n_vus = int(row.get("n_vus", row["n_total_perf"] - row["n_classified"]))

        mat = np.array([[tp, fn], [fp, tn]], dtype=float)
        mat_norm = mat / mat.sum() if mat.sum() > 0 else mat

        ax.imshow(mat_norm, cmap="RdYlGn", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                label = f"{int(mat[i, j])} ({mat_norm[i, j]:.1%})"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if mat_norm[i, j] < 0.3 or mat_norm[i, j] > 0.7 else "black")

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Pathogenic", "Pred: Benign"])
        ax.set_yticklabels(["True: Pathogenic", "True: Benign"])
        ax.set_title(
            f"{row['model']}\n"
            f"F1={row['f1']:.3f}  Acc={row['accuracy']:.3f}  "
            f"MCC={float(row['mcc']):.3f}\n"
            f"VUS={n_vus} ({row['vus_rate']:.1%} of test set)"
        )
        ax.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    out_path = output_dir / "intermediate_confusion_matrices.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_bootstrap_distributions(output_dir, test_dom_mask: bool, dpi: int = 600):
    """
    Boxplots of F1 / MCC / VUS-rate across all bootstrap iterations (Step 8),
    plus a mean +/- std summary table for GEMVAP models only.

    Returns (plot_path, gemvap_summary_df). Raises FileNotFoundError if the
    bootstrap metrics CSV for the current test_dom_mask setting hasn't been
    generated yet.
    """
    output_dir = Path(output_dir)
    suffix = "_noncys_nondom" if test_dom_mask else ""
    bs_path = output_dir / f"bootstrap_performance_metrics{suffix}.csv"
    if not bs_path.exists():
        raise FileNotFoundError(f"{bs_path} not found -- run the bootstrap cell first.")

    bs = pd.read_csv(bs_path)
    model_order = sorted(bs["model"].unique(), key=lambda m: (0 if m.startswith("GEMVAP") else 1, m))
    colors = ["#ff7f0e" if m.startswith("GEMVAP") else "#aec7e8" for m in model_order]

    metrics = [("f1", "F1 score"), ("mcc", "MCC"), ("vus_rate", "VUS rate")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Bootstrap performance distributions ({bs['iteration'].nunique()} iterations)",
                 fontsize=12, fontweight="bold")

    for ax, (col, ylabel) in zip(axes, metrics):
        data_list = [bs.loc[bs["model"] == m, col].dropna().values for m in model_order]
        bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_xticks(range(1, len(model_order) + 1))
        ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "intermediate_bootstrap_distributions.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    gemvap_bs = bs[bs["model"].str.startswith("GEMVAP")]
    summary = None
    if not gemvap_bs.empty:
        summary = (gemvap_bs.groupby("model")[["f1", "mcc", "vus_rate"]]
                   .agg(["mean", "std"])
                   .round(4))
        summary.columns = [f"{c}_{s}" for c, s in summary.columns]

    return str(out_path), summary


def plot_horizontal_gemvap3(base, dom, df, output_path: str, dpi: int = 600) -> None:
    from packages.package1.predictor_selection import build_trace

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    top_predictors = dom["top_predictors"]
    thresholds = {k: dom["rbc"]["threshold"]["case"][k] for k in top_predictors}
    training_data = base["training_data"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace_1 = np.array(build_trace(
            training_data[~df.is_case & df.is_ctrl][top_predictors],
            thresholds, name="1", text=False, y_percentage=False))
        trace_2 = np.array(build_trace(
            training_data[df.is_case & ~df.is_ctrl][top_predictors],
            thresholds, name="2", text=False, y_percentage=False))

    test_bar = list(range(len(top_predictors) + 1))
    sorted_dict = _aa_consensus(training_data, df, top_predictors, thresholds)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10), layout="constrained")

    ax1.bar(test_bar, list(trace_1), color="#1876BD", label="Control")
    ax1.bar(test_bar, list(trace_2), bottom=list(trace_1), color="#FAA71A", label="Pathogenic")
    max_height = max(trace_1 + trace_2)
    ax1.vlines(dom["ci_data_ks"]["ks"]["cons"] - 0.5, 0, max_height, transform=ax1.get_xaxis_transform(), colors="r")
    ax1.set_xlabel("Consensus Score (GEMVAP 3)")
    ax1.set_ylabel("Number of variants")
    ax1.set_xticks(test_bar)
    ax1.set_xticklabels([str(i) for i in test_bar])
    ax1.legend(loc="best")

    ax2.bar(sorted_dict.keys(), sorted_dict.values(), width=0.6, align="center", color="#1876BD")
    ax2.set_xlabel("Amino Acid")
    ax2.set_ylabel("Level of Consensus (GEMVAP 3)")
    ax2.set_xticks(range(len(sorted_dict)))
    ax2.set_xticklabels(sorted_dict.keys())
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels([0, 1, 2, 3, 4])
    ax2.legend(handles=[
        Patch(facecolor="#1876BD", label=r"Average number of predictors predicting $\bf{\mathit{Pathogenic}}$"),
    ], loc="upper left")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
