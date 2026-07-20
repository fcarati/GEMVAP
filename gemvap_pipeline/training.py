"""
Per-model training steps for the GEMVAP variants (1, 2, 3): fitting, exporting
KS/F1 statistics, and producing the associated plots.
"""
from pathlib import Path

import pandas as pd

from .verbose import info, result, section, step, substep
from .data import annotate_hgvsp, compute_grouped_data, load_processor
from .model import (
    annotate_consensus,
    extract_ks_values,
    fit_gemvap,
    fit_gemvap2,
    fit_gemvap3,
    load_fit_result,
    save_fit_result,
)
from .visualization import (
    plot_figure4,
    plot_horizontal_gemvap1,
    plot_horizontal_gemvap2,
    plot_horizontal_gemvap3,
    plot_ks_f1_curve,
)


def _export_ks_f1(fit_result, output_dir, prefix: str):
    ks_values = extract_ks_values(fit_result["ks_case_ctrl_bf"])
    ks_df = pd.DataFrame(
        {"model": list(ks_values.keys()), "ks_statistic": list(ks_values.values())}
    ).sort_values(by="ks_statistic", ascending=True)
    ks_df.to_csv(Path(output_dir) / f"{prefix}_KS_values.csv", index=False)

    stats = fit_result["ci_data_ks"]["ks"]["ci"]["stats"]
    stats.to_csv(Path(output_dir) / f"{prefix}_F1_score.csv", index=False)
    return ks_values, stats


def train_gemvap1(df, training_filter, args, output_dir) -> dict:
    section("Step 2a — Training GEMVAP 1  (all missense variants)")
    info("GEMVAP 1 is the broadest model: it trains on ALL missense variants in the")
    info("training set, regardless of amino-acid type or domain position.")
    info("It serves as the baseline model and the source of the Figure 4 residue-level map.")
    fit_path = Path(output_dir) / "GEMVAP_1_fit.pkl"
    if fit_path.exists():
        step(f"Cached fit found at {fit_path} — loading without re-training")
        info("Delete this file to force re-training from scratch.")
        return load_fit_result(fit_path)

    step("Fitting GEMVAP 1 — KS ranking -> ROC thresholding -> consensus optimisation")
    base = fit_gemvap(df, training_filter, it=args.model_iteration, binary=True)
    save_fit_result(base, fit_path)
    result(f"Fit saved to {fit_path}")

    step("Exporting KS statistics and incremental-consensus F1 curve to CSV")
    info("These CSVs record how F1 evolves as predictors are added one by one —")
    info("useful for inspecting which predictors contribute most to discrimination.")
    ks_values, stats = _export_ks_f1(base, output_dir, "GEMVAP_1")

    step("Generating visualisation plots")
    info("KS + F1 curve: shows predictor rankings and the optimal consensus cut-off.")
    info("Horizontal score plot: pathogenic vs control score distributions per predictor.")
    plot_ks_f1_curve(
        ks_values, stats, str(Path(output_dir) / "Combined_KS_F1_SharedAxis.png"),
        f1_ylim=(0.65, 0.75),
    )
    plot_horizontal_gemvap1(base, df, str(Path(output_dir) / "Combined_Horizontal_GEMVAP_1.png"))

    return base


def train_gemvap2(df, training_filter, base, args, output_dir) -> dict:
    section("Step 2b — Training GEMVAP 2  (non-cysteine variants only)")
    info("Rationale: cysteine variants in FBN1 EGF/cbEGF domains are almost always")
    info("pathogenic because they disrupt structural disulfide bonds. Including them")
    info("in training would let the model learn that 'cysteine change = pathogenic',")
    info("which is trivially obvious and does not generalise to non-cysteine cases.")
    info("GEMVAP 2 therefore excludes cysteine variants from training to focus the")
    info("predictor selection on harder, non-cysteine missense variants.")
    fit_path = Path(output_dir) / "GEMVAP_2_fit.pkl"
    if fit_path.exists():
        step(f"Cached fit found at {fit_path} — loading without re-training")
        return load_fit_result(fit_path)

    step("Fitting GEMVAP 2 — non-cysteine variants only")
    cyst = fit_gemvap2(df, training_filter, it=args.model_iteration, binary=True)
    save_fit_result(cyst, fit_path)
    result(f"Fit saved to {fit_path}")

    step("Exporting GEMVAP 2 KS statistics and F1 curve")
    ks_values_v2, stats_v2 = _export_ks_f1(cyst, output_dir, "GEMVAP_2")

    step("Generating GEMVAP 2 visualisation plots")
    info("The horizontal plot also shows GEMVAP 1 scores for comparison — this lets")
    info("you see how the predictor panel shifts when cysteine variants are excluded.")
    plot_ks_f1_curve(
        ks_values_v2, stats_v2, str(Path(output_dir) / "GEMVAP_2_Combined_KS_F1.png"),
        f1_ylim=(0.3, 0.55),
    )
    plot_horizontal_gemvap2(
        base, cyst, df, args.conserved_data_path,
        str(Path(output_dir) / "Combined_Horizontal_GEMVAP_2.png"),
    )

    return cyst


def train_gemvap3(df, training_filter, base, args, output_dir) -> dict:
    section("Step 2c — Training GEMVAP 3  (non-cysteine + outside conserved domains)")
    info("GEMVAP 3 further restricts to variants that are BOTH non-cysteine AND outside")
    info("positions conserved across FBN1 EGF-Ca2+-binding domain cores.")
    info("Conserved-domain positions are under strong structural constraint: computational")
    info("predictors trained on general missense variation can systematically mis-score")
    info("them because the structural context is atypical. Excluding them lets GEMVAP 3")
    info("focus on variants where predictor combination adds genuine value.")
    info("This subset is the most diagnostically challenging class in FBN1 variant interpretation.")
    fit_path = Path(output_dir) / "GEMVAP_3_fit.pkl"
    if fit_path.exists():
        step(f"Cached fit found at {fit_path} — loading without re-training")
        return load_fit_result(fit_path)

    step("Fitting GEMVAP 3 — non-cysteine variants outside conserved domain positions")
    dom = fit_gemvap3(
        df, training_filter, args.conserved_data_path,
        it=args.model_iteration, binary=True,
    )
    save_fit_result(dom, fit_path)
    result(f"Fit saved to {fit_path}")

    step("Exporting GEMVAP 3 KS statistics and F1 curve")
    ks_values_v3, stats_v3 = _export_ks_f1(dom, output_dir, "GEMVAP_3")

    step("Generating GEMVAP 3 visualisation plots")
    info("The horizontal plot overlays GEMVAP 1 and GEMVAP 3 score distributions to")
    info("highlight how the predictor panel changes for this most-restricted subset.")
    plot_ks_f1_curve(ks_values_v3, stats_v3, str(Path(output_dir) / "GEMVAP_3_Combined_KS_F1.png"))
    plot_horizontal_gemvap3(base, dom, df, str(Path(output_dir) / "Combined_Horizontal_GEMVAP_3.png"))

    return dom


def load_full_dataset(data_path: str, seed: int, base: dict):
    """Load the full (unfiltered) dataset and annotate it with GEMVAP 1's
    consensus score, for the residue-level Figure 4 plots."""
    df_full = load_processor(data_path, seed=seed)
    annotate_hgvsp(df_full)
    annotate_consensus(df_full, base, pathogenic_threshold=7)
    return df_full


def generate_figure4(df_full, output_dir, filename: str, variant_filter=None, xlim=None) -> None:
    dfgrouped_by_proteicpos, _, _ = compute_grouped_data(df_full, variant_filter=variant_filter)
    plot_figure4(dfgrouped_by_proteicpos, str(Path(output_dir) / filename), xlim=xlim)


def generate_figure4_if_missing(get_full_dataset, output_dir, filename: str, filter_fn=None, xlim=None) -> None:
    """Skip regenerating a Figure 4 variant if it's already on disk; otherwise
    lazily load the full dataset (via get_full_dataset, a zero-arg callable)
    and build it. filter_fn, if given, receives the full dataset and returns
    the variant_filter mask — deferred so it can itself depend on the loaded data.
    xlim, if given, restricts the plotted residue-position range (e.g. for a
    zoomed-in variant of the figure)."""
    target = Path(output_dir) / filename
    if target.exists():
        print(f"Found cached {filename} — skipping.")
        return

    print(f"Creating {filename}...")
    df_full = get_full_dataset()
    variant_filter = filter_fn(df_full) if filter_fn else None
    generate_figure4(df_full, output_dir, filename, variant_filter=variant_filter, xlim=xlim)
