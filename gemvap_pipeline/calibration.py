"""
Pejaver et al. (2022) calibration of GEMVAP consensus scores, and the
resulting performance-metrics comparison against individual predictors.
"""
from pathlib import Path

import pandas as pd

from gemvap_pejaver_calibration import run_calibration as run_pejaver_calibration

from .verbose import info, note, result, section, step, substep
from .data import conserved_domain_mask
from .metrics import compute_calibrated_metrics
from .pejaver_tools import compute_individual_tool_metrics
from .visualization import plot_horizontal_gemvap1_calibrated, plot_metrics_comparison


def _build_labels(df, df_test, test_dom_mask: bool, conserved_data_path: str):
    train_label = pd.Series("unlabelled", index=df.data.index, dtype=object)
    train_label[df.is_case] = "pathogenic"
    train_label[df.is_ctrl] = "benign"

    # Optionally restrict the validation labels to non-cysteine / non-conserved-domain variants
    if test_dom_mask:
        dom_mask = conserved_domain_mask(df_test, conserved_data_path)
        test_case_mask = df_test.is_case & ~df_test.is_ctrl & dom_mask
        test_ctrl_mask = ~df_test.is_case & df_test.is_ctrl & dom_mask
        print(f"  Test dom_mask: {test_case_mask.sum()} pathogenic, "
              f"{test_ctrl_mask.sum()} control variants retained.")
    else:
        test_case_mask = df_test.is_case & ~df_test.is_ctrl
        test_ctrl_mask = ~df_test.is_case & df_test.is_ctrl

    test_label = pd.Series("unlabelled", index=df_test.data.index, dtype=object)
    test_label[test_case_mask] = "test_pathogenic"
    test_label[test_ctrl_mask] = "test_benign"
    return train_label, test_label


def _make_calib_df(train_label, test_label, train_scores, test_scores, col):
    train_part = pd.DataFrame({"label": train_label, col: train_scores})
    test_part = pd.DataFrame({"label": test_label, col: test_scores})
    return pd.concat([train_part, test_part], ignore_index=True)


def _load_thresholds(calib_dir: Path, model_name: str):
    """Load previously calibrated PP3/BP4 thresholds for one model, or None
    if run_pejaver_calibration hasn't been run for it yet in this calib_dir."""
    path = calib_dir / f"{model_name}_thresholds.csv"
    if not path.exists():
        return None
    thr = pd.read_csv(path)
    return {
        row["evidence_level"]: None if pd.isna(row["score_threshold"]) else row["score_threshold"]
        for _, row in thr.iterrows()
    }


def run_calibration_and_metrics(args, output_dir, df, df_test, models):
    """
    Calibrate each trained GEMVAP model against Pejaver et al. (2022) evidence
    thresholds (fit from the training set's pathogenic/benign consensus
    scores), validate against the test set (an internal diagnostic only —
    doesn't affect the thresholds), then compare F1/Accuracy/MCC/VUS-rate to
    individual predictors on that same test set (overall, and restricted to
    the non-cysteine / non-conserved-domain subset).

    `models` is an ordered list of dicts, one per trained GEMVAP variant, each
    with keys: name ("GEMVAP_1"/"GEMVAP_2"/"GEMVAP_3"), col (score column name,
    e.g. "gemvap1"), fit (the fit_gemvap* result), train_scores, test_scores.
    The first entry must be GEMVAP_1 — it gets the extra calibrated
    horizontal-scores plot.

    Returns (all_acc_df, all_acc_dom_df).
    """
    section("Step 3 — Calibration against Pejaver et al. (2022) ACMG/ClinGen thresholds")
    info("Pejaver et al. (2022) derived evidence-strength thresholds for computational")
    info("predictors within the ACMG/ClinGen variant classification framework.")
    info("A variant's consensus score is mapped to an ACMG evidence code:")
    info("  PP3_Supporting / PP3_Moderate / PP3_Strong / PP3_VeryStrong  -> pathogenic evidence")
    info("  BP4_Supporting / BP4_Moderate / BP4_Strong / BP4_VeryStrong  -> benign evidence")
    info("  Indeterminate (score between thresholds)                      -> VUS (variant of uncertain significance)")
    info("Thresholds are derived from likelihood ratios (LR) fitted to the TRAINING set,")
    info("so the test set is never used to choose thresholds — it is only used to evaluate them.")

    output_dir = Path(output_dir)
    perf_metrics_path = output_dir / "performance_metrics.csv"
    perf_metrics_dom_path = output_dir / "performance_metrics_noncys_nondom.csv"
    if perf_metrics_path.exists() and perf_metrics_dom_path.exists():
        step(f"Cached performance metrics found in {output_dir} — skipping calibration")
        info("Delete performance_metrics.csv / performance_metrics_noncys_nondom.csv to re-run.")
        return pd.read_csv(perf_metrics_path), pd.read_csv(perf_metrics_dom_path)

    substep("Fitting PP3/BP4 likelihood-ratio thresholds from the training set")
    calib_dir = output_dir / "calibration"
    prior_override = None if args.prior_override == 0 else args.prior_override

    train_label, test_label = _build_labels(df, df_test, args.test_dom_mask, args.conserved_data_path)

    calib_results = {}
    for m in models:
        cached_thresholds = _load_thresholds(calib_dir, m["name"])
        if cached_thresholds is not None:
            step(f"Cached thresholds found for {m['name']} — loading without re-fitting")
            calib_results[m["name"]] = {"thresholds": cached_thresholds}
            continue

        step(f"Calibrating {m['name']} — bootstrapped LR threshold fitting")
        info(f"The pathogenic and control consensus-score distributions from the training set")
        info(f"are used to estimate likelihood ratios at each score value.")
        info(f"Bootstrap resampling ({args.n_bootstrap} iterations) builds confidence intervals")
        info(f"around each LR estimate, and thresholds are placed where LR bounds cross")
        info(f"the ACMG evidence-level cutoffs (e.g., LR >= 8 for 'Moderate' pathogenic evidence).")
        calib_df = _make_calib_df(train_label, test_label, m["train_scores"], m["test_scores"], m["col"])
        calib_results[m["name"]] = run_pejaver_calibration(
            calib_df,
            score_col=m["col"],
            label_col="label",
            prior_override=prior_override,
            n_bootstrap=args.n_bootstrap,
            min_window=10,
            output_dir=str(calib_dir),
            model_name=m["name"],
        )

    # Calibrated horizontal score-distribution plot for every trained model
    # (not just GEMVAP_1) -- same layout, just parameterised by model name.
    # GEMVAP_1's fit holds the full training set (all missense variants,
    # cysteines and conserved-domain positions included). GEMVAP_2/3's own
    # fits are restricted to the subsets they were trained on, but their
    # amino-acid average-score panel (ax2) should still be computed over the
    # full training set, not their own restricted one.
    full_training_data = models[0]["fit"]["training_data"]
    for m in models:
        calib_df_m = _make_calib_df(train_label, test_label, m["train_scores"], m["test_scores"], m["col"])
        m_calib_path = calib_df_m[calib_df_m["label"] == "pathogenic"][m["col"]].dropna().values.astype(float)
        m_calib_benign = calib_df_m[calib_df_m["label"] == "benign"][m["col"]].dropna().values.astype(float)
        plot_horizontal_gemvap1_calibrated(
            m["fit"], df,
            m_calib_path, m_calib_benign,
            calib_results[m["name"]]["thresholds"],
            str(output_dir / f"Combined_Horizontal_{m['name']}_Calibrated.png"),
            model_name=m["name"].replace("_", " "),
            center_legends=m["name"] in ("GEMVAP_1", "GEMVAP_2"),
            aa_training_data=full_training_data if m["name"] in ("GEMVAP_2", "GEMVAP_3") else None,
        )

    result(f"Calibration outputs saved to: {(output_dir / 'calibration').resolve()}")

    substep("Evaluating calibrated models on the held-out test set")
    step("Computing F1, Accuracy, MCC, and VUS rate for each GEMVAP variant")
    info("F1       : harmonic mean of precision and recall for the pathogenic class")
    info("Accuracy : fraction of classified variants correctly called (VUS excluded)")
    info("MCC      : Matthews Correlation Coefficient — balanced measure robust to class imbalance")
    info("VUS rate : fraction of test variants that fall between PP3 and BP4 thresholds")
    info("           (i.e. receive no ACMG evidence code) — lower is better")
    strict = getattr(args, "strict_evaluation", False)
    if strict:
        info("Strict evaluation mode: pathogenic variants classified as Indeterminate")
        info("are counted as False Negatives (missed diagnoses), not excluded.")

    test_true = pd.Series("benign", index=df_test.data.index, dtype=object)
    test_true[df_test.is_case & ~df_test.is_ctrl] = "pathogenic"

    gemvap_acc_df = pd.DataFrame([
        compute_calibrated_metrics(
            m["test_scores"].values, test_true.values,
            calib_results[m["name"]]["thresholds"], m["name"],
            strict=strict,
        )
        for m in models
    ])

    substep("Comparing against individual predictors (Pejaver et al. 2022 thresholds)")
    step("Computing metrics for each individual tool using published PP3/BP4 thresholds")
    tool_acc_df = compute_individual_tool_metrics(df_test.data, test_true.values, args.pejaver_thresholds, strict=strict)

    all_acc_df = (
        pd.concat([gemvap_acc_df, tool_acc_df], ignore_index=True)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    all_acc_df.to_csv(output_dir / "performance_metrics.csv", index=False)
    print("\nMetrics — calibrated GEMVAP vs individual Pejaver tools:")
    print(all_acc_df[["model", "f1", "accuracy", "mcc", "vus_rate", "n_classified", "n_total_perf"]].to_string(index=False))
    plot_metrics_comparison(all_acc_df, str(output_dir / "performance_metrics_comparison.png"))

    # Accuracy restricted to non-cysteine, non-conserved-domain variants
    substep("Evaluating on non-cysteine / non-conserved-domain subset")
    step("Restricting to the hardest-to-classify test variants")
    info("This subset excludes: (a) cysteine variants (near-universally pathogenic in FBN1)")
    info("and (b) variants in conserved EGF-Ca2+ domain cores (structurally constrained).")
    info("Performance on this subset is the most clinically meaningful benchmark,")
    info("as these are the variants where GEMVAP adds the most value over simple rules.")
    dom_mask = conserved_domain_mask(df_test, args.conserved_data_path)
    dom_idx = df_test.data.index[dom_mask]
    test_true_dom = test_true.loc[dom_idx]
    print(
        f"  Subset: {len(dom_idx)} variants "
        f"({test_true_dom.eq('pathogenic').sum()} pathogenic, "
        f"{test_true_dom.eq('benign').sum()} control)"
    )

    acc_dom_df = pd.DataFrame([
        compute_calibrated_metrics(
            m["test_scores"].loc[dom_idx].values, test_true_dom.values,
            calib_results[m["name"]]["thresholds"], m["name"],
            strict=strict,
        )
        for m in models
    ])
    tool_acc_dom_df = compute_individual_tool_metrics(
        df_test.data.loc[dom_idx], test_true_dom.values, args.pejaver_thresholds, strict=strict
    )

    all_acc_dom_df = (
        pd.concat([acc_dom_df, tool_acc_dom_df], ignore_index=True)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    all_acc_dom_df.to_csv(output_dir / "performance_metrics_noncys_nondom.csv", index=False)
    print("\nMetrics — non-cysteine / non-conserved-domain subset:")
    print(all_acc_dom_df[["model", "f1", "accuracy", "mcc", "vus_rate", "n_classified", "n_total_perf"]].to_string(index=False))
    plot_metrics_comparison(
        all_acc_dom_df,
        str(output_dir / "performance_metrics_comparison_noncys_nondom.png"),
        title=(
            "F1 / Accuracy / MCC — Non-Cysteine, Non-Conserved-Domain Variants\n"
            "GEMVAP models vs individual Pejaver et al. (2022) tools"
        ),
    )

    return all_acc_df, all_acc_dom_df
