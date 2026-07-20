"""
Robustness check via repeated stratified subsampling: in each iteration,
randomly take a fraction (default 80%) of the training set — stratified so
that fraction is preserved within pathogenic and within control variants
separately — to fit PP3/BP4 thresholds for each GEMVAP model, and the same
fraction of the held-out test set (likewise stratified) to evaluate GEMVAP
and individual-predictor performance. The underlying GEMVAP fits (top
predictors, ROC thresholds) are fixed — only the calibration thresholds and
the reported metrics vary across iterations.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gemvap_pejaver_calibration import (
    bootstrap_lr,
    compute_lr_thresholds,
    derive_thresholds,
    estimate_prior,
)

from .verbose import info, note, result, section, step, substep
from .data import conserved_domain_mask, load_dataset_csv, subsample_stratified
from .evaluation import denovo_pathogenic_overrides
from .metrics import compute_calibrated_metrics
from .model import apply_consensus_score, load_fit_result
from .pejaver_tools import compute_individual_tool_metrics

GEMVAP_NAMES = ("GEMVAP_1", "GEMVAP_2", "GEMVAP_3")


def _fit_thresholds_from_calib_set(calib_path, calib_benign, prior_override, n_bootstrap, min_window, random_state):
    if prior_override:
        prior = prior_override
    else:
        prior, _, _ = estimate_prior(calib_path, calib_benign, None)

    lr_thresholds = compute_lr_thresholds(prior)

    calib_scores = np.concatenate([calib_path, calib_benign])
    calib_labels = np.concatenate([
        np.ones(len(calib_path), dtype=int),
        np.zeros(len(calib_benign), dtype=int),
    ])
    score_values = np.sort(np.unique(calib_scores))

    lr_lower, lr_upper = bootstrap_lr(
        calib_scores, calib_labels, score_values, prior,
        n_bootstrap=n_bootstrap, min_variants_in_window=min_window, random_state=random_state,
    )
    thresholds = derive_thresholds(score_values, lr_lower, lr_upper, lr_thresholds)
    return thresholds, prior


def run_bootstrap(df_train, df_test, models, args):
    """
    df_train: full training set DataProcessor (calibration source population).
    df_test: full held-out test set DataProcessor (performance source population).
    models: list of dicts with keys "name" and "fit" (a fit_gemvap*() result —
        top_predictors + per-predictor case thresholds).

    Each of args.n_iterations: subsamples args.train_fraction of df_train
    (stratified by pathogenic/control) to fit thresholds, and
    args.test_fraction of df_test (likewise stratified) to evaluate
    performance.

    Returns (thresholds_df, metrics_df): one row per iteration/model/evidence
    level, and one row per iteration/model, respectively.
    """
    section("Step 5 — Bootstrap robustness check")
    info("The bootstrap tests whether GEMVAP's performance is stable across random")
    info("subsets of the training and test data — guarding against overfitting to")
    info("the specific variants that happened to be in the train/test split.")
    info(f"Each of the {args.n_iterations} iterations:")
    info(f"  1. Draws {args.train_fraction:.0%} of training pathogenic + {args.train_fraction:.0%} of training control variants")
    info(f"     (stratified: the class ratio is preserved within each draw).")
    info(f"  2. Re-fits PP3/BP4 LR thresholds from this subsample using {args.n_bootstrap} inner bootstrap rounds.")
    info(f"  3. Draws {args.test_fraction:.0%} of test variants (stratified) and evaluates F1/Accuracy/MCC/VUS rate.")
    info(f"The GEMVAP predictor panels and case thresholds are FIXED across all iterations;")
    info(f"only the calibration thresholds and the evaluated test subset vary.")
    model_names = [m['name'] for m in models]
    step(f"Models to bootstrap: {model_names}")
    step(f"Training set size: {df_train.is_case.sum()} pathogenic  |  {df_train.is_ctrl.sum()} control")
    step(f"Test set size    : {df_test.is_case.sum()} pathogenic   |  {df_test.is_ctrl.sum()} control")
    n_calib_case = round(args.train_fraction * int((df_train.is_case & ~df_train.is_ctrl).sum()))
    n_calib_ctrl = round(args.train_fraction * int((df_train.is_ctrl & ~df_train.is_case).sum()))
    n_perf_case = round(args.test_fraction * int((df_test.is_case & ~df_test.is_ctrl).sum()))
    n_perf_ctrl = round(args.test_fraction * int((df_test.is_ctrl & ~df_test.is_case).sum()))
    result(f"Each iteration's calibration subsample: {n_calib_case} pathogenic  |  {n_calib_ctrl} control "
           f"(from training, {args.train_fraction:.0%})")
    result(f"Each iteration's performance subsample: {n_perf_case} pathogenic  |  {n_perf_ctrl} control "
           f"(from test, {args.test_fraction:.0%})")

    prior_override = None if args.prior_override == 0 else args.prior_override
    strict = getattr(args, "strict_evaluation", False)
    if strict:
        info("Strict evaluation mode active: pathogenic Indeterminate variants counted as FN.")

    threshold_dfs = []
    metric_dfs = []

    for i in range(args.n_iterations):
        if i == 0 or (i + 1) % 20 == 0 or (i + 1) == args.n_iterations:
            print(f"  Bootstrap iteration {i + 1}/{args.n_iterations}...")

        seed = args.base_seed + i
        df_calib = subsample_stratified(df_train, fraction=args.train_fraction, seed=seed)
        df_perf = subsample_stratified(df_test, fraction=args.test_fraction, seed=seed)

        perf_true = pd.Series("benign", index=df_perf.data.index, dtype=object)
        perf_true[df_perf.is_case & ~df_perf.is_ctrl] = "pathogenic"

        gemvap_metrics = []
        for m in models:
            top_predictors = m["fit"]["top_predictors"]
            case_thresholds = m["fit"]["rbc"]["threshold"]["case"]

            calib_scores = apply_consensus_score(df_calib.data, top_predictors, case_thresholds)
            perf_scores = apply_consensus_score(df_perf.data, top_predictors, case_thresholds)

            calib_path = calib_scores[df_calib.is_case & ~df_calib.is_ctrl].dropna().values.astype(float)
            calib_benign = calib_scores[df_calib.is_ctrl & ~df_calib.is_case].dropna().values.astype(float)

            thresholds, prior = _fit_thresholds_from_calib_set(
                calib_path, calib_benign, prior_override, args.n_bootstrap, args.min_window,
                random_state=seed,
            )

            thr_df = pd.DataFrame(
                {"evidence_level": list(thresholds.keys()), "score_threshold": list(thresholds.values())}
            )
            thr_df.insert(0, "model", m["name"])
            thr_df["prior"] = prior
            thr_df.insert(0, "iteration", i)
            threshold_dfs.append(thr_df)

            gemvap_metrics.append(compute_calibrated_metrics(perf_scores.values, perf_true.values, thresholds, m["name"], strict=strict))

        tool_metrics = compute_individual_tool_metrics(df_perf.data, perf_true.values, args.pejaver_thresholds, strict=strict)
        iter_metrics = pd.concat([pd.DataFrame(gemvap_metrics), tool_metrics], ignore_index=True)
        iter_metrics.insert(0, "iteration", i)
        metric_dfs.append(iter_metrics)

    thresholds_df = pd.concat(threshold_dfs, ignore_index=True)
    metrics_df = pd.concat(metric_dfs, ignore_index=True)

    substep("Bootstrap complete — summary across all iterations")
    for m in models:
        m_metrics = metrics_df[metrics_df["model"] == m["name"]]
        result(f"{m['name']}: F1 = {m_metrics['f1'].mean():.3f} +/- {m_metrics['f1'].std():.3f}  "
               f"|  VUS rate = {m_metrics['vus_rate'].mean():.3f} +/- {m_metrics['vus_rate'].std():.3f}")
    info("Low std values indicate stable performance — the model is not sensitive to which")
    info("specific variants happened to be selected in the train/test split.")
    return thresholds_df, metrics_df


def run_bootstrap_step(args, output_dir, data_path: str, seed: int, gemvap_names=GEMVAP_NAMES):
    """
    Step 8 orchestration: reload the cached train/test CSVs and GEMVAP fits,
    optionally restrict the test pool to the non-cysteine/non-conserved-domain
    subset (args.test_dom_mask), run the bootstrap, and save
    bootstrap_thresholds{suffix}.csv / bootstrap_performance_metrics{suffix}.csv
    / bootstrap_performance_summary{suffix}.csv under output_dir.

    Returns the GEMVAP-only mean/std summary DataFrame, or None if no cached
    GEMVAP fits were found.

    args.new_denovo_path/new_denovo_path_2/pathogenic_overrides must match
    whatever was passed to build_train_test_sets in Step 2, so purely-de-novo
    pathogenic variants keep their pathogenic label after the reload (see
    evaluation.denovo_pathogenic_overrides for why this is necessary).
    """
    output_dir = Path(output_dir)

    overrides = denovo_pathogenic_overrides(
        getattr(args, "new_denovo_path", None),
        getattr(args, "new_denovo_path_2", None),
        getattr(args, "pathogenic_overrides", None),
    )
    df_train_bs = load_dataset_csv(output_dir / "training_dataset.csv", data_path, seed=seed, pathogenic_overrides=overrides)
    df_test_bs = load_dataset_csv(output_dir / "test_dataset.csv", data_path, seed=seed, pathogenic_overrides=overrides)

    if args.test_dom_mask:
        step("test_dom_mask is True -- restricting the bootstrap test pool to non-cys/non-domain variants")
        n_before = len(df_test_bs.data)
        mask = conserved_domain_mask(df_test_bs, args.conserved_data_path)
        df_test_bs.data = df_test_bs.data[mask]
        df_test_bs.create_filters()
        result(f"{n_before - len(df_test_bs.data)} variant(s) removed (cysteine or conserved-domain)  |  "
               f"{n_before} -> {len(df_test_bs.data)} rows remain")

    bs_models = []
    for name in gemvap_names:
        fit_path = output_dir / f"{name}_fit.pkl"
        if fit_path.exists():
            bs_models.append({"name": name, "fit": load_fit_result(fit_path)})

    if not bs_models:
        note("No trained models found -- run Steps 2-6 first.")
        return None

    step(f"Running bootstrap ({args.n_iterations} iterations) for: {[m['name'] for m in bs_models]}")
    thresholds_df, metrics_df = run_bootstrap(df_train_bs, df_test_bs, bs_models, args)

    suffix = "_noncys_nondom" if args.test_dom_mask else ""
    thresholds_df.to_csv(output_dir / f"bootstrap_thresholds{suffix}.csv", index=False)
    metrics_df.to_csv(output_dir / f"bootstrap_performance_metrics{suffix}.csv", index=False)

    summary = (
        metrics_df.groupby("model")[["f1", "accuracy", "mcc", "vus_rate"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.round(4).sort_values("f1_mean", ascending=False).reset_index()
    summary.to_csv(output_dir / f"bootstrap_performance_summary{suffix}.csv", index=False)

    result(f"Completed {args.n_iterations} iterations -- saved bootstrap_performance_summary{suffix}.csv")
    return summary
