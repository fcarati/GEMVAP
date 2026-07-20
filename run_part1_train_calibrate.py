#!/usr/bin/env python3
"""Standalone runner reproducing GEMVAP_Pipeline_gnomAD4_part1_train_calibrate.ipynb
(Steps 1-6: config, train/test split, train GEMVAP 1/2/3, Pejaver calibration)
plus the notebook's intermediate visualisations -- no Jupyter required.

Usage (from anywhere -- the script chdirs to its own folder first):
    python run_part1_train_calibrate.py
    python run_part1_train_calibrate.py --archive          # wipe OUTPUT_DIR and regenerate everything
    python run_part1_train_calibrate.py --config other.yaml

Writes the same files as the notebook to OUTPUT_DIR (see config_gnomad4.yaml):
training_dataset.csv, test_dataset.csv, GEMVAP_{1,2,3}_fit.pkl, calibration/,
performance_metrics*.csv, full_dataset_predictions.csv (every variant in the
original dataset scored by every trained GEMVAP model: consensus score +
highest Pejaver et al. 2022 evidence category), and all PNG figures (KS/F1
curves, Figure_4 variants, calibrated score distributions, publication
boxplots). Every step is cached -- re-running only regenerates what's missing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

# The pipeline modules print Unicode arrows/dashes (originally only ever seen
# through Jupyter's UTF-8 output). A plain Windows console defaults to the
# cp1252/OEM codepage, which crashes on those characters -- force UTF-8 here
# so the script runs the same everywhere.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))

from pipeline_log import step, inputs, info, subsection, action, result, warning, summary_table


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", default="config_gnomad4.yaml",
        help="Path to config YAML, relative to gemvap_clean_pipeline/ (default: config_gnomad4.yaml)",
    )
    p.add_argument(
        "--archive", action="store_true",
        help="Archive current OUTPUT_DIR contents to output/_archive/ and regenerate "
             "everything from scratch. Default: keep cached outputs, verifying them "
             "against the last saved manifest.",
    )
    p.add_argument(
        "--figures-dir", default="figures",
        help="Folder (relative to gemvap_clean_pipeline/) to copy every PNG generated "
             "under OUTPUT_DIR into, flattened into one place for easy browsing "
             "(default: figures/). Pass '' to skip this step.",
    )
    return p.parse_args()


def main():
    cli_args = parse_args()

    # ================================================================= Step 1
    step(1, "Environment & configuration setup")
    inputs([cli_args.config])
    info("Every path below is relative to this script's own folder, which must be\n"
         "gemvap_clean_pipeline/ -- checked before anything else runs.")

    subsection("Working-directory check")
    cwd = Path(os.getcwd())
    action(f"Checking that '{cwd}' contains a gemvap_pipeline/ package")
    if not (cwd / "gemvap_pipeline").is_dir():
        raise RuntimeError(
            f"Working directory is '{cwd}' after chdir to the script's own folder.\n"
            "gemvap_pipeline/ is missing -- has this script been moved out of "
            "gemvap_clean_pipeline/?"
        )
    result(f"Working directory confirmed: {cwd}")

    subsection("Loading config")
    cfg_path = Path(cli_args.config)
    action(f"Reading {cfg_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path.resolve()}\n"
            "Expected next to this script in gemvap_clean_pipeline/."
        )
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    DATA_PATH = cfg["data_path"]
    CONSERVED_DATA_PATH = cfg["conserved_data_path"]
    PEJAVER_THRESHOLDS = cfg["pejaver_thresholds"]
    NEW_DENOVO_PATH = cfg.get("new_denovo_path") or None
    NEW_DENOVO_PATH_2 = cfg.get("new_denovo_path_2") or None
    STRICT_EVALUATION = bool(cfg.get("strict_evaluation", False))
    DDGUN_DATA_PATH = cfg.get("ddgun_data_path") or None
    ALPHAMISSENSE_DATA_PATH = cfg.get("alphamissense_data_path") or None

    OUTPUT_DIR = Path(cfg["output_dir"])

    SEED = int(cfg["seed"])
    TRAIN_GEMVAP2 = bool(cfg["train_gemvap2"])
    TRAIN_GEMVAP3 = bool(cfg["train_gemvap3"])

    PRIOR_OVERRIDE = float(cfg["prior_override"])
    N_BOOTSTRAP_CALIB = int(cfg["n_bootstrap_calib"])

    MODEL_ITERATION = int(cfg["model_iteration"])

    args = SimpleNamespace(
        data_path=DATA_PATH,
        conserved_data_path=CONSERVED_DATA_PATH,
        pejaver_thresholds=PEJAVER_THRESHOLDS,
        new_denovo_path=NEW_DENOVO_PATH,
        new_denovo_path_2=NEW_DENOVO_PATH_2,
        strict_evaluation=STRICT_EVALUATION,
        output_dir=str(OUTPUT_DIR),
        seed=SEED,
        seeds=[SEED],
        train_gemvap2=TRAIN_GEMVAP2,
        train_gemvap3=TRAIN_GEMVAP3,
        prior_override=PRIOR_OVERRIDE,
        n_bootstrap=N_BOOTSTRAP_CALIB,
        run_calibration=True,
        test_dom_mask=False,
        model_iteration=MODEL_ITERATION,
        pathogenic_overrides=None,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result(f"Config loaded; output will be written to {OUTPUT_DIR.resolve()}")

    summary_table(
        "Key Configuration Settings",
        ["Setting", "Value"],
        [
            ("data_path", DATA_PATH),
            ("seed", SEED),
            ("train_gemvap2", TRAIN_GEMVAP2),
            ("train_gemvap3", TRAIN_GEMVAP3),
            ("prior_override", PRIOR_OVERRIDE),
            ("strict_evaluation", STRICT_EVALUATION),
            ("new_denovo_path_2", NEW_DENOVO_PATH_2),
        ],
    )

    subsection("Archive intermediate files & images?")
    info("Every run can either start clean -- old outputs moved into a timestamped\n"
         "folder so everything below regenerates from scratch -- or keep reusing what\n"
         "is already cached in OUTPUT_DIR. If you keep the cache, its contents are\n"
         "verified against the manifest saved at the end of the last full run, so silent\n"
         "drift (a file changed/removed outside the pipeline) does not go unnoticed.")

    from output_archive import archive_output_dir
    from verify_outputs import verify as verify_cached_outputs

    ARCHIVE_ROOT = Path("output") / "_archive"

    if cli_args.archive:
        action(f"--archive passed: archiving current contents of {OUTPUT_DIR}")
        archive_dest = archive_output_dir(OUTPUT_DIR, ARCHIVE_ROOT)
        if archive_dest is None:
            result(f"{OUTPUT_DIR} was already empty -- nothing to archive")
        else:
            result(f"Archived previous outputs -> {archive_dest}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        action(f"Keeping cached outputs in {OUTPUT_DIR} -- verifying against last saved manifest")
        outputs_clean = verify_cached_outputs(OUTPUT_DIR)
        if not outputs_clean:
            warning(
                "Cached files differ from the last saved manifest -- if this is unexpected, "
                "re-run with --archive to regenerate from scratch."
            )

    # ================================================================= Step 2
    from gemvap_pipeline.data import (
        build_train_test_sets, load_cached_datasets,
        save_denovo_overrides, load_denovo_overrides,
        build_intersection_ac_table,
    )

    step(2, "Build the train / test split")
    inputs([
        (DATA_PATH, "main FBN1 missense variant table -- pathogenic + control"),
        (NEW_DENOVO_PATH or "(unset)", "de novo pathogenic variants, 'cDNA' column"),
        (NEW_DENOVO_PATH_2 or "(unset)", "2026 NGS de novo report, 'Mutation c.' column"),
    ])
    info("The dataset is split into a TRAINING set (used to fit each model) and a\n"
         "TEST set (held out entirely for final performance evaluation). De novo\n"
         "pathogenic variants are reserved for the test set so the model is never\n"
         "fit on the same variants it is later judged against.")

    new_denovo_path = args.new_denovo_path
    new_denovo_path_2 = args.new_denovo_path_2
    use_cache = new_denovo_path is None and new_denovo_path_2 is None

    if use_cache:
        action("No de novo path set -- looking for a cached split and its saved override set")
    else:
        action("De novo path(s) set -- forcing a rebuild so new variants are included")

    recovered_denovo = load_denovo_overrides(OUTPUT_DIR) if use_cache else None
    cached = (
        load_cached_datasets(OUTPUT_DIR, DATA_PATH, seed=SEED, pathogenic_overrides=recovered_denovo)
        if use_cache else None
    )

    if cached is not None:
        df, df_test = cached

        subsection("Row-count audit trail (recomputed fresh every run, not read from cache)")
        info("This stage-by-stage breakdown is always recomputed from DATA_PATH, even on a\n"
             "cache hit -- it's a cheap pandas filter pass, no model fitting -- so the\n"
             "row-count impact of every operation stays visible on every run. The\n"
             "training/test DataFrames actually used below still come from the cached\n"
             "CSVs, unchanged.")
        fresh_df, fresh_df_test, _ = build_train_test_sets(
            DATA_PATH, seed=SEED, extra_denovo_cdna=recovered_denovo,
        )
        if len(fresh_df.data) != len(df.data) or len(fresh_df_test.data) != len(df_test.data):
            warning(
                f"Cached split ({len(df.data)} train / {len(df_test.data)} test) differs from a "
                f"fresh rebuild ({len(fresh_df.data)} train / {len(fresh_df_test.data)} test) -- "
                "DATA_PATH or the de novo override set may have changed since the cache was built. "
                "Delete training_dataset.csv / test_dataset.csv in OUTPUT_DIR to force a rebuild."
            )
        result(f"Reloaded cached split from {OUTPUT_DIR}/training_dataset.csv + test_dataset.csv")
    else:
        action("Building train / test split from scratch")
        df, df_test, _ = build_train_test_sets(
            DATA_PATH,
            seed=SEED,
            new_denovo_path=new_denovo_path,
            extra_denovo_paths=[new_denovo_path_2] if new_denovo_path_2 else None,
        )
        df.data.to_csv(OUTPUT_DIR / "training_dataset.csv", index=False)
        df_test.data.to_csv(OUTPUT_DIR / "test_dataset.csv", index=False)

        from gemvap_pipeline.evaluation import denovo_pathogenic_overrides
        all_overrides = denovo_pathogenic_overrides(new_denovo_path, new_denovo_path_2)
        save_denovo_overrides(OUTPUT_DIR, all_overrides)

        result(f"Saved training_dataset.csv ({len(df.data)} rows) and "
               f"test_dataset.csv ({len(df_test.data)} rows) to {OUTPUT_DIR}")

    summary_table(
        "Dataset Composition Summary",
        ["Set", "Pathogenic", "Control", "Total"],
        [
            ("Training", int(df.is_case.sum()), int(df.is_ctrl.sum()), len(df.data)),
            ("Test", int(df_test.is_case.sum()), int(df_test.is_ctrl.sum()), len(df_test.data)),
        ],
    )

    subsection("Intersection variants with high gnomAD allele count")
    info("Variants flagged in BOTH a pathogenic database (HGMD/UMD/FRANKEN/MUTDB/PARIS/GENT)\n"
         "and a gnomAD control database carry contradictory evidence -- DataProcessor's is_inte,\n"
         "the same label _exclude_intersection_variants() strips from the training pool above.\n"
         "Restricting to gnomad4_joint_AC > 11 (the same 'recurrently observed' cutoff used by\n"
         "run_part3_venn.py) surfaces the ones with the strongest population-frequency evidence\n"
         "against pathogenicity, alongside which pathogenic database(s) called them.")
    action("Scanning the full dataset (every consequence type, unfiltered) for "
           "is_inte & gnomad4_joint_AC > 11")
    intersection_ac_table = build_intersection_ac_table(DATA_PATH, seed=SEED)
    intersection_ac_path = OUTPUT_DIR / "intersection_ac_gt11_with_pathdb.csv"
    intersection_ac_table.to_csv(intersection_ac_path, index=False)
    result(f"{len(intersection_ac_table)} variant(s) -> {intersection_ac_path}")

    # ================================================================= Step 3
    from gemvap_pipeline.training import (
        generate_figure4_if_missing, load_full_dataset,
        train_gemvap1, train_gemvap2, train_gemvap3,
    )
    from gemvap_pipeline.model import apply_consensus_score
    from gemvap_pipeline.data import conserved_domain_mask
    from gemvap_pipeline.pipeline import _lazy, _model_scores

    step(3, "Train GEMVAP 1")
    inputs([
        (str(OUTPUT_DIR / "training_dataset.csv"), "training set built in Step 2"),
        (DATA_PATH, "full predictor score table, for KS ranking"),
    ])
    info("GEMVAP 1 trains on every missense variant not reserved for testing. Predictors\n"
         "are ranked by KS statistic (separation between pathogenic/control score\n"
         "distributions), then added incrementally until training-set F1 stops improving.")

    training_filter = ~df.training_sets & df.is_mis & ~df.is_denovo
    action("Fitting GEMVAP 1 (predictor selection + consensus scoring)")
    base = train_gemvap1(df, training_filter, args, OUTPUT_DIR)
    models = [_model_scores("GEMVAP_1", "gemvap1", base, df, df_test)]
    result(f"GEMVAP 1 fit: {len(base['top_predictors'])} predictors selected -- "
           f"saved to {OUTPUT_DIR}/GEMVAP_1_fit.pkl")

    get_full_dataset = _lazy(lambda: load_full_dataset(DATA_PATH, SEED, base))
    generate_figure4_if_missing(get_full_dataset, OUTPUT_DIR, "Figure_4.png")
    generate_figure4_if_missing(
        get_full_dataset, OUTPUT_DIR, "Figure_4_zoom_300_500.png", xlim=(300, 500),
    )
    generate_figure4_if_missing(
        get_full_dataset, OUTPUT_DIR, "Figure_4_zoom_2687_2871.png", xlim=(2687, 2871),
    )

    _list_figures(OUTPUT_DIR, [
        "Combined_KS_F1_SharedAxis.png", "Combined_Horizontal_GEMVAP_1.png",
        "Figure_4.png", "Figure_4_zoom_300_500.png", "Figure_4_zoom_2687_2871.png",
    ])

    subsection("Classifying wild-type amino acids by mean GEMVAP 1 consensus score")
    info("Ranks each wild-type amino acid by mean per-variant consensus score and\n"
         "splits into tiers at the largest gaps between consecutive sorted means\n"
         "(a simple natural-breaks clustering) -- e.g. cysteine/tryptophan/glycine\n"
         "losses carry the strongest pathogenicity signal, lysine/glutamine/\n"
         "methionine the weakest.")
    from gemvap_pipeline.visualization import classify_aa_by_consensus

    action("Grouping GEMVAP 1 consensus scores by wild-type amino acid into tiers")
    tiers_path = OUTPUT_DIR / "gemvap1_aa_tiers.csv"
    tiers_df = classify_aa_by_consensus(base, DATA_PATH, str(tiers_path))
    result(f"Saved: {tiers_path}")
    _print_df(tiers_df.reset_index().round(3))

    subsection("DDGun-Seq stability vs. GEMVAP 1 pathogenicity, per wild-type amino acid")
    info("Cross-checks GEMVAP 1's per-residue pathogenicity signal against an\n"
         "independent, sequence-only stability predictor (DDGun-Seq, hhblits +\n"
         "uniclust30 profile) -- set 'ddgun_data_path' in config_gnomad4.yaml to\n"
         "enable. See archive/plot_sddg_vs_gemvap1.py for the original script.")
    if DDGUN_DATA_PATH and Path(DDGUN_DATA_PATH).exists():
        from gemvap_pipeline.visualization import plot_ddgun_vs_gemvap1

        action("Grouping GEMVAP 1 consensus score and DDGun-Seq S_DDG by wild-type amino acid")
        ddgun_plot_path = OUTPUT_DIR / "sddg_vs_gemvap1_per_aa.png"
        if not ddgun_plot_path.exists():
            plot_ddgun_vs_gemvap1(base, DATA_PATH, DDGUN_DATA_PATH, str(ddgun_plot_path))
        result(f"Saved: {ddgun_plot_path}")
    else:
        warning(
            f"ddgun_data_path not set or file missing ({DDGUN_DATA_PATH}) -- "
            "skipping DDGun-Seq comparison plot."
        )

    # ================================================================= Step 4
    step(4, "Train GEMVAP 2 (non-cysteine variants)")
    inputs([
        (str(OUTPUT_DIR / "training_dataset.csv"), "training set built in Step 2"),
        (str(OUTPUT_DIR / "GEMVAP_1_fit.pkl"), "GEMVAP 1 fit produced in Step 3"),
    ])
    info("Controlled by 'train_gemvap2' in config_gnomad4.yaml. When enabled, this model\n"
         "re-runs predictor selection on the non-cysteine subset of the training filter.")

    if TRAIN_GEMVAP2:
        action("Fitting GEMVAP 2 (non-cysteine training subset)")
        cyst = train_gemvap2(df, training_filter, base, args, OUTPUT_DIR)
        models.append(_model_scores("GEMVAP_2", "gemvap2", cyst, df, df_test))
        generate_figure4_if_missing(
            get_full_dataset, OUTPUT_DIR, "Figure_4.1.png", filter_fn=lambda d: ~d.is_cys,
        )
        generate_figure4_if_missing(
            get_full_dataset, OUTPUT_DIR, "Figure_4.1_zoom_300_500.png",
            filter_fn=lambda d: ~d.is_cys, xlim=(300, 500),
        )
        generate_figure4_if_missing(
            get_full_dataset, OUTPUT_DIR, "Figure_4.1_zoom_2687_2871.png",
            filter_fn=lambda d: ~d.is_cys, xlim=(2687, 2871),
        )
        result(f"GEMVAP 2 fit: {len(cyst['top_predictors'])} predictors selected -- "
               f"saved to {OUTPUT_DIR}/GEMVAP_2_fit.pkl")
        _list_figures(OUTPUT_DIR, [
            "GEMVAP_2_Combined_KS_F1.png", "Combined_Horizontal_GEMVAP_2.png",
            "Figure_4.1.png", "Figure_4.1_zoom_300_500.png", "Figure_4.1_zoom_2687_2871.png",
        ])
    else:
        result("Skipped -- train_gemvap2 is False in config_gnomad4.yaml")

    # ================================================================= Step 5
    step(5, "Train GEMVAP 3 (non-cysteine, outside conserved domains)")
    inputs([
        (str(OUTPUT_DIR / "training_dataset.csv"), "training set built in Step 2"),
        (str(OUTPUT_DIR / "GEMVAP_1_fit.pkl"), "GEMVAP 1 fit produced in Step 3"),
        (CONSERVED_DATA_PATH, "conserved EGF-Ca2+ domain annotation, for the domain mask"),
    ])
    info("Controlled by 'train_gemvap3' in config_gnomad4.yaml. Excludes both cysteine\n"
         "variants and positions inside the conserved domain from the training filter.")

    if TRAIN_GEMVAP3:
        action("Fitting GEMVAP 3 (non-cysteine, non-domain training subset)")
        dom = train_gemvap3(df, training_filter, base, args, OUTPUT_DIR)
        models.append(_model_scores("GEMVAP_3", "gemvap3", dom, df, df_test))
        generate_figure4_if_missing(
            get_full_dataset, OUTPUT_DIR, "Figure_4.2.png",
            filter_fn=lambda d: conserved_domain_mask(d, CONSERVED_DATA_PATH),
        )
        generate_figure4_if_missing(
            get_full_dataset, OUTPUT_DIR, "Figure_4.2_zoom_300_500.png",
            filter_fn=lambda d: conserved_domain_mask(d, CONSERVED_DATA_PATH), xlim=(300, 500),
        )
        generate_figure4_if_missing(
            get_full_dataset, OUTPUT_DIR, "Figure_4.2_zoom_2687_2871.png",
            filter_fn=lambda d: conserved_domain_mask(d, CONSERVED_DATA_PATH), xlim=(2687, 2871),
        )
        result(f"GEMVAP 3 fit: {len(dom['top_predictors'])} predictors selected -- "
               f"saved to {OUTPUT_DIR}/GEMVAP_3_fit.pkl")
        _list_figures(OUTPUT_DIR, [
            "GEMVAP_3_Combined_KS_F1.png", "Combined_Horizontal_GEMVAP_3.png",
            "Figure_4.2.png", "Figure_4.2_zoom_300_500.png", "Figure_4.2_zoom_2687_2871.png",
        ])
    else:
        result("Skipped -- train_gemvap3 is False in config_gnomad4.yaml")

    # ================================================================= Step 6
    from gemvap_pipeline.calibration import run_calibration_and_metrics

    step(6, "Calibrate against Pejaver et al. (2022) thresholds")
    inputs([
        (str(OUTPUT_DIR / "training_dataset.csv"), "training set built in Step 2, for fitting thresholds"),
        (str(OUTPUT_DIR / "test_dataset.csv"), "test set built in Step 2, for diagnostic metrics"),
        (PEJAVER_THRESHOLDS, "published per-tool Pejaver thresholds, for the individual-predictor comparison"),
        ("GEMVAP_{1,2,3}_fit.pkl", "model fits produced in Steps 3-5"),
    ])
    info("A likelihood-ratio calibration maps each model's consensus score to an ACMG\n"
         "evidence level (PP3 = pathogenic evidence, BP4 = benign evidence). Thresholds are\n"
         "fit only from training-set scores, so the test set stays unseen until scoring.")

    action("Running LR calibration and computing performance metrics for each model")
    all_acc_df, all_acc_dom_df = run_calibration_and_metrics(args, OUTPUT_DIR, df, df_test, models)
    result(f"Calibrated {len(models)} model(s); wrote performance_metrics.csv and "
           f"performance_metrics_noncys_nondom.csv to {OUTPUT_DIR}")

    subsection("Performance metrics -- all variants (test set)")
    _print_df(all_acc_df[["model", "f1", "accuracy", "mcc", "vus_rate", "n_classified", "n_total_perf"]])

    # ======================================================= Intermediate viz
    subsection("PP3/BP4 calibration thresholds per model")
    action("Reading per-model threshold CSVs from the calibration/ subfolder")

    from gemvap_pipeline.visualization import (
        build_thresholds_table, plot_test_score_distributions, plot_publication_boxplot,
    )

    pivot = build_thresholds_table(models, OUTPUT_DIR / "calibration")
    if not pivot.empty:
        _print_df(pivot)
        result(f"Thresholds recovered for {len(pivot)} model(s)")
    else:
        warning("No threshold CSV files found -- run Step 6 (calibration) first.")

    subsection("Test-set score distributions with thresholds overlaid")
    action("Plotting per-model consensus score histograms with PP3/BP4 lines")
    plot_path = plot_test_score_distributions(models, OUTPUT_DIR)
    if plot_path is None:
        warning("test_scores.csv not found -- run Step 2 and Step 6 first.")
    else:
        result(f"Saved: {plot_path}")

    subsection("Performance metrics -- non-cysteine / non-conserved-domain subset")
    _print_df(all_acc_dom_df[["model", "f1", "accuracy", "mcc", "vus_rate", "n_classified", "n_total_perf"]])

    subsection("Cached calibration plots")
    action("Listing cached calibration comparison figures")
    _list_figures(OUTPUT_DIR, [
        *[f"Combined_Horizontal_{m['name']}_Calibrated.png" for m in models],
        "performance_metrics_comparison.png",
        "performance_metrics_comparison_noncys_nondom.png",
    ])

    subsection("Per-model calibration curves")
    action("Listing every PNG written under calibration/")
    for p in sorted((OUTPUT_DIR / "calibration").glob("*.png")):
        result(str(p))

    subsection("Publication-style performance boxplot")
    action("Comparing GEMVAP models against the interquartile range of individual predictors")
    path_full, path_noncys_nondom = plot_publication_boxplot(all_acc_df, all_acc_dom_df, OUTPUT_DIR)
    result(f"Saved: {path_full}")
    result(f"Saved: {path_noncys_nondom}")

    subsection("Publication-style performance boxplot, with AlphaMissense + REVEL highlighted")
    info("AlphaMissense (Cheng et al. 2023) predates Pejaver et al. (2022), so it has no\n"
         "published PP3/BP4 thresholds and isn't part of the individual-predictor pool\n"
         "above. It's scored separately here against its own published thresholds\n"
         "(score > 0.5642 = Pathogenic, < 0.34 = Benign) and overlaid as its own point\n"
         "alongside GEMVAP 1/2/3 (but not added to the interquartile-range pool, since it\n"
         "was never part of it). REVEL already has a row in all_acc_df/all_acc_dom_df and\n"
         "stays in the interquartile-range pool -- it's a genuine individual predictor,\n"
         "unlike GEMVAP -- but is additionally re-plotted as its own highlighted point.")
    if ALPHAMISSENSE_DATA_PATH and Path(ALPHAMISSENSE_DATA_PATH).exists():
        from gemvap_pipeline.pejaver_tools import compute_alphamissense_metrics
        from gemvap_pipeline.visualization import plot_publication_boxplot_with_alphamissense

        action(f"Scoring AlphaMissense from {ALPHAMISSENSE_DATA_PATH}")
        test_true_am = pd.Series("benign", index=df_test.data.index, dtype=object)
        test_true_am[df_test.is_case & ~df_test.is_ctrl] = "pathogenic"

        am_full_df = compute_alphamissense_metrics(
            df_test.data, test_true_am.values, ALPHAMISSENSE_DATA_PATH, strict=STRICT_EVALUATION,
        )
        am_dom_mask = conserved_domain_mask(df_test, CONSERVED_DATA_PATH)
        am_dom_idx = df_test.data.index[am_dom_mask]
        am_dom_df = compute_alphamissense_metrics(
            df_test.data.loc[am_dom_idx], test_true_am.loc[am_dom_idx].values,
            ALPHAMISSENSE_DATA_PATH, strict=STRICT_EVALUATION,
        )

        am_metrics_full = am_full_df.iloc[0].to_dict() if not am_full_df.empty else None
        am_metrics_dom = am_dom_df.iloc[0].to_dict() if not am_dom_df.empty else None

        am_path_full, am_path_dom = plot_publication_boxplot_with_alphamissense(
            all_acc_df, all_acc_dom_df, am_metrics_full, am_metrics_dom, OUTPUT_DIR,
        )
        result(f"Saved: {am_path_full}")
        result(f"Saved: {am_path_dom}")
    else:
        warning(
            f"alphamissense_data_path not set or file missing ({ALPHAMISSENSE_DATA_PATH}) -- "
            "skipping AlphaMissense comparison boxplot."
        )

    # ================================================================= Step 7
    step(7, "Build the full-dataset prediction table (all GEMVAP models)")
    inputs([
        (DATA_PATH, "full FBN1 missense variant table -- same source as Step 1, unfiltered"),
        ("GEMVAP_{1,2,3}_fit.pkl", "model fits produced in Steps 3-5"),
        (str(OUTPUT_DIR / "calibration"), "PP3/BP4 thresholds fitted in Step 6"),
    ])
    info("For every variant in the original dataset (not just the train/test split),\n"
         "each trained GEMVAP model scores a consensus score (how many of its top\n"
         "predictors call the variant pathogenic) and the highest Pejaver et al. (2022)\n"
         "evidence category that score reaches -- PP3_VeryStrong down to BP4_VeryStrong,\n"
         "or Indeterminate/VUS -- using the thresholds calibrated in Step 6.")

    from gemvap_pejaver_calibration import annotate_variant
    from gemvap_pipeline.calibration import _load_thresholds

    action("Loading the full, unfiltered dataset")
    df_full = get_full_dataset()

    predictions_df = pd.DataFrame(index=df_full.data.index)
    for id_col in ("cDNA", "HGVSp", "variantvcf"):
        if id_col in df_full.data.columns:
            predictions_df[id_col] = df_full.data[id_col]
    predictions_df["label"] = "unlabelled"
    predictions_df.loc[df_full.is_case & ~df_full.is_ctrl, "label"] = "pathogenic"
    predictions_df.loc[~df_full.is_case & df_full.is_ctrl, "label"] = "benign"

    calib_dir = OUTPUT_DIR / "calibration"
    for m in models:
        thresholds = _load_thresholds(calib_dir, m["name"])
        if thresholds is None:
            warning(f"No calibrated thresholds found for {m['name']} -- skipping its columns")
            continue
        action(f"Scoring every variant with {m['name']} (consensus score + Pejaver category)")
        scores = apply_consensus_score(
            df_full.data, m["fit"]["top_predictors"], m["fit"]["rbc"]["threshold"]["case"],
        )
        predictions_df[f"{m['col']}_consensus_score"] = scores
        predictions_df[f"{m['col']}_pejaver_category"] = scores.apply(
            lambda s: annotate_variant(float(s), thresholds)
        )

    predictions_path = OUTPUT_DIR / "full_dataset_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    result(f"Saved {len(predictions_df)} variant(s) x {len(models)} model(s) -> {predictions_path}")

    # ============================================================= Figures dir
    if cli_args.figures_dir:
        subsection("Collecting all generated images into one folder")
        figures_dir = Path(cli_args.figures_dir)
        action(f"Copying every PNG under {OUTPUT_DIR} (incl. calibration/) into {figures_dir}")
        n_copied = _collect_figures(OUTPUT_DIR, figures_dir)
        result(f"Copied {n_copied} PNG(s) to {figures_dir.resolve()}")

    # ================================================================ Wrap-up
    subsection("End of Part 1 (Steps 1-7)")
    info(f"{OUTPUT_DIR} now contains the train/test split, GEMVAP_{{1,2,3}}_fit.pkl,\n"
         "calibration/, performance_metrics*.csv, full_dataset_predictions.csv, and\n"
         "every PNG figure listed above.")
    info("Next: run the evaluate/bootstrap and Venn-diagram steps against this same\n"
         "config_gnomad4.yaml (see README.md), or open the PNGs above directly.")


def _list_figures(output_dir, filenames):
    """Print the path of each named PNG under output_dir that exists (script
    equivalent of the notebook's inline `display(Image(...))` calls)."""
    output_dir = Path(output_dir)
    for fname in filenames:
        p = output_dir / fname
        if p.exists():
            result(str(p))


def _collect_figures(output_dir, figures_dir) -> int:
    """Copy every PNG found anywhere under output_dir (including calibration/)
    into a single flat figures_dir, for easy browsing without hunting through
    OUTPUT_DIR's mix of CSVs/pickles/PNGs. Overwrites on re-run so figures_dir
    always reflects the latest outputs. Returns the number of files copied."""
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for src in sorted(output_dir.rglob("*.png")):
        dest = figures_dir / src.name
        if dest.exists() and dest.resolve() != src.resolve():
            warning(f"{src.name} already exists in {figures_dir} from another source file -- overwriting")
        shutil.copy2(src, dest)
        n += 1
    return n


def _print_df(df: pd.DataFrame):
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
