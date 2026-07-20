#!/usr/bin/env python3
"""Standalone runner producing four specific Part 3 deliverables -- no
Jupyter, and no longer the full GEMVAP_Pipeline_gnomAD4_part3_venn.ipynb
reproduction (trimmed down from the original Steps 11-16 to just these four
outputs, dropping every other intermediate Venn panel/CSV that script used
to also produce).

Requires Part 1 (run_part1_train_calibrate.py) to have already been run with
the same config_gnomad4.yaml -- this script only reloads the GEMVAP model
fits and PP3/BP4 thresholds Part 1 wrote to OUTPUT_DIR. It does not need
Part 2's outputs.

Usage (from anywhere -- the script chdirs to its own folder first):
    python run_part3_venn.py
    python run_part3_venn.py --config other.yaml

Writes to OUTPUT_DIR/venn/ (see config_gnomad4.yaml):
  venn_predictor_tier_counts.csv     -- Pathogenic/Ambiguous/Benign counts per
                                        predictor (AlphaMissense, REVEL,
                                        GEMVAP 1/2/3, GEMVAP Union, GEMVAP
                                        Intersection), for the 1345-variant
                                        pool and all FBN1 missense variants.
  venn_test_confusion_matrices.png   -- TP/FP/FN/TN heatmap per predictor
                                        (GEMVAP 1/2/3, Any GEMVAP, REVEL,
                                        AlphaMissense) on the held-out test
                                        set, ambiguous calls excluded.
  venn_test_true_label_grid.png      -- 2x2 fixed-layout Venn grid: true-
                                        pathogenic/true-control test-set
                                        variants x predicted-pathogenic/
                                        predicted-control.
  venn_new_format_pool_full_control_ac11.png
                                     -- 3x2 fixed-layout Venn grid: all
                                        missense / 1345-pool / gnomAD-control
                                        (AC>11) populations x predicted-
                                        pathogenic/predicted-control.
Every run regenerates every output from scratch, then refreshes OUTPUT_DIR's
manifest for the next cache check.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

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
        "--figures-dir", default="figures",
        help="Folder (relative to gemvap_clean_pipeline/) to copy every Venn PNG generated "
             "under OUTPUT_DIR/venn/ into, flattened into one place for easy browsing "
             "(default: figures/). Pass '' to skip this step.",
    )
    return p.parse_args()


def main():
    cli_args = parse_args()

    # ================================================================= Step 1
    step(1, "Environment & configuration setup (reloaded for Part 3)")
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
    PEJAVER_THRESHOLDS = cfg["pejaver_thresholds"]
    OUTPUT_DIR = Path(cfg["output_dir"])
    SEED = int(cfg.get("seed", 42))
    NEW_DENOVO_PATH = cfg.get("new_denovo_path") or None
    NEW_DENOVO_PATH_2 = cfg.get("new_denovo_path_2") or None

    TRAIN_GEMVAP2 = bool(cfg["train_gemvap2"])
    TRAIN_GEMVAP3 = bool(cfg["train_gemvap3"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result(f"Config loaded; output will be written to {OUTPUT_DIR.resolve()}")

    summary_table(
        "Key Configuration Settings",
        ["Setting", "Value"],
        [
            ("data_path", DATA_PATH),
            ("train_gemvap2", TRAIN_GEMVAP2),
            ("train_gemvap3", TRAIN_GEMVAP3),
            ("output_dir", str(OUTPUT_DIR)),
        ],
    )

    subsection("Prerequisite check")
    action(f"Verifying that Part 1 (Steps 1-6) has already written its outputs to {OUTPUT_DIR}")

    required = [OUTPUT_DIR / "GEMVAP_1_fit.pkl", OUTPUT_DIR / "calibration" / "GEMVAP_1_thresholds.csv"]
    if TRAIN_GEMVAP2:
        required += [OUTPUT_DIR / "GEMVAP_2_fit.pkl", OUTPUT_DIR / "calibration" / "GEMVAP_2_thresholds.csv"]
    if TRAIN_GEMVAP3:
        required += [OUTPUT_DIR / "GEMVAP_3_fit.pkl", OUTPUT_DIR / "calibration" / "GEMVAP_3_thresholds.csv"]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Missing expected output(s) from Part 1 (Steps 1-6):\n  "
            + "\n  ".join(missing)
            + f"\n\nRun run_part1_train_calibrate.py first, with this same "
              f"{cfg_path}, so these files exist in {OUTPUT_DIR}."
        )
    result(f"All Part 1 outputs found in {OUTPUT_DIR} -- continuing.")

    from gemvap_pipeline.venn import (
        load_gemvap_fits, load_venn_cohort, load_all_missense_variants,
        pejaver_supporting_thresholds, bp4_supporting_thresholds,
        pathogenic_set, benign_set, draw_true_label_venn_grid, draw_confusion_matrix_grid,
        load_alphamissense_hg38, build_full_predictor_table, predictor_tier_counts,
        AM_BENIGN_THRESHOLD, AM_PATHOGENIC_THRESHOLD,
    )
    from gemvap_pipeline.pejaver_tools import parse_pejaver_thresholds
    from gemvap_pipeline.data import load_processor, load_dataset_csv
    from gemvap_pipeline.evaluation import denovo_pathogenic_overrides

    EXCEL_PATH = Path("../data/raw/20250609 FELIX 1346 AM_REVEL_GEMVAP_Analysis_20250609_full_outcomes.xlsx")
    ALPHAMISSENSE_PATH = Path("data/raw/AlphaMissense_hg38_FBN1.tsv")
    VENN_DIR = OUTPUT_DIR / "venn"
    VENN_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================= Step 2
    step(2, "Load model fits and build the pathogenic/benign call sets for each population")
    inputs([
        (str(EXCEL_PATH), "1346-variant annotated file with pre-called REVEL_Prediction column"),
        (DATA_PATH, "predictor score table, merged in by cDNA"),
        (str(ALPHAMISSENSE_PATH), "FBN1-region extract (chr15:48.40-48.65Mb) of AlphaMissense's genome-wide "
                                   "hg38 predictions, joined by (chr, pos, ref, alt)"),
        ("GEMVAP_{1,2,3}_fit.pkl", "model fits produced in Part 1, Steps 3-5"),
        ("calibration/GEMVAP_{1,2,3}_thresholds.csv", "PP3/BP4 thresholds fit in Part 1, Step 6"),
    ])
    info("REVEL's classification is pre-called in the Excel file (confirmed equivalent to its own\n"
         "Pejaver PP3_Supporting threshold); AlphaMissense is always scored from\n"
         "AlphaMissense_hg38_FBN1.tsv's own 'pathogenicity score' (>0.5642 Pathogenic / <0.34\n"
         "Benign), joined by genomic coordinates. GEMVAP classifications are recomputed here from\n"
         "each model's fit + threshold, so all four sets are compared consistently.")

    subsection("Loading and joining input data")
    fits = load_gemvap_fits(OUTPUT_DIR)
    merged, set_am, set_revel, n = load_venn_cohort(EXCEL_PATH, DATA_PATH, fits, ALPHAMISSENSE_PATH)

    pej_thr = pejaver_supporting_thresholds(fits, OUTPUT_DIR / "calibration")
    bp4_thr = bp4_supporting_thresholds(fits, OUTPUT_DIR / "calibration")
    revel_thr = parse_pejaver_thresholds(PEJAVER_THRESHOLDS)["REVEL"]
    revel_pp3, revel_bp4 = revel_thr["PP3_Supporting"], revel_thr["BP4_Supporting"]
    ps_cohort = {name: pathogenic_set(merged, fits[name], pej_thr[name]) for name in fits}
    any_cohort = ps_cohort.get("GEMVAP_1", set()) | ps_cohort.get("GEMVAP_2", set()) | ps_cohort.get("GEMVAP_3", set())

    subsection("1345-variant pathogenic-cases pool (1346-variant cohort minus one flagged variant)")
    excl_mask_cohort = merged["cDNA"].astype(str).str.match(r"c\.141[456]([^0-9]|$)")
    excl_idx_cohort = set(merged.index[excl_mask_cohort])
    action(f"Dropping {len(excl_idx_cohort)} flagged variant(s) from the cohort: "
           f"{sorted(merged.loc[list(excl_idx_cohort), 'cDNA'].tolist())}")
    pool = merged.drop(index=excl_idx_cohort)
    n_pool = len(pool)
    result(f"{n_pool} variants remain in the pathogenic-cases pool")

    set_am_pool = set_am - excl_idx_cohort
    set_revel_pool = set_revel - excl_idx_cohort
    any_gemvap_pool = any_cohort - excl_idx_cohort

    subsection("Joining REVEL_score onto the pool (not present on the Excel-derived frame)")
    tsv_annot = pd.read_csv(DATA_PATH, sep="\t", low_memory=False, na_values=["."])
    tsv_annot.columns = tsv_annot.columns.str.lstrip("#").str.strip()
    tsv_annot["REVEL_score"] = pd.to_numeric(tsv_annot["REVEL_score"], errors="coerce")
    tsv_lookup = tsv_annot.drop_duplicates(subset="cDNA").set_index("cDNA")[["REVEL_score"]]
    pool_annot = pool.join(tsv_lookup, on="cDNA")

    subsection("All FBN1 missense variants")
    missense_all, set_am_all, set_revel_all, n_all = load_all_missense_variants(
        DATA_PATH, ALPHAMISSENSE_PATH, PEJAVER_THRESHOLDS,
    )
    excl_mask = missense_all["cDNA"].astype(str).str.match(r"c\.141[456]([^0-9]|$)")
    excl_idx = set(missense_all.index[excl_mask])
    action(f"Dropping {len(excl_idx)} variant(s) at cDNA positions 1414/1415/1416 from the all-missense set: "
           f"{sorted(missense_all.loc[list(excl_idx), 'cDNA'].tolist())}")
    missense_all = missense_all.drop(index=excl_idx)
    set_am_all = set_am_all - excl_idx
    set_revel_all = set_revel_all - excl_idx
    n_all = n_all - len(excl_idx)
    result(f"{n_all} variants remain in the all-missense set")

    ps_all = {name: pathogenic_set(missense_all, fits[name], pej_thr[name]) for name in fits}
    any_all = ps_all.get("GEMVAP_1", set()) | ps_all.get("GEMVAP_2", set()) | ps_all.get("GEMVAP_3", set())

    subsection("Benign (BP4_Supporting) call sets -- pool and all-missense")
    set_am_benign_pool = set(pool_annot.index[
        pd.to_numeric(pool_annot["pathogenicity score"], errors="coerce") < AM_BENIGN_THRESHOLD
    ])
    set_revel_benign_pool = set(pool_annot.index[
        pd.to_numeric(pool_annot["REVEL_score"], errors="coerce") <= revel_bp4
    ])
    ps_benign_pool = {name: benign_set(pool_annot, fits[name], bp4_thr[name]) for name in fits}
    any_benign_pool = (
        ps_benign_pool.get("GEMVAP_1", set())
        | ps_benign_pool.get("GEMVAP_2", set())
        | ps_benign_pool.get("GEMVAP_3", set())
    )

    set_am_benign_all = set(missense_all.index[
        pd.to_numeric(missense_all["pathogenicity score"], errors="coerce") < AM_BENIGN_THRESHOLD
    ])
    set_revel_benign_all = set(missense_all.index[
        pd.to_numeric(missense_all["REVEL_score"], errors="coerce") <= revel_bp4
    ])
    ps_benign_all = {name: benign_set(missense_all, fits[name], bp4_thr[name]) for name in fits}
    any_benign_all = (
        ps_benign_all.get("GEMVAP_1", set())
        | ps_benign_all.get("GEMVAP_2", set())
        | ps_benign_all.get("GEMVAP_3", set())
    )
    result(f"Any GEMVAP benign -- pool: {len(any_benign_pool)}/{n_pool}; "
           f"all-missense: {len(any_benign_all)}/{n_all}")

    subsection("gnomAD-control population, joint AC > 11")
    action(f"Loading {DATA_PATH} via DataProcessor to get its is_ctrl label")
    dp_full = load_processor(DATA_PATH, seed=SEED)
    ctrl_cdnas = set(dp_full.data.loc[dp_full.is_ctrl, "cDNA"])
    idx_ctrl_all = set(missense_all.index[missense_all["cDNA"].isin(ctrl_cdnas)])
    n_ctrl_all = len(idx_ctrl_all)

    ac_all = pd.to_numeric(missense_all["gnomad4_joint_AC"], errors="coerce")
    idx_ac_gt11 = set(missense_all.index[ac_all > 11])
    idx_ctrl_ac11 = idx_ctrl_all & idx_ac_gt11
    n_ctrl_ac11 = len(idx_ctrl_ac11)
    result(f"{n_ctrl_ac11}/{n_ctrl_all} gnomAD-control variants have joint AC > 11")

    # ================================================================= Step 3
    step(3, "Predictor Pathogenic/Ambiguous/Benign tier counts -- pool vs all missense")
    action("Building a per-variant table with each predictor's raw score and 3-way "
           "Pathogenic/Ambiguous/Benign call (AlphaMissense, REVEL, GEMVAP 1/2/3), then "
           "tabulating tier counts per predictor plus GEMVAP Union/Intersection")

    full_table = build_full_predictor_table(missense_all, fits, pej_thr, bp4_thr, revel_pp3, revel_bp4)
    pool_full_table = build_full_predictor_table(pool_annot, fits, pej_thr, bp4_thr, revel_pp3, revel_bp4)

    tier_counts_pool = predictor_tier_counts(pool_full_table)
    tier_counts_pool.insert(0, "Dataset", f"1345-variant pathogenic-cases pool (n={n_pool})")
    tier_counts_all = predictor_tier_counts(full_table)
    tier_counts_all.insert(0, "Dataset", f"All FBN1 missense variants (n={n_all})")

    tier_counts = pd.concat([tier_counts_pool, tier_counts_all], ignore_index=True)
    tier_counts_out = VENN_DIR / "venn_predictor_tier_counts.csv"
    tier_counts.to_csv(tier_counts_out, index=False)
    _print_df(tier_counts)
    result(f"Saved -> {tier_counts_out}")

    # ================================================================= Step 4
    step(4, "Test-set confusion matrices -- GEMVAP 1/2/3, Any GEMVAP, REVEL, AlphaMissense")
    info("Reloads test_dataset.csv (the held-out, 1:1 case/control-balanced set Part 1's GEMVAP\n"
         "models were evaluated on), true-labelled via is_case/is_ctrl. Each predictor's own\n"
         "BP4_Supporting threshold defines its predicted-control (benign) set, and any variant\n"
         "that is neither PP3_Supporting-pathogenic nor BP4_Supporting-benign for that predictor\n"
         "(i.e. ambiguous / VUS-range) is excluded from that predictor's matrix.")

    action("Reloading test_dataset.csv with the same de novo pathogenic overrides Part 1 used")
    test_overrides = denovo_pathogenic_overrides(NEW_DENOVO_PATH, NEW_DENOVO_PATH_2)
    df_test = load_dataset_csv(OUTPUT_DIR / "test_dataset.csv", DATA_PATH, seed=SEED, pathogenic_overrides=test_overrides)
    n_test = len(df_test.data)
    result(f"{n_test} test-set variants reloaded: {int(df_test.is_case.sum())} pathogenic, "
           f"{int(df_test.is_ctrl.sum())} control")

    action(f"Joining AlphaMissense score onto the test set from {ALPHAMISSENSE_PATH.name}")
    test_data = df_test.data.merge(
        load_alphamissense_hg38(ALPHAMISSENSE_PATH), how="left",
        left_on=["#chr", "pos(1-based)", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"],
    )
    am_scores_test = pd.to_numeric(test_data["pathogenicity score"], errors="coerce")
    revel_scores_test = pd.to_numeric(test_data["REVEL_score"], errors="coerce")
    set_am_test = set(test_data.index[am_scores_test > AM_PATHOGENIC_THRESHOLD])
    set_revel_test = set(test_data.index[revel_scores_test >= revel_pp3])

    ps_test = {name: pathogenic_set(test_data, fits[name], pej_thr[name]) for name in fits}
    any_gemvap_test = ps_test.get("GEMVAP_1", set()) | ps_test.get("GEMVAP_2", set()) | ps_test.get("GEMVAP_3", set())

    ps_benign_test = {name: benign_set(test_data, fits[name], bp4_thr[name]) for name in fits}
    any_gemvap_benign_test = (
        ps_benign_test.get("GEMVAP_1", set())
        | ps_benign_test.get("GEMVAP_2", set())
        | ps_benign_test.get("GEMVAP_3", set())
    )
    set_revel_benign_test = set(test_data.index[revel_scores_test <= revel_bp4])
    set_am_benign_test = set(test_data.index[am_scores_test < AM_BENIGN_THRESHOLD])

    true_pathogenic_idx = set(df_test.data.index[df_test.is_case & ~df_test.is_ctrl])
    true_control_idx = set(df_test.data.index[df_test.is_ctrl & ~df_test.is_case])
    result(f"{len(true_pathogenic_idx)} true-pathogenic, {len(true_control_idx)} true-control in the "
           f"test set (Any GEMVAP predicted-pathogenic: {len(any_gemvap_test)}/{n_test})")

    predictor_call_sets = {
        "GEMVAP 1": (ps_test.get("GEMVAP_1", set()), ps_benign_test.get("GEMVAP_1", set())),
        "GEMVAP 2": (ps_test.get("GEMVAP_2", set()), ps_benign_test.get("GEMVAP_2", set())),
        "GEMVAP 3": (ps_test.get("GEMVAP_3", set()), ps_benign_test.get("GEMVAP_3", set())),
        "Any GEMVAP": (any_gemvap_test, any_gemvap_benign_test),
        "REVEL": (set_revel_test, set_revel_benign_test),
        "AlphaMissense": (set_am_test, set_am_benign_test),
    }
    confusion_matrices = {}
    for name, (pathogenic_call_set, benign_call_set) in predictor_call_sets.items():
        benign_call_set = benign_call_set - pathogenic_call_set  # pathogenic call wins any overlap
        tp = len(true_pathogenic_idx & pathogenic_call_set)
        fn = len(true_pathogenic_idx & benign_call_set)
        fp = len(true_control_idx & pathogenic_call_set)
        tn = len(true_control_idx & benign_call_set)
        n_ambiguous = len(true_pathogenic_idx | true_control_idx) - (tp + fn + fp + tn)
        confusion_matrices[name] = (tp, fp, fn, tn)
        result(f"{name}: TP={tp} FN={fn} FP={fp} TN={tn} ({n_ambiguous} ambiguous excluded)")

    confusion_out = VENN_DIR / "venn_test_confusion_matrices.png"
    draw_confusion_matrix_grid(confusion_matrices, output_path=confusion_out)
    _list_figures(VENN_DIR, ["venn_test_confusion_matrices.png"])

    # ================================================================= Step 5
    step(5, "Test-set true-label Venn grid -- true-pathogenic/true-control x predicted class")
    info("2x2 fixed-layout Venn grid: each predictor's own pathogenic- or benign-call set drawn\n"
         "as its own circle against the full true-pathogenic or true-control population (not just\n"
         "discrepancy cases), so all three circles are informative, including GEMVAP's.")

    n_true_path = len(true_pathogenic_idx)
    n_true_ctrl = len(true_control_idx)

    true_label_panels = [
        ("True-pathogenic test variants, predicted pathogenic (PP3_Supporting)",
         any_gemvap_test & true_pathogenic_idx, set_am_test & true_pathogenic_idx,
         set_revel_test & true_pathogenic_idx, n_true_path),
        ("True-pathogenic test variants, predicted control (BP4_Supporting)",
         any_gemvap_benign_test & true_pathogenic_idx, set_am_benign_test & true_pathogenic_idx,
         set_revel_benign_test & true_pathogenic_idx, n_true_path),
        ("True-control test variants, predicted control (BP4_Supporting)",
         any_gemvap_benign_test & true_control_idx, set_am_benign_test & true_control_idx,
         set_revel_benign_test & true_control_idx, n_true_ctrl),
        ("True-control test variants, predicted pathogenic (PP3_Supporting)",
         any_gemvap_test & true_control_idx, set_am_test & true_control_idx,
         set_revel_test & true_control_idx, n_true_ctrl),
    ]
    for subtitle, gemvap_set, am_set, revel_set, n_panel in true_label_panels:
        result(f"{subtitle}: n={n_panel}")

    action("Drawing all 4 panels into one 2x2 grid, sharing a fixed circle layout (same\n"
           "position/size per predictor in every panel), no titles, single shared legend")
    grid_out = VENN_DIR / "venn_test_true_label_grid.png"
    grid_panel_order = [
        true_label_panels[0],  # top-left: true-pathogenic / predicted-pathogenic
        true_label_panels[1],  # top-right: true-pathogenic / predicted-control
        true_label_panels[3],  # bottom-left: true-control / predicted-pathogenic
        true_label_panels[2],  # bottom-right: true-control / predicted-control
    ]
    draw_true_label_venn_grid(
        [(g, a, r, n) for _, g, a, r, n in grid_panel_order],
        output_path=grid_out,
    )
    info("Panel order (row-major): top-left = true-pathogenic/predicted-pathogenic, "
         "top-right = true-pathogenic/predicted-control, bottom-left = "
         "true-control/predicted-pathogenic, bottom-right = true-control/predicted-control.")
    _list_figures(VENN_DIR, ["venn_test_true_label_grid.png"])

    # ================================================================= Step 6
    step(6, "New-format Venn grid -- all-missense, 1345-pool, and gnomAD-control (AC>11) populations")
    info("Same fixed-layout/count-shaded/named-circle format as the test-set grid above, applied\n"
         "to three other populations: all FBN1 missense variants, the 1345-variant pathogenic-\n"
         "cases pool, and gnomAD-control variants with joint AC > 11. Each population gets a\n"
         "predicted-pathogenic (PP3_Supporting) panel and a predicted-control/benign\n"
         "(BP4_Supporting) panel, laid out 3 rows x 2 columns (rows = population, columns =\n"
         "predicted class).")

    pathogenic_ac11 = (any_all & idx_ctrl_ac11, set_am_all & idx_ctrl_ac11, set_revel_all & idx_ctrl_ac11, n_ctrl_ac11)
    benign_ac11 = (
        any_benign_all & idx_ctrl_ac11, set_am_benign_all & idx_ctrl_ac11,
        set_revel_benign_all & idx_ctrl_ac11, n_ctrl_ac11,
    )

    new_format_panels = [
        ("All FBN1 missense, predicted pathogenic", any_all, set_am_all, set_revel_all, n_all),
        ("All FBN1 missense, predicted control", any_benign_all, set_am_benign_all, set_revel_benign_all, n_all),
        ("1345-variant pool, predicted pathogenic", any_gemvap_pool, set_am_pool, set_revel_pool, n_pool),
        ("1345-variant pool, predicted control", any_benign_pool, set_am_benign_pool, set_revel_benign_pool, n_pool),
        ("gnomAD-control AC>11, predicted pathogenic", *pathogenic_ac11),
        ("gnomAD-control AC>11, predicted control", *benign_ac11),
    ]
    for subtitle, gemvap_set, am_set, revel_set, n_panel in new_format_panels:
        result(f"{subtitle}: n={n_panel}")

    action("Drawing all 6 panels into one 3x2 grid (rows = population, columns = predicted class), "
           "sharing a fixed circle layout, one title per row, single shared legend")
    new_format_out = VENN_DIR / "venn_new_format_pool_full_control_ac11.png"
    draw_true_label_venn_grid(
        [(g, a, r, n) for _, g, a, r, n in new_format_panels],
        output_path=new_format_out,
        ncols=2,
        row_titles=[
            f"All FBN1 missense variants (n={n_all})",
            f"1345-variant pathogenic-cases pool (n={n_pool})",
            f"gnomAD-control variants, AC>11 (n={n_ctrl_ac11})",
        ],
    )
    info("Panel order (row-major, 3x2): row 1 = all missense (predicted pathogenic, predicted "
         "control); row 2 = 1345-variant pool, same column order; row 3 = gnomAD-control AC>11, "
         "same column order.")
    _list_figures(VENN_DIR, ["venn_new_format_pool_full_control_ac11.png"])

    # ================================================================ Manifest
    subsection("Saving output manifest")
    action(f"Hashing files in {OUTPUT_DIR} for next run's change-verification baseline")

    from output_archive import compute_manifest, save_manifest

    manifest = compute_manifest(OUTPUT_DIR)
    save_manifest(OUTPUT_DIR, manifest)
    result(
        f"Manifest saved for {len(manifest)} file(s) -- next run's cache-verification "
        "check will compare cached outputs against this state"
    )

    # ============================================================= Figures dir
    if cli_args.figures_dir:
        subsection("Collecting all generated images into one folder")
        figures_dir = Path(cli_args.figures_dir)
        action(f"Copying every PNG under {VENN_DIR} into {figures_dir}")
        n_copied = _collect_figures(VENN_DIR, figures_dir)
        result(f"Copied {n_copied} PNG(s) to {figures_dir.resolve()}")

    # ================================================================ Wrap-up
    subsection("End of Part 3")
    info(f"{VENN_DIR} now contains the three Venn-diagram figures and one tier-counts CSV "
         "listed at the top of this script's docstring.")


def _list_figures(output_dir, filenames):
    """Print the path of each named PNG under output_dir that exists (script
    equivalent of the notebook's inline `display(Image(...))` calls)."""
    output_dir = Path(output_dir)
    for fname in filenames:
        p = output_dir / fname
        if p.exists():
            result(str(p))


def _collect_figures(output_dir, figures_dir) -> int:
    """Copy every PNG found under output_dir into a single flat figures_dir,
    for easy browsing without hunting through OUTPUT_DIR/venn. Overwrites on
    re-run so figures_dir always reflects the latest outputs. Returns the
    number of files copied."""
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
