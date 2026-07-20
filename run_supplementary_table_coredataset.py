#!/usr/bin/env python3
"""Standalone runner reproducing GEMVAP_Supplementary_Table_CoreDataset.ipynb --
no Jupyter required.

Standalone companion to run_part1_train_calibrate.py / Part 1 of the notebook
split. Does not touch the train/test split, model fits, or calibration -- it
only re-reads cached outputs already written by Part 1 (training_dataset.csv,
test_dataset.csv, GEMVAP_{1,2,3}_fit.pkl, calibration/{name}_thresholds.csv,
_denovo_overrides.json) plus the raw core dataset at data_path, and produces
six supplementary tables (each a CSV + a publication-style PNG table figure):

  1. Supplementary_Table_CoreDataset               -- pathogenic/control/intersection/other
                                                       counts, all-consequences vs. missense-only
  2. Supplementary_Table_PredictorThresholds        -- all 38 candidate predictors' ROC case
                                                       thresholds for GEMVAP 1/2/3, selected flags
  3. Supplementary_Table_DenovoPathogenicVariants   -- the 66 de novo pathogenic variants in the
                                                       Part 1 test set
     Supplementary_Table_DenovoPathogenicVariants_Compact -- same variants, CSV only, columns
                                                       hg38/Ref/Alt/cDNA/Ref aa/Alt aa/Protein
  4. Supplementary_Table_TrainingComposition        -- per-model training subset sizes
  5. Supplementary_Table_PredictedPathogenic        -- per-model predicted-pathogenic counts on
                                                       that model's own training subset
  6. Supplementary_Table_PredictedPathogenic_FullTrainingSet -- same, but all three models
                                                       scored against the common full training set
  7. Supplementary_Table_LRTargets                  -- ACMG/AMP evidence-level LR targets
                                                       (Pejaver et al. 2022 framework) alongside
                                                       each model's fitted score threshold
  8. Supplementary_Table_TestComposition            -- test-set cysteine / in-domain conserved /
                                                       others counts, pathogenic vs. control

Requires Part 1 (run_part1_train_calibrate.py) to have already been run with
the same config, so its cached outputs exist under OUTPUT_DIR.

Usage (from anywhere -- the script chdirs to its own folder first):
    python run_supplementary_table_coredataset.py
    python run_supplementary_table_coredataset.py --config other.yaml
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))

from pipeline_log import step, inputs, info, subsection, action, result, summary_table, warning

MODEL_NAMES = ["GEMVAP_1", "GEMVAP_2", "GEMVAP_3"]


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
        help="Folder (relative to gemvap_clean_pipeline/) to copy every PNG generated "
             "under OUTPUT_DIR into, flattened into one place for easy browsing "
             "(default: figures/, same folder Part 1 uses). Pass '' to skip this step.",
    )
    return p.parse_args()


def _save_table(df_display, csv_path, png_path, title, figsize, fontsize=10.5, scale_y=1.6,
                cell_fmt=None, extra_style=None, footnote=None):
    """Shared CSV + matplotlib-table-figure export used by every supplementary table below.

    The CSV holds exactly the same formatted strings as the PNG's cells (not the raw,
    unrounded/unformatted values), so the two are always the same "final numbers".
    """
    col_labels = [df_display.index.name or "Category"] + list(df_display.columns)
    if cell_fmt is None:
        def cell_fmt(idx, col):
            try:
                return f"{df_display.loc[idx, col]:,}"
            except (TypeError, ValueError):
                return str(df_display.loc[idx, col])
    cell_text = [[str(idx)] + [cell_fmt(idx, col) for col in df_display.columns] for idx in df_display.index]

    pd.DataFrame(cell_text, columns=col_labels).to_csv(csv_path, index=False)
    result(f"Saved {csv_path}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, scale_y)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dddddd")
        elif extra_style is not None:
            extra_style(row, col, cell, df_display)

    ax.set_title(title, pad=14, fontsize=12)
    if footnote:
        fig.text(0.5, 0.005, footnote, ha="center", fontsize=8, style="italic")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    result(f"Saved {png_path}")


def main():
    cli_args = parse_args()

    # ================================================================= Step 1
    step(1, "Environment & configuration setup")
    cwd = Path(os.getcwd())
    if not (cwd / "gemvap_pipeline").is_dir():
        raise RuntimeError(f"Working directory is '{cwd}' -- gemvap_pipeline/ is missing.")

    cfg_path = Path(cli_args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")
    cfg = yaml.safe_load(open(cfg_path))
    DATA_PATH = cfg["data_path"]
    SEED = int(cfg["seed"])
    OUTPUT_DIR = Path(cfg["output_dir"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs([(DATA_PATH, "core dataset -- raw FBN1 variant table, pathogenic + control + unlabeled")])
    result(f"Config loaded. data_path={DATA_PATH}  seed={SEED}  output_dir={OUTPUT_DIR}")

    # ================================================================= Step 2
    step(2, "Load the core dataset")
    info("The core dataset is the raw table as read by DataProcessor -- before missense\n"
         "filtering, training-cohort exclusion, or the de novo/test-set split applied in\n"
         "Part 1. is_case / is_ctrl / is_inte come directly from DataProcessor.create_filters().")

    from gemvap_pipeline.data import load_processor

    action(f"Reading {DATA_PATH}")
    dp = load_processor(DATA_PATH, seed=SEED)
    result(f"Core dataset loaded: {len(dp.data)} rows")

    # ================================================================= Step 3
    step(3, "Classify variants: pathogenic / control / intersection / others")
    info("Pathogenic  = in a pathogenic source DB (HGMD/UMD/FRANKEN/MUTDB/PARIS/GENT), not in gnomAD.\n"
         "Control     = in gnomAD (exomes/genomes), not in a pathogenic source DB.\n"
         "Intersection= in BOTH a pathogenic source DB and gnomAD (contradictory labels).\n"
         "Others      = in neither -- unlabeled.")

    def composition_table(mask: pd.Series, label: str) -> pd.DataFrame:
        n_path = int((dp.is_case & mask).sum())
        n_ctrl = int((dp.is_ctrl & mask).sum())
        n_inte = int((dp.is_inte & mask).sum())
        n_total = int(mask.sum())
        n_other = n_total - n_path - n_ctrl - n_inte
        rows = [
            ("Pathogenic", n_path), ("Control", n_ctrl), ("Intersection", n_inte),
            ("Others", n_other), ("Total", n_total),
        ]
        return pd.DataFrame(rows, columns=["Category", label]).set_index("Category")

    table_all = composition_table(pd.Series(True, index=dp.data.index), "N (all consequences)")
    table_mis = composition_table(dp.is_mis, "N (missense only)")

    action("Counting variants by class -- all consequence types")
    summary_table("Core Dataset Composition -- All Consequence Types", ["Category", "N"],
                  [(idx, int(row.iloc[0])) for idx, row in table_all.iterrows()])
    action("Counting variants by class -- missense only")
    summary_table("Core Dataset Composition -- Missense Only", ["Category", "N"],
                  [(idx, int(row.iloc[0])) for idx, row in table_mis.iterrows()])

    combined = table_all.join(table_mis)

    # ================================================================= Step 4
    step(4, "Render and save the supplementary table")
    _save_table(
        combined,
        OUTPUT_DIR / "Supplementary_Table_CoreDataset.csv",
        OUTPUT_DIR / "Supplementary_Table_CoreDataset.png",
        "Supplementary Table -- Core Dataset Composition (FBN1, gnomAD v4)",
        figsize=(6, 2.2), fontsize=11,
        cell_fmt=lambda idx, col: f"{combined.loc[idx, col]:,}",
        extra_style=lambda row, col, cell, df: cell.set_text_props(weight="bold") if row == len(df) else None,
    )

    # ================================================================= Step 5
    step(5, "Per-predictor ROC thresholds for GEMVAP 1, 2, 3")
    info("Reloading the cached fit objects from Part 1 (Steps 3-5) -- no retraining.\n"
         "Each fit stores rbc['threshold']['case']: a case threshold per candidate\n"
         "predictor, fitted via 10-fold ROC cross-validation on that model's own\n"
         "training subset -- plus top_predictors: the panel actually used for voting.")

    if "pandas.core.indexes.numeric" not in sys.modules:
        # Compatibility shim for fits pickled under pandas < 2.0 (Int64Index etc.
        # were folded into the generic Index class and the module removed).
        shim = types.ModuleType("pandas.core.indexes.numeric")
        shim.Int64Index = pd.Index
        shim.Float64Index = pd.Index
        shim.UInt64Index = pd.Index
        sys.modules["pandas.core.indexes.numeric"] = shim

    from gemvap_pipeline.model import load_fit_result, apply_consensus_score
    from packages.package1.predictor_selection import DataProcessor

    fits = {}
    for name in MODEL_NAMES:
        fit_path = OUTPUT_DIR / f"{name}_fit.pkl"
        action(f"Loading {fit_path}")
        fits[name] = load_fit_result(fit_path)
        result(f"{name}: {len(fits[name]['top_predictors'])} predictors in panel")

    subsection("Per-model predictor panels, in KS-selection order")
    for name in MODEL_NAMES:
        top_predictors = fits[name]["top_predictors"]
        thresholds = fits[name]["rbc"]["threshold"]["case"]
        rows = [(rank + 1, predictor, thresholds[predictor]) for rank, predictor in enumerate(top_predictors)]
        summary_table(f"{name} -- predictor panel and case thresholds",
                      ["Rank", "Predictor", "Threshold"],
                      [(r, p, f"{t:.4f}") for r, p, t in rows])

    subsection("Combined table: ALL 38 candidate predictors across GEMVAP 1, 2, 3")
    all_predictors = fits["GEMVAP_1"]["ordered_predictors_by_ks"]
    assert set(all_predictors) == set(fits["GEMVAP_2"]["ordered_predictors_by_ks"])
    assert set(all_predictors) == set(fits["GEMVAP_3"]["ordered_predictors_by_ks"])

    combined_rows = []
    for predictor in all_predictors:
        row = {"Predictor": predictor}
        for name in MODEL_NAMES:
            thresholds = fits[name]["rbc"]["threshold"]["case"]
            top_predictors = fits[name]["top_predictors"]
            row[f"{name}_threshold"] = thresholds[predictor]
            row[f"{name}_selected"] = predictor in top_predictors
        combined_rows.append(row)

    combined_thresholds = pd.DataFrame(combined_rows).set_index("Predictor")
    n_sel = {name: int(combined_thresholds[f"{name}_selected"].sum()) for name in MODEL_NAMES}
    result(f"Combined table: {len(combined_thresholds)} candidate predictors "
           f"(selected -- GEMVAP_1: {n_sel['GEMVAP_1']}, GEMVAP_2: {n_sel['GEMVAP_2']}, "
           f"GEMVAP_3: {n_sel['GEMVAP_3']})")

    subsection("Saving the full threshold table")

    def _fmt(predictor, name):
        value = combined_thresholds.loc[predictor, f"{name}_threshold"]
        selected = combined_thresholds.loc[predictor, f"{name}_selected"]
        return f"{value:.2f}*" if selected else f"{value:.2f}"

    cell_text = [[predictor] + [_fmt(predictor, name) for name in MODEL_NAMES]
                 for predictor in combined_thresholds.index]
    col_labels = ["Predictor"] + MODEL_NAMES

    csv_path = OUTPUT_DIR / "Supplementary_Table_PredictorThresholds.csv"
    pd.DataFrame(cell_text, columns=col_labels).to_csv(csv_path, index=False)
    result(f"Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(combined_thresholds) + 1.4))
    ax.axis("off")

    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="upper center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.3)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dddddd")
        elif col > 0:
            predictor = combined_thresholds.index[row - 1]
            name = MODEL_NAMES[col - 1]
            if combined_thresholds.loc[predictor, f"{name}_selected"]:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#eaf4ea")

    ax.set_title("Supplementary Table -- Per-predictor ROC case thresholds, all 38 candidates (GEMVAP 1, 2, 3)",
                 pad=18, fontsize=11)
    fig.text(0.5, 0.005, "* / bold = predictor selected into that model's voting panel (top_predictors)",
              ha="center", fontsize=8, style="italic")

    png_path = OUTPUT_DIR / "Supplementary_Table_PredictorThresholds.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    result(f"Saved {png_path}")

    # ================================================================= Step 6
    step(6, "De novo pathogenic FBN1 missense variants in the test set")
    info("Reloading test_dataset.csv with the saved de novo override set, so is_case /\n"
         "is_denovo are reconstructed exactly as in Part 1, Step 2 -- then filtering to\n"
         "variants that are BOTH de novo AND pathogenic: the test set's pathogenic half.")

    from gemvap_pipeline.data import load_cached_datasets, load_denovo_overrides

    action("Reloading cached train/test split")
    denovo_overrides = load_denovo_overrides(OUTPUT_DIR)
    _, df_test = load_cached_datasets(OUTPUT_DIR, DATA_PATH, seed=SEED, pathogenic_overrides=denovo_overrides)

    denovo_case_mask = df_test.is_denovo & df_test.is_case
    result(f"{int(denovo_case_mask.sum())} de novo pathogenic variants found in the test set")

    subsection("Building the variant table")
    denovo_pathogenic = df_test.data.loc[denovo_case_mask, [
        "cDNA", "HGVSp_VEP", "aaref", "aapos", "aaalt", "EXON",
        "#chr", "pos(1-based)", "ref", "alt",
    ]].copy()
    denovo_pathogenic["Cysteine_variant"] = df_test.is_cys.loc[denovo_case_mask].values
    denovo_pathogenic["aapos"] = pd.to_numeric(denovo_pathogenic["aapos"], errors="coerce")
    denovo_pathogenic = denovo_pathogenic.sort_values("aapos").reset_index(drop=True)
    denovo_pathogenic.columns = [
        "cDNA", "Protein change", "Ref AA", "Position", "Alt AA", "Exon",
        "Chr", "Genomic pos (hg38)", "Ref allele", "Alt allele", "Cysteine variant",
    ]
    n_cys = int(denovo_pathogenic["Cysteine variant"].sum())
    result(f"{len(denovo_pathogenic)} variants  |  {n_cys} cysteine  |  {len(denovo_pathogenic) - n_cys} non-cysteine")

    subsection("Saving the de novo pathogenic variant table")
    csv_path = OUTPUT_DIR / "Supplementary_Table_DenovoPathogenicVariants.csv"
    denovo_pathogenic.to_csv(csv_path, index=False)
    result(f"Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(11, 0.28 * len(denovo_pathogenic) + 1.2))
    ax.axis("off")
    display_cols = ["cDNA", "Protein change", "Exon", "Genomic pos (hg38)", "Cysteine variant"]
    cell_text = denovo_pathogenic[display_cols].astype(str).values.tolist()

    tbl = ax.table(cellText=cell_text, colLabels=display_cols, cellLoc="center", loc="upper center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.25)
    tbl.auto_set_column_width(col=list(range(len(display_cols))))
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dddddd")

    ax.set_title(f"Supplementary Table -- De novo pathogenic FBN1 missense variants in the test set (n={len(denovo_pathogenic)})",
                 pad=18, fontsize=11)
    png_path = OUTPUT_DIR / "Supplementary_Table_DenovoPathogenicVariants.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    result(f"Saved {png_path}")

    subsection("Saving the compact de novo pathogenic variant table (CSV only)")
    compact_columns = {
        "Genomic pos (hg38)": "hg38",
        "Ref allele": "Ref",
        "Alt allele": "Alt",
        "cDNA": "cDNA",
        "Ref AA": "Ref aa",
        "Alt AA": "Alt aa",
        "Protein change": "Protein",
    }
    denovo_pathogenic_compact = denovo_pathogenic[list(compact_columns)].rename(columns=compact_columns)
    csv_path = OUTPUT_DIR / "Supplementary_Table_DenovoPathogenicVariants_Compact.csv"
    denovo_pathogenic_compact.to_csv(csv_path, index=False)
    result(f"Saved {csv_path}")

    # ================================================================= Step 7
    step(7, "Training dataset composition per GEMVAP model")
    info("Re-deriving is_case/is_ctrl on each model's own training_data (already cached\n"
         "inside fits[name] from Step 5) via the same DataProcessor.create_filters()\n"
         "logic used during training -- not re-reading the raw table.")

    composition_rows = []
    for name in MODEL_NAMES:
        model_dp = DataProcessor(DATA_PATH, seed=SEED)
        model_dp.data = fits[name]["training_data"]
        model_dp.create_filters()
        composition_rows.append((
            name, int(model_dp.is_case.sum()), int(model_dp.is_ctrl.sum()),
            len(model_dp.data), len(fits[name]["top_predictors"]),
        ))

    summary_table("GEMVAP training dataset composition",
                  ["Model", "Pathogenic", "Control", "Total", "Predictors used"], composition_rows)

    training_composition = pd.DataFrame(
        composition_rows, columns=["Model", "Pathogenic", "Control", "Total", "Predictors used"],
    ).set_index("Model")

    subsection("Saving the training composition table")
    _save_table(
        training_composition,
        OUTPUT_DIR / "Supplementary_Table_TrainingComposition.csv",
        OUTPUT_DIR / "Supplementary_Table_TrainingComposition.png",
        "Supplementary Table -- Training dataset composition per GEMVAP model",
        figsize=(6.5, 2.2), fontsize=10.5,
    )

    # ================================================================= Step 8
    step(8, "Predicted-pathogenic counts on the training set")
    info("Re-scoring each model's own training_data with its own top_predictors +\n"
         "per-predictor thresholds (apply_consensus_score), then classifying against\n"
         "the calibrated PP3_Supporting threshold from Step 6 -- score >= threshold\n"
         "counts as 'predicted pathogenic', matching annotate_variant()'s own rule.")

    pp3_supporting, bp4_supporting = {}, {}
    for name in MODEL_NAMES:
        thr = pd.read_csv(OUTPUT_DIR / "calibration" / f"{name}_thresholds.csv")
        thr = thr.set_index("evidence_level")["score_threshold"]
        pp3_supporting[name] = thr["PP3_Supporting"]
        bp4_supporting[name] = thr["BP4_Supporting"]

    predicted_rows = []
    for name in MODEL_NAMES:
        model_dp = DataProcessor(DATA_PATH, seed=SEED)
        model_dp.data = fits[name]["training_data"]
        model_dp.create_filters()

        scores = apply_consensus_score(model_dp.data, fits[name]["top_predictors"], fits[name]["rbc"]["threshold"]["case"])
        path_scores = scores[model_dp.is_case]
        pp3, bp4 = pp3_supporting[name], bp4_supporting[name]

        n_total = len(path_scores)
        n_pred_path = int((path_scores >= pp3).sum())
        n_vus = int(((path_scores > bp4) & (path_scores < pp3)).sum())
        n_pred_benign = int((path_scores <= bp4).sum())
        predicted_rows.append((name, n_total, n_pred_path, n_vus, n_pred_benign))

    predicted_pathogenic = pd.DataFrame(
        predicted_rows,
        columns=["Model", "Pathogenic (training)", "Predicted pathogenic", "VUS (indeterminate)", "Predicted benign"],
    ).set_index("Model")
    predicted_pathogenic["% predicted pathogenic"] = (
        100 * predicted_pathogenic["Predicted pathogenic"] / predicted_pathogenic["Pathogenic (training)"]
    ).round(1)

    summary_table("Predicted-pathogenic counts on the training set",
                  list(predicted_pathogenic.reset_index().columns),
                  [tuple(r) for r in predicted_pathogenic.reset_index().itertuples(index=False)])

    subsection("Saving the predicted-pathogenic table")
    _save_table(
        predicted_pathogenic,
        OUTPUT_DIR / "Supplementary_Table_PredictedPathogenic.csv",
        OUTPUT_DIR / "Supplementary_Table_PredictedPathogenic.png",
        "Supplementary Table -- Predicted-pathogenic counts on the training set (GEMVAP 1, 2, 3)",
        figsize=(9, 2.2), fontsize=9.5,
        cell_fmt=lambda idx, col: (
            f"{predicted_pathogenic.loc[idx, col]:.1f}%" if col == "% predicted pathogenic"
            else f"{predicted_pathogenic.loc[idx, col]:,}"
        ),
    )

    # ================================================================= Step 9
    step(9, "Predicted-pathogenic counts against the full training set")
    info("Reusing GEMVAP_1's training_data (the full pathogenic/control training set) as a\n"
         "common denominator, and re-scoring it with EACH model's own predictor panel +\n"
         "thresholds -- including GEMVAP_2/3 on variants (cysteine, conserved-domain) they\n"
         "were not trained on.")

    full_dp = DataProcessor(DATA_PATH, seed=SEED)
    full_dp.data = fits["GEMVAP_1"]["training_data"]
    full_dp.create_filters()
    n_full_path = int(full_dp.is_case.sum())
    result(f"Full training set: {len(full_dp.data)} variants  |  {n_full_path} pathogenic  |  "
           f"{int(full_dp.is_ctrl.sum())} control")

    full_predicted_rows = []
    for name in MODEL_NAMES:
        scores = apply_consensus_score(full_dp.data, fits[name]["top_predictors"], fits[name]["rbc"]["threshold"]["case"])
        path_scores = scores[full_dp.is_case]
        pp3, bp4 = pp3_supporting[name], bp4_supporting[name]

        n_pred_path = int((path_scores >= pp3).sum())
        n_vus = int(((path_scores > bp4) & (path_scores < pp3)).sum())
        n_pred_benign = int((path_scores <= bp4).sum())
        full_predicted_rows.append((name, n_full_path, n_pred_path, n_vus, n_pred_benign))

    full_predicted_pathogenic = pd.DataFrame(
        full_predicted_rows,
        columns=["Model", "Pathogenic (full training set)", "Predicted pathogenic", "VUS (indeterminate)", "Predicted benign"],
    ).set_index("Model")
    full_predicted_pathogenic["% predicted pathogenic"] = (
        100 * full_predicted_pathogenic["Predicted pathogenic"] / full_predicted_pathogenic["Pathogenic (full training set)"]
    ).round(1)

    summary_table("Predicted-pathogenic counts against the full training set",
                  list(full_predicted_pathogenic.reset_index().columns),
                  [tuple(r) for r in full_predicted_pathogenic.reset_index().itertuples(index=False)])

    subsection("Saving the full-training-set predicted-pathogenic table")
    _save_table(
        full_predicted_pathogenic,
        OUTPUT_DIR / "Supplementary_Table_PredictedPathogenic_FullTrainingSet.csv",
        OUTPUT_DIR / "Supplementary_Table_PredictedPathogenic_FullTrainingSet.png",
        "Supplementary Table -- Predicted-pathogenic counts, all models vs. the full training set",
        figsize=(9.5, 2.2), fontsize=9.5,
        cell_fmt=lambda idx, col: (
            f"{full_predicted_pathogenic.loc[idx, col]:.1f}%" if col == "% predicted pathogenic"
            else f"{full_predicted_pathogenic.loc[idx, col]:,}"
        ),
    )

    # ================================================================ Step 10
    step(10, "LR (likelihood-ratio) targets for each ACMG/AMP evidence level")
    info("calibration/{name}_thresholds.csv (written by Part 1's calibration step) already\n"
         "stores 'lr_target' alongside each fitted 'score_threshold' -- the target LR that\n"
         "compute_lr_thresholds() derived from the prior used for that run (Pejaver et al.\n"
         "2022 / Tavtigian et al. 2018 framework). LR targets are a function of the prior\n"
         "only, not of any model's own scores, so they are identical across GEMVAP_1/2/3;\n"
         "this table lists them once, next to each model's fitted score threshold that\n"
         "achieves them.")

    action("Reloading cached LR targets from calibration/{name}_thresholds.csv")
    thr_by_model = {
        name: pd.read_csv(OUTPUT_DIR / "calibration" / f"{name}_thresholds.csv").set_index("evidence_level")
        for name in MODEL_NAMES
    }
    evidence_order = list(thr_by_model["GEMVAP_1"].index)
    lr_reference = thr_by_model["GEMVAP_1"]["lr_target"]

    for name in MODEL_NAMES[1:]:
        mismatch = (thr_by_model[name]["lr_target"] - lr_reference).abs().max()
        if mismatch > 1e-6:
            warning(f"{name}'s lr_target differs from GEMVAP_1's by up to {mismatch:.2e} "
                    "-- these models were calibrated against different priors.")

    lr_targets = pd.DataFrame({"LR target": lr_reference})
    for name in MODEL_NAMES:
        lr_targets[f"{name}_score_threshold"] = thr_by_model[name]["score_threshold"]
    lr_targets.index.name = "Evidence level"
    lr_targets = lr_targets.loc[evidence_order]

    summary_table(
        "LR targets per ACMG/AMP evidence level",
        ["Evidence level", "LR target"] + [f"{name} threshold" for name in MODEL_NAMES],
        [
            (
                level,
                f"{lr_targets.loc[level, 'LR target']:.4f}",
                *[
                    "--" if pd.isna(lr_targets.loc[level, f"{name}_score_threshold"])
                    else f"{lr_targets.loc[level, f'{name}_score_threshold']:.2f}"
                    for name in MODEL_NAMES
                ],
            )
            for level in evidence_order
        ],
    )

    subsection("Saving the LR targets table")
    _save_table(
        lr_targets,
        OUTPUT_DIR / "Supplementary_Table_LRTargets.csv",
        OUTPUT_DIR / "Supplementary_Table_LRTargets.png",
        "Supplementary Table -- LR targets per ACMG/AMP evidence level (Pejaver et al. 2022)",
        figsize=(7.5, 2.6), fontsize=10,
        cell_fmt=lambda idx, col: (
            f"{lr_targets.loc[idx, col]:.4f}" if col == "LR target"
            else ("--" if pd.isna(lr_targets.loc[idx, col]) else f"{lr_targets.loc[idx, col]:.2f}")
        ),
    )

    # ================================================================ Step 11
    step(11, "Test dataset composition: cysteine / in-domain conserved / others")
    info("Partitioning the Part 1 test set (df_test, already loaded in Step 6) into the\n"
         "same three mutually-exclusive structural classes used for GEMVAP 1/2/3 training:\n"
         "Cysteine (is_cys), In-domain conserved (non-cysteine, inside a conserved FBN1\n"
         "EGF-Ca2+-binding domain site per the conserved-domain annotation table), and\n"
         "Others (non-cysteine, outside conserved sites -- the GEMVAP 3 subset) -- split\n"
         "by pathogenic (is_case) vs. control (is_ctrl).")

    from gemvap_pipeline.data import conserved_domain_mask

    action(f"Reading {cfg['conserved_data_path']}")
    gemvap3_mask = conserved_domain_mask(df_test, cfg["conserved_data_path"])
    indom_conserved_mask = ~df_test.is_cys & ~gemvap3_mask

    def test_composition_column(mask: pd.Series, label: str) -> pd.DataFrame:
        n_cys = int((mask & df_test.is_cys).sum())
        n_indom = int((mask & indom_conserved_mask).sum())
        n_other = int((mask & gemvap3_mask).sum())
        n_total = int(mask.sum())
        rows = [
            ("Cysteine", n_cys), ("In-domain conserved", n_indom),
            ("Others", n_other), ("Total", n_total),
        ]
        return pd.DataFrame(rows, columns=["Category", label]).set_index("Category")

    test_composition = test_composition_column(df_test.is_case, "Pathogenic").join(
        test_composition_column(df_test.is_ctrl, "Control"))

    action("Counting test-set variants by structural class -- pathogenic vs. control")
    summary_table("Test Dataset Composition", ["Category", "Pathogenic", "Control"],
                  [(idx, int(row["Pathogenic"]), int(row["Control"])) for idx, row in test_composition.iterrows()])

    subsection("Saving the test dataset composition table")
    _save_table(
        test_composition,
        OUTPUT_DIR / "Supplementary_Table_TestComposition.csv",
        OUTPUT_DIR / "Supplementary_Table_TestComposition.png",
        "Supplementary Table -- Test Dataset Composition (Cysteine / In-domain conserved / Others)",
        figsize=(6, 2.4), fontsize=11,
        cell_fmt=lambda idx, col: f"{test_composition.loc[idx, col]:,}",
        extra_style=lambda row, col, cell, df: cell.set_text_props(weight="bold") if row == len(df) else None,
    )

    # ============================================================= Figures dir
    if cli_args.figures_dir:
        subsection("Collecting all generated images into one folder")
        figures_dir = Path(cli_args.figures_dir)
        action(f"Copying every Supplementary_Table_*.png under {OUTPUT_DIR} into {figures_dir}")
        figures_dir.mkdir(parents=True, exist_ok=True)
        n_copied = 0
        for src in sorted(OUTPUT_DIR.glob("Supplementary_Table_*.png")):
            shutil.copy2(src, figures_dir / src.name)
            n_copied += 1
        result(f"Copied {n_copied} PNG(s) to {figures_dir.resolve()}")

    subsection("Done")
    info(f"{OUTPUT_DIR} now contains all 8 Supplementary_Table_*.csv/.png files\n"
         "plus the compact CSV-only de novo pathogenic variant table.")


if __name__ == "__main__":
    main()
