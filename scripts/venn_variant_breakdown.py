#!/usr/bin/env python3
"""
Detailed Pathogenic/Ambiguous/Benign breakdown behind venn_all_missense_vs_cohort.png,
plus two supporting variant lists on the "pathogenic cases" cohort:

  1. Summary counts table: AlphaMissense, REVEL, GEMVAP 1/2/3, G1 AND G2 AND G3,
     G1 OR G2 OR G3 -- each split Pathogenic / Ambiguous / Benign -- across two
     populations: all FBN1 missense variants, and the curated pathogenic-cases
     cohort (the 1346-variant Excel file minus the one manually-flagged variant
     it contains, c.1416C>G -- the same exclusion rule applied to the
     all-missense population -- leaving 1345).

     GEMVAP's Ambiguous bucket uses each model's own calibrated PP3_Supporting /
     BP4_Supporting consensus-score thresholds (score >= PP3_Supporting ->
     Pathogenic, score <= BP4_Supporting -> Benign, otherwise -> Ambiguous --
     the same "indeterminate" zone annotate_variant() in
     gemvap_pejaver_calibration.py falls back to). G1 x G2 x G3 is Pathogenic
     only when all three agree Pathogenic and Benign only when all three agree
     Benign (Ambiguous on any disagreement); G1 + G2 + G3 is Pathogenic if any
     one is Pathogenic and Benign only when all three agree Benign.
  2. Variants in the 1345-case pool called pathogenic by exactly one of
     {AlphaMissense, REVEL, Any GEMVAP}.
  3. Variants in the 1345-case pool called benign by all three.

Requires Part 1 (run_part1_train_calibrate.py) to have been run first.
Run from gemvap_clean_pipeline/:  python scripts/venn_variant_breakdown.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))

from gemvap_pipeline.venn import (
    load_gemvap_fits, load_venn_cohort, load_all_missense_variants,
    pejaver_supporting_thresholds, bp4_supporting_thresholds, pathogenic_set,
    alphamissense_three_way, three_way, consensus_scores,
    build_variant_rows, exclusive_single_predictor_table,
)
from gemvap_pipeline.pejaver_tools import parse_pejaver_thresholds


def counts(series: pd.Series) -> dict:
    vc = series.value_counts()
    return {k: int(vc.get(k, 0)) for k in ("Pathogenic", "Ambiguous", "Benign")}


def combine_and(*labels: pd.Series) -> pd.Series:
    """3-way AND (unanimous-pathogenic) combination: Pathogenic only if every
    input is Pathogenic, Benign only if every input is Benign, Ambiguous
    otherwise (disagreement, or any Ambiguous involved)."""
    df = pd.concat(labels, axis=1)
    out = pd.Series("Ambiguous", index=df.index, dtype=object)
    out[(df == "Pathogenic").all(axis=1)] = "Pathogenic"
    out[(df == "Benign").all(axis=1)] = "Benign"
    return out


def combine_or(*labels: pd.Series) -> pd.Series:
    """3-way OR (any-pathogenic) combination: Pathogenic if any input is
    Pathogenic, Benign only if every input is Benign, Ambiguous otherwise
    (no Pathogenic call, but not unanimous Benign either)."""
    df = pd.concat(labels, axis=1)
    out = pd.Series("Ambiguous", index=df.index, dtype=object)
    out[(df == "Benign").all(axis=1)] = "Benign"
    out[(df == "Pathogenic").any(axis=1)] = "Pathogenic"
    return out


def main():
    cfg = yaml.safe_load(open("config_gnomad4.yaml"))
    DATA_PATH = cfg["data_path"]
    PEJAVER_THRESHOLDS = cfg["pejaver_thresholds"]
    OUTPUT_DIR = Path(cfg["output_dir"])
    ALPHAMISSENSE_PATH = Path("data/raw/AlphaMissense_hg38_FBN1.tsv")
    EXCEL_PATH = Path("../data/raw/20250609 FELIX 1346 AM_REVEL_GEMVAP_Analysis_20250609_full_outcomes.xlsx")

    VENN_DIR = OUTPUT_DIR / "venn"
    VENN_DIR.mkdir(parents=True, exist_ok=True)

    revel_thr = parse_pejaver_thresholds(PEJAVER_THRESHOLDS)["REVEL"]
    revel_pp3, revel_bp4 = revel_thr["PP3_Supporting"], revel_thr["BP4_Supporting"]

    fits = load_gemvap_fits(OUTPUT_DIR)
    pej_thr = pejaver_supporting_thresholds(fits, OUTPUT_DIR / "calibration")
    bp4_thr = bp4_supporting_thresholds(fits, OUTPUT_DIR / "calibration")

    # ---------------------------------------------------------- Population A
    print("\n=== Population A: all FBN1 missense variants ===")
    missense_all, set_am_all, set_revel_all, n_all = load_all_missense_variants(
        DATA_PATH, ALPHAMISSENSE_PATH, PEJAVER_THRESHOLDS,
    )
    excl_mask_a = missense_all["cDNA"].astype(str).str.match(r"c\.141[456]([^0-9]|$)")
    missense_all = missense_all.drop(index=missense_all.index[excl_mask_a])
    n_all = len(missense_all)
    print(f"n = {n_all} (after excluding c.1414/1415/1416)")

    am_a = alphamissense_three_way(pd.to_numeric(missense_all["pathogenicity score"], errors="coerce"))
    revel_scores_a = pd.to_numeric(missense_all["REVEL_score"], errors="coerce")
    revel_a = three_way(revel_scores_a, revel_pp3, revel_bp4)

    g1_a = three_way(consensus_scores(missense_all, fits["GEMVAP_1"]), pej_thr["GEMVAP_1"], bp4_thr["GEMVAP_1"])
    g2_a = three_way(consensus_scores(missense_all, fits["GEMVAP_2"]), pej_thr["GEMVAP_2"], bp4_thr["GEMVAP_2"])
    g3_a = three_way(consensus_scores(missense_all, fits["GEMVAP_3"]), pej_thr["GEMVAP_3"], bp4_thr["GEMVAP_3"])
    and_a = combine_and(g1_a, g2_a, g3_a)
    or_a = combine_or(g1_a, g2_a, g3_a)

    rows_a = {
        "AlphaMissense": counts(am_a),
        "REVEL": counts(revel_a),
        "GEMVAP 1 (G1)": counts(g1_a),
        "GEMVAP 2 (G2)": counts(g2_a),
        "GEMVAP 3 (G3)": counts(g3_a),
        "G1 x G2 x G3": counts(and_a),
        "G1 + G2 + G3": counts(or_a),
    }

    # ---------------------------------------------------------- Population B
    print("\n=== Population B: pathogenic-cases cohort ===")
    merged, set_am, set_revel, n_cohort = load_venn_cohort(EXCEL_PATH, DATA_PATH, fits, ALPHAMISSENSE_PATH)

    # Bring in the continuous REVEL score (not necessarily a top_predictor of
    # every model) -- AlphaMissense's raw score is already joined in on
    # merged by load_venn_cohort (same source it derives set_am from).
    tsv = pd.read_csv(DATA_PATH, sep="\t", low_memory=False, na_values=["."])
    tsv.columns = tsv.columns.str.lstrip("#").str.strip()
    tsv["REVEL_score"] = pd.to_numeric(tsv["REVEL_score"], errors="coerce")
    tsv_dd = tsv.drop_duplicates(subset="cDNA")
    merged = merged.merge(tsv_dd[["cDNA", "REVEL_score"]], on="cDNA", how="left", suffixes=("", "_full"))

    excl_mask_b = merged["cDNA"].astype(str).str.match(r"c\.141[456]([^0-9]|$)")
    excl_idx_b = set(merged.index[excl_mask_b])
    print(f"Excluding {len(excl_idx_b)} flagged variant(s) from the cohort: "
          f"{sorted(merged.loc[list(excl_idx_b), 'cDNA'].tolist())}")
    pool = merged.drop(index=excl_idx_b)
    n_pool = len(pool)
    print(f"n = {n_pool} (1346-variant cohort minus flagged variant(s))")

    set_am_b = set_am - excl_idx_b
    set_revel_b = set_revel - excl_idx_b

    am_b = alphamissense_three_way(pd.to_numeric(pool["pathogenicity score"], errors="coerce"))
    revel_scores_b = pd.to_numeric(pool["REVEL_score"], errors="coerce")
    revel_b = three_way(revel_scores_b, revel_pp3, revel_bp4)

    # Binary pathogenic-call sets (PP3_Supporting only) -- used below for the
    # exclusive-single-predictor / missed-by-all-three variant lists, which
    # stay in the same Pathogenic-vs-not framework as AM_SCORE/REVEL_Prediction.
    g1_b = pathogenic_set(pool, fits["GEMVAP_1"], pej_thr["GEMVAP_1"])
    g2_b = pathogenic_set(pool, fits["GEMVAP_2"], pej_thr["GEMVAP_2"])
    g3_b = pathogenic_set(pool, fits["GEMVAP_3"], pej_thr["GEMVAP_3"])

    # 3-way (Pathogenic/Ambiguous/Benign) labels -- used for the summary table.
    g1_b_tw = three_way(consensus_scores(pool, fits["GEMVAP_1"]), pej_thr["GEMVAP_1"], bp4_thr["GEMVAP_1"])
    g2_b_tw = three_way(consensus_scores(pool, fits["GEMVAP_2"]), pej_thr["GEMVAP_2"], bp4_thr["GEMVAP_2"])
    g3_b_tw = three_way(consensus_scores(pool, fits["GEMVAP_3"]), pej_thr["GEMVAP_3"], bp4_thr["GEMVAP_3"])
    and_b_tw = combine_and(g1_b_tw, g2_b_tw, g3_b_tw)
    or_b_tw = combine_or(g1_b_tw, g2_b_tw, g3_b_tw)

    rows_b = {
        "AlphaMissense": counts(am_b),
        "REVEL": counts(revel_b),
        "GEMVAP 1 (G1)": counts(g1_b_tw),
        "GEMVAP 2 (G2)": counts(g2_b_tw),
        "GEMVAP 3 (G3)": counts(g3_b_tw),
        "G1 x G2 x G3": counts(and_b_tw),
        "G1 + G2 + G3": counts(or_b_tw),
    }

    # -------------------------------------------------------------- Table
    print(f"\n{'Predictor':<16}"
          f"{'A-Path':>10}{'A-Amb':>10}{'A-Ben':>10}   "
          f"{'B-Path':>10}{'B-Amb':>10}{'B-Ben':>10}")
    for name in rows_a:
        a, b = rows_a[name], rows_b[name]
        fa = lambda v: "NA" if v is None else str(v)
        print(f"{name:<16}"
              f"{fa(a['Pathogenic']):>10}{fa(a['Ambiguous']):>10}{fa(a['Benign']):>10}   "
              f"{fa(b['Pathogenic']):>10}{fa(b['Ambiguous']):>10}{fa(b['Benign']):>10}")

    summary_df = pd.DataFrame({
        "Predictor": list(rows_a.keys()),
        "A_Pathogenic": [rows_a[k]["Pathogenic"] for k in rows_a],
        "A_Ambiguous": [rows_a[k]["Ambiguous"] for k in rows_a],
        "A_Benign": [rows_a[k]["Benign"] for k in rows_a],
        "B_Pathogenic": [rows_b[k]["Pathogenic"] for k in rows_b],
        "B_Ambiguous": [rows_b[k]["Ambiguous"] for k in rows_b],
        "B_Benign": [rows_b[k]["Benign"] for k in rows_b],
    })
    summary_path = VENN_DIR / "venn_missense_vs_cohort_breakdown.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path}")

    # ---------------------------------------------- Exclusively-one-predictor
    any_gemvap_b = g1_b | g2_b | g3_b
    caller_sets = {"AlphaMissense": set_am_b, "REVEL": set_revel_b, "GEMVAP": any_gemvap_b}

    excl_df = exclusive_single_predictor_table(pool, caller_sets)
    excl_path = VENN_DIR / "venn_exclusive_single_predictor.csv"
    excl_df.to_csv(excl_path, index=False)
    print(f"\n{len(excl_df)} variant(s) called pathogenic by exactly one predictor -> {excl_path}")
    print(excl_df["Pathogenic by"].value_counts().to_string())

    # ------------------------------------------------- Missed-by-all-three
    missed_idx = [i for i in pool.index
                  if i not in set_am_b and i not in set_revel_b and i not in any_gemvap_b]
    missed_df = build_variant_rows(pool, missed_idx).sort_values("cDNA")
    missed_path = VENN_DIR / "venn_missed_by_all_three.csv"
    missed_df.to_csv(missed_path, index=False)
    print(f"\n{len(missed_df)} variant(s) called benign by all three (AlphaMissense, REVEL, GEMVAP) -> {missed_path}")


if __name__ == "__main__":
    main()
