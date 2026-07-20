"""
Venn diagram (Any GEMVAP vs AlphaMissense vs REVEL, predicted-PATHOGENIC
calls, Pejaver PP3_Supporting thresholds) on the GEMVAP pipeline's own
held-out TEST SET (output/gemvap_notebook_gnomad4/test_dataset.csv, 132
variants: 66 pathogenic-source-DB cases, 66 gnomAD controls) -- one panel
per ground-truth label, side by side, so the left panel reads as each
predictor's sensitivity (recall) and the right panel as its false-positive
rate on this held-out set.

Reuses the same helpers as run_part3_venn.py (gemvap_pipeline.venn) so the
pathogenic-call convention (score thresholding, not a pre-baked label) is
identical to the rest of the pipeline.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from gemvap_pipeline.data import load_cached_datasets, load_denovo_overrides
from gemvap_pipeline.venn import (
    load_gemvap_fits, pejaver_supporting_thresholds, pathogenic_set,
    draw_missense_vs_cohort_venn, draw_any_gemvap_venn, AM_PATHOGENIC_THRESHOLD,
)
from gemvap_pipeline.pejaver_tools import parse_pejaver_thresholds

OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHAMISSENSE_PATH = HERE / "data" / "raw" / "AlphaMissense-Search-P35555.tsv"


def main():
    cfg = yaml.safe_load(open(HERE / "config_gnomad4.yaml"))
    data_path = cfg["data_path"]
    seed = int(cfg.get("seed", 42))
    output_dir = Path(cfg["output_dir"])
    pejaver_thresholds = cfg["pejaver_thresholds"]

    denovo_overrides = load_denovo_overrides(output_dir)
    _, df_test = load_cached_datasets(output_dir, data_path, seed=seed, pathogenic_overrides=denovo_overrides)
    data = df_test.data.copy()
    n_case, n_ctrl = int(df_test.is_case.sum()), int(df_test.is_ctrl.sum())
    print(f"Test set: {len(data)} variants -- {n_case} pathogenic-source-DB cases, {n_ctrl} gnomAD controls")

    am = pd.read_csv(ALPHAMISSENSE_PATH, sep="\t")
    am_lookup = dict(zip(am["protein variant"], am["pathogenicity score"]))
    data["pathogenicity score"] = data["HGVSp_VEP"].map(am_lookup)
    n_am_missing = data["pathogenicity score"].isna().sum()
    if n_am_missing:
        print(f"WARNING: {n_am_missing} test-set variants had no AlphaMissense match by protein change")

    fits = load_gemvap_fits(output_dir)
    pej_thr = pejaver_supporting_thresholds(fits, output_dir / "calibration")
    revel_pp3 = parse_pejaver_thresholds(pejaver_thresholds)["REVEL"]["PP3_Supporting"]

    am_scores = pd.to_numeric(data["pathogenicity score"], errors="coerce")
    set_am = set(data.index[am_scores > AM_PATHOGENIC_THRESHOLD])
    revel_scores = pd.to_numeric(data["REVEL_score"], errors="coerce")
    set_revel = set(data.index[revel_scores >= revel_pp3])
    ps = {name: pathogenic_set(data, fits[name], pej_thr[name]) for name in fits}
    any_gemvap = ps.get("GEMVAP_1", set()) | ps.get("GEMVAP_2", set()) | ps.get("GEMVAP_3", set())

    idx_case = set(data.index[df_test.is_case.values])
    idx_ctrl = set(data.index[df_test.is_ctrl.values])

    panel_case = (any_gemvap & idx_case, set_am & idx_case, set_revel & idx_case, n_case)
    panel_ctrl = (any_gemvap & idx_ctrl, set_am & idx_ctrl, set_revel & idx_ctrl, n_ctrl)

    print(f"Pathogenic cases called pathogenic -- GEMVAP: {len(panel_case[0])}/{n_case}, "
          f"AlphaMissense: {len(panel_case[1])}/{n_case}, REVEL: {len(panel_case[2])}/{n_case}")
    print(f"Controls called pathogenic (false positives) -- GEMVAP: {len(panel_ctrl[0])}/{n_ctrl}, "
          f"AlphaMissense: {len(panel_ctrl[1])}/{n_ctrl}, REVEL: {len(panel_ctrl[2])}/{n_ctrl}")

    out_path = OUT_DIR / "venn_test_set_case_vs_control.png"
    draw_missense_vs_cohort_venn(
        panel_case, panel_ctrl, out_path,
        label_a="Test-set pathogenic cases (sensitivity)",
        label_b="Test-set gnomAD controls (false-positive rate)",
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
