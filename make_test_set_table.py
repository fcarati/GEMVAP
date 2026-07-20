"""
Build a table figure (REVEL / AlphaMissense / GEMVAP 1-3, + concordance
columns) for the GEMVAP pipeline's own held-out TEST SET
(output/gemvap_notebook_gnomad4/test_dataset.csv, 132 variants), following
the same visual style as make_cysteine_loss_table.py's tables for the
expert_missense.xlsx cohort.

Unlike the expert_missense.xlsx tables, this dataset has no hand-curated
"EXPERT CLASSIFICATION" or pre-scored REVEL/AlphaMissense P columns, so:
  - REVEL tier is derived from the raw REVEL_score using the published
    Pejaver et al. (2022) thresholds (pathogenic >=0.644, benign <=0.290,
    else ambiguous) -- data/raw/pejaver2022_thresholds.csv.
  - AlphaMissense tier comes from data/raw/AlphaMissense-Search-P35555.tsv (a
    saturation-mutagenesis AlphaMissense+REVEL call file for every possible
    FBN1 protein variant), matched by protein change (HGVSp_VEP).
  - GEMVAP 1/2/3 are scored fresh from the currently fitted models, exactly
    like score_expert_missense.py did for the expert cohort.
  - The "ground truth" for the concordance/match columns is the pipeline's
    own is_case (pathogenic source DB) / is_ctrl (gnomAD control) label --
    what GEMVAP itself was evaluated against -- not an expert curation.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gemvap_pipeline.data import load_cached_datasets, load_denovo_overrides
from gemvap_pipeline.model import load_fit_result, apply_consensus_score
from gemvap_pejaver_calibration import annotate_variant

from make_cysteine_loss_table import TIER_COLOR, concordance_color

OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHAMISSENSE_PATH = HERE / "data" / "raw" / "AlphaMissense-Search-P35555.tsv"
OUTPUT_DIR = HERE / "output" / "gemvap_notebook_gnomad4"
CALIB_DIR = OUTPUT_DIR / "calibration"
MODELS = ["GEMVAP_1", "GEMVAP_2", "GEMVAP_3"]

REVEL_PATHOGENIC_THRESHOLD = 0.644
REVEL_BENIGN_THRESHOLD = 0.290

AM_TIER = {
    "likely_pathogenic": "pathogenic",
    "likely_benign": "benign",
    "ambiguous": "ambiguous",
}


def revel_tier(score) -> str | None:
    if pd.isna(score):
        return None
    if score >= REVEL_PATHOGENIC_THRESHOLD:
        return "pathogenic"
    if score <= REVEL_BENIGN_THRESHOLD:
        return "benign"
    return "ambiguous"


def evidence_to_tier(evidence: str) -> str:
    if evidence.startswith("PP3"):
        return "pathogenic"
    if evidence.startswith("BP4"):
        return "benign"
    return "ambiguous"


def load_thresholds(model_name: str) -> dict:
    thr = pd.read_csv(CALIB_DIR / f"{model_name}_thresholds.csv")
    return dict(zip(thr["evidence_level"], thr["score_threshold"]))


def main():
    cfg = yaml.safe_load(open(HERE / "config_gnomad4.yaml"))
    data_path, seed, output_dir = cfg["data_path"], int(cfg["seed"]), Path(cfg["output_dir"])

    denovo_overrides = load_denovo_overrides(output_dir)
    _, df_test = load_cached_datasets(output_dir, data_path, seed=seed, pathogenic_overrides=denovo_overrides)
    data = df_test.data.copy()
    data["ground_truth"] = None
    data.loc[df_test.is_case.values, "ground_truth"] = "pathogenic"
    data.loc[df_test.is_ctrl.values, "ground_truth"] = "benign"
    n_unlabelled = data["ground_truth"].isna().sum()
    if n_unlabelled:
        print(f"WARNING: {n_unlabelled} test-set variants are neither is_case nor is_ctrl")

    data["REVEL_tier"] = data["REVEL_score"].apply(revel_tier)

    am = pd.read_csv(ALPHAMISSENSE_PATH, sep="\t")
    am_lookup = dict(zip(am["protein variant"], am["pathogenicity class"]))
    data["AM_class_raw"] = data["HGVSp_VEP"].map(am_lookup)
    data["AlphaMissense_tier"] = data["AM_class_raw"].map(AM_TIER)
    n_am_missing = data["AlphaMissense_tier"].isna().sum()
    if n_am_missing:
        print(f"WARNING: {n_am_missing} test-set variants had no AlphaMissense match by protein change")

    for model_name in MODELS:
        fit = load_fit_result(OUTPUT_DIR / f"{model_name}_fit.pkl")
        scores = apply_consensus_score(data, fit["top_predictors"], fit["rbc"]["threshold"]["case"])
        thresholds = load_thresholds(model_name)
        tiers = [
            evidence_to_tier(annotate_variant(float(s), thresholds)) if pd.notna(s) else None
            for s in scores
        ]
        key = model_name.replace("GEMVAP_", "GEMVAP ")
        data[f"{key} class"] = tiers

    for source_col, match_col in [
        ("REVEL_tier", "REVEL match"), ("AlphaMissense_tier", "AlphaMissense match"),
        ("GEMVAP 1 class", "GEMVAP 1 match"), ("GEMVAP 2 class", "GEMVAP 2 match"),
        ("GEMVAP 3 class", "GEMVAP 3 match"),
    ]:
        data[match_col] = [
            concordance_color(pred, gt) for pred, gt in zip(data[source_col], data["ground_truth"])
        ]

    data["hg_38"] = "chr" + data["#chr"].astype(str) + ":" + data["pos(1-based)"].astype(str)
    data["Position"] = pd.to_numeric(data["aapos"], errors="coerce")
    ground_truth_order = {"benign": 0, "pathogenic": 1}
    data["ground_truth_order"] = data["ground_truth"].map(ground_truth_order)
    data = data.sort_values(["ground_truth_order", "Position"]).reset_index(drop=True)

    df = pd.DataFrame({
        "cDNA": data["cDNA"],
        "Protein change": data["HGVSp_VEP"],
        "hg_38": data["hg_38"],
        "Variant classification": data["ground_truth"],
        "REVEL": data["REVEL_tier"],
        "AlphaMissense": data["AlphaMissense_tier"],
        "GEMVAP 1": data["GEMVAP 1 class"],
        "GEMVAP 2": data["GEMVAP 2 class"],
        "GEMVAP 3": data["GEMVAP 3 class"],
    })
    for c in ["REVEL match", "AlphaMissense match", "GEMVAP 1 match", "GEMVAP 2 match", "GEMVAP 3 match"]:
        df[c] = ""
        df[f"{c}__hex"] = data[c]

    csv_path = OUT_DIR / "FBN1_TestSet_Variants.csv"
    display_cols = [
        "cDNA", "Protein change", "hg_38", "Variant classification", "REVEL", "AlphaMissense",
        "GEMVAP 1", "GEMVAP 2", "GEMVAP 3",
        "REVEL match", "AlphaMissense match", "GEMVAP 1 match", "GEMVAP 2 match", "GEMVAP 3 match",
    ]
    tier_cols = ["Variant classification", "REVEL", "AlphaMissense", "GEMVAP 1", "GEMVAP 2", "GEMVAP 3"]
    match_cols = ["REVEL match", "AlphaMissense match", "GEMVAP 1 match", "GEMVAP 2 match", "GEMVAP 3 match"]

    df[display_cols].to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    cell_text = df[display_cols].astype(str).values.tolist()

    fontsize = 8.5
    row_height_in = fontsize / 72 * 1.6
    n_total_rows = len(df) + 1
    table_height_in = row_height_in * n_total_rows
    title_margin_in = 0.55
    footnote_margin_in = 0.5
    fig_width = 12 + 0.55 * len(display_cols)
    fig_height = table_height_in + title_margin_in + footnote_margin_in

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0.01, footnote_margin_in / fig_height, 0.98, table_height_in / fig_height])
    ax.axis("off")

    tbl = ax.table(cellText=cell_text, colLabels=display_cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.auto_set_column_width(col=list(range(len(display_cols))))

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("black")
            continue
        col_name = display_cols[col]
        if col_name in tier_cols:
            raw_value = df.iloc[row - 1][col_name]
            color = TIER_COLOR.get(raw_value)
            if color:
                cell.set_facecolor(color)
        elif col_name in match_cols:
            color = df.iloc[row - 1][f"{col_name}__hex"]
            if isinstance(color, str):
                cell.set_facecolor(color)

    fig.text(
        0.5, 1 - title_margin_in * 0.45 / fig_height,
        f"GEMVAP test-set FBN1 missense variants (n={len(df)})", ha="center", va="center", fontsize=13,
    )
    fig.text(
        0.5, footnote_margin_in * 0.32 / fig_height,
        "Classification columns: orange = pathogenic-tier, blue = benign-tier, grey = ambiguous/uncertain.  "
        "\"match\" columns: green = agrees with the variant classification, yellow = disagrees but one call was ambiguous, "
        "red = disagrees and neither was ambiguous.",
        ha="center", va="center", fontsize=8, style="italic",
    )
    fig.text(
        0.5, footnote_margin_in * 0.68 / fig_height,
        "Variant classification = the GEMVAP pipeline's own pathogenic-source-DB / gnomAD-control test-set label (not an "
        "expert curation). REVEL tier from Pejaver et al. (2022) published thresholds on the raw REVEL score. "
        "AlphaMissense tier from a saturation-mutagenesis AlphaMissense+REVEL call file, matched by protein change.",
        ha="center", va="center", fontsize=8, style="italic",
    )

    png_path = OUT_DIR / "FBN1_TestSet_Variants.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
