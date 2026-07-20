"""
Comparison of GEMVAP against the individual predictors tabulated in
Pejaver et al. (2022), using that paper's published PP3/BP4 thresholds.
"""
from pathlib import Path

import pandas as pd

from gemvap_pejaver_calibration import annotate_variant

from .verbose import info, result, step
from .metrics import metrics_from_classification_counts

# AlphaMissense score thresholds (Cheng et al. 2023): >0.5642 = Pathogenic,
# <0.34 = Benign, the closed interval in between = VUS. AlphaMissense predates
# Pejaver et al. (2022) so it has no entry in PEJAVER_COLUMN_MAP / pejaver
# thresholds.csv -- it's scored separately below, against its own thresholds.
AM_PATHOGENIC_THRESHOLD = 0.5642
AM_BENIGN_THRESHOLD = 0.34

# Maps Pejaver CSV tool names → (dataset column, optional score transform).
# Transforms ensure scores are in the same direction (higher = more pathogenic)
# as the published PP3 thresholds before comparison.
PEJAVER_COLUMN_MAP = {
    "REVEL":                   ("REVEL_score",                  None),
    "VEST4":                   ("VEST4_score",                  None),
    "MutPred2":                ("MutPred_score",                None),
    "SIFT_converted":          ("SIFT_score",                   lambda x: 1.0 - x),
    "FATHMM_converted":        ("FATHMM_converted_rankscore",   None),
    "CADD_raw":                ("CADD_phred",                   None),
    "MPC":                     ("MPC_score",                    None),
    "PrimateAI":               ("PrimateAI_score",              None),
    # Polyphen2_HVAR and Polyphen2_HDIV are all-NaN in this dataset — omitted
    "GERP++_RS":               ("GERP++_RS",                    None),
    "phyloP100way_vertebrate": ("phyloP100way_vertebrate",      None),
}


def parse_pejaver_thresholds(csv_path):
    """
    Parse pejaver2022_thresholds.csv into per-tool threshold dicts compatible
    with annotate_variant(). Each dict maps evidence level → numeric threshold
    (or None when the level is not defined for that tool).
    """
    df = pd.read_csv(csv_path)
    level_cols = [
        "PP3_Supporting", "PP3_Moderate", "PP3_Strong", "PP3_VeryStrong",
        "BP4_Supporting", "BP4_Moderate", "BP4_Strong", "BP4_VeryStrong",
    ]
    result = {}
    for _, row in df.iterrows():
        thresholds = {}
        for level in level_cols:
            cell = str(row[level]).strip()
            if cell in ("—", "nan", ""):
                thresholds[level] = None
            else:
                thresholds[level] = float(cell.lstrip("≥≤").replace("−", "-"))
        result[row["Tool"]] = thresholds
    return result


def compute_individual_tool_metrics(df_perf, perf_true_labels, pejaver_csv_path, strict: bool = False):
    """
    For each individual predictor in pejaver2022_thresholds.csv, classify
    performance test variants using the published thresholds and compute
    F1 (pathogenic class), Accuracy, MCC, and VUS rate.

    strict=False (default): Indeterminate variants are excluded from
        F1/Accuracy/MCC and counted in VUS rate only.
    strict=True: truly pathogenic Indeterminate variants count as FN.
        Truly benign Indeterminate variants are still excluded (VUS).
    """
    info("Scoring individual predictors using Pejaver et al. (2022) published thresholds.")
    info("Each predictor classifies variants via PP3/BP4 evidence codes; VUS = indeterminate.")
    pejaver_thresholds = parse_pejaver_thresholds(pejaver_csv_path)
    records = []
    n_total = len(perf_true_labels)

    for tool_name, (col, transform) in PEJAVER_COLUMN_MAP.items():
        tool_thresholds = pejaver_thresholds.get(tool_name)
        if tool_thresholds is None:
            continue

        scores = df_perf[col].values if col in df_perf.columns else None
        if scores is None:
            continue

        tp = tn = fp = fn = n_vus = 0
        for score_raw, true_label in zip(scores, perf_true_labels):
            if pd.isna(score_raw):
                if strict and true_label == "pathogenic":
                    fn += 1
                else:
                    n_vus += 1
                continue
            # dbNSFP may store multiple transcript scores separated by ";"
            # take the max (most pathogenic) after applying any transform
            try:
                parts = [float(v) for v in str(score_raw).split(";") if v.strip() not in ("", ".")]
            except ValueError:
                if strict and true_label == "pathogenic":
                    fn += 1
                else:
                    n_vus += 1
                continue
            if not parts:
                if strict and true_label == "pathogenic":
                    fn += 1
                else:
                    n_vus += 1
                continue
            score = max(transform(v) for v in parts) if transform else max(parts)
            evidence = annotate_variant(score, tool_thresholds)
            if evidence == "Indeterminate":
                if strict and true_label == "pathogenic":
                    fn += 1
                else:
                    n_vus += 1
                continue
            predicted = "pathogenic" if evidence.startswith("PP3") else "benign"
            if predicted == "pathogenic" and true_label == "pathogenic":
                tp += 1
            elif predicted == "benign" and true_label == "benign":
                tn += 1
            elif predicted == "pathogenic" and true_label == "benign":
                fp += 1
            else:
                fn += 1

        m = metrics_from_classification_counts(tp, tn, fp, fn, n_vus, n_total)
        mcc_str = str(m["mcc"]) if not isinstance(m["mcc"], float) else f"{m['mcc']:.3f}"
        step(f"{tool_name:<28}  F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}  "
             f"MCC={mcc_str}  VUS={m['vus_rate']:.3f}  "
             f"(classified {m['n_classified']}/{n_total})")
        records.append({"model": tool_name, **m})

    return pd.DataFrame(records)


def compute_alphamissense_metrics(df_perf, perf_true_labels, alphamissense_path, strict: bool = False):
    """
    Score AlphaMissense (Cheng et al. 2023) on the same performance-test rows
    used by compute_individual_tool_metrics, joining its genome-wide hg38
    FBN1-region extract onto df_perf by (chr, pos, ref, alt) -- the same join
    venn.load_alphamissense_hg38 uses -- and classifying with its own
    published thresholds (score > 0.5642 = Pathogenic, < 0.34 = Benign, else
    VUS), since it isn't one of the tools calibrated in pejaver_thresholds.csv.

    Returns a single-row DataFrame in the same shape as
    compute_individual_tool_metrics's output (model="AlphaMissense"), or an
    empty DataFrame if alphamissense_path is unset / doesn't exist.
    """
    if not alphamissense_path or not Path(alphamissense_path).exists():
        return pd.DataFrame()

    info("Scoring AlphaMissense using its own published thresholds (Cheng et al. 2023).")
    am = pd.read_csv(alphamissense_path, sep="\t")
    am.columns = am.columns.str.lstrip("#").str.strip()
    am["CHROM"] = pd.to_numeric(am["CHROM"].str.replace("chr", "", regex=False), errors="coerce")
    am = am.rename(columns={"am_pathogenicity": "pathogenicity score"})
    am = am[["CHROM", "POS", "REF", "ALT", "pathogenicity score"]]

    merged = df_perf.merge(
        am, how="left",
        left_on=["#chr", "pos(1-based)", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"],
    )
    scores = pd.to_numeric(merged["pathogenicity score"], errors="coerce").values

    tp = tn = fp = fn = n_vus = 0
    n_total = len(perf_true_labels)
    for score, true_label in zip(scores, perf_true_labels):
        if pd.isna(score):
            predicted = None
        elif score > AM_PATHOGENIC_THRESHOLD:
            predicted = "pathogenic"
        elif score < AM_BENIGN_THRESHOLD:
            predicted = "benign"
        else:
            predicted = None

        if predicted is None:
            if strict and true_label == "pathogenic":
                fn += 1
            else:
                n_vus += 1
            continue
        if predicted == "pathogenic" and true_label == "pathogenic":
            tp += 1
        elif predicted == "benign" and true_label == "benign":
            tn += 1
        elif predicted == "pathogenic" and true_label == "benign":
            fp += 1
        else:
            fn += 1

    m = metrics_from_classification_counts(tp, tn, fp, fn, n_vus, n_total)
    mcc_str = str(m["mcc"]) if not isinstance(m["mcc"], float) else f"{m['mcc']:.3f}"
    step(f"{'AlphaMissense':<28}  F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}  "
         f"MCC={mcc_str}  VUS={m['vus_rate']:.3f}  "
         f"(classified {m['n_classified']}/{n_total}, "
         f"{int(pd.notna(scores).sum())}/{n_total} matched)")
    return pd.DataFrame([{"model": "AlphaMissense", **m}])
