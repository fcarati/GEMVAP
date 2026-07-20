import math
from typing import Dict, Tuple

from gemvap_pejaver_calibration import annotate_variant


def matthews_correlation_coefficient(TP: int, TN: int, FP: int, FN: int) -> float:
    numerator = (TP * TN) - (FP * FN)
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_f1_score(TP: int, FP: int, FN: int) -> float:
    if TP + FP == 0 or TP + FN == 0:
        return 0.0
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_accuracy(TP: int, TN: int, FP: int, FN: int) -> float:
    total = TP + TN + FP + FN
    return 0.0 if total == 0 else (TP + TN) / total


def confusion_metrics_from_counts(path, contr, threshold: int) -> Dict[str, float]:
    TP = TN = FP = FN = 0
    for idx, a in enumerate(contr):
        if idx > threshold:
            FP += a
        else:
            TN += a
    for idx, a in enumerate(path):
        if idx <= threshold:
            FN += a
        else:
            TP += a
    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": calculate_accuracy(TP, TN, FP, FN),
        "f1_score": calculate_f1_score(TP, FP, FN),
        "mcc": matthews_correlation_coefficient(TP, TN, FP, FN),
    }


def metrics_from_classification_counts(
    tp: int, tn: int, fp: int, fn: int, n_vus: int, n_total: int
) -> Dict:
    """
    F1 (pathogenic class), Accuracy, MCC and VUS rate from raw confusion counts.
    Shared by compute_calibrated_metrics and pejaver_tools.compute_individual_tool_metrics
    so the two evidence-based metric computations stay numerically consistent.
    """
    def _safe(num, denom):
        return round(num / denom, 4) if denom else float("nan")

    n_classified = tp + tn + fp + fn
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "f1": _safe(2 * tp, 2 * tp + fp + fn),
        "accuracy": _safe(tp + tn, n_classified),
        "mcc": round((tp * tn - fp * fn) / mcc_denom, 4) if mcc_denom else float("nan"),
        "vus_rate": _safe(n_vus, n_total),
        "n_classified": n_classified,
        "n_vus_excluded": n_vus,
        "n_total_perf": n_total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def compute_calibrated_metrics(
    perf_scores,
    perf_true_labels,
    thresholds,
    model_name: str,
    strict: bool = False,
) -> Dict:
    """
    Classify performance test variants with calibrated thresholds and compute
    F1 (pathogenic class), Accuracy, MCC, and VUS rate.

    Standard mode (strict=False):
        Indeterminate variants are excluded from F1/Accuracy/MCC but counted in
        VUS rate.  This is the optimistic interpretation: VUS are simply
        unresolved, not wrong.

    Strict mode (strict=True):
        A truly pathogenic variant classified as Indeterminate is counted as a
        false negative (FN).  The model failed to provide actionable evidence for
        a pathogenic variant, which is a clinically meaningful error.
        Truly benign Indeterminate variants are still excluded (VUS rate counts
        only those benign-VUS).

    PP3_* = pathogenic evidence; BP4_* = benign evidence.
    """
    tp = tn = fp = fn = n_vus = 0

    for score, true in zip(perf_scores, perf_true_labels):
        evidence = annotate_variant(float(score), thresholds)
        if evidence == "Indeterminate":
            if strict and true == "pathogenic":
                fn += 1   # missed pathogenic variant counts as false negative
            else:
                n_vus += 1
            continue
        predicted = "pathogenic" if evidence.startswith("PP3") else "benign"
        if predicted == "pathogenic" and true == "pathogenic":
            tp += 1
        elif predicted == "benign" and true == "benign":
            tn += 1
        elif predicted == "pathogenic" and true == "benign":
            fp += 1
        else:
            fn += 1

    return {
        "model": model_name,
        **metrics_from_classification_counts(tp, tn, fp, fn, n_vus, len(perf_true_labels)),
    }
