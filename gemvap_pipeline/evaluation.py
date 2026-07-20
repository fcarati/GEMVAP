"""
Step 7 cross-check: re-score the held-out test set independently of the
in-memory df_test built in Step 2/6, reloading test_dataset.csv from disk and
re-applying the thresholds already fit in Step 6. This is a sanity check, not
a new calibration -- it should reproduce Step 6's numbers.
"""
from pathlib import Path

import pandas as pd

from .verbose import note, result, step
from .data import load_dataset_csv, load_new_denovo_cdna
from .metrics import compute_calibrated_metrics
from .model import apply_consensus_score, load_fit_result
from .pejaver_tools import compute_individual_tool_metrics

GEMVAP_NAMES = ("GEMVAP_1", "GEMVAP_2", "GEMVAP_3")


def denovo_pathogenic_overrides(new_denovo_path=None, new_denovo_path_2=None, pathogenic_overrides=None) -> set:
    """
    Rebuild the merged de novo cDNA override set used by build_train_test_sets.

    De novo pathogenic status is stamped onto is_case/is_ctrl in memory only
    (see _apply_pathogenic_overrides in data.py) -- it is never written back
    as a column into training_dataset.csv/test_dataset.csv. Any code that
    reloads those CSVs via load_dataset_csv() must re-derive this override
    set and pass it back in, or purely-de-novo pathogenic variants (not
    already flagged in a historical database) silently revert to "not
    pathogenic" after create_filters() recomputes is_case from scratch.
    """
    denovo_cdna = load_new_denovo_cdna(new_denovo_path) if new_denovo_path else set()
    if new_denovo_path_2:
        denovo_cdna |= load_new_denovo_cdna(new_denovo_path_2)
    return (pathogenic_overrides or set()) | denovo_cdna


def _load_thresholds(calib_dir: Path, model_name: str):
    path = calib_dir / f"{model_name}_thresholds.csv"
    if not path.exists():
        return None
    thr = pd.read_csv(path)
    return {
        row["evidence_level"]: None if pd.isna(row["score_threshold"]) else row["score_threshold"]
        for _, row in thr.iterrows()
    }


def evaluate_full_test_set(
    output_dir,
    data_path: str,
    seed: int,
    pejaver_thresholds: str,
    strict_evaluation: bool = False,
    gemvap_names=GEMVAP_NAMES,
    new_denovo_path=None,
    new_denovo_path_2=None,
    pathogenic_overrides=None,
) -> pd.DataFrame:
    """
    Reload test_dataset.csv, re-score each cached GEMVAP fit + calibrated
    threshold, compute individual-predictor metrics via the same Pejaver
    thresholds, and save the combined table to performance_metrics_full_test.csv.

    new_denovo_path/new_denovo_path_2/pathogenic_overrides must match whatever
    was passed to build_train_test_sets in Step 2, so purely-de-novo pathogenic
    variants keep their pathogenic label after the reload (see
    denovo_pathogenic_overrides for why this is necessary).
    """
    output_dir = Path(output_dir)
    calib_dir = output_dir / "calibration"

    step("Reloading test_dataset.csv independently of the in-memory df_test")
    overrides = denovo_pathogenic_overrides(new_denovo_path, new_denovo_path_2, pathogenic_overrides)
    df_test_full = load_dataset_csv(output_dir / "test_dataset.csv", data_path, seed=seed, pathogenic_overrides=overrides)
    result(f"Full test set: {df_test_full.is_case.sum()} pathogenic, {df_test_full.is_ctrl.sum()} control")

    true_labels = pd.Series("benign", index=df_test_full.data.index, dtype=object)
    true_labels[df_test_full.is_case & ~df_test_full.is_ctrl] = "pathogenic"

    step("Re-scoring each GEMVAP model and each individual Pejaver-style predictor")
    gemvap_records = []
    for name in gemvap_names:
        fit_path = output_dir / f"{name}_fit.pkl"
        thresholds = _load_thresholds(calib_dir, name)
        if not fit_path.exists() or thresholds is None:
            note(f"Skipping {name}: missing fit or thresholds.")
            continue
        fit_result = load_fit_result(fit_path)
        scores = apply_consensus_score(
            df_test_full.data,
            fit_result["top_predictors"],
            fit_result["rbc"]["threshold"]["case"],
        )
        gemvap_records.append(
            compute_calibrated_metrics(scores.values, true_labels.values, thresholds, name, strict=strict_evaluation)
        )

    tool_df = compute_individual_tool_metrics(
        df_test_full.data, true_labels.values, pejaver_thresholds, strict=strict_evaluation
    )

    full_test_df = (
        pd.concat([pd.DataFrame(gemvap_records), tool_df], ignore_index=True)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    full_test_df.to_csv(output_dir / "performance_metrics_full_test.csv", index=False)
    result(f"Scored {len(gemvap_records)} GEMVAP model(s) + {len(tool_df)} individual predictors -- "
           f"saved performance_metrics_full_test.csv")
    return full_test_df
