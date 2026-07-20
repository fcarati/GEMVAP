import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from gemvap_pipeline.verbose import info, note, result, section, step, substep
from packages.package1.predictor_selection import (
    build_trace,
    consensus_incremental_data,
    predictors_computation,
    roc_based_classification,
    top_predictors_by_ks,
)
from packages.package1.predictor_selection import DataProcessor


def fit_gemvap(
    df: DataProcessor,
    training_filter,
    it: int = 2,
    binary: bool = True,
) -> Dict:
    n_case = int(df.is_case[training_filter].sum())
    n_ctrl = int(df.is_ctrl[training_filter].sum())
    step(f"Training data: {n_case} pathogenic  |  {n_ctrl} control variants")

    substep("Sub-step A: Kolmogorov-Smirnov predictor ranking")
    info("For each candidate predictor (dbNSFP rankscores), a two-sample KS test")
    info("measures how well its score distribution differs between pathogenic and control variants.")
    info("A higher KS statistic means the predictor better separates the two classes.")
    data, predictors, ks_case_ctrl_bf = predictors_computation(df, training_filter, it)
    result(f"{len(predictors)} candidate predictors evaluated")

    item = {
        "training_data": df.data[training_filter].copy(),
        "predictors": predictors,
        "ks_case_ctrl_bf": ks_case_ctrl_bf,
    }
    item["ordered_predictors_by_ks"] = top_predictors_by_ks(
        ks_case_ctrl_bf,
        n=len(predictors),
        reverse=True,
        sort_by=0,
    )
    top5 = item["ordered_predictors_by_ks"][:5]
    result(f"Top 5 predictors by KS: {', '.join(top5)}")

    substep("Sub-step B: 10-fold ROC thresholding")
    info("The training set is split into 10 folds. For each predictor, a score threshold")
    info("is found via ROC analysis on 9 folds and validated on the 10th.")
    info("This gives a per-predictor 'case threshold': a score above which a variant")
    info("is called pathogenic by that predictor alone.")
    item["rbc"] = roc_based_classification(
        item["training_data"],
        predictors,
        df.pathogenic,
        df.control,
        nfold=10,
    )
    result("Per-predictor case thresholds fitted via 10-fold cross-validation")

    substep("Sub-step C: Incremental consensus optimisation")
    info("Predictors are added one by one in KS rank order. After each addition,")
    info("a 'consensus score' is computed (how many predictors call a variant pathogenic).")
    info("The optimal number of predictors is chosen where the consensus F1 score peaks.")
    info("This prevents over-fitting from including weakly discriminating predictors.")
    item["ci_data_ks"] = {
        "ks": consensus_incremental_data(
            item["training_data"][df.is_case],
            item["training_data"][df.is_ctrl],
            item["ordered_predictors_by_ks"],
            item["rbc"]["threshold"]["case"],
            "greater",
            binary,
        )
    }
    item["top_predictors"] = item["ci_data_ks"]["ks"]["top"]
    n_top = item["ci_data_ks"]["ks"]["npreds"]
    best_cons = item["ci_data_ks"]["ks"]["cons"]
    result(f"Optimal number of predictors: {n_top}  |  consensus threshold: {best_cons}")
    result(f"Final predictor panel: {item['top_predictors']}")

    substep("Sub-step D: Annotating training data with consensus scores")
    info("Each training variant receives a consensus score = number of top predictors")
    info("that exceed their individual case threshold for that variant.")
    item["training_data"]["consensus_score"] = df.top_pred_cons(
        item["top_predictors"], item["rbc"]["threshold"]["case"]
    )
    result("Consensus scores written to training_data['consensus_score']")
    return item


def fit_gemvap2(
    df: DataProcessor,
    training_filter,
    it: int = 2,
    binary: bool = True,
) -> Dict:
    """
    Train GEMVAP 2 model on non-cysteine variants only.

    This is a specialized version of GEMVAP that excludes cysteine variants
    from training to create a complementary model for cysteine-specific predictions.
    """
    substep("Applying non-cysteine filter")
    info("In FBN1, missense mutations that create or destroy a cysteine residue in EGF-like")
    info("or cbEGF domains almost universally cause Marfan syndrome — they disrupt disulfide")
    info("bonds critical for domain folding. Because computational predictors trained on")
    info("general missense data are less informative for this structurally obvious class,")
    info("GEMVAP 2 trains exclusively on non-cysteine variants, where the pathogenicity")
    info("signal is subtler and predictor discrimination matters more.")
    # Filter out cysteine variants for training
    non_cys_filter = training_filter & ~df.is_cys

    n_case = int(df.is_case[non_cys_filter].sum())
    n_ctrl = int(df.is_ctrl[non_cys_filter].sum())
    n_cys_excluded = int((training_filter & df.is_cys).sum())
    result(f"Non-cysteine training data: {n_case} pathogenic  |  {n_ctrl} control  "
           f"({n_cys_excluded} cysteine variants excluded)")

    step("Running KS ranking, ROC thresholding, and consensus optimisation on non-cysteine variants")
    info("(Same four sub-steps as GEMVAP 1 — see above for explanations.)")
    data, predictors, ks_case_ctrl_bf = predictors_computation(df, non_cys_filter, it)

    item = {
        "training_data": df.data[non_cys_filter].copy(),
        "predictors": predictors,
        "ks_case_ctrl_bf": ks_case_ctrl_bf,
    }
    item["ordered_predictors_by_ks"] = top_predictors_by_ks(
        ks_case_ctrl_bf,
        n=len(predictors),
        reverse=True,
        sort_by=0,
    )
    item["rbc"] = roc_based_classification(
        item["training_data"],
        predictors,
        df.pathogenic,
        df.control,
        nfold=10,
    )
    item["ci_data_ks"] = {
        "ks": consensus_incremental_data(
            item["training_data"][df.is_case],
            item["training_data"][df.is_ctrl],
            item["ordered_predictors_by_ks"],
            item["rbc"]["threshold"]["case"],
            "greater",
            binary,
        )
    }
    item["top_predictors"] = item["ci_data_ks"]["ks"]["top"]
    n_top = item["ci_data_ks"]["ks"]["npreds"]
    result(f"GEMVAP 2 optimal predictors: {n_top}  ->  {item['top_predictors']}")
    item["training_data"]["consensus_score"] = df.top_pred_cons(
        item["top_predictors"], item["rbc"]["threshold"]["case"]
    )
    return item


def fit_gemvap3(
    df: DataProcessor,
    training_filter,
    conserved_data_path: str,
    it: int = 2,
    binary: bool = True,
) -> Dict:
    """
    Train GEMVAP 3 model on non-cysteine variants that are not in conserved domain sites.

    GEMVAP 3 is a specialized dataset derived by filtering the training set for non-cysteine
    variants and excluding variants that fall in conserved domain regions.
    """
    substep("Applying non-cysteine + non-conserved-domain filter")
    info("GEMVAP 3 extends GEMVAP 2 by also excluding variants in positions that are")
    info("conserved across FBN1 EGF-Ca2+-binding domain cores. These positions are under")
    info("strong structural constraint, so even 'neutral-looking' amino-acid changes")
    info("tend to be pathogenic — which can mislead predictor-based classifiers.")
    info("The remaining variants (non-cysteine AND outside conserved sites) are the")
    info("hardest subset to classify and the primary target of GEMVAP 3.")
    tableS1_cons = pd.read_csv(conserved_data_path, sep="\t", low_memory=False, na_values=["."])
    tableS1 = df.data.join(tableS1_cons.set_index("variantvcf"), on="variantvcf")
    dom_filter = tableS1["in_dom_conserved"].fillna(1) == 0
    non_cys_dom_filter = training_filter & ~df.is_cys & dom_filter

    n_case = int(df.is_case[non_cys_dom_filter].sum())
    n_ctrl = int(df.is_ctrl[non_cys_dom_filter].sum())
    n_dom_excluded = int((training_filter & ~df.is_cys & ~dom_filter).sum())
    result(f"Non-cys/non-domain training data: {n_case} pathogenic  |  {n_ctrl} control  "
           f"({n_dom_excluded} conserved-domain variants excluded)")

    step("Running KS ranking, ROC thresholding, and consensus optimisation on this subset")
    info("(Same four sub-steps as GEMVAP 1 — see above for explanations.)")
    data, predictors, ks_case_ctrl_bf = predictors_computation(df, non_cys_dom_filter, it)

    item = {
        "training_data": df.data[non_cys_dom_filter].copy(),
        "predictors": predictors,
        "ks_case_ctrl_bf": ks_case_ctrl_bf,
    }
    item["ordered_predictors_by_ks"] = top_predictors_by_ks(
        ks_case_ctrl_bf,
        n=len(predictors),
        reverse=True,
        sort_by=0,
    )
    item["rbc"] = roc_based_classification(
        item["training_data"],
        predictors,
        df.pathogenic,
        df.control,
        nfold=10,
    )
    item["ci_data_ks"] = {
        "ks": consensus_incremental_data(
            item["training_data"][df.is_case],
            item["training_data"][df.is_ctrl],
            item["ordered_predictors_by_ks"],
            item["rbc"]["threshold"]["case"],
            "greater",
            binary,
        )
    }
    item["top_predictors"] = item["ci_data_ks"]["ks"]["top"]
    n_top = item["ci_data_ks"]["ks"]["npreds"]
    result(f"GEMVAP 3 optimal predictors: {n_top}  ->  {item['top_predictors']}")
    item["training_data"]["consensus_score"] = df.top_pred_cons(
        item["top_predictors"], item["rbc"]["threshold"]["case"]
    )
    return item


def annotate_consensus(
    df,
    item: Dict,
    pathogenic_threshold: int = 7,
) -> None:
    top_predictors = item["top_predictors"]
    thresholds = item["rbc"]["threshold"]["case"]

    df.data["top_predictor_count"] = df.data.apply(
        lambda row: sum(row.loc[tp] for tp in top_predictors), axis=1
    )
    df.data["consensus_score"] = df.data.apply(
        lambda row: sum(1 if row.loc[tp] >= thresholds[tp] else 0 for tp in top_predictors),
        axis=1,
    )
    df.data["consensus_score_norm"] = df.data.apply(
        lambda row: row.consensus_score / row.top_predictor_count
        if row.top_predictor_count else 0,
        axis=1,
    )
    df.data["consensus_score_scaled"] = df.data.apply(
        lambda row: np.rint(row.consensus_score_norm * len(top_predictors)), axis=1
    )
    df.data["Pathogenic"] = 0
    df.data.loc[df.data["consensus_score"] > pathogenic_threshold, "Pathogenic"] = 1


def extract_ks_values(ks_case_ctrl_bf) -> Dict:
    return {key: value[0] for key, value in ks_case_ctrl_bf.items()}


def apply_consensus_score(data_df: pd.DataFrame, top_predictors, thresholds) -> pd.Series:
    """Score variants using a fitted GEMVAP model's top predictors and thresholds."""
    return data_df.apply(
        lambda row: sum(
            1 if pd.notna(row[tp]) and row[tp] >= thresholds[tp] else 0
            for tp in top_predictors
        ),
        axis=1,
    )


def save_fit_result(fit_result: Dict, path) -> None:
    """Persist a fit_gemvap*() result so a later run can reuse it without
    re-fitting (predictor selection, ROC thresholding, and the bootstrapped
    consensus curve are the expensive parts of training)."""
    with open(path, "wb") as f:
        pickle.dump(fit_result, f)


def load_fit_result(path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_trace_series(data, version, filter_1, filter_2):
    trace_1 = build_trace(
        data[filter_1][version["top_predictors"]],
        {k: version["rbc"]["threshold"]["case"][k] for k in version["top_predictors"]},
        name="1",
        text=False,
        y_percentage=False,
    )
    trace_2 = build_trace(
        data[filter_2][version["top_predictors"]],
        {k: version["rbc"]["threshold"]["case"][k] for k in version["top_predictors"]},
        name="2",
        text=False,
        y_percentage=False,
    )
    return trace_1, trace_2
