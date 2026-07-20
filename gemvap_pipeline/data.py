import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from packages.package1.predictor_selection import DataProcessor
from .verbose import info, note, result, step, substep


def create_output_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_processor(data_path: str, seed: int = 42) -> DataProcessor:
    processor = DataProcessor(data_path, seed=seed)
    processor.read_data()
    processor.create_filters()
    return processor


def count_pathogenic_and_control(dp: DataProcessor, available_filter) -> Tuple[int, int]:
    """Count pathogenic (is_case) and control (is_ctrl) variants within
    available_filter, using the case/control classification already computed
    by DataProcessor.create_filters()."""
    n_pathogenic = int((dp.is_case & available_filter).sum())
    n_control = int((dp.is_ctrl & available_filter).sum())
    return n_pathogenic, n_control


_PATHOGENIC_DB_COLUMNS = ("HGMD", "UMD", "FRANKEN", "MUTDB", "PARIS", "GENT")


def build_intersection_ac_table(
    data_path: str,
    seed: int = 42,
    ac_col: str = "gnomad4_joint_AC",
    ac_threshold: int = 11,
) -> pd.DataFrame:
    """Intersection variants (is_inte -- flagged in both a pathogenic database
    and a gnomAD control database, per DataProcessor.create_filters(), i.e.
    contradictory evidence) whose ac_col exceeds ac_threshold: pathogenic-
    labelled variants that are also recurrently observed in the general
    population.

    Loads data_path fresh via load_processor -- the full dataset, every
    consequence type, no missense/training-set filtering -- so is_inte matches
    exactly what the train/test split (build_train_test_sets) and
    run_part3_venn.py's is_ctrl-based panels are built from.

    Returns one row per matching variant (cDNA, aaref/aapos/aaalt, ac_col,
    'pathogenic_dbs' -- the '+'-joined subset of _PATHOGENIC_DB_COLUMNS
    flagging it), sorted by ac_col descending.
    """
    step(f"Identifying intersection variants with {ac_col} > {ac_threshold}")
    dp = load_processor(data_path, seed=seed)

    ac = pd.to_numeric(dp.data[ac_col], errors="coerce")
    inte = dp.data[dp.is_inte & (ac > ac_threshold)].copy()
    inte[ac_col] = ac.loc[inte.index]
    inte["pathogenic_dbs"] = inte.apply(
        lambda row: "+".join(c for c in _PATHOGENIC_DB_COLUMNS if row[c] == 1), axis=1,
    )

    out = inte[["cDNA", "aaref", "aapos", "aaalt", ac_col, "pathogenic_dbs"]]
    out = out.sort_values(ac_col, ascending=False).reset_index(drop=True)
    result(f"{len(out)} intersection variant(s) with {ac_col} > {ac_threshold}")
    return out


def select_highest_an_controls(control_pool: pd.DataFrame, n: int, an_col: str = "gnomAD_exomes_AN") -> List:
    """Select the n control variants with the highest gnomAD_exomes_AN (allele
    number) — the most confidently genotyped controls — to hold out for the
    test set. Rows with a missing an_col are never selected."""
    step(f"Selecting {n} controls for the test set by highest {an_col}")
    info("AN = allele number: how many chromosomes were successfully sequenced at this site in gnomAD.")
    info("Higher AN -> the variant was observed (or absent) in more individuals -> stronger evidence of benignity.")
    if n > len(control_pool):
        raise ValueError("Number of controls requested exceeds the available control pool.")
    selected = control_pool.nlargest(n, an_col)
    an_min = selected[an_col].min()
    an_max = selected[an_col].max()
    result(f"{n} controls selected  |  {an_col} range: {an_min:,.0f} – {an_max:,.0f}")
    return selected.index.tolist()


def _drop_duplicate_variants(df: DataProcessor) -> None:
    step("Deduplicating variants on (genomic position, aaref, aaalt)")
    info("The same amino-acid substitution can appear multiple times in the input table")
    info("if it was reported by more than one database or annotation source.")
    info("Duplicates are dropped to avoid inflating training counts.")
    before = len(df.data)
    for mask in [df.is_case, df.is_ctrl]:
        subset = df.data[mask]
        duplicates = subset[subset.duplicated(subset=["pos(1-based)", "aaref", "aaalt"])]
        df.data = df.data.drop(duplicates.index, errors="ignore")
    df.create_filters()
    after = len(df.data)
    result(f"{before - after} duplicate rows removed  |  {after} variants remain")


def _exclude_intersection_variants(dp: DataProcessor) -> int:
    """Remove variants that appear in both pathogenic and control databases
    from dp.data. Variants already overridden to pathogenic (is_case=True) are
    kept. Returns the number of excluded variants."""
    step("Checking for intersection variants (present in both pathogenic and control databases)")
    info("A variant reported as pathogenic in HGMD/UMD AND observed in gnomAD carries")
    info("contradictory evidence: it cannot be reliably labelled for training.")
    info("Variants already force-flagged as pathogenic via overrides are kept.")
    exclude_mask = dp.is_inte & ~dp.is_case
    n_excluded = int(exclude_mask.sum())
    if n_excluded:
        dp.data = dp.data[~exclude_mask]
        dp.create_filters()
        result(f"{n_excluded} intersection variant(s) excluded from the training set")
    else:
        result("No intersection variants found — all training variants have unambiguous labels")
    return n_excluded


def _exclude_unlabeled_variants(dp: DataProcessor) -> int:
    """Remove variants that are neither pathogenic nor control from dp.data.
    Uses the current dp.is_case/dp.is_ctrl (not the stale dp.is_left computed
    at create_filters() time) so that variants force-flagged pathogenic via
    pathogenic_overrides after the last create_filters() call are correctly
    kept. Returns the number of excluded variants."""
    step("Excluding variants that are neither pathogenic nor control")
    info("A variant absent from every pathogenic database (HGMD/UMD/FRANKEN/MUTDB/PARIS/GENT)")
    info("and every control database (gnomAD exomes/genomes) carries no usable label for training.")
    exclude_mask = ~dp.is_case & ~dp.is_ctrl
    n_excluded = int(exclude_mask.sum())
    if n_excluded:
        dp.data = dp.data[~exclude_mask]
        dp.create_filters()
        result(f"{n_excluded} unlabeled variant(s) excluded from the training set")
    else:
        result("No unlabeled variants found — every training variant is pathogenic or control")
    return n_excluded


def _apply_pathogenic_overrides(dp: DataProcessor, cdna_overrides: set) -> None:
    """Force-flag variants whose cDNA notation is in cdna_overrides as pathogenic.
    Removes them from the control pool so they cannot also be treated as controls."""
    if not cdna_overrides:
        return
    mask = dp.data["cDNA"].isin(cdna_overrides)
    dp.is_case = dp.is_case | mask
    dp.is_ctrl = dp.is_ctrl & ~mask


def _normalise_cdna(cdna: str) -> str:
    """Normalise the substitution allele (after '>') to uppercase.
    Handles typos like 'c.2659T>c' → 'c.2659T>C'."""
    if ">" in cdna:
        prefix, alt = cdna.rsplit(">", 1)
        return prefix + ">" + alt.upper()
    return cdna


_CDNA_COLUMN_CANDIDATES = ("cDNA", "Mutation c.")
_DENOVO_INDEX_CASE_COLUMN = "variant identifié de novo chez un cas index"


def load_new_denovo_cdna(path: str) -> set:
    """Load cDNA variant IDs from a de novo Excel file.

    Supports two column naming conventions:
    - 'cDNA' (original New_denovo.xlsx format)
    - 'Mutation c.' (NGS clinical report format)
    Column names are stripped of whitespace before matching.
    Values are normalised with _normalise_cdna for case consistency.

    If a 'variant identifié de novo chez un cas index' column is present
    (NGS clinical report format), only rows flagged with 'x' in that column
    are confirmed de novo in the index case; all other rows (e.g. mosaic in
    a parent, mosaic in the index case, unconfirmed) are discarded.
    """
    df_nd = pd.read_excel(path)
    df_nd.columns = df_nd.columns.str.strip()
    if _DENOVO_INDEX_CASE_COLUMN in df_nd.columns:
        n_before = len(df_nd)
        flag = df_nd[_DENOVO_INDEX_CASE_COLUMN].astype(str).str.strip().str.lower()
        df_nd = df_nd[flag == "x"]
        print(
            f"De novo index-case filter applied on {path!r}: "
            f"{len(df_nd)}/{n_before} rows flagged 'x' in {_DENOVO_INDEX_CASE_COLUMN!r}"
        )
    for col in _CDNA_COLUMN_CANDIDATES:
        if col in df_nd.columns:
            cdna_set = {_normalise_cdna(v) for v in df_nd[col].dropna().astype(str).str.strip()}
            print(f"De novo variants loaded: {len(cdna_set)} from {path!r} (column: {col!r})")
            return cdna_set
    raise ValueError(
        f"Cannot find a cDNA column in {path!r}. "
        f"Expected one of: {_CDNA_COLUMN_CANDIDATES}"
    )


def _apply_new_denovo_overrides(dp: DataProcessor, cdna_set: set) -> None:
    """Stamp new de novos into the raw DataFrame before any split.

    Sets DENOVO=1 in dp.data so create_filters() permanently inherits is_denovo.
    is_case/is_ctrl are also patched here; callers that need them to survive a
    subsequent create_filters() call should pass new_denovo_cdna via the merged
    pathogenic_overrides argument to _apply_pathogenic_overrides instead.
    """
    if not cdna_set:
        return
    mask = dp.data["cDNA"].apply(_normalise_cdna).isin(cdna_set)
    dp.data.loc[mask, "DENOVO"] = 1   # persistent: create_filters() reads this column
    dp.is_denovo = dp.is_denovo | mask
    dp.is_case   = dp.is_case   | mask
    dp.is_ctrl   = dp.is_ctrl   & ~mask


def _print_split_summary(stages: List[Tuple[str, int, int]]) -> None:
    """Print a dataset composition table.

    stages: list of (label, n_pathogenic, n_control) tuples printed in order.
    """
    col_label = max(len(r[0]) for r in stages) + 2
    header = f"{'Set':<{col_label}} {'Pathogenic':>12} {'Control':>12} {'Total':>10}"
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print(f"{'Dataset Composition Summary':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print(sep)
    for label, n_p, n_c in stages:
        print(f"{label:<{col_label}} {n_p:>12} {n_c:>12} {n_p + n_c:>10}")
    print("=" * len(header))
    print()


def build_train_test_sets(
    data_path: str,
    seed: int = 42,
    pathogenic_overrides: set = None,
    new_denovo_path: str = None,
    extra_denovo_paths: list = None,
    extra_denovo_cdna: set = None,
) -> Tuple[DataProcessor, DataProcessor, List]:
    info("Pathogenic source databases : HGMD, UMD, FRANKEN, MUTDB, PARIS, GENT")
    info("Control source databases    : gnomAD exomes, gnomAD genomes")
    new_denovo_cdna = load_new_denovo_cdna(new_denovo_path) if new_denovo_path else set()
    for extra_path in (extra_denovo_paths or []):
        if extra_path:
            new_denovo_cdna |= load_new_denovo_cdna(extra_path)
    new_denovo_cdna |= (extra_denovo_cdna or set())

    # Merge new de novos into pathogenic_overrides so _apply_pathogenic_overrides
    # re-asserts is_case=True after every create_filters() call automatically.
    all_pathogenic_overrides = (pathogenic_overrides or set()) | new_denovo_cdna

    df = load_processor(data_path, seed=seed)
    df_test = load_processor(data_path, seed=seed)

    # Stamp DENOVO=1 into the raw DataFrames BEFORE any split so that all
    # subsequent create_filters() calls inherit is_denovo correctly.
    _apply_new_denovo_overrides(df, new_denovo_cdna)
    _apply_new_denovo_overrides(df_test, new_denovo_cdna)
    _apply_pathogenic_overrides(df, all_pathogenic_overrides)
    _apply_pathogenic_overrides(df_test, all_pathogenic_overrides)

    # Population available for the case/control split: missense variants not
    # already reserved for other training cohorts (ESP/EXAC/1000G/HGMD_2016).
    substep("Defining the available variant pool")
    n_total = len(df.data)
    result(f"Starting from {n_total} rows in the loaded dataset")

    step("Filtering to missense variants only (Consequence == 'missense_variant')")
    info("Stop-gained, frameshift, splice-site etc. are excluded — GEMVAP is designed for missense.")
    n_missense = int(df.is_mis.sum())
    result(f"{n_missense} rows remain  |  {n_total - n_missense} rows dropped (non-missense)")

    step("Excluding variants already used in pre-existing training cohorts (ESP, ExAC, 1000G, HGMD_2016)")
    info("These cohorts were used to train several individual predictors that are GEMVAP features.")
    info("Keeping them would cause data leakage: the predictor scores would have been trained on")
    info("the same variants we are evaluating, inflating apparent performance.")
    available_filter = ~df.training_sets & df.is_mis
    n_available = int(available_filter.sum())
    result(f"{n_available} rows remain  |  {n_missense - n_available} rows dropped "
           f"(already in ESP/ExAC/1000G/HGMD_2016)")

    n_pathogenic, n_control = count_pathogenic_and_control(df, available_filter)
    result(f"Available pool: {n_pathogenic} pathogenic  |  {n_control} control variants")

    substep("Defining the test set — de novo pathogenic variants + matched controls")
    step("Assigning ALL de novo pathogenic variants to the test set")
    info("De novo variants were not present in the historical databases used to train individual")
    info("predictors, so they provide a leakage-free evaluation benchmark.")
    n_denovo_case = int((available_filter & df.is_denovo & df.is_case).sum())
    result(f"{n_denovo_case} de novo pathogenic variants -> test set")

    # Test set is balanced 1:1 — all de novo pathogenic variants plus an equal
    # number of controls selected by highest gnomAD_exomes_AN.
    control_count = n_denovo_case

    substep("Defining the training pool")
    step("All non-de-novo variants from the available pool form the training candidate set")
    training_filter = available_filter & ~df.is_denovo
    n_nondenovopool_path, n_nondenovopool_ctrl = count_pathogenic_and_control(df, training_filter)
    result(f"Training candidate pool: {n_nondenovopool_path} pathogenic  |  {n_nondenovopool_ctrl} control variants")

    df.data = df.data[training_filter]
    df.create_filters()
    _apply_pathogenic_overrides(df, all_pathogenic_overrides)
    n_excluded_intersection = _exclude_intersection_variants(df)
    _apply_pathogenic_overrides(df, all_pathogenic_overrides)
    n_excluded_unlabeled = _exclude_unlabeled_variants(df)
    _apply_pathogenic_overrides(df, all_pathogenic_overrides)

    index_train = select_highest_an_controls(df.data[df.is_ctrl], control_count)
    df.data = df.data[~df.data.index.isin(index_train)]
    df.create_filters()
    _apply_pathogenic_overrides(df, all_pathogenic_overrides)

    test_filter = ~df_test.training_sets & df_test.is_mis
    df_test.data = df_test.data[test_filter]
    df_test.create_filters()
    _apply_pathogenic_overrides(df_test, all_pathogenic_overrides)

    test_selection = (df_test.is_denovo & df_test.is_case) | (
        df_test.data.index.isin(index_train) & df_test.is_ctrl
    )
    df_test.data = df_test.data[test_selection]
    df_test.create_filters()
    _apply_pathogenic_overrides(df_test, all_pathogenic_overrides)

    substep("Final deduplication of the training set")
    _drop_duplicate_variants(df)
    _apply_pathogenic_overrides(df, all_pathogenic_overrides)

    _print_split_summary([
        ("Available pool (missense, non-training sets)", n_pathogenic, n_control),
        ("  of which de novo pathogenic  [-> test]", n_denovo_case, 0),
        ("  of which non-de-novo  [-> training pool]", n_nondenovopool_path, n_nondenovopool_ctrl),
        ("Test set (1:1 balanced)", int(df_test.is_case.sum()), int(df_test.is_ctrl.sum())),
        ("Training set (final, after deduplication)", int(df.is_case.sum()), int(df.is_ctrl.sum())),
    ])
    if n_excluded_intersection:
        note(f"{n_excluded_intersection} intersection variant(s) were excluded from training "
             "(present in both pathogenic and control databases — contradictory labels).")
    if n_excluded_unlabeled:
        note(f"{n_excluded_unlabeled} unlabeled variant(s) were excluded from training "
             "(present in neither a pathogenic nor a control database).")
    substep("Train / test split complete")
    result(f"Training set : {int(df.is_case.sum())} pathogenic  |  {int(df.is_ctrl.sum())} control")
    result(f"Test set     : {int(df_test.is_case.sum())} pathogenic  |  {int(df_test.is_ctrl.sum())} control  (1:1 balanced)")

    return df, df_test, index_train


def load_dataset_csv(
    csv_path,
    data_path: str,
    seed: int = 42,
    pathogenic_overrides: set = None,
) -> DataProcessor:
    """Reload a DataProcessor from a CSV previously saved via df.data.to_csv()
    (e.g. training_dataset.csv). is_case/is_ctrl/is_cys/etc. are recomputed by
    create_filters(), which is deterministic given the row contents, so this
    reproduces the same filters as the in-memory object that produced the CSV."""
    dp = DataProcessor(data_path, seed=seed)
    dp.data = pd.read_csv(csv_path)
    dp.create_filters()
    _apply_pathogenic_overrides(dp, pathogenic_overrides)
    return dp


def load_cached_datasets(
    output_dir,
    data_path: str,
    seed: int = 42,
    pathogenic_overrides: set = None,
):
    """Reload (df, df_test) from a previous run's saved CSVs in output_dir,
    or return None if either is missing."""
    output_dir = Path(output_dir)
    train_path = output_dir / "training_dataset.csv"
    test_path = output_dir / "test_dataset.csv"
    if not (train_path.exists() and test_path.exists()):
        return None

    return tuple(
        load_dataset_csv(p, data_path, seed=seed, pathogenic_overrides=pathogenic_overrides)
        for p in (train_path, test_path)
    )


_DENOVO_OVERRIDES_FILENAME = "_denovo_overrides.json"


def save_denovo_overrides(output_dir, cdna_set: set) -> None:
    """Persist the merged de novo/pathogenic override cDNA set used to build
    training_dataset.csv/test_dataset.csv, so a later run can recover it (and
    correctly recompute is_case for purely-de-novo variants -- see
    evaluation.denovo_pathogenic_overrides) without needing the original de
    novo Excel files to still be present or referenced in config."""
    path = Path(output_dir) / _DENOVO_OVERRIDES_FILENAME
    path.write_text(json.dumps(sorted(cdna_set)))


def load_denovo_overrides(output_dir) -> set:
    """Counterpart to save_denovo_overrides(). Returns an empty set if no
    override file has been saved yet (e.g. an output_dir from before this
    mechanism existed)."""
    path = Path(output_dir) / _DENOVO_OVERRIDES_FILENAME
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def save_dataframe(df, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_parquet(path)


def annotate_hgvsp(
    df: DataProcessor,
    prefix: str = "ENSP00000325527.5:p.",
) -> None:
    df.data["HGVSp_bis"] = df.data["HGVSp"].apply(lambda x: x.replace(prefix, ""))
    df.data["HGVSp_bis"] = df.data["HGVSp_bis"][:-1]
    df.data["HGVSp_bis"] = df.data["HGVSp_bis"].apply(
        lambda x: "".join(re.findall(r"\d+", str(x)))
    )
    df.data["proteic_pos"] = df.data["HGVSp_bis"]
    df.data["exon"] = df.data.apply(
        lambda row: int(row["EXON"].split("/")[0]) if row["EXON"] != "-" else np.nan,
        axis=1,
    )


def _grouped_to_df(grouped, groupkey: str, aggfunctions: Dict, sortby: List) -> pd.DataFrame:
    dfgrouped = grouped.agg(aggfunctions)
    dfgrouped.reset_index(inplace=True)
    if groupkey in ["proteic_pos"]:
        dfgrouped[groupkey] = pd.to_numeric(dfgrouped[groupkey])
    dfgrouped.sort_values(by=sortby, inplace=True)
    return dfgrouped


def compute_grouped_data(
    df: DataProcessor,
    variant_filter=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask = df.is_mis if variant_filter is None else df.is_mis & variant_filter
    grouped_by_proteicpos = df.data[mask].groupby("proteic_pos")
    grouped_by_exon = df.data[mask].groupby("exon")
    grouped_by_aaref = df.data[mask].groupby("aaref")

    dfgrouped_by_proteicpos = _grouped_to_df(
        grouped_by_proteicpos,
        "proteic_pos",
        {
            "exon": "mean",
            "consensus_score_norm": "mean",
            "Pathogenic": "mean",
            "gnomAD_exomes_AC": "sum",
            "gnomAD_genomes_AC": "sum",
            "gnomAD_exomes_AF": "mean",
        },
        ["exon", "proteic_pos"],
    )
    dfgrouped_by_aaref = _grouped_to_df(
        grouped_by_aaref,
        "aaref",
        {
            "consensus_score_norm": "mean",
            "gnomAD_exomes_AC": "sum",
            "gnomAD_genomes_AC": "sum",
            "gnomAD_exomes_AF": "mean",
        },
        ["gnomAD_exomes_AC"],
    )
    dfgrouped_by_exon = _grouped_to_df(
        grouped_by_exon,
        "exon",
        {
            "consensus_score_norm": "mean",
            "gnomAD_exomes_AC": "sum",
            "gnomAD_genomes_AC": "sum",
            "gnomAD_exomes_AF": "mean",
        },
        ["exon"],
    )
    return dfgrouped_by_proteicpos, dfgrouped_by_aaref, dfgrouped_by_exon


def subsample_stratified(dp: DataProcessor, fraction: float, seed: int = 42) -> DataProcessor:
    """Randomly subsample dp, keeping `fraction` of its pathogenic (is_case-only)
    and `fraction` of its control (is_ctrl-only) variants independently — so the
    case:control ratio is preserved — without replacement."""
    rng = random.Random(seed)

    case_idx = list(dp.data[dp.is_case & ~dp.is_ctrl].index)
    ctrl_idx = list(dp.data[dp.is_ctrl & ~dp.is_case].index)

    n_case = round(fraction * len(case_idx))
    n_ctrl = round(fraction * len(ctrl_idx))
    sampled_idx = rng.sample(case_idx, n_case) + rng.sample(ctrl_idx, n_ctrl)

    sub = copy.deepcopy(dp)
    sub.data = dp.data.loc[sampled_idx]
    sub.create_filters()
    return sub


def build_calibration_subsample(
    df_train: DataProcessor,
    df_test: DataProcessor,
    train_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[DataProcessor, DataProcessor, DataProcessor]:
    """One calibration draw: a stratified train_fraction subsample of df_train
    and a stratified test_fraction subsample of df_test (both via
    subsample_stratified, same seed) — the calibration set — plus
    test_complement, the test set rows NOT drawn into the test subsample,
    held out for performance evaluation."""
    calib_train = subsample_stratified(df_train, fraction=train_fraction, seed=seed)
    calib_test = subsample_stratified(df_test, fraction=test_fraction, seed=seed)

    test_complement = copy.deepcopy(df_test)
    test_complement.data = df_test.data.drop(calib_test.data.index)
    test_complement.create_filters()
    return calib_train, calib_test, test_complement


def conserved_domain_mask(dp: DataProcessor, conserved_data_path: str) -> pd.Series:
    """Boolean mask (aligned to dp.data.index) for variants that are non-cysteine
    and fall outside conserved domain sites, per the FBN1 conserved-domain
    annotation table — the subset used for GEMVAP 3 training and evaluation."""
    table = pd.read_csv(conserved_data_path, sep="\t", low_memory=False, na_values=["."])
    joined = dp.data.join(table.set_index("variantvcf"), on="variantvcf")
    return ~dp.is_cys & (joined["in_dom_conserved"].fillna(1) == 0)


def filter_controls_by_an(
    dp: DataProcessor,
    min_an: int,
    an_col: str = "gnomAD_exomes_AN",
) -> DataProcessor:
    """Remove training controls whose gnomAD exomes allele number (AN) is below
    min_an, then refresh all DataProcessor filters.

    Controls with a missing AN value are also removed: a site with no AN
    information gives no population-frequency evidence of benignity.
    Pathogenic variants (is_case) are never removed, regardless of AN.

    Returns dp (modified in-place) for chaining.
    """
    if an_col not in dp.data.columns:
        raise ValueError(
            f"AN column '{an_col}' not found.  Available columns (first 10): "
            + str(list(dp.data.columns[:10]))
        )

    n_before = len(dp.data)
    low_an_ctrl = dp.is_ctrl & (dp.data[an_col].fillna(0) < min_an)
    n_removed = int(low_an_ctrl.sum())

    if n_removed:
        dp.data = dp.data[~low_an_ctrl]
        dp.create_filters()
        result(f"{an_col} < {min_an}: {n_removed} control(s) removed  |  "
               f"{n_before} -> {len(dp.data)} rows remain")
    else:
        result(f"{an_col} < {min_an}: no controls removed  |  {n_before} rows remain")

    return dp


def verify_against_comp(df_train, df_test, comp_dir: Path = Path("data/comp")) -> None:
    """Compare generated training/test sets against the reference datasets in
    data/comp, raising if they diverge. Used as a sanity check during development."""
    comp_train_path = comp_dir / "training_dataset.parquet"
    comp_test_path = comp_dir / "test_dataset.parquet"

    if not comp_train_path.exists() or not comp_test_path.exists():
        raise FileNotFoundError(
            f"Expected comparison datasets not found in '{comp_dir}'. "
            "Please ensure 'training_dataset.parquet' and 'test_dataset.parquet' exist."
        )

    comp_train = pd.read_parquet(comp_train_path)
    comp_test = pd.read_parquet(comp_test_path)

    def _compare(generated, expected, name):
        if not generated.columns.equals(expected.columns):
            raise ValueError(
                f"{name} columns do not match data/comp: "
                f"generated={list(generated.columns)}, expected={list(expected.columns)}"
            )

        if generated.shape != expected.shape:
            raise ValueError(
                f"{name} shape mismatch: generated={generated.shape}, expected={expected.shape}"
            )

        if not generated.index.equals(expected.index):
            generated = generated.reset_index(drop=True)
            expected = expected.reset_index(drop=True)
            # Sort by key columns to ensure consistent ordering
            sort_cols = ['#chr', 'pos(1-based)', 'ref', 'alt']
            generated = generated.sort_values(by=sort_cols).reset_index(drop=True)
            expected = expected.sort_values(by=sort_cols).reset_index(drop=True)

        diff = generated.compare(expected)
        if not diff.empty:
            print(f"First differences in {name}:")
            print(diff.head(10))
            raise ValueError(
                f"{name} values differ from the corresponding dataset in data/comp. "
                "Verify the input data and the selection procedure."
            )
        # If compare is empty, values match (dtypes may differ but values are same)

    _compare(df_train.data, comp_train, "Training set")
    _compare(df_test.data, comp_test, "Test set")
    print("Verification passed: generated training/test sets match data/comp.")
