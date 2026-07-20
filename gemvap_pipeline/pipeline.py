"""
Single-seed run of the GEMVAP pipeline: build the train/test split, fit the
requested GEMVAP variants, optionally calibrate them, and save all outputs.
"""
from pathlib import Path

import pandas as pd

from .calibration import run_calibration_and_metrics
from .data import build_train_test_sets, conserved_domain_mask, load_cached_datasets
from .model import apply_consensus_score
from .training import generate_figure4_if_missing, load_full_dataset, train_gemvap1, train_gemvap2, train_gemvap3


def _load_pathogenic_overrides(path):
    if not path:
        return None
    overrides_df = pd.read_csv(path, header=None, names=["cDNA"])
    overrides = set(overrides_df["cDNA"].str.strip())
    print(f"Pathogenic overrides loaded: {len(overrides)} variants from {path}")
    return overrides


def _lazy(fn):
    """Memoize a zero-arg callable so it runs at most once, on first use."""
    cache = {}

    def wrapper():
        if "value" not in cache:
            cache["value"] = fn()
        return cache["value"]

    return wrapper


def _model_scores(name, col, fit_result, df, df_test):
    top_predictors = fit_result["top_predictors"]
    case_thresholds = fit_result["rbc"]["threshold"]["case"]
    return {
        "name": name,
        "col": col,
        "fit": fit_result,
        # Computed directly rather than read from df.data["consensus_score"]:
        # that column is only populated as a side effect of fit_gemvap*()
        # actually running, which doesn't happen when training is skipped
        # because a cached fit was loaded.
        "train_scores": apply_consensus_score(df.data, top_predictors, case_thresholds),
        "test_scores": apply_consensus_score(df_test.data, top_predictors, case_thresholds),
    }


def run_seed(args, seed: int, output_dir):
    """Run the full pipeline for a single seed. Returns (all_acc_df, all_acc_dom_df).
    Both are None when --run-calibration is not set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pathogenic_overrides = _load_pathogenic_overrides(args.pathogenic_overrides)

    new_denovo_path = args.new_denovo_path if args.new_denovo_path else None

    # Skip the cache when new de novo overrides are active: the saved CSVs
    # may have been built without those variants and would produce a stale split.
    use_cache = new_denovo_path is None
    cached = load_cached_datasets(output_dir, args.data_path, seed=seed, pathogenic_overrides=pathogenic_overrides) if use_cache else None
    if cached is not None:
        print(f"Found cached training/test datasets in {output_dir} — skipping dataset build.")
        df, df_test = cached
    else:
        print("Building training and test sets...")
        df, df_test, _ = build_train_test_sets(
            args.data_path,
            seed=seed,
            pathogenic_overrides=pathogenic_overrides,
            new_denovo_path=new_denovo_path,
        )

        print("Saving training and test datasets...")
        df.data.to_csv(output_dir / "training_dataset.csv", index=False)
        df_test.data.to_csv(output_dir / "test_dataset.csv", index=False)

    training_filter = ~df.training_sets & df.is_mis & ~df.is_denovo

    base = train_gemvap1(df, training_filter, args, output_dir)
    models = [_model_scores("GEMVAP_1", "gemvap1", base, df, df_test)]

    get_full_dataset = _lazy(lambda: load_full_dataset(args.data_path, seed, base))
    generate_figure4_if_missing(get_full_dataset, output_dir, "Figure_4.png")

    if args.train_gemvap2:
        cyst = train_gemvap2(df, training_filter, base, args, output_dir)
        models.append(_model_scores("GEMVAP_2", "gemvap2", cyst, df, df_test))

        generate_figure4_if_missing(
            get_full_dataset, output_dir, "Figure_4.1.png",
            filter_fn=lambda d: ~d.is_cys,
        )

    if args.train_gemvap3:
        dom = train_gemvap3(df, training_filter, base, args, output_dir)
        models.append(_model_scores("GEMVAP_3", "gemvap3", dom, df, df_test))

        generate_figure4_if_missing(
            get_full_dataset, output_dir, "Figure_4.2.png",
            filter_fn=lambda d: conserved_domain_mask(d, args.conserved_data_path),
        )

    all_acc_df = None
    all_acc_dom_df = None
    if args.run_calibration:
        all_acc_df, all_acc_dom_df = run_calibration_and_metrics(
            args, output_dir, df, df_test, models
        )

    print("Saving test set scores...")
    test_scores_df = pd.DataFrame(index=df_test.data.index)
    for m in models:
        test_scores_df[m["col"]] = m["test_scores"]
    test_label = pd.Series("unlabelled", index=df_test.data.index, dtype=object)
    test_label[df_test.is_case & ~df_test.is_ctrl] = "pathogenic"
    test_label[~df_test.is_case & df_test.is_ctrl] = "benign"
    test_scores_df.insert(0, "label", test_label)
    test_scores_df.to_csv(output_dir / "test_scores.csv", index=False)

    print("Pipeline complete.")
    print(f"Results saved in: {output_dir.resolve()}")
    return all_acc_df, all_acc_dom_df
