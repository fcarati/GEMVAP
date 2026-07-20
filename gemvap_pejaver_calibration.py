"""
gemvap_pejaver_calibration.py
─────────────────────────────────────────────────────────────────────────────
Applies the Pejaver et al. (2022) local-posterior-probability calibration
framework to GEMVAP consensus scores, mapping them to ACMG/AMP PP3/BP4
evidence strengths.

Methodology reference:
  Pejaver et al. (2022) Am J Hum Genet 109:2163-2177
  https://doi.org/10.1016/j.ajhg.2022.10.013

GEMVAP reference:
  Carati et al. (2025) – GEMVAP article draft

─────────────────────────────────────────────────────────────────────────────
Pipeline overview
─────────────────────────────────────────────────────────────────────────────
1.  Load calibration data  (training set: labelled pathogenic / control)
2.  Estimate FBN1-specific prior probability of pathogenicity
3.  Compute LR thresholds for each ACMG/AMP evidence level from the prior
4.  Estimate local positive likelihood ratio  lrⁿ(s)  via sliding window
    over the GEMVAP consensus score
5.  Derive score thresholds (PP3 and BP4) with 95 % one-sided confidence
    bounds via bootstrapping
6.  Validate on an independent test set (interval-based LR check)
7.  Annotate any new variants with their evidence level
8.  Plot calibration curves

─────────────────────────────────────────────────────────────────────────────
Expected input format
─────────────────────────────────────────────────────────────────────────────
A CSV / TSV / Excel file with at least these columns:

  variant_id   – any unique identifier  (e.g. cDNA change)
  label        – "pathogenic" | "benign"   (training / calibration set)
                 "test_pathogenic" | "test_benign"  (held-out test set)
                 "unlabelled"  (gnomAD-style population reference set,
                                used for prior estimation and overprediction
                                check; optional but recommended)
  gemvap1      – GEMVAP 1 consensus score  (integer 0-11)
  gemvap2      – GEMVAP 2 consensus score  (integer 0-7 , optional)
  gemvap3      – GEMVAP 3 consensus score  (integer 0-4 , optional)

All other columns are carried through unchanged and appear in the output.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ACMG/AMP evidence-strength framework
# ─────────────────────────────────────────────────────────────────────────────

# Pejaver Table 1 – LR thresholds derived with prior = 0.0441 (genome-wide).
# These are *recomputed* below for a user-supplied gene-specific prior.
# The exponential model: LR_vs = c, LR_st = c^(1/2), LR_mo = c^(1/4),
#                        LR_su = c^(1/8)
# with c chosen so that combining vs + st + mo + su evidence pushes
# posterior P(path) to >= 0.99.

EVIDENCE_LEVELS_PP3 = ["PP3_Supporting", "PP3_Moderate", "PP3_Strong", "PP3_VeryStrong"]
EVIDENCE_LEVELS_BP4 = ["BP4_VeryStrong", "BP4_Strong", "BP4_Moderate", "BP4_Supporting"]
INDETERMINATE = "Indeterminate"


def compute_lr_thresholds(prior: float) -> dict[str, float]:
    """
    Derive positive-LR thresholds for each evidence strength from a given
    prior probability of pathogenicity, following Tavtigian et al. (2018)
    and Pejaver et al. (2022).

    The constant c satisfies:
        posterior_odds(pathogenic) >= 0.99 / 0.01  when all four
        evidence levels for pathogenicity are combined.

    Parameters
    ----------
    prior : float
        Estimated prevalence of pathogenic variants in the reference set.

    Returns
    -------
    dict mapping evidence-level name → LR threshold (float).
    For BP4 the values are the *reciprocals* (< 1).
    """
    prior_odds = prior / (1.0 - prior)
    # Target: posterior odds >= 99 (i.e. posterior P >= 0.99) for pathogenic
    target_posterior_odds_path = 0.99 / 0.01

    # c^(1 + 1/2 + 1/4 + 1/8) * prior_odds = target
    # c^(15/8) * prior_odds = target
    c = (target_posterior_odds_path / prior_odds) ** (8.0 / 15.0)

    lr_vs = c
    lr_st = c ** (1.0 / 2.0)
    lr_mo = c ** (1.0 / 4.0)
    lr_su = c ** (1.0 / 8.0)

    return {
        "PP3_VeryStrong": lr_vs,
        "PP3_Strong":     lr_st,
        "PP3_Moderate":   lr_mo,
        "PP3_Supporting": lr_su,
        # BP4 thresholds are reciprocals (evidence of benignity)
        "BP4_Supporting": 1.0 / lr_su,
        "BP4_Moderate":   1.0 / lr_mo,
        "BP4_Strong":     1.0 / lr_st,
        "BP4_VeryStrong": 1.0 / lr_vs,
    }


def posterior_from_lr(lr: float | np.ndarray, prior: float) -> float | np.ndarray:
    """Convert local LR to posterior probability of pathogenicity."""
    prior_odds = prior / (1.0 - prior)
    posterior_odds = lr * prior_odds
    return posterior_odds / (1.0 + posterior_odds)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Prior estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_prior(
    scores_path: np.ndarray,
    scores_benign: np.ndarray,
    scores_unlabelled: Optional[np.ndarray] = None,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Estimate the prior probability of pathogenicity in the reference set.

    Strategy A (preferred): if an unlabelled population set is supplied
    (analogous to gnomAD in Pejaver et al.), use a simplified DistCurve-
    inspired nearest-neighbour approach.

    Strategy B (fallback): use the empirical fraction of pathogenic variants
    weighted by a conservative factor to account for ClinVar enrichment.
    Pejaver et al. used 4.41 % genome-wide; for a disease gene like FBN1
    the prior may be higher (they assumed 10 % for clinical sequencing).

    Returns (prior_estimate, ci_lower, ci_upper) – 95 % bootstrap CI.
    """
    rng = np.random.default_rng(random_state)

    if scores_unlabelled is not None and len(scores_unlabelled) > 0:
        # Simplified DistCurve: sample P variants, find nearest neighbour
        # in unlabelled set, record when distances inflate.
        # The inflection point fraction ≈ prior.
        # We repeat over bootstrap resamples for a CI.
        estimates = []
        for _ in range(n_bootstrap):
            path_sample = rng.choice(scores_path, size=len(scores_path), replace=True)
            fracs = np.linspace(0.01, 0.30, 60)
            nn_dists = []
            unlab = scores_unlabelled.copy()
            for frac in fracs:
                n_remove = max(1, int(frac * len(unlab)))
                removed_idx = rng.choice(len(unlab), size=n_remove, replace=False)
                remaining = np.delete(unlab, removed_idx)
                if len(remaining) == 0:
                    nn_dists.append(np.inf)
                    continue
                # nearest-neighbour distance for each path sample
                diffs = np.abs(path_sample[:, None] - remaining[None, :])
                nn_dists.append(np.mean(np.min(diffs, axis=1)))
            # Inflection: largest second derivative
            nn_dists = np.array(nn_dists)
            second_deriv = np.diff(np.diff(nn_dists))
            inflection_idx = np.argmax(second_deriv) + 1
            estimates.append(fracs[inflection_idx])
        prior_est = float(np.median(estimates))
        ci_lo = float(np.percentile(estimates, 2.5))
        ci_hi = float(np.percentile(estimates, 97.5))
    else:
        # Fallback: empirical fraction with bootstrap
        total = len(scores_path) + len(scores_benign)
        all_labels = np.array([1] * len(scores_path) + [0] * len(scores_benign))
        estimates = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, total, size=total)
            estimates.append(float(all_labels[idx].mean()))
        prior_est = float(np.mean(estimates))
        ci_lo = float(np.percentile(estimates, 2.5))
        ci_hi = float(np.percentile(estimates, 97.5))
        warnings.warn(
            f"No unlabelled population set supplied. Falling back to empirical "
            f"fraction ({prior_est:.3f}). This likely over-estimates the true "
            f"prior due to ClinVar enrichment. Consider supplying gnomAD FBN1 "
            f"variants as 'unlabelled' rows."
        )

    return prior_est, ci_lo, ci_hi


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Local likelihood ratio estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_local_lr(
    scores: np.ndarray,
    labels: np.ndarray,
    score_values: np.ndarray,
    prior: float,
    min_variants_in_window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate lrⁿ(s) for each unique score value using an adaptive sliding
    window (Pejaver et al. Eq. 6).

    Because GEMVAP consensus scores are integers (0-11), the sliding window
    naturally collapses to exact-score bins.  For continuous scores (e.g.
    a softened GEMVAP score or REVEL), a bandwidth-adaptive kernel is used.

    Parameters
    ----------
    scores : 1-D array of scores for all calibration variants
    labels : 1-D array of {1 = pathogenic, 0 = benign} for each variant
    score_values : unique score values at which to evaluate lrⁿ
    prior : estimated prior probability of pathogenicity
    min_variants_in_window : minimum combined variants required per window

    Returns
    -------
    lr_point : point estimate of lrⁿ at each score_value
    posterior_point : corresponding posterior P(pathogenic|score)
    """
    # Weight benign variants to account for ClinVar enrichment
    n_path = np.sum(labels == 1)
    n_benign = np.sum(labels == 0)
    prior_odds = prior / (1.0 - prior)
    empirical_odds = n_path / n_benign if n_benign > 0 else 1.0
    benign_weight = empirical_odds / prior_odds  # Pejaver et al. weighting

    lr_point = np.full(len(score_values), np.nan)

    is_integer_score = np.all(scores == scores.astype(int))

    for i, sv in enumerate(score_values):
        if is_integer_score:
            # Exact bin for integer scores
            mask = scores == sv
        else:
            # Adaptive window: expand epsilon until min_variants_in_window met
            epsilon = 0.0
            step = (scores.max() - scores.min()) / 100.0 if scores.max() > scores.min() else 0.01
            for _ in range(500):
                mask = (scores >= sv - epsilon) & (scores <= sv + epsilon)
                n_path_w = np.sum(labels[mask] == 1)
                n_ben_w = np.sum(labels[mask] == 0) * benign_weight
                if (n_path_w + n_ben_w) >= min_variants_in_window:
                    break
                epsilon += step
            else:
                mask = np.ones(len(scores), dtype=bool)  # fallback: all data

        path_in_window = np.sum(labels[mask] == 1)
        benign_in_window = np.sum(labels[mask] == 0) * benign_weight

        total_path = n_path
        total_benign = n_benign * benign_weight

        if path_in_window == 0 or benign_in_window == 0 or total_path == 0 or total_benign == 0:
            lr_point[i] = np.nan
            continue

        tpr = path_in_window / total_path
        fpr = benign_in_window / total_benign
        lr_point[i] = tpr / fpr if fpr > 0 else np.inf

    posterior_point = posterior_from_lr(lr_point, prior)
    return lr_point, posterior_point


def bootstrap_lr(
    scores: np.ndarray,
    labels: np.ndarray,
    score_values: np.ndarray,
    prior: float,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_state: int = 42,
    min_variants_in_window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided 95 % confidence bound on lrⁿ(s) via bootstrapping.
    For PP3, we use the *lower* bound (more stringent threshold).
    For BP4, we use the *upper* bound.

    Returns (lr_lower_bound, lr_upper_bound) arrays over score_values.
    """
    rng = np.random.default_rng(random_state)
    n = len(scores)
    boot_lrs = np.full((n_bootstrap, len(score_values)), np.nan)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores = scores[idx]
        boot_labels = labels[idx]
        lr_b, _ = compute_local_lr(
            boot_scores, boot_labels, score_values, prior,
            min_variants_in_window=min_variants_in_window
        )
        boot_lrs[b] = lr_b

    alpha = 1.0 - confidence
    lr_lower = np.nanpercentile(boot_lrs, alpha * 100, axis=0)   # e.g. 5th pct
    lr_upper = np.nanpercentile(boot_lrs, (1 - alpha) * 100, axis=0)  # 95th pct
    return lr_lower, lr_upper


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Threshold derivation
# ─────────────────────────────────────────────────────────────────────────────

def derive_thresholds(
    score_values: np.ndarray,
    lr_lower_bound: np.ndarray,
    lr_upper_bound: np.ndarray,
    lr_thresholds: dict[str, float],
) -> dict[str, Optional[float]]:
    """
    Determine score thresholds for each evidence level using the conservative
    one-sided confidence bound (Pejaver et al. Eqs. 8–9).

    PP3 threshold for level L: the *smallest* score s such that
      lr_lower_bound(s') >= LR_L  for all s' >= s.

    BP4 threshold for level L: the *largest* score s such that
      lr_upper_bound(s') <= LR_L  for all s' <= s.

    Returns a dict: evidence_level → threshold score (or None if not reached).
    """
    thresholds = {}
    sorted_idx = np.argsort(score_values)
    sv_sorted = score_values[sorted_idx]
    lr_lo_sorted = lr_lower_bound[sorted_idx]
    lr_hi_sorted = lr_upper_bound[sorted_idx]

    # PP3: from highest score downward – find smallest s where all higher
    # scores satisfy the lower-bound LR condition.
    for level in EVIDENCE_LEVELS_PP3:
        target = lr_thresholds[level]
        threshold = None
        # Scan from high to low: the threshold is the lowest score where
        # the conservative lower bound still meets the target
        for j in range(len(sv_sorted) - 1, -1, -1):
            if np.isnan(lr_lo_sorted[j]):
                continue
            if lr_lo_sorted[j] >= target:
                threshold = sv_sorted[j]
            else:
                break  # once we go below target, stop
        thresholds[level] = threshold

    # BP4: from lowest score upward – find largest s where all lower
    # scores satisfy the upper-bound LR condition.
    for level in EVIDENCE_LEVELS_BP4:
        target = lr_thresholds[level]
        threshold = None
        for j in range(len(sv_sorted)):
            if np.isnan(lr_hi_sorted[j]):
                continue
            if lr_hi_sorted[j] <= target:
                threshold = sv_sorted[j]
            else:
                break
        thresholds[level] = threshold

    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Variant annotation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_variant(score: float, thresholds: dict[str, Optional[float]]) -> str:
    """
    Assign a single variant to its highest achieved evidence level.
    Priority: PP3_VeryStrong > PP3_Strong > PP3_Moderate > PP3_Supporting
              > Indeterminate
              > BP4_Supporting > BP4_Moderate > BP4_Strong > BP4_VeryStrong
    """
    # PP3 (higher score = more pathogenic)
    for level in EVIDENCE_LEVELS_PP3[::-1]:  # VS > St > Mo > Su
        t = thresholds.get(level)
        if t is not None and score >= t:
            return level

    # BP4 (lower score = more benign)
    for level in EVIDENCE_LEVELS_BP4[::-1]:  # VS > St > Mo > Su (reversed)
        t = thresholds.get(level)
        if t is not None and score <= t:
            return level

    return INDETERMINATE


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_on_test_set(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    thresholds: dict[str, Optional[float]],
    lr_thresholds: dict[str, float],
) -> pd.DataFrame:
    """
    Compute interval-based LR on the held-out test set (Pejaver Eq. 10).
    For each evidence interval, compute:
        LR_interval = (fraction of pathogenic in interval) /
                      (fraction of benign in interval)
    and check whether it meets the expected LR target.
    """
    annotations = np.array([annotate_variant(s, thresholds) for s in test_scores])

    path_mask = test_labels == 1
    benign_mask = test_labels == 0
    n_path = path_mask.sum()
    n_benign = benign_mask.sum()

    records = []
    all_levels = EVIDENCE_LEVELS_PP3 + EVIDENCE_LEVELS_BP4 + [INDETERMINATE]

    for level in all_levels:
        in_interval = annotations == level
        if not in_interval.any():
            records.append({
                "evidence_level": level,
                "n_path_in_interval": 0,
                "n_benign_in_interval": 0,
                "tpr": np.nan,
                "fpr": np.nan,
                "interval_LR": np.nan,
                "target_LR": lr_thresholds.get(level, np.nan),
                "meets_target": np.nan,
            })
            continue

        n_path_in = path_mask[in_interval].sum()
        n_benign_in = benign_mask[in_interval].sum()
        tpr = n_path_in / n_path if n_path > 0 else np.nan
        fpr = n_benign_in / n_benign if n_benign > 0 else np.nan

        if level.startswith("PP3"):
            interval_lr = tpr / fpr if fpr and fpr > 0 else np.inf
        elif level.startswith("BP4"):
            interval_lr = fpr / tpr if tpr and tpr > 0 else np.inf  # reciprocal for benign
        else:
            interval_lr = np.nan

        target = lr_thresholds.get(level, np.nan)
        if level.startswith("PP3"):
            meets = bool(interval_lr >= target) if not np.isnan(interval_lr) else False
        elif level.startswith("BP4"):
            meets = bool(interval_lr <= target) if not np.isnan(interval_lr) else False
        else:
            meets = np.nan

        records.append({
            "evidence_level": level,
            "n_path_in_interval": int(n_path_in),
            "n_benign_in_interval": int(n_benign_in),
            "tpr": round(tpr, 4) if not np.isnan(tpr) else np.nan,
            "fpr": round(fpr, 4) if not np.isnan(fpr) else np.nan,
            "interval_LR": round(interval_lr, 3) if np.isfinite(interval_lr) else interval_lr,
            "target_LR": round(target, 3) if not np.isnan(target) else np.nan,
            "meets_target": meets,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_COLORS = {
    "PP3_VeryStrong": "#8B0000",
    "PP3_Strong":     "#D43F3A",
    "PP3_Moderate":   "#E8876A",
    "PP3_Supporting": "#F5C6B1",
    "Indeterminate":  "#DDDDDD",
    "BP4_Supporting": "#B1C6F5",
    "BP4_Moderate":   "#6A9CE8",
    "BP4_Strong":     "#3A5AD4",
    "BP4_VeryStrong": "#00008B",
}


def _place_legend_clear_of_lines(
    ax,
    vertical_xs: list[float],
    horizontal_ys: list[float],
    curve_x: np.ndarray,
    curve_ylo: np.ndarray,
    curve_yhi: np.ndarray,
    ncol: int = 2,
    base_fontsize: float = 7,
    label_order: list[str] | None = None,
):
    """
    Place the axes legend centered in whichever gap between the vertical
    threshold lines / horizontal reference lines / plotted curve is largest,
    verifying the actual rendered legend box against those lines and
    shrinking it if needed so it never overlaps them.
    """
    fig = ax.figure
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    def _gaps(values, lo, hi):
        bounds = [lo] + sorted(values) + [hi]
        spans = [
            (bounds[i + 1] - bounds[i], (bounds[i] + bounds[i + 1]) / 2)
            for i in range(len(bounds) - 1)
        ]
        return sorted(spans, key=lambda s: s[0], reverse=True)

    x_gaps = _gaps(vertical_xs, *xlim)
    y_gaps = _gaps(horizontal_ys, *ylim)

    handles, labels = ax.get_legend_handles_labels()
    if label_order is not None:
        by_label = dict(zip(labels, handles))
        handles = [by_label[l] for l in label_order if l in by_label]
        labels = [l for l in label_order if l in by_label]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    fontsize = base_fontsize
    legend = None
    for _ in range(6):
        for _, x_center in x_gaps[:3]:
            for _, y_center in y_gaps[:2]:
                legend = ax.legend(
                    handles, labels, fontsize=fontsize, ncol=ncol,
                    loc="center", bbox_to_anchor=(x_center, y_center),
                    bbox_transform=ax.transData,
                )
                fig.canvas.draw()
                bbox = legend.get_window_extent(renderer)
                (x0, y0), (x1, y1) = ax.transData.inverted().transform(bbox)
                hits_vline = any(x0 <= vx <= x1 for vx in vertical_xs)
                hits_hline = any(y0 <= hy <= y1 for hy in horizontal_ys)
                hits_curve = bool(np.any(
                    (curve_x >= x0) & (curve_x <= x1) & (curve_yhi >= y0) & (curve_ylo <= y1)
                ))
                if not (hits_vline or hits_hline or hits_curve):
                    return legend
                legend.remove()
        fontsize -= 0.5

    # Best effort: use the largest available gap even if a residual overlap remains.
    legend = ax.legend(
        handles, labels, fontsize=max(fontsize, 4), ncol=ncol,
        loc="center", bbox_to_anchor=(x_gaps[0][1], y_gaps[0][1]),
        bbox_transform=ax.transData,
    )
    return legend


def plot_calibration_curve(
    score_values: np.ndarray,
    lr_point: np.ndarray,
    lr_lower: np.ndarray,
    lr_upper: np.ndarray,
    thresholds: dict[str, Optional[float]],
    lr_thresholds: dict[str, float],
    prior: float,
    model_name: str = "GEMVAP",
    output_path: Optional[str] = None,
) -> None:
    """
    Reproduce the left (full-range) panel of the calibration curve from
    Pejaver et al. Fig. 3: posterior probability vs score.
    """
    posterior_point = posterior_from_lr(lr_point, prior)
    posterior_lower = posterior_from_lr(lr_lower, prior)
    posterior_upper = posterior_from_lr(lr_upper, prior)

    # PP3 posterior thresholds
    lr_su_thr = lr_thresholds["PP3_Supporting"]
    posterior_levels_pp3 = {
        "PP3_Supporting": posterior_from_lr(lr_su_thr, prior),
        "PP3_Moderate":   posterior_from_lr(lr_thresholds["PP3_Moderate"], prior),
        "PP3_Strong":     posterior_from_lr(lr_thresholds["PP3_Strong"], prior),
        "PP3_VeryStrong": posterior_from_lr(lr_thresholds["PP3_VeryStrong"], prior),
    }
    posterior_levels_bp4 = {
        "BP4_Supporting": posterior_from_lr(lr_thresholds["BP4_Supporting"], prior),
        "BP4_Moderate":   posterior_from_lr(lr_thresholds["BP4_Moderate"], prior),
        "BP4_Strong":     posterior_from_lr(lr_thresholds["BP4_Strong"], prior),
        "BP4_VeryStrong": posterior_from_lr(lr_thresholds["BP4_VeryStrong"], prior),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    ylim = (0, 0.8) if model_name in ("GEMVAP_1", "GEMVAP_2", "GEMVAP_3") else (0, 1)
    valid = ~np.isnan(lr_point)
    ax.plot(score_values[valid], posterior_point[valid],
            color="black", lw=2, label="Point estimate", zorder=5)
    ax.fill_between(score_values[valid],
                    np.clip(posterior_lower[valid], 0, 1),
                    np.clip(posterior_upper[valid], 0, 1),
                    color="gray", alpha=0.25, label="95% CI")

    # Horizontal reference lines
    horizontal_ys = []
    for level, pval in {**posterior_levels_pp3, **posterior_levels_bp4}.items():
        if ylim[0] <= pval <= ylim[1]:
            color = LEVEL_COLORS.get(level, "gray")
            style = "--" if level.startswith("PP3") else ":"
            ax.axhline(pval, color=color, ls=style, lw=1.2, alpha=0.8,
                       label=level)
            horizontal_ys.append(pval)

    # Vertical threshold lines
    vertical_xs = []
    for level, tval in thresholds.items():
        if tval is not None:
            color = LEVEL_COLORS.get(level, "gray")
            ax.axvline(tval, color=color, lw=1.0, alpha=0.6)
            vertical_xs.append(tval)

    ax.set_xlim(score_values.min() - 0.5, score_values.max() + 0.5)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"{model_name.replace('_', ' ')} Consensus Score", fontsize=11)
    ax.set_ylabel("Posterior P(Pathogenic | Score)", fontsize=11)
    ax.set_xticks(score_values)
    # Same legend order for every model (previously GEMVAP_2/3 listed PP3 before
    # BP4, GEMVAP_1 the reverse) so all three calibration curves are directly
    # comparable at a glance.
    legend_label_order = [
        "Point estimate", "BP4_VeryStrong", "BP4_Strong", "BP4_Moderate", "BP4_Supporting",
        "95% CI", "PP3_Supporting", "PP3_Moderate", "PP3_Strong", "PP3_VeryStrong",
    ]
    _place_legend_clear_of_lines(
        ax,
        vertical_xs=vertical_xs,
        horizontal_ys=horizontal_ys,
        curve_x=score_values[valid],
        curve_ylo=np.clip(posterior_lower[valid], 0, 1),
        curve_yhi=np.clip(posterior_upper[valid], 0, 1),
        ncol=2,
        base_fontsize=7,
        label_order=legend_label_order,
    )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"  Calibration curve saved -> {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_score_distribution(
    scores_path: np.ndarray,
    scores_benign: np.ndarray,
    thresholds: dict[str, Optional[float]],
    model_name: str = "GEMVAP",
    output_path: Optional[str] = None,
    score_range: Optional[tuple] = None,
    path_label: str = "Pathogenic",
    benign_label: str = "Control",
    title_prefix: str = "Score Distribution",
) -> None:
    """Bar chart of score distributions coloured by evidence interval."""
    if score_range is not None:
        score_values = np.arange(int(score_range[0]), int(score_range[1]) + 1)
    else:
        all_scores = np.concatenate([scores_path, scores_benign])
        score_values = np.arange(int(all_scores.min()), int(all_scores.max()) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.4

    # Match Combined_Horizontal color scheme: #1876BD = control, #FAA71A = pathogenic
    ax.bar(score_values - width / 2,
           [np.sum(scores_path == sv) for sv in score_values],
           width=width, color="#FAA71A", alpha=0.9, label=path_label, zorder=3)
    ax.bar(score_values + width / 2,
           [np.sum(scores_benign == sv) for sv in score_values],
           width=width, color="#1876BD", alpha=0.9, label=benign_label, zorder=3)

    # Background shading by evidence zone (drawn before bars via zorder)
    x_min = score_values.min() - 0.5
    x_max = score_values.max() + 0.5

    bp4_vals = [thresholds.get(l) for l in EVIDENCE_LEVELS_BP4]
    pp3_vals = [thresholds.get(l) for l in EVIDENCE_LEVELS_PP3]

    for tval, level in zip(bp4_vals, EVIDENCE_LEVELS_BP4):
        if tval is not None:
            ax.axvspan(x_min, tval + 0.5, color=LEVEL_COLORS[level], alpha=0.18, zorder=1)
    for tval, level in zip(pp3_vals, EVIDENCE_LEVELS_PP3):
        if tval is not None:
            ax.axvspan(tval - 0.5, x_max, color=LEVEL_COLORS[level], alpha=0.18, zorder=1)

    ax.set_xlabel(f"{model_name} Consensus Score", fontsize=11)
    ax.set_ylabel("Variant Count", fontsize=11)
    ax.set_title(f"{title_prefix} with PP3/BP4 Evidence Zones – {model_name}", fontsize=12)
    ax.set_xticks(score_values)

    # Legend: bars first, then background zones
    bar_handles = [
        mpatches.Patch(facecolor="#FAA71A", alpha=0.9, label=path_label),
        mpatches.Patch(facecolor="#1876BD", alpha=0.9, label=benign_label),
    ]
    zone_handles = []
    active_bp4 = [(l, t) for l, t in zip(EVIDENCE_LEVELS_BP4, bp4_vals) if t is not None]
    active_pp3 = [(l, t) for l, t in zip(EVIDENCE_LEVELS_PP3, pp3_vals) if t is not None]
    for level, _ in active_bp4:
        label = level.replace("BP4_", "BP4 ").replace("VeryStrong", "Very Strong")
        zone_handles.append(mpatches.Patch(facecolor=LEVEL_COLORS[level], alpha=0.4, label=label))
    zone_handles.append(mpatches.Patch(facecolor=LEVEL_COLORS[INDETERMINATE], alpha=0.4, label="Indeterminate"))
    for level, _ in active_pp3:
        label = level.replace("PP3_", "PP3 ").replace("VeryStrong", "Very Strong")
        zone_handles.append(mpatches.Patch(facecolor=LEVEL_COLORS[level], alpha=0.4, label=label))

    first_legend = ax.legend(handles=bar_handles, loc="upper left", fontsize=9, title="Variants")
    ax.add_artist(first_legend)
    ax.legend(handles=zone_handles, loc="upper right", fontsize=9, title="ACMG/AMP evidence zone")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"  Distribution plot saved -> {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    df: pd.DataFrame,
    score_col: str = "gemvap1",
    label_col: str = "label",
    prior_override: Optional[float] = None,
    n_bootstrap: int = 10_000,
    min_window: int = 30,
    output_dir: str = ".",
    model_name: str = "GEMVAP_1",
) -> dict:
    """
    Full calibration pipeline for one GEMVAP model.

    Parameters
    ----------
    df : DataFrame with columns [label_col, score_col, ...]
    score_col : column name for the GEMVAP consensus score to calibrate
    label_col : column with "pathogenic"/"benign"/"test_pathogenic"/
                "test_benign"/"unlabelled"
    prior_override : if supplied, skip prior estimation and use this value
    n_bootstrap : number of bootstrap iterations for CI estimation
    min_window : minimum variants in sliding window
    output_dir : directory for output files
    model_name : label for plots and output files

    Returns
    -------
    dict with keys: thresholds, lr_thresholds, prior, validation_df,
                    annotated_df
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Partition data ────────────────────────────────────────────────────────
    calib_path = df[df[label_col] == "pathogenic"][score_col].dropna().values.astype(float)
    calib_benign = df[df[label_col] == "benign"][score_col].dropna().values.astype(float)
    test_path = df[df[label_col] == "test_pathogenic"][score_col].dropna().values.astype(float)
    test_benign = df[df[label_col] == "test_benign"][score_col].dropna().values.astype(float)
    unlabelled = df[df[label_col] == "unlabelled"][score_col].dropna().values.astype(float)

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}  |  Score column: {score_col}")
    print(f"{'='*60}")
    print(f"  Calibration – pathogenic: {len(calib_path)}, benign: {len(calib_benign)}")
    print(f"  Test set    – pathogenic: {len(test_path)}, benign: {len(test_benign)}")
    print(f"  Unlabelled (population): {len(unlabelled)}")

    if len(calib_path) == 0 or len(calib_benign) == 0:
        raise ValueError("Need at least some 'pathogenic' and 'benign' rows for calibration.")

    # ── Prior ─────────────────────────────────────────────────────────────────
    if prior_override is not None:
        prior = prior_override
        print(f"\n  Prior (user-supplied): {prior:.4f}")
    else:
        prior, ci_lo, ci_hi = estimate_prior(
            calib_path, calib_benign,
            unlabelled if len(unlabelled) > 0 else None,
        )
        print(f"\n  Prior estimate: {prior:.4f}  (95% CI: {ci_lo:.4f} – {ci_hi:.4f})")

    # ── LR thresholds ─────────────────────────────────────────────────────────
    lr_thresholds = compute_lr_thresholds(prior)
    print("\n  LR thresholds for evidence levels:")
    for k, v in lr_thresholds.items():
        print(f"    {k:<25} {v:.4f}")

    # ── Score values to evaluate ──────────────────────────────────────────────
    all_calib = np.concatenate([calib_path, calib_benign])
    score_values = np.sort(np.unique(all_calib))

    # ── Local LR (point estimate) ─────────────────────────────────────────────
    print(f"\n  Computing local LR (point estimate) …")
    calib_scores = np.concatenate([calib_path, calib_benign])
    calib_labels = np.concatenate([
        np.ones(len(calib_path), dtype=int),
        np.zeros(len(calib_benign), dtype=int),
    ])
    lr_point, post_point = compute_local_lr(
        calib_scores, calib_labels, score_values, prior, min_window
    )

    # ── Bootstrap CI ──────────────────────────────────────────────────────────
    print(f"  Bootstrapping ({n_bootstrap} iterations) for CI …")
    lr_lower, lr_upper = bootstrap_lr(
        calib_scores, calib_labels, score_values, prior,
        n_bootstrap=n_bootstrap, min_variants_in_window=min_window,
    )

    # ── Threshold derivation ──────────────────────────────────────────────────
    thresholds = derive_thresholds(score_values, lr_lower, lr_upper, lr_thresholds)

    print("\n  +- Derived score thresholds ------------------------------+")
    for level in EVIDENCE_LEVELS_PP3 + EVIDENCE_LEVELS_BP4:
        t = thresholds.get(level)
        status = f"{t}" if t is not None else "NOT REACHED"
        print(f"  |  {level:<25} score threshold: {status}")
    print("  +---------------------------------------------------------+")

    # ── Validation ────────────────────────────────────────────────────────────
    val_df = None
    if len(test_path) > 0 and len(test_benign) > 0:
        print("\n  Validating on held-out test set …")
        test_scores = np.concatenate([test_path, test_benign])
        test_labels = np.concatenate([
            np.ones(len(test_path), dtype=int),
            np.zeros(len(test_benign), dtype=int),
        ])
        val_df = validate_on_test_set(test_scores, test_labels, thresholds, lr_thresholds)
        print(val_df[["evidence_level", "n_path_in_interval", "n_benign_in_interval",
                       "interval_LR", "target_LR", "meets_target"]].to_string(index=False))
        val_path = out / f"{model_name}_validation.csv"
        val_df.to_csv(val_path, index=False)
        print(f"\n  Validation table saved -> {val_path}")
    else:
        print("\n  No test set found – skipping validation.")

    # ── Annotate all variants ─────────────────────────────────────────────────
    valid_rows = df[score_col].notna()
    df = df.copy()
    df.loc[valid_rows, f"{score_col}_evidence"] = df.loc[valid_rows, score_col].apply(
        lambda s: annotate_variant(float(s), thresholds)
    )
    annotated_path = out / f"{model_name}_annotated.csv"
    df.to_csv(annotated_path, index=False)
    print(f"  Annotated variants saved -> {annotated_path}")

    # ── Save threshold table ──────────────────────────────────────────────────
    thr_records = []
    for level in EVIDENCE_LEVELS_PP3 + EVIDENCE_LEVELS_BP4:
        t = thresholds.get(level)
        thr_records.append({
            "model": model_name,
            "evidence_level": level,
            "score_threshold": t,
            "lr_target": lr_thresholds.get(level),
        })
    thr_df = pd.DataFrame(thr_records)
    thr_path = out / f"{model_name}_thresholds.csv"
    thr_df.to_csv(thr_path, index=False)
    print(f"  Threshold table saved -> {thr_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_calibration_curve(
        score_values, lr_point, lr_lower, lr_upper,
        thresholds, lr_thresholds, prior,
        model_name=model_name,
        output_path=str(out / f"{model_name}_calibration_curve.png"),
    )
    calib_score_range = (score_values.min(), score_values.max())

    plot_score_distribution(
        calib_path, calib_benign, thresholds,
        model_name=model_name,
        output_path=str(out / f"{model_name}_score_distribution.png"),
        score_range=calib_score_range,
        title_prefix="Calibration Set Score Distribution",
    )

    if len(test_path) > 0 and len(test_benign) > 0:
        plot_score_distribution(
            test_path, test_benign, thresholds,
            model_name=model_name,
            output_path=str(out / f"{model_name}_test_score_distribution.png"),
            score_range=calib_score_range,
            path_label="Test Pathogenic",
            benign_label="Test Control",
            title_prefix="Test Set Score Distribution",
        )

    return {
        "thresholds": thresholds,
        "lr_thresholds": lr_thresholds,
        "prior": prior,
        "score_values": score_values,
        "lr_point": lr_point,
        "lr_lower": lr_lower,
        "lr_upper": lr_upper,
        "validation_df": val_df,
        "annotated_df": df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Convenience: annotate new variants from a pre-computed threshold table
# ─────────────────────────────────────────────────────────────────────────────

def annotate_from_threshold_file(
    variants_df: pd.DataFrame,
    threshold_csv: str,
    score_col: str = "gemvap1",
) -> pd.DataFrame:
    """
    Annotate a new batch of variants using previously computed thresholds
    (output of run_calibration), without re-running the bootstrap.
    """
    thr = pd.read_csv(threshold_csv)
    thresholds = dict(zip(thr["evidence_level"], thr["score_threshold"]))

    variants_df = variants_df.copy()
    valid = variants_df[score_col].notna()
    variants_df.loc[valid, f"{score_col}_evidence"] = (
        variants_df.loc[valid, score_col]
        .apply(lambda s: annotate_variant(float(s), thresholds))
    )
    return variants_df


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Apply Pejaver et al. (2022) calibration to GEMVAP consensus scores "
            "to derive PP3/BP4 ACMG/AMP evidence thresholds."
        )
    )
    p.add_argument("input", help="Input CSV/TSV/XLSX with variant data")
    p.add_argument(
        "--score-cols", nargs="+", default=["gemvap1"],
        help="GEMVAP score column(s) to calibrate (default: gemvap1)"
    )
    p.add_argument("--label-col", default="label", help="Column with variant labels")
    p.add_argument(
        "--prior", type=float, default=None,
        help="Override prior probability of pathogenicity (0–1). "
             "If omitted, estimated from data."
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=10_000,
        help="Bootstrap iterations for CI (default: 10000)"
    )
    p.add_argument(
        "--min-window", type=int, default=30,
        help="Minimum variants in sliding window (default: 30)"
    )
    p.add_argument("--output-dir", default="gemvap_calibration_output",
                   help="Output directory")
    p.add_argument(
        "--annotate-only", default=None,
        help="Path to pre-computed threshold CSV; skip calibration and just annotate."
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Load data
    path = Path(args.input)
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    print(f"Loaded {len(df)} variants from {path.name}")

    if args.annotate_only:
        # Quick annotation mode
        for sc in args.score_cols:
            df = annotate_from_threshold_file(df, args.annotate_only, score_col=sc)
        out_file = Path(args.output_dir) / "annotated_variants.csv"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        print(f"Annotated variants saved -> {out_file}")
        return

    # Full calibration for each requested score column
    for sc in args.score_cols:
        if sc not in df.columns:
            print(f"WARNING: column '{sc}' not found, skipping.")
            continue
        model_name = sc.upper().replace("_", "")
        run_calibration(
            df,
            score_col=sc,
            label_col=args.label_col,
            prior_override=args.prior,
            n_bootstrap=args.n_bootstrap,
            min_window=args.min_window,
            output_dir=args.output_dir,
            model_name=model_name,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Quick demo with synthetic data (run when no arguments given)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_demo_data(seed: int = 0) -> pd.DataFrame:
    """
    Generate synthetic data that roughly mirrors the GEMVAP 1 score
    distributions described in the paper (410 pathogenic, 349 control,
    38 test pathogenic, 38 test control, 200 unlabelled).
    """
    rng = np.random.default_rng(seed)

    def score_dist(n, mean, std, lo=0, hi=11):
        raw = rng.normal(mean, std, n * 5)
        raw = np.round(raw).astype(int)
        raw = raw[(raw >= lo) & (raw <= hi)]
        return rng.choice(raw, size=n, replace=False)

    path_scores   = score_dist(410, 7.5, 2.0)
    benign_scores = score_dist(349, 2.5, 2.0)
    test_p_scores = score_dist(38,  7.5, 2.0)
    test_b_scores = score_dist(38,  2.5, 2.0)
    unlab_scores  = score_dist(200, 3.0, 2.5)

    frames = []
    for sc, lab in [
        (path_scores,   "pathogenic"),
        (benign_scores, "benign"),
        (test_p_scores, "test_pathogenic"),
        (test_b_scores, "test_benign"),
        (unlab_scores,  "unlabelled"),
    ]:
        frames.append(pd.DataFrame({"variant_id": [f"{lab}_{i}" for i in range(len(sc))],
                                    "gemvap1": sc, "label": lab}))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No arguments supplied — running demo with synthetic data.\n")
        demo_df = _generate_demo_data()
        run_calibration(
            demo_df,
            score_col="gemvap1",
            label_col="label",
            prior_override=None,
            n_bootstrap=500,       # small for demo speed
            min_window=20,
            output_dir="gemvap_calibration_output",
            model_name="GEMVAP1_demo",
        )
        print("\nDemo complete. Check gemvap_calibration_output/ for results.")
    else:
        main()
