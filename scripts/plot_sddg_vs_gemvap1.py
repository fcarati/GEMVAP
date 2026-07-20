import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = r"c:\Users\Admin\Desktop\PhD\VSC_2"
sys.path.insert(0, ROOT + r"\gemvap_clean_pipeline")

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# ── Load GEMVAP 1 model ────────────────────────────────────────────────────────
with open(ROOT + r"\gemvap_clean_pipeline\output\gemvap_v2\GEMVAP_1_fit.pkl", "rb") as f:
    fit = pickle.load(f)
top_predictors = fit["top_predictors"]
thresholds     = fit["rbc"]["threshold"]["case"]

# ── Load all FBN1 missense variants and compute GEMVAP 1 score ─────────────────
tsv = pd.read_csv(
    ROOT + r"\gemvap_clean_pipeline\data\raw\FBN1_tableS1_allmissense.tsv",
    sep="\t", low_memory=False, na_values=["."],
)
tsv.columns = tsv.columns.str.lstrip("#").str.strip()

def gemvap1_score(row):
    return sum(
        1 if pd.notna(row[tp]) and row[tp] >= thresholds[tp] else 0
        for tp in top_predictors
    )

tsv["gemvap1"] = tsv.apply(gemvap1_score, axis=1)

# Filter to missense variants only
missense = tsv[
    tsv["Consequence"].str.contains("missense_variant", na=False) &
    tsv["aaref"].isin(AA_ORDER) &
    tsv["aaalt"].isin(AA_ORDER) &
    (tsv["aaref"] != tsv["aaalt"])
].copy()

gemvap_summary = (
    missense.groupby("aaref")["gemvap1"]
    .agg(["mean", "sem"])
    .reindex(AA_ORDER)
)

# ── Load DDGun S_DDG scores ────────────────────────────────────────────────────
ddg = pd.read_csv(
    ROOT + r"\output\P35555_ddgun_hhblits_all.tsv",
    sep="\t", comment="#",
    names=["seqfile", "variant", "s_ddg", "t_ddg", "stability"],
)
ddg["wt_aa"] = ddg["variant"].str[0]

ddg_summary = (
    ddg.groupby("wt_aa")["s_ddg"]
    .agg(["mean", "sem"])
    .reindex(AA_ORDER)
)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.subplots_adjust(hspace=0.08)

x = np.arange(len(AA_ORDER))
bar_kw = dict(width=0.65, edgecolor="white", linewidth=0.4,
              error_kw=dict(elinewidth=0.8, capsize=3, ecolor="grey"))

# GEMVAP 1 (top panel)
g_vals = gemvap_summary["mean"].values
g_norm = (g_vals - g_vals.min()) / (g_vals.max() - g_vals.min())
colors_g = plt.cm.Oranges(0.35 + 0.55 * g_norm)
ax1.bar(x, gemvap_summary["mean"], yerr=gemvap_summary["sem"],
        color=colors_g, **bar_kw)
ax1.set_ylabel("Mean GEMVAP 1 score", fontsize=11)
ax1.set_title(
    "Average GEMVAP 1 score vs. DDGun-Seq S_DDG per wild-type amino acid\n"
    "P35555 (FBN1) — all missense variants",
    fontsize=12,
)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(bottom=False)

# DDGun S_DDG (bottom panel)
d_vals = ddg_summary["mean"].values
d_norm = np.clip((d_vals - d_vals.min()) / (d_vals.max() - d_vals.min()), 0, 1)
colors_d = plt.cm.Blues_r(0.25 + 0.65 * d_norm)
ax2.bar(x, ddg_summary["mean"], yerr=ddg_summary["sem"],
        color=colors_d, **bar_kw)
ax2.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
ax2.set_ylabel("Mean S_DDG[SEQ]", fontsize=11)
ax2.set_xlabel("Wild-type amino acid", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(AA_ORDER, fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

out = ROOT + r"\output\sddg_vs_gemvap1_per_aa.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
