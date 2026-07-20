import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

df = pd.read_csv(
    r"c:\Users\Admin\Desktop\PhD\VSC_2\output\P35555_ddgun_hhblits_all.tsv",
    sep="\t", comment="#",
    names=["seqfile", "variant", "s_ddg", "t_ddg", "stability"]
)

df["mut_aa"] = df["variant"].str[0]

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
summary = (
    df.groupby("mut_aa")["s_ddg"]
    .agg(["mean", "sem"])
    .reindex(AA_ORDER)
)

cmap = plt.cm.RdYlGn
vmin = summary["mean"].min()
vmax = max(summary["mean"].max(), abs(vmin) * 0.1)  # ensure vmax > 0
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
colors = [cmap(norm(v)) for v in summary["mean"]]

fig, ax = plt.subplots(figsize=(12, 5))

bars = ax.bar(AA_ORDER, summary["mean"], yerr=summary["sem"],
              color=colors, edgecolor="white", linewidth=0.5,
              error_kw=dict(elinewidth=0.8, capsize=3, ecolor="grey"))

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

ax.set_xlabel("Wild-type amino acid (mutated from)", fontsize=12)
ax.set_ylabel("Mean S_DDG[SEQ]", fontsize=12)
ax.set_title("Average DDGun-Seq score per wild-type amino acid\nP35555 (FBN1) — hhblits + uniclust30 profile", fontsize=13)
ax.tick_params(axis="x", labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Mean S_DDG", fontsize=10)

out = r"c:\Users\Admin\Desktop\PhD\VSC_2\output\sddg_per_mutant_aa.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
