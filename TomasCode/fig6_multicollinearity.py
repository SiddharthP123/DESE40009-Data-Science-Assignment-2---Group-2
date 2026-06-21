
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plot_style
from model import prepare, FEATURES

plot_style.apply()
r = prepare()

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# left: correlations between the six symptoms (lower triangle only)
corr = r.df[FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"label": "Pearson r"},
            ax=axes[0], annot_kws={"size": 12})
axes[0].set_title("Inter-Symptom Correlation")

# right: VIF per feature, red if it crosses the usual threshold of 5
vif_sorted = dict(sorted(r.vif.items(), key=lambda kv: kv[1]))
colors = ["#c0392b" if v > 5 else "#5499c7" for v in vif_sorted.values()]
axes[1].barh(list(vif_sorted.keys()), list(vif_sorted.values()), color=colors)
axes[1].axvline(5, color="red", ls="--", lw=1.5, label="VIF = 5")
for i, v in enumerate(vif_sorted.values()):
    axes[1].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=13)
axes[1].set_xlabel("VIF")
axes[1].set_title("Variance Inflation Factors")
axes[1].legend()

fig.suptitle("Multicollinearity Diagnostics")
fig.tight_layout()
fig.savefig(f"{plot_style.FIG_DIR}/6_multicollinearity.png", bbox_inches="tight")
print("saved 6_multicollinearity.png")
