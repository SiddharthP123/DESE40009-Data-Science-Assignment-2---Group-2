
import numpy as np
import matplotlib.pyplot as plt

import plot_style
from model import prepare, Z95

plot_style.apply()
r = prepare()

# sort by OR so the biggest effect sits at the top
rows = sorted(
    ((name, np.exp(r.ord_coef[i]),
      np.exp(r.ord_coef[i] - Z95 * r.ord_bse[i]),
      np.exp(r.ord_coef[i] + Z95 * r.ord_bse[i]))
     for i, name in enumerate(r.selected)),
    key=lambda t: t[1])

fig, ax = plt.subplots(figsize=(9, 5.5))
for i, (name, orr, lo, hi) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color="#2b7bba", lw=2.5)          # 95% CI
    ax.plot(orr, i, "o", color="#1b4f72", ms=11)               # the OR itself
    # white box behind the label so it doesn't sit on a grid line
    ax.annotate(f"{orr:.1f}", (orr, i), textcoords="offset points",
                xytext=(0, 13), ha="center", fontsize=14,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

ax.axvline(1.0, color="black", ls="--", alpha=0.6)             # OR = 1 is "no effect"
ax.set_xscale("log")                                           # otherwise CoB dwarfs the rest
ax.set_yticks(range(len(rows)), labels=[name for name, *_ in rows])
ax.set_xlabel("Odds Ratio (log scale, per +1 std-dev) - 95% CI")
ax.set_title("Ordinal Model: Symptom Odds Ratios")

fig.tight_layout()
fig.savefig(f"{plot_style.FIG_DIR}/3_odds_ratios.png", bbox_inches="tight")
print("saved 3_odds_ratios.png")
