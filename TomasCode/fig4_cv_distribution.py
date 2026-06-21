
import matplotlib.pyplot as plt
import seaborn as sns

import plot_style
from model import prepare, repeated_cv, SEEDS

plot_style.apply()
r = prepare()
accs = repeated_cv(r)   # 5 folds x each seed

fig, ax = plt.subplots(figsize=(7.5, 6))
sns.histplot(accs, bins=12, kde=True, color="#5499c7", ax=ax)
ax.axvline(accs.mean(), color="#1b4f72", lw=2, label=f"mean {accs.mean():.3f}")
ax.axvline(r.no_info_rate, color="red", ls="--", lw=1.5,
           label=f"baseline {r.no_info_rate:.2f}")

# start the axis near 0.30 so the bars aren't squashed against the right edge,
# but keep the 0.36 baseline line in view
ax.set_xlim(0.30, 0.96)
ax.annotate("all 25 fits >= 0.85", xy=(accs.min(), 0), xytext=(0.55, 0.9),
            textcoords="axes fraction", fontsize=14, color="#1b4f72",
            arrowprops=dict(arrowstyle="->", color="#1b4f72", alpha=0.7))

ax.set_xlabel("Fold accuracy")
ax.set_ylabel("Count")
ax.set_title(f"Repeated CV ({len(accs)} fits over {len(SEEDS)} seeds)")
ax.legend()

fig.tight_layout()
fig.savefig(f"{plot_style.FIG_DIR}/4_cv_distribution.png", bbox_inches="tight")
print("saved 4_cv_distribution.png")
