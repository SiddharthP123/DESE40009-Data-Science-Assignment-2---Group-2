
import matplotlib.pyplot as plt

import plot_style
from model import prepare

plot_style.apply()
r = prepare()

steps = range(1, len(r.history) + 1)
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.plot(steps, [a for _, a in r.history], marker="o", lw=2.5, color="#2b7bba")
for i, (feat, acc) in enumerate(r.history, 1):
    ax.annotate(f"{feat}\n{acc:.3f}", (i, acc), textcoords="offset points",
                xytext=(0, 16), ha="center", fontsize=15)

ax.axhline(r.no_info_rate, color="red", ls="--", lw=1.5,
           label=f"baseline {r.no_info_rate:.2f}")
ax.set_xticks(list(steps))
ax.set_xlabel("Features selected")
ax.set_ylabel("CV accuracy")
ax.set_ylim(r.no_info_rate - 0.05, 1.0)
ax.set_title("Forward Selection Path")
ax.legend(loc="lower right")

fig.tight_layout()
fig.savefig(f"{plot_style.FIG_DIR}/5_forward_selection.png", bbox_inches="tight")
print("saved 5_forward_selection.png")
