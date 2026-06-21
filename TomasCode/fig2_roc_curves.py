
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import plot_style
from model import prepare, ORDER, PALETTE

plot_style.apply()
r = prepare()

# test_proba columns are in code order 0/1/2 = Low/Medium/High
y_bin = label_binarize(r.yte, classes=[0, 1, 2])

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for i, cls in enumerate(ORDER):
    fpr, tpr, _ = roc_curve(y_bin[:, i], r.test_proba[:, i])
    ax.plot(fpr, tpr, lw=2.5, color=PALETTE[cls], label=f"{cls} (AUC = {auc(fpr, tpr):.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (Ordinal Model, One-vs-Rest)")
ax.legend(loc="lower right")

fig.tight_layout()
fig.savefig(f"{plot_style.FIG_DIR}/2_roc_curves.png", bbox_inches="tight")
print("saved 2_roc_curves.png")
