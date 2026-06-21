
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plot_style
from model import prepare, ORDER

plot_style.apply()
r = prepare()

cm = r.cm
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=ORDER, yticklabels=ORDER, ax=axes[0],
            annot_kws={"size": 15})
axes[0].set_title("Confusion Matrix (counts)")

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            xticklabels=ORDER, yticklabels=ORDER, ax=axes[1],
            annot_kws={"size": 15})
axes[1].set_title("Row-Normalised (recall)")

for ax in axes:
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

fig.suptitle("Ordinal Model - Test Confusion Matrices")
fig.tight_layout()
fig.savefig(f"{plot_style.FIG_DIR}/1_confusion_matrix.png", bbox_inches="tight")
print("saved 1_confusion_matrix.png")
