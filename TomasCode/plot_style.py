"""One place for the plot styling so every figure comes out consistent."""

import os
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = "figures"


def apply():
    """Plain look, fonts bumped up a bit so they survive being shrunk to one column."""
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 15,
        "axes.titleweight": "normal",
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })
