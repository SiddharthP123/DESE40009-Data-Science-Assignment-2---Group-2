
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

DATA_FILE = "cancer patient data sets.csv"
FEATURES = ["Chest Pain", "Coughing of Blood", "Shortness of Breath",
            "Wheezing", "Dry Cough", "Frequent Cold"]
ORDER = ["Low", "Medium", "High"]
CODE = {"Low": 0, "Medium": 1, "High": 2}
PALETTE = {"Low": "#2e8b57", "Medium": "#e0a200", "High": "#c0392b"}

MIN_IMPROVEMENT = 0.005   # min CV gain before we bother adding another feature
Z95 = 1.959963985         # 95% CI z-value
SEEDS = [42, 1, 7, 13, 99]


def _cv_accuracy(X_train, y_train, feature_names, seed=42):
    """Mean 5-fold CV accuracy for a logit on the given features."""
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_validate(pipe, X_train[feature_names], y_train, cv=cv,
                          scoring="accuracy")["test_score"].mean()


def prepare():
    """Run the whole pipeline once and hand everything back in one namespace."""
    df = pd.read_csv(DATA_FILE)
    X, y = df[FEATURES], df["Level"]
    counts = y.value_counts()
    no_info_rate = counts.max() / len(y)

    # check for multicollinearity before fitting anything
    X_vif = sm.add_constant(StandardScaler().fit_transform(X))
    vif = {FEATURES[i - 1]: variance_inflation_factor(X_vif, i)
           for i in range(1, X_vif.shape[1])}

    # 80/20, keep the class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # greedily add whichever feature helps most; stop once it stops helping.
    # ends up with the smallest useful set and drops the redundant symptoms.
    selected, remaining, best_cv, history = [], list(FEATURES), 0.0, []
    while remaining:
        cand, cand_cv = max(
            ((f, _cv_accuracy(X_train, y_train, selected + [f])) for f in remaining),
            key=lambda t: t[1])
        if cand_cv - best_cv >= MIN_IMPROVEMENT:
            selected.append(cand)
            remaining.remove(cand)
            best_cv = cand_cv
            history.append((cand, cand_cv))
        else:
            break

    # scale on the training data only so the test set doesn't leak in
    scaler = StandardScaler().fit(X_train[selected])
    Xtr = scaler.transform(X_train[selected])
    Xte = scaler.transform(X_test[selected])
    ytr = y_train.map(CODE).values
    yte = y_test.map(CODE).values

    # the actual model. keep any convergence warnings around instead of hiding them.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ord_res = OrderedModel(ytr, Xtr, distr="logit").fit(method="bfgs", disp=False)
        fit_warnings = sorted({f"{w.category.__name__}: {w.message}" for w in caught})

    k = len(selected)
    ord_coef = np.asarray(ord_res.params)[:k]
    ord_bse = np.asarray(ord_res.bse)[:k]
    ord_p = np.asarray(ord_res.pvalues)[:k]

    # rough proportional-odds check: fit the two cut-points separately and see
    # if the slopes line up (Brant-style). big gap = assumption is shaky.
    coefA = LogisticRegression(max_iter=1000).fit(Xtr, (ytr >= 1).astype(int)).coef_[0]
    coefB = LogisticRegression(max_iter=1000).fit(Xtr, (ytr >= 2).astype(int)).coef_[0]

    # predict the held-out test set
    test_proba = ord_res.model.predict(ord_res.params, exog=Xte)
    test_code = test_proba.argmax(axis=1)
    test_pred = pd.Series(test_code).map({v: k for k, v in CODE.items()})
    cm = confusion_matrix(yte, test_code, labels=[0, 1, 2])

    # sanity check: does a plain multinomial land in the same place?
    mult = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, random_state=42))])
    mult.fit(X_train[selected], y_train)
    mult_acc = accuracy_score(y_test, mult.predict(X_test[selected]))

    return SimpleNamespace(
        df=df, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        counts=counts, no_info_rate=no_info_rate, vif=vif,
        selected=selected, history=history,
        ord_res=ord_res, ord_coef=ord_coef, ord_bse=ord_bse, ord_p=ord_p,
        fit_warnings=fit_warnings, coefA=coefA, coefB=coefB,
        test_proba=test_proba, test_code=test_code, test_pred=test_pred,
        yte=yte, cm=cm, mult_acc=mult_acc,
    )


def repeated_cv(r):
    """5-fold ordinal CV over a few seeds, so it's not just one lucky split."""
    accs = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in cv.split(r.X_train[r.selected], r.y_train):
            sc = StandardScaler().fit(r.X_train[r.selected].iloc[tr])
            Xt = sc.transform(r.X_train[r.selected].iloc[tr])
            Xv = sc.transform(r.X_train[r.selected].iloc[va])
            yt = r.y_train.map(CODE).values[tr]
            res = OrderedModel(yt, Xt, distr="logit").fit(method="bfgs", disp=False)
            pred = res.predict(Xv).argmax(axis=1)
            accs.append(accuracy_score(r.y_train.map(CODE).values[va], pred))
    return np.array(accs)


def _print_report(r):
    """Print the numbers we actually quote in the report."""
    print("=" * 72)
    print("ORDINAL LOGISTIC REGRESSION | Respiratory Symptoms -> Cancer Level")
    print("=" * 72)
    for c in ORDER:
        print(f"  {c:<8}{r.counts[c]:>5} ({r.counts[c] / len(r.df) * 100:.1f}%)")
    print(f"Baseline (majority class): {r.no_info_rate:.4f}\n")

    print("VIF (multicollinearity; >5 moderate, >10 serious)")
    for f, v in sorted(r.vif.items(), key=lambda kv: -kv[1]):
        print(f"  {f:<22}{v:>6.2f}")

    print(f"\nForward selection kept {len(r.selected)}: {r.selected}")
    print(f"Dropped: {[f for f in FEATURES if f not in r.selected]}")

    print(f"\nOrdinal model | McFadden pseudo-R^2 = {r.ord_res.prsquared:.4f}")
    print("  warnings:", r.fit_warnings or "none (clean convergence)")
    print(f"  {'Symptom':<22}{'OR':>9}{'95% CI':>20}{'p':>11}")
    for i, name in enumerate(r.selected):
        orr = np.exp(r.ord_coef[i])
        lo = np.exp(r.ord_coef[i] - Z95 * r.ord_bse[i])
        hi = np.exp(r.ord_coef[i] + Z95 * r.ord_bse[i])
        print(f"  {name:<22}{orr:>9.2f}  [{lo:>6.2f}, {hi:>6.2f}]{r.ord_p[i]:>11.1e}")

    # if the two slopes are close the single OR is fine; a big gap means it's
    # only an approximation for that symptom (Wheezing is the one to watch).
    print("\nProportional-odds check (slope per cumulative split)")
    print(f"  {'Symptom':<22}{'>=Med':>8}{'>=High':>9}{'|diff|':>9}")
    for i, name in enumerate(r.selected):
        diff = abs(r.coefA[i] - r.coefB[i])
        flag = "" if diff < 1.0 else "  diverges"
        print(f"  {name:<22}{r.coefA[i]:>8.2f}{r.coefB[i]:>9.2f}{diff:>9.2f}{flag}")

    acc = accuracy_score(r.yte, r.test_code)
    print(f"\nTest accuracy {acc:.4f} (ordinal) vs {r.mult_acc:.4f} (multinomial)")
    print("Per-class report:")
    print(classification_report(r.y_test, r.test_pred, labels=ORDER, digits=3))

    errors = r.cm.sum() - np.trace(r.cm)
    non_adj = r.cm[0, 2] + r.cm[2, 0]
    print(f"Errors: {errors} | severe (Low<->High): {non_adj} "
          f"({non_adj / len(r.y_test) * 100:.1f}%)")


if __name__ == "__main__":
    _print_report(prepare())
