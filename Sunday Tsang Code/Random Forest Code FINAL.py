import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import os
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


# load the dataset here
df = pd.read_csv("cancer patient data sets.csv")

features = ["Alcohol use", "Smoking", "Balanced Diet",
            "Obesity", "Genetic Risk", "chronic Lung Disease"]

X = df[features]
y = df["Level"]

# encode target labels
le = LabelEncoder()
y = le.fit_transform(y)


# split randomly 70% training, 15% validation and 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(
X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# filter using SMOTE to training set only
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)


# hyperparameter tuning graphs
# maximum depth
depths = range(1, 20)
train_acc = []
val_acc = []

for d in depths:
    rf = RandomForestClassifier(max_depth=d, n_estimators=200, random_state=42)
    rf.fit(X_train_sm, y_train_sm)
    train_acc.append(accuracy_score(y_train_sm, rf.predict(X_train_sm)))
    val_acc.append(accuracy_score(y_val, rf.predict(X_val)))

plt.figure(figsize=(7,5))
plt.plot(depths, train_acc, label="Training Accuracy")
plt.plot(depths, val_acc, label="Validation Accuracy")
plt.title("Max Depth vs Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# minimum sample split
splits = range(2, 20)
train_acc = []
val_acc = []

for s in splits:
    rf = RandomForestClassifier(min_samples_split=s, n_estimators=200, random_state=42)
    rf.fit(X_train_sm, y_train_sm)
    train_acc.append(accuracy_score(y_train_sm, rf.predict(X_train_sm)))
    val_acc.append(accuracy_score(y_val, rf.predict(X_val)))

plt.figure(figsize=(7,5))
plt.plot(splits, train_acc, label="Training Accuracy")
plt.plot(splits, val_acc, label="Validation Accuracy")
plt.title("Min Samples Split vs Accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# minimum impurity increase
impurities = np.linspace(0, 0.05, 20)
train_acc = []
val_acc = []

for imp in impurities:
    rf = RandomForestClassifier(min_impurity_decrease=imp, n_estimators=200, random_state=42)
    rf.fit(X_train_sm, y_train_sm)
    train_acc.append(accuracy_score(y_train_sm, rf.predict(X_train_sm)))
    val_acc.append(accuracy_score(y_val, rf.predict(X_val)))

plt.figure(figsize=(7,5))
plt.plot(impurities, train_acc, label="Training Accuracy")
plt.plot(impurities, val_acc, label="Validation Accuracy")
plt.title("Min Impurity Decrease vs Accuracy")
plt.xlabel("min_impurity_decrease")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# n estimators 
trees = range(10, 310, 20)
train_acc = []
val_acc = []

for t in trees:
    rf = RandomForestClassifier(n_estimators=t, random_state=42)
    rf.fit(X_train_sm, y_train_sm)
    train_acc.append(accuracy_score(y_train_sm, rf.predict(X_train_sm)))
    val_acc.append(accuracy_score(y_val, rf.predict(X_val)))

plt.figure(figsize=(7,5))
plt.plot(trees, train_acc, label="Training Accuracy")
plt.plot(trees, val_acc, label="Validation Accuracy")
plt.title("N Estimators vs Accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# accuracy comparison tables

models = {
    "Model 1": RandomForestClassifier(max_depth=2, min_samples_split=5,
                                      min_impurity_decrease=0.1, n_estimators=200, random_state=42),

    "Model 2": RandomForestClassifier(max_depth=7, min_samples_split=5,
                                      min_impurity_decrease=0, n_estimators=200, random_state=42),

    "Model 3": RandomForestClassifier(max_depth=7, min_samples_split=2,
                                      min_impurity_decrease=0, n_estimators=200, random_state=42),

    "Model 4": RandomForestClassifier(max_depth=7, min_samples_split=2,
                                      min_impurity_decrease=0, n_estimators=20, random_state=42)
}

results = []

for name, clf in models.items():
    clf.fit(X_train_sm, y_train_sm)

    train_acc = accuracy_score(y_train_sm, clf.predict(X_train_sm))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    results.append([name, train_acc, val_acc, test_acc])

df_results = pd.DataFrame(results, columns=["Model", "Training", "Validation", "Test"])
print(df_results)


# show random forest model (updated depending on parameters)
model = RandomForestClassifier(
    max_depth=7,
    n_estimators=200,
    min_samples_split=2,
    min_impurity_decrease=0
)

model.fit(X_train_sm, y_train_sm)
# save a PNG for every tree in the forest
tree_dir = "forest_trees"
class_names = [str(label) for label in le.classes_]

for i, tree in enumerate(model.estimators_):
    if (i+1) % 10 != 0:
        continue
    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(
        tree,
        feature_names=features,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=9,
    )
    ax.set_title(f"Decision Tree {i + 1} of {len(model.estimators_)}")
    fig.tight_layout()
    #fig.savefig(os.path.join(tree_dir, f"tree_{i + 1:03d}.png"), dpi=100, bbox_inches="tight") #Used to save the figure for every tenth estimator
    plt.show() #Used to show/see tree for every tenth estimator
    #plt.close(fig)


# generate learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_sm, y_train_sm, cv=5, scoring='accuracy'
)

plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Accuracy")
plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation Accuracy")
plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# final results table: accuracy, precision and recall
def get_metrics(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, average='macro', zero_division=0)
    rec = recall_score(y, preds, average='macro', zero_division=0)
    return acc, prec, rec

train_acc, train_prec, train_rec = get_metrics(model, X_train_sm, y_train_sm)
val_acc, val_prec, val_rec = get_metrics(model, X_val, y_val)
test_acc, test_prec, test_rec = get_metrics(model, X_test, y_test)

df_final = pd.DataFrame({
    "Dataset": ["Training", "Validation", "Testing"],
    "Accuracy": [train_acc, val_acc, test_acc],
    "Precision": [train_prec, val_prec, test_prec],
    "Recall": [train_rec, val_rec, test_rec]
})

print("\n=== TABLE 4: Final Model Results ===")
print(df_final)


# confusion matrix for the test set
cm = confusion_matrix(y_test, model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.show()


# feature importance plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(7,5))
plt.bar(range(len(features)), importances[indices])
plt.xticks(range(len(features)), np.array(features)[indices], rotation=45)
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.show()