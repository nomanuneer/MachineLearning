import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

np.random.seed(42)

X, y = make_classification(
    n_samples=1000,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.2,
    random_state=42,
)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = y
df.head()


X_train, X_val, y_train, y_val = train_test_split(
    df.drop(columns="target"), df["target"], test_size=0.25, random_state=42, stratify=df["target"]
)

# Standardize + logistic regression
log_reg = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, solver="lbfgs")
)
log_reg.fit(X_train, y_train)

y_val_pred = log_reg.predict(X_val)
y_val_proba = log_reg.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_val_pred)
roc = roc_auc_score(y_val, y_val_proba)
print(f"Accuracy: {acc:.3f}  |  ROC-AUC: {roc:.3f}")


fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, ax=ax[0], colorbar=False)
ax[0].set_title("Confusion Matrix")

RocCurveDisplay.from_predictions(y_val, y_val_proba, ax=ax[1])
ax[1].set_title("ROC Curve")

plt.tight_layout()
plt.show()


new_points = pd.DataFrame(
    [[0.5, -1.0, 0.3, 1.2, -0.7, 0.1],
     [-1.2, 0.8, -0.5, -0.9, 0.6, -0.2]],
    columns=[f"f{i}" for i in range(X.shape[1])]
)
proba = log_reg.predict_proba(new_points)[:, 1]
pd.DataFrame({"probability_class_1": proba})