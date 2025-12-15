"""
Logistic Regression Classification Pipeline

This script generates a synthetic dataset, trains a Logistic Regression
model using a preprocessing pipeline, and evaluates it using
accuracy and ROC-AUC metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)


def generate_dataset(
    n_samples: int = 1000,
    n_features: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic binary classification dataset.

    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        random_state (int): Random seed

    Returns:
        pd.DataFrame: Feature dataframe with target column
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        class_sep=1.2,
        random_state=random_state,
    )

    feature_columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_columns)
    df["target"] = y

    return df


def train_model(X_train, y_train):
    """
    Train a Logistic Regression model using a pipeline.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained pipeline model
    """
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs"),
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_val, y_val) -> dict:
    """
    Evaluate the trained model.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels

    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
    }

    return metrics


def plot_results(model, X_val, y_val):
    """
    Plot ROC curve and confusion matrix.
    """
    RocCurveDisplay.from_estimator(model, X_val, y_val)
    plt.title("ROC Curve")
    plt.show()

    ConfusionMatrixDisplay.from_estimator(model, X_val, y_val)
    plt.title("Confusion Matrix")
    plt.show()


def main():
    """
    Main execution function.
    """
    # Generate data
    df = generate_dataset()

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        df.drop(columns="target"),
        df["target"],
        test_size=0.25,
        random_state=42,
        stratify=df["target"],
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_val, y_val)

    print("Model Evaluation Results")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    # Plot evaluation results
    plot_results(model, X_val, y_val)


if __name__ == "__main__":
    main()
