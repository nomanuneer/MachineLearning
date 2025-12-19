import os
import sys
import joblib
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass
class Config:
    DATA_PATH: str = "data/dataset.csv"
    MODEL_PATH: str = "artifacts/random_forest.pkl"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    N_ESTIMATORS: int = 300


config = Config()


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)



def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading dataset")
    if not os.path.exists(path):
        logger.error("Dataset not found")
        sys.exit(1)
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y



def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )
    return preprocessor



def build_model(preprocessor: ColumnTransformer) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )

    return pipeline



def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    logger.info("Training Random Forest model")
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model: Pipeline, X_test, y_test) -> None:
    logger.info("Evaluating model")
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    logger.info(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))


def save_model(model: Pipeline) -> None:
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    logger.info(f"Model saved at {config.MODEL_PATH}")



def main():
    logger.info("Starting training pipeline")

    df = load_data(config.DATA_PATH)
    X, y = split_features_target(df, target='target')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    pipeline = build_model(preprocessor)

    trained_model = train_model(pipeline, X_train, y_train)
    evaluate_model(trained_model, X_test, y_test)
    save_model(trained_model)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
