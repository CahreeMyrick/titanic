from __future__ import annotations

import joblib
import pandas as pd

from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocess import build_preprocessor, drop_columns


try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


class Experiment:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

        self.name = cfg["experiment"]["name"]
        self.active_model = cfg["experiment"]["active_model"]
        self.target = cfg["experiment"]["target"]
        self.id_column = cfg["experiment"]["id_column"]
        self.random_state = cfg["experiment"].get("random_state", 42)

        self.outputs_dir = Path(cfg["paths"]["outputs_dir"])
        self.models_dir = self.outputs_dir / "models"
        self.predictions_dir = self.outputs_dir / "predictions"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def make_model(self):
        if self.active_model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {self.active_model}")

        model_class = MODEL_REGISTRY[self.active_model]

        if model_class is None:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            )

        params = self.cfg["models"][self.active_model].get("params", {})

        return model_class(**params)

    def make_pipeline(self) -> Pipeline:
        preprocessor = build_preprocessor(self.cfg)
        model = self.make_model()

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def prepare_train_data(self, data: pd.DataFrame):
        drop_cols = self.cfg["preprocessing"].get("drop_columns", [])

        data = drop_columns(data, drop_cols)

        X = data.drop(columns=[self.target])
        y = data[self.target]

        return X, y

    def prepare_test_data(self, data: pd.DataFrame):
        drop_cols = self.cfg["preprocessing"].get("drop_columns", [])

        data = drop_columns(data, drop_cols)

        return data

    def train(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        test_size = self.cfg["training"].get("test_size", 0.2)
        stratify_enabled = self.cfg["training"].get("stratify", True)

        stratify = y if stratify_enabled else None

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        pipeline = self.make_pipeline()
        pipeline.fit(X_train, y_train)

        val_preds = pipeline.predict(X_val)

        print()
        print(f"Experiment: {self.name}")
        print(f"Model: {self.active_model}")
        print()
        print("Validation Accuracy:")
        print(accuracy_score(y_val, val_preds))
        print()
        print("Classification Report:")
        print(classification_report(y_val, val_preds))

        return pipeline

    def get_predictions(self, model: Pipeline, X_test: pd.DataFrame):
        return model.predict(X_test)

    def save_model(self, model: Pipeline):
        path = self.models_dir / f"{self.name}_{self.active_model}.joblib"
        joblib.dump(model, path)
        print(f"Saved model to: {path}")

    def save_predictions(self, passenger_ids: pd.Series, preds):
        submission = pd.DataFrame(
            {
                self.id_column: passenger_ids,
                self.target: preds,
            }
        )

        path = self.predictions_dir / f"{self.name}_{self.active_model}_submission.csv"
        submission.to_csv(path, index=False)

        print(f"Saved predictions to: {path}")
