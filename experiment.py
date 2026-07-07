from __future__ import annotations

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold,  GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocess import build_preprocessor, prepare_features

from xgboost import XGBClassifier


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
        self.mode = cfg["experiment"].get("mode", "single")
        self.target = cfg["experiment"]["target"]
        self.id_column = cfg["experiment"]["id_column"]
        self.random_state = cfg["experiment"].get("random_state", 42)

        self.outputs_dir = Path(cfg["paths"]["outputs_dir"])
        self.models_dir = self.outputs_dir / "models"
        self.predictions_dir = self.outputs_dir / "predictions"
        self.evaluation_dir = self.outputs_dir / "evaluation"
        self.cv_dir = self.outputs_dir / "cv"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.cv_dir.mkdir(parents=True, exist_ok=True)

        self.search_dir = self.outputs_dir / "search"

        self.search_dir.mkdir(parents=True, exist_ok=True)

    def make_model(self, model_name: str):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = MODEL_REGISTRY[model_name]
        params = self.cfg["models"][model_name].get("params", {})

        return model_class(**params)

    def make_pipeline(self, model_name: str) -> Pipeline:
        preprocessor = build_preprocessor(self.cfg)
        model = self.make_model(model_name)

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
         )

    def get_model_names(self) -> list[str]:
        if self.mode == "all":
            return list(self.cfg["models"].keys())

        return [self.active_model]

    def prepare_train_data(self, data: pd.DataFrame):
        data = prepare_features(data, self.cfg)

        X = data.drop(columns=[self.target])
        y = data[self.target]

        return X, y

    def prepare_test_data(self, data: pd.DataFrame):
        data = prepare_features(data, self.cfg)
        return data
    
    def run_hyperparameter_search(
    self,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    ):
        search_cfg = self.cfg["training"].get("hyperparameter_search", {})

        method = search_cfg.get("method", "grid")
        folds = search_cfg.get("folds", 5)
        scoring = search_cfg.get("scoring", "accuracy")
        n_iter = search_cfg.get("n_iter", 20)

        param_grid = self.cfg["models"][model_name].get("search_params", {})

        if not param_grid:
            raise ValueError(f"No search_params defined for model: {model_name}")

        pipeline = self.make_pipeline(model_name)

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.random_state,
        )

        if method == "grid":
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )

        elif method == "random":
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=self.random_state,
                n_jobs=-1,
            )

        else:
            raise ValueError(f"Unknown search method: {method}")

        search.fit(X, y)

        print()
        print("=" * 60)
        print(f"Hyperparameter Search: {model_name}")
        print("=" * 60)
        print(f"Best Score: {search.best_score_:.4f}")
        print(f"Best Params: {search.best_params_}")

        self.save_search_results(model_name, search)

        return search.best_estimator_

    def train_one_model(
    self,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:

        search_cfg = self.cfg["training"].get("hyperparameter_search", {})

        if search_cfg.get("enabled", False):
            return self.run_hyperparameter_search(model_name, X, y)

        cv_cfg = self.cfg["training"].get("cross_validation", {})
        cv_enabled = cv_cfg.get("enabled", False)

        pipeline = self.make_pipeline(model_name)

        if cv_enabled:
            folds = cv_cfg.get("folds", 5)
            scoring = cv_cfg.get("scoring", "accuracy")

            cv = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=self.random_state,
            )

            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
            )

            if self.cfg["outputs"].get("cross_validation", {}).get("print", True):

                self.save_cv_results(
                    model_name=model_name,
                    scores=scores,
                    scoring=scoring,
                    folds=folds,
                )

            print()
            print("=" * 60)
            print(f"Experiment: {self.name}")
            print(f"Model: {model_name}")
            print("=" * 60)
            print()
            print(f"Cross Validation ({folds}-fold, scoring={scoring}):")
            print(f"Scores: {scores}")
            print(f"Mean: {scores.mean():.4f}")
            print(f"Std:  {scores.std():.4f}")

        else:
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

            pipeline.fit(X_train, y_train)

            val_preds = pipeline.predict(X_val)

            accuracy = accuracy_score(y_val, val_preds)
            report = classification_report(y_val, val_preds)

            print()
            print("=" * 60)
            print(f"Experiment: {self.name}")
            print(f"Model: {model_name}")
            print("=" * 60)

            if self.cfg["outputs"].get("evaluation", {}).get("print", True):
                print()
                print("Validation Accuracy:")
                print(accuracy)

                print()
                print("Classification Report:")
                print(report)

            self.save_evaluation(
                model_name=model_name,
                accuracy=accuracy,
                report=report,
            )

        pipeline.fit(X, y)

        return pipeline

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Pipeline]:
        trained_models = {}

        for model_name in self.get_model_names():
            model = self.train_one_model(model_name, X, y)
            trained_models[model_name] = model

        return trained_models

    def get_predictions(self, model: Pipeline, X_test: pd.DataFrame):
        return model.predict(X_test)

    def save_model(self, model: Pipeline, model_name: str):
        path = self.models_dir / f"{self.name}_{model_name}.joblib"
        joblib.dump(model, path)
        print(f"Saved model to: {path}")

    def save_predictions(
        self,
        passenger_ids: pd.Series,
        preds,
        model_name: str,
    ):
        submission = pd.DataFrame(
            {
                self.id_column: passenger_ids,
                self.target: preds,
            }
        )

        path = self.predictions_dir / f"{self.name}_{model_name}_submission.csv"
        submission.to_csv(path, index=False)

        print(f"Saved predictions to: {path}")

        
    def save_evaluation( self,
        model_name: str,
        accuracy: float,
        report: str,
    ):
        eval_cfg = self.cfg["outputs"].get("evaluation", {})

        if not eval_cfg.get("save", False):
            return

        path = (
            self.evaluation_dir
            / f"{self.name}_{model_name}_evaluation.txt"
        )

        with open(path, "w") as f:
            f.write(f"Experiment: {self.name}\n")
            f.write(f"Model: {model_name}\n\n")

            f.write(f"Validation Accuracy: {accuracy:.6f}\n\n")

            f.write("Classification Report\n")
            f.write("=====================\n\n")
            f.write(report)

        
        print(f"Saved evaluation to: {path}") 

    
    def save_cv_results(
        self,
        model_name: str,
        scores,
        scoring: str,
        folds: int,
    ):
        cv_output_cfg = self.cfg["outputs"].get("cross_validation", {})

        if not cv_output_cfg.get("save", False):
            return

        results = {
            "experiment": self.name,
            "model": model_name,
            "scoring": scoring,
            "folds": folds,
            "scores": scores.tolist(),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

        path = self.cv_dir / f"{self.name}_{model_name}_cv.json"

        with open(path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Saved CV results to: {path}")

    def save_search_results(
    self,
    model_name: str,
    search,
    ):
        search_cfg = self.cfg["training"].get("hyperparameter_search", {})

        if not search_cfg.get("save", True):
            return

        results = {
            "experiment": self.name,
            "model": model_name,
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        }

        path = self.search_dir / f"{self.name}_{model_name}_search.json"

        with open(path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Saved search results to: {path}")

    def run(self):
        train_path = self.cfg["paths"]["train_data"]
        test_path = self.cfg["paths"]["test_data"]

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        passenger_ids = test_df[self.id_column]

        X, y = self.prepare_train_data(train_df)
        X_test = self.prepare_test_data(test_df)

        if self.id_column in X.columns:
            X = X.drop(columns=[self.id_column])

        if self.id_column in X_test.columns:
            X_test = X_test.drop(columns=[self.id_column])

        trained_models = self.train(X, y)

        for model_name, model in trained_models.items():
            preds = self.get_predictions(model, X_test)

            if self.cfg["outputs"].get("save_predictions", True):
                self.save_predictions(passenger_ids, preds, model_name)

            if self.cfg["outputs"].get("save_model", False):
                self.save_model(model, model_name)
