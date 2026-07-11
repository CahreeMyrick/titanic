from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocess import (
    build_column_dropper,
    build_feature_engineer,
    build_preprocessor,
)


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
        self.search_dir = self.outputs_dir / "search"

        for directory in (
            self.models_dir,
            self.predictions_dir,
            self.evaluation_dir,
            self.cv_dir,
            self.search_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def make_model(self, model_name: str):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = MODEL_REGISTRY[model_name]
        params = self.cfg["models"][model_name].get("params", {})
        return model_class(**params)

    def _make_shared_pipeline_steps(self) -> list[tuple[str, Any]]:
        return [
            ("feature_engineering", build_feature_engineer(self.cfg)),
            ("drop_columns", build_column_dropper(self.cfg)),
            ("preprocessor", build_preprocessor(self.cfg)),
        ]

    def make_ensemble(self) -> Pipeline:
        ensemble_cfg = self.cfg.get("ensemble", {})
        voting_cfg = ensemble_cfg.get("voting", {})

        if not voting_cfg.get("enabled", False):
            raise ValueError("Voting ensemble is disabled.")

        model_names = voting_cfg.get("models", [])
        voting_type = voting_cfg.get("type", "soft")
        if not model_names:
            raise ValueError("No voting models were configured.")

        if voting_type == "soft":
            for model_name in model_names:
                params = self.cfg["models"][model_name].get("params", {})
                if model_name == "svm" and not params.get("probability", False):
                    raise ValueError(
                        "Soft voting requires svm.params.probability: true."
                    )

        voting_model = VotingClassifier(
            estimators=[
                (model_name, self.make_model(model_name))
                for model_name in model_names
            ],
            voting=voting_type,
        )

        return Pipeline(
            self._make_shared_pipeline_steps() + [("model", voting_model)]
        )

    def make_pipeline(self, model_name: str) -> Pipeline:
        if model_name == "ensemble":
            return self.make_ensemble()

        return Pipeline(
            self._make_shared_pipeline_steps()
            + [("model", self.make_model(model_name))]
        )

    def get_model_names(self) -> list[str]:
        if self.cfg.get("ensemble", {}).get("enabled", False):
            return ["ensemble"]
        if self.mode == "all":
            return list(self.cfg["models"].keys())
        return [self.active_model]

    def prepare_train_data(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        X = data.drop(columns=[self.target])
        y = data[self.target]
        return X, y

    @staticmethod
    def prepare_test_data(data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    def _make_cv(self, folds: int) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.random_state,
        )

    def run_hyperparameter_search(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Pipeline:
        search_cfg = self.cfg["training"].get("hyperparameter_search", {})
        method = search_cfg.get("method", "grid")
        folds = search_cfg.get("folds", 5)
        scoring = search_cfg.get("scoring", "accuracy")
        n_iter = search_cfg.get("n_iter", 20)

        param_grid = self.cfg["models"][model_name].get("search_params", {})
        if not param_grid:
            raise ValueError(f"No search_params defined for model: {model_name}")

        pipeline = self.make_pipeline(model_name)
        cv = self._make_cv(folds)

        common = {
            "estimator": pipeline,
            "cv": cv,
            "scoring": scoring,
            "n_jobs": -1,
            "refit": True,
            "error_score": "raise",
        }

        if method == "grid":
            search = GridSearchCV(
                param_grid=param_grid,
                **common,
            )
        elif method == "random":
            search = RandomizedSearchCV(
                param_distributions=param_grid,
                n_iter=n_iter,
                random_state=self.random_state,
                **common,
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
        if model_name != "ensemble" and search_cfg.get("enabled", False):
            return self.run_hyperparameter_search(model_name, X, y)

        cv_cfg = self.cfg["training"].get("cross_validation", {})
        cv_enabled = cv_cfg.get("enabled", False)
        pipeline = self.make_pipeline(model_name)

        if cv_enabled:
            folds = cv_cfg.get("folds", 5)
            scoring = cv_cfg.get("scoring", "accuracy")
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=self._make_cv(folds),
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
            )

            cv_output_cfg = self.cfg["outputs"].get("cross_validation", {})
            if cv_output_cfg.get("save", False):
                self.save_cv_results(
                    model_name=model_name,
                    scores=scores,
                    scoring=scoring,
                    folds=folds,
                )

            if cv_output_cfg.get("print", True):
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
                print(f"Min:  {scores.min():.4f}")
                print(f"Max:  {scores.max():.4f}")
        else:
            test_size = self.cfg["training"].get("test_size", 0.2)
            stratify = (
                y if self.cfg["training"].get("stratify", True) else None
            )

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
            report = classification_report(
                y_val,
                val_preds,
                zero_division=0,
            )

            eval_cfg = self.cfg["outputs"].get("evaluation", {})
            if eval_cfg.get("print", True):
                print()
                print("=" * 60)
                print(f"Experiment: {self.name}")
                print(f"Model: {model_name}")
                print("=" * 60)
                print(f"Validation Accuracy: {accuracy:.4f}")
                print()
                print(report)

            if eval_cfg.get("save", False):
                self.save_evaluation(model_name, accuracy, report)

        # Pipeline.fit invokes the feature engineer's fit_transform method.
        # With add_group_survival enabled, training rows receive inner
        # out-of-fold encodings; the fitted mappings are retained for test.
        pipeline.fit(X, y)
        return pipeline

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, Pipeline]:
        return {
            model_name: self.train_one_model(model_name, X, y)
            for model_name in self.get_model_names()
        }

    @staticmethod
    def get_predictions(model: Pipeline, X_test: pd.DataFrame):
        return model.predict(X_test)

    def save_model(self, model: Pipeline, model_name: str) -> None:
        path = self.models_dir / f"{self.name}_{model_name}.joblib"
        joblib.dump(model, path)
        print(f"Saved model to: {path}")

    def save_predictions(
        self,
        passenger_ids: pd.Series,
        preds,
        model_name: str,
    ) -> None:
        submission = pd.DataFrame(
            {
                self.id_column: passenger_ids,
                self.target: np.asarray(preds, dtype=int),
            }
        )

        if submission[self.id_column].duplicated().any():
            raise ValueError("Submission contains duplicate passenger IDs.")
        if not set(submission[self.target].unique()).issubset({0, 1}):
            raise ValueError("Submission predictions must contain only 0 and 1.")

        path = self.predictions_dir / f"{self.name}_{model_name}_submission.csv"
        submission.to_csv(path, index=False)
        print(f"Saved predictions to: {path}")

    def save_evaluation(
        self,
        model_name: str,
        accuracy: float,
        report: str,
    ) -> None:
        path = self.evaluation_dir / f"{self.name}_{model_name}_evaluation.txt"
        with path.open("w", encoding="utf-8") as file:
            file.write(f"Experiment: {self.name}\n")
            file.write(f"Model: {model_name}\n\n")
            file.write(f"Validation Accuracy: {accuracy:.6f}\n\n")
            file.write(report)
        print(f"Saved evaluation to: {path}")

    def save_cv_results(
        self,
        model_name: str,
        scores,
        scoring: str,
        folds: int,
    ) -> None:
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
        with path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)
        print(f"Saved CV results to: {path}")

    def save_search_results(self, model_name: str, search) -> None:
        output_cfg = self.cfg["outputs"].get("hyperparameter_search", {})
        if not output_cfg.get("save", False):
            return

        results = {
            "experiment": self.name,
            "model": model_name,
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        }
        path = self.search_dir / f"{self.name}_{model_name}_search.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)
        print(f"Saved search results to: {path}")

    def run(self) -> None:
        train_df = pd.read_csv(self.cfg["paths"]["train_data"])
        test_df = pd.read_csv(self.cfg["paths"]["test_data"])

        passenger_ids = test_df[self.id_column].copy()
        X, y = self.prepare_train_data(train_df)
        X_test = self.prepare_test_data(test_df)

        X = X.drop(columns=[self.id_column], errors="ignore")
        X_test = X_test.drop(columns=[self.id_column], errors="ignore")

        trained_models = self.train(X, y)

        for model_name, model in trained_models.items():
            predictions = self.get_predictions(model, X_test)

            if self.cfg["outputs"].get("save_predictions", True):
                self.save_predictions(
                    passenger_ids,
                    predictions,
                    model_name,
                )

            if self.cfg["outputs"].get("save_model", False):
                self.save_model(model, model_name)
