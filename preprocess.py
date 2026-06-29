from __future__ import annotations

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def drop_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data.drop(columns=columns, errors="ignore")


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    preprocessing_cfg = cfg["preprocessing"]

    numerical_columns = preprocessing_cfg.get("numerical", [])
    categorical_columns = preprocessing_cfg.get("categorical", [])

    numerical_impute_strategy = preprocessing_cfg.get("impute", {}).get("numerical", "median")
    categorical_impute_strategy = preprocessing_cfg.get("impute", {}).get("categorical", "most_frequent")

    scale_enabled = preprocessing_cfg.get("scale", {}).get("enabled", True)
    encode_enabled = preprocessing_cfg.get("encode", {}).get("enabled", True)

    numerical_steps = [
        ("imputer", SimpleImputer(strategy=numerical_impute_strategy))
    ]

    if scale_enabled:
        numerical_steps.append(("scaler", StandardScaler()))

    numerical_pipeline = Pipeline(steps=numerical_steps)

    categorical_steps = [
        ("imputer", SimpleImputer(strategy=categorical_impute_strategy))
    ]

    if encode_enabled:
        categorical_steps.append(
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        )

    categorical_pipeline = Pipeline(steps=categorical_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor
