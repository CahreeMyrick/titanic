from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def fill_age_by_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    df["Age"] = df["Age"].fillna(df["Age"].median())

    return df


def fill_embarked(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


def fill_fare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    return df


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def add_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def add_family_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def group(size: int) -> str:
        if size == 1:
            return "Alone"
        elif size <= 4:
            return "Small"
        else:
            return "Large"

    df["FamilyGroup"] = df["FamilySize"].apply(group)
    return df


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    rare_titles = [
        "Lady", "Countess", "Capt", "Col", "Don",
        "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"
    ]

    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    df["Title"] = df["Title"].fillna("Unknown")

    return df


def add_surname(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Surname"] = df["Name"].str.split(",").str[0]
    return df


def add_ticket_prefix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TicketPrefix"] = (
        df["Ticket"]
        .astype(str)
        .str.replace(r"\d+", "", regex=True)
        .str.replace(".", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.strip()
    )

    df["TicketPrefix"] = df["TicketPrefix"].replace("", "None")

    return df


def add_ticket_group_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ticket_counts = df["Ticket"].value_counts()
    df["TicketGroupSize"] = df["Ticket"].map(ticket_counts)

    return df


def add_has_cabin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


def add_deck(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Deck"] = df["Cabin"].str[0]
    df["Deck"] = df["Deck"].fillna("Unknown")

    return df


def add_fare_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FareLog"] = np.log1p(df["Fare"])
    return df


def add_age_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
        include_lowest=True
    )

    return df


def add_fare_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["FareBin"] = pd.qcut(
        df["Fare"],
        q=4,
        labels=["Low", "Medium", "High", "VeryHigh"],
        duplicates="drop"
    )

    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["FemaleFirstClass"] = (
        (df["Sex"] == "female") & (df["Pclass"] == 1)
    ).astype(int)

    df["Child"] = (df["Age"] < 12).astype(int)

    df["Mother"] = (
        (df["Sex"] == "female") &
        (df["Parch"] > 0) &
        (df["Age"] > 18) &
        (df["Title"] != "Miss")
    ).astype(int)

    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    return df


FEATURE_FUNCTIONS = {
    "fill_age_by_group": fill_age_by_group,
    "fill_embarked": fill_embarked,
    "fill_fare": fill_fare,
    "add_family_size": add_family_size,
    "add_is_alone": add_is_alone,
    "add_family_group": add_family_group,
    "add_title": add_title,
    "add_surname": add_surname,
    "add_ticket_prefix": add_ticket_prefix,
    "add_ticket_group_size": add_ticket_group_size,
    "add_has_cabin": add_has_cabin,
    "add_deck": add_deck,
    "add_fare_log": add_fare_log,
    "add_age_bin": add_age_bin,
    "add_fare_bin": add_fare_bin,
    "add_interactions": add_interactions,
}


def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    engineer_cfg = cfg["preprocessing"].get("engineer_features", {})

    if not engineer_cfg.get("enabled", False):
        return df

    features = engineer_cfg.get("features", [])

    for feature_name in features:
        if feature_name not in FEATURE_FUNCTIONS:
            raise ValueError(f"Unknown feature engineering function: {feature_name}")

        df = FEATURE_FUNCTIONS[feature_name](df)

    return df


def drop_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data.drop(columns=columns, errors="ignore")


def prepare_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    df = engineer_features(df, cfg)
    df = drop_columns(df, cfg["preprocessing"].get("drop_columns", []))

    return df


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    preprocessing_cfg = cfg["preprocessing"]

    numerical_columns = preprocessing_cfg.get("numerical", [])
    categorical_columns = preprocessing_cfg.get("categorical", [])

    numerical_impute_strategy = preprocessing_cfg.get("impute", {}).get(
        "numerical", "median"
    )
    categorical_impute_strategy = preprocessing_cfg.get("impute", {}).get(
        "categorical", "most_frequent"
    )

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