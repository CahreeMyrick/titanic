from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Leakage-safe Titanic feature engineering.

    The transformer supports both ordinary feature engineering and target-aware
    family/ticket survival encoding.

    Important behavior
    ------------------
    * ``fit`` learns statistics only from the data passed to that fit call.
    * ``fit_transform`` creates out-of-fold target encodings for training rows.
    * ``transform`` applies mappings learned during ``fit`` to validation/test
      rows.
    * Therefore, when this transformer is placed inside an sklearn Pipeline,
      outer cross-validation remains leakage-safe.

    Feature order matters. A typical target-aware order is:

    ``add_title -> fill_* -> add_family_size -> add_surname ->
    add_family_id -> add_passenger_group -> add_group_survival``.
    """

    TARGET_FEATURE = "add_group_survival"

    TARGET_OUTPUT_COLUMNS = (
        "FallbackSurvivalRate",
        "TicketSurvivalRate",
        "FamilySurvivalRate",
        "CombinedGroupSurvivalRate",
        "GroupSurvivalSignal",
        "TicketOutcomeCount",
        "FamilyOutcomeCount",
        "HasTicketOutcome",
        "HasFamilyOutcome",
    )

    SUPPORTED_FEATURES = {
        "add_title",
        "fill_embarked",
        "fill_fare",
        "fill_age_by_title_class",
        "fill_age_by_group",  # Backward-compatible alias.
        "add_family_size",
        "add_is_alone",
        "add_family_group",
        "add_surname",
        "add_family_id",
        "add_ticket_prefix",
        "add_ticket_group_size",
        "add_family_group_size",
        "add_fare_frequency",
        "add_has_cabin",
        "add_deck",
        "add_fare_log",
        "add_age_bin",
        "add_fare_bin",
        "add_female_first_class",
        "add_child",
        "add_mother",
        "add_fare_per_person",
        "add_fare_per_ticket",
        "add_passenger_group",
        "add_interactions",  # Backward-compatible bundled feature.
        TARGET_FEATURE,
    }

    def __init__(
        self,
        features: tuple[str, ...] = (),
        target_encoding_folds: int = 5,
        target_smoothing: float = 2.0,
        random_state: int = 42,
    ):
        # Store constructor parameters unchanged so sklearn.clone works.
        self.features = features
        self.target_encoding_folds = target_encoding_folds
        self.target_smoothing = target_smoothing
        self.random_state = random_state

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None):
        self._validate_input(X)
        self._validate_feature_names()
        self._validate_configuration()

        base = self._fit_and_transform_base(X)

        if self._uses_target_encoding:
            target = self._validate_target(y, X.index)
            self.target_maps_ = self._build_target_maps(base, target)

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        output_columns = list(base.columns)
        if self._uses_target_encoding:
            output_columns.extend(
                column
                for column in self.TARGET_OUTPUT_COLUMNS
                if column not in output_columns
            )
        self.output_columns_ = tuple(output_columns)
        self.is_fitted_ = True
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params,
    ) -> pd.DataFrame:
        del fit_params
        self.fit(X, y)

        base = self._transform_base(X)
        if not self._uses_target_encoding:
            return base

        target = self._validate_target(y, X.index)
        return self._add_oof_group_survival(base, target)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")
        self._validate_input(X)

        base = self._transform_base(X)
        if self._uses_target_encoding:
            base = self._apply_target_maps(base, self.target_maps_)
        return base

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        del input_features
        check_is_fitted(self, "is_fitted_")
        return np.asarray(self.output_columns_, dtype=object)

    # ------------------------------------------------------------------
    # Validation and feature execution
    # ------------------------------------------------------------------
    @property
    def _uses_target_encoding(self) -> bool:
        return self.TARGET_FEATURE in self.features

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "TitanicFeatureEngineer requires a pandas DataFrame."
            )

    def _validate_feature_names(self) -> None:
        unknown = sorted(set(self.features) - self.SUPPORTED_FEATURES)
        if unknown:
            raise ValueError(
                "Unknown feature engineering function(s): "
                + ", ".join(unknown)
            )

    def _validate_configuration(self) -> None:
        if self.target_encoding_folds < 2:
            raise ValueError("target_encoding_folds must be at least 2.")
        if self.target_smoothing < 0:
            raise ValueError("target_smoothing cannot be negative.")

        if self._uses_target_encoding:
            required_predecessors = {
                "add_title",
                "add_family_size",
                "add_surname",
                "add_family_id",
                "add_passenger_group",
            }
            missing = required_predecessors.difference(self.features)
            if missing:
                raise ValueError(
                    "add_group_survival requires these configured features: "
                    + ", ".join(sorted(missing))
                )

            target_position = self.features.index(self.TARGET_FEATURE)
            for predecessor in required_predecessors:
                if self.features.index(predecessor) > target_position:
                    raise ValueError(
                        f"'{predecessor}' must appear before "
                        f"'{self.TARGET_FEATURE}'."
                    )

    @staticmethod
    def _validate_target(
        y: pd.Series | np.ndarray | None,
        index: pd.Index,
    ) -> pd.Series:
        if y is None:
            raise ValueError(
                "Target y is required when add_group_survival is enabled."
            )

        values = np.asarray(y)
        if values.ndim != 1 or len(values) != len(index):
            raise ValueError("Target y must be one-dimensional and match X.")

        target = pd.Series(values, index=index, dtype=float)
        if target.isna().any():
            raise ValueError("Target y cannot contain missing values.")
        return target

    @staticmethod
    def _require_columns(
        df: pd.DataFrame,
        columns: Iterable[str],
        feature: str,
    ) -> None:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(
                f"Feature '{feature}' requires missing column(s): {missing}. "
                "Check feature order in the configuration."
            )

    def _fit_and_transform_base(self, X: pd.DataFrame) -> pd.DataFrame:
        working = X.copy()
        for feature_name in self.features:
            if feature_name == self.TARGET_FEATURE:
                continue
            working = self._apply_feature(
                working,
                feature_name=feature_name,
                fitting=True,
            )
        return working

    def _transform_base(self, X: pd.DataFrame) -> pd.DataFrame:
        working = X.copy()
        for feature_name in self.features:
            if feature_name == self.TARGET_FEATURE:
                continue
            working = self._apply_feature(
                working,
                feature_name=feature_name,
                fitting=False,
            )
        return working

    def _apply_feature(
        self,
        df: pd.DataFrame,
        feature_name: str,
        fitting: bool,
    ) -> pd.DataFrame:
        method_name = (
            "_fill_age_by_title_class"
            if feature_name == "fill_age_by_group"
            else f"_{feature_name}"
        )
        return getattr(self, method_name)(df, fitting=fitting)

    # ------------------------------------------------------------------
    # Basic feature engineering
    # ------------------------------------------------------------------
    @staticmethod
    def _title_from_name(name: pd.Series) -> pd.Series:
        title = name.astype("string").str.extract(
            r" ([A-Za-z]+)\.",
            expand=False,
        )

        rare_titles = [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ]

        title = title.replace(rare_titles, "Rare")
        title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        return title.fillna("Unknown")

    def _add_title(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["Name"], "add_title")
        df["Title"] = self._title_from_name(df["Name"])
        return df

    def _fill_embarked(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        self._require_columns(df, ["Embarked"], "fill_embarked")
        if fitting:
            modes = df["Embarked"].dropna().mode()
            self.embarked_mode_ = modes.iloc[0] if not modes.empty else "S"
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)
        return df

    def _fill_fare(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        self._require_columns(df, ["Fare", "Pclass"], "fill_fare")
        if fitting:
            self.fare_by_pclass_ = (
                df.groupby("Pclass", dropna=False)["Fare"].median().to_dict()
            )
            median = df["Fare"].median()
            self.global_fare_median_ = (
                float(median) if pd.notna(median) else 0.0
            )

        class_medians = df["Pclass"].map(self.fare_by_pclass_)
        df["Fare"] = df["Fare"].fillna(class_medians)
        df["Fare"] = df["Fare"].fillna(self.global_fare_median_)
        return df

    def _fill_age_by_title_class(
        self,
        df: pd.DataFrame,
        fitting: bool,
    ) -> pd.DataFrame:
        self._require_columns(
            df,
            ["Age", "Pclass", "Title", "Sex"],
            "fill_age_by_title_class",
        )

        if fitting:
            self.age_by_title_class_ = (
                df.groupby(["Pclass", "Title"], dropna=False)["Age"]
                .median()
                .to_dict()
            )
            self.age_by_sex_class_ = (
                df.groupby(["Sex", "Pclass"], dropna=False)["Age"]
                .median()
                .to_dict()
            )
            median = df["Age"].median()
            self.global_age_median_ = (
                float(median) if pd.notna(median) else 28.0
            )

        title_keys = pd.Series(
            list(zip(df["Pclass"], df["Title"])),
            index=df.index,
        )
        df["Age"] = df["Age"].fillna(title_keys.map(self.age_by_title_class_))

        sex_keys = pd.Series(
            list(zip(df["Sex"], df["Pclass"])),
            index=df.index,
        )
        df["Age"] = df["Age"].fillna(sex_keys.map(self.age_by_sex_class_))
        df["Age"] = df["Age"].fillna(self.global_age_median_)
        return df

    def _add_family_size(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["SibSp", "Parch"], "add_family_size")
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        return df

    def _add_is_alone(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["FamilySize"], "add_is_alone")
        df["IsAlone"] = (df["FamilySize"] == 1).astype("int8")
        return df

    def _add_family_group(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["FamilySize"], "add_family_group")
        df["FamilyGroup"] = pd.cut(
            df["FamilySize"],
            bins=[-np.inf, 1, 4, np.inf],
            labels=["Alone", "Small", "Large"],
        )
        return df

    def _add_surname(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["Name"], "add_surname")
        df["Surname"] = (
            df["Name"]
            .astype("string")
            .str.split(",", n=1)
            .str[0]
            .str.strip()
            .fillna("Unknown")
        )
        return df

    @staticmethod
    def _family_key(df: pd.DataFrame) -> pd.Series:
        key = (
            df["Surname"].astype("string")
            + "_"
            + df["FamilySize"].astype("Int64").astype("string")
        )
        # Do not let all solo passengers become one artificial family.
        return key.where(df["FamilySize"] > 1, "__NO_FAMILY__")

    def _add_family_id(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(
            df,
            ["Surname", "FamilySize"],
            "add_family_id",
        )
        df["FamilyID"] = self._family_key(df)
        return df

    def _add_ticket_prefix(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["Ticket"], "add_ticket_prefix")
        df["TicketPrefix"] = (
            df["Ticket"]
            .astype("string")
            .str.replace(r"\d+", "", regex=True)
            .str.replace(r"[./]", "", regex=True)
            .str.replace(r"\s+", "", regex=True)
            .str.upper()
            .replace("", "None")
            .fillna("None")
        )
        return df

    def _add_ticket_group_size(
        self,
        df: pd.DataFrame,
        fitting: bool,
    ) -> pd.DataFrame:
        self._require_columns(df, ["Ticket"], "add_ticket_group_size")
        if fitting:
            self.ticket_counts_ = (
                df["Ticket"].value_counts(dropna=False).to_dict()
            )
        df["TicketGroupSize"] = (
            df["Ticket"].map(self.ticket_counts_).fillna(1).astype("int64")
        )
        return df

    def _add_family_group_size(
        self,
        df: pd.DataFrame,
        fitting: bool,
    ) -> pd.DataFrame:
        self._require_columns(
            df,
            ["FamilyID"],
            "add_family_group_size",
        )

        if fitting:
            valid = df.loc[df["FamilyID"] != "__NO_FAMILY__", "FamilyID"]
            self.family_counts_ = valid.value_counts().to_dict()

        df["FamilyGroupSize"] = (
            df["FamilyID"].map(self.family_counts_).fillna(1).astype("int64")
        )
        return df

    def _add_fare_frequency(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        self._require_columns(df, ["Fare"], "add_fare_frequency")
        if fitting:
            self.fare_counts_ = (
                df["Fare"].value_counts(dropna=False).to_dict()
            )
        df["FareFrequency"] = (
            df["Fare"].map(self.fare_counts_).fillna(1).astype("int64")
        )
        return df

    def _add_has_cabin(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["Cabin"], "add_has_cabin")
        df["HasCabin"] = df["Cabin"].notna().astype("int8")
        return df

    def _add_deck(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["Cabin"], "add_deck")
        df["Deck"] = df["Cabin"].astype("string").str[0].fillna("Unknown")
        return df

    def _add_fare_log(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["Fare"], "add_fare_log")
        df["FareLog"] = np.log1p(df["Fare"].clip(lower=0))
        return df

    def _add_age_bin(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["Age"], "add_age_bin")
        df["AgeBin"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 35, 60, np.inf],
            labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
        )
        return df

    def _add_fare_bin(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        self._require_columns(df, ["Fare"], "add_fare_bin")
        if fitting:
            quantiles = df["Fare"].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
            edges = np.unique(quantiles.to_numpy(dtype=float))
            if len(edges) < 2:
                edges = np.asarray([-np.inf, np.inf])
            else:
                edges[0] = -np.inf
                edges[-1] = np.inf
            labels = ("Low", "Medium", "High", "VeryHigh")
            self.fare_bin_edges_ = edges
            self.fare_bin_labels_ = labels[: len(edges) - 1]

        df["FareBin"] = pd.cut(
            df["Fare"],
            bins=self.fare_bin_edges_,
            labels=self.fare_bin_labels_,
            include_lowest=True,
        )
        return df

    def _add_female_first_class(
        self,
        df: pd.DataFrame,
        fitting: bool,
    ) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["Sex", "Pclass"], "add_female_first_class")
        df["FemaleFirstClass"] = (
            (df["Sex"] == "female") & (df["Pclass"] == 1)
        ).astype("int8")
        return df

    def _add_child(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(df, ["Age"], "add_child")
        df["Child"] = (df["Age"] < 14).astype("int8")
        return df

    def _add_mother(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        del fitting
        self._require_columns(
            df,
            ["Sex", "Parch", "Age", "Title"],
            "add_mother",
        )
        df["Mother"] = (
            (df["Sex"] == "female")
            & (df["Parch"] > 0)
            & (df["Age"] > 18)
            & (df["Title"] != "Miss")
        ).astype("int8")
        return df

    def _add_fare_per_person(
        self,
        df: pd.DataFrame,
        fitting: bool,
    ) -> pd.DataFrame:
        del fitting
        self._require_columns(
            df,
            ["Fare", "FamilySize"],
            "add_fare_per_person",
        )
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"].clip(lower=1)
        return df

    def _add_fare_per_ticket(
        self,
        df: pd.DataFrame,
        fitting: bool,
    ) -> pd.DataFrame:
        del fitting
        self._require_columns(
            df,
            ["Fare", "TicketGroupSize"],
            "add_fare_per_ticket",
        )
        df["FarePerTicket"] = (
            df["Fare"] / df["TicketGroupSize"].clip(lower=1)
        )
        return df

    def _add_passenger_group(
        self,
        df: pd.DataFrame,
        fitting: bool,
    ) -> pd.DataFrame:
        """Create the five passenger populations discussed in the notebook.

        These are approximations because Titanic exposes aggregate SibSp/Parch,
        not exact relationship types for every passenger.
        """
        del fitting
        self._require_columns(
            df,
            ["Sex", "Title", "Age", "SibSp", "Parch"],
            "add_passenger_group",
        )

        is_male = df["Sex"].eq("male")
        is_boy = is_male & (df["Title"].eq("Master") | (df["Age"] < 14))
        is_adult_male = is_male & ~is_boy

        is_female = df["Sex"].eq("female")
        female_alone = is_female & df["SibSp"].eq(0) & df["Parch"].eq(0)

        # Parch is the strongest available signal for children/parents.
        # Multiple female family members also tend to be represented through
        # SibSp, but exact sister/mother identity is unavailable.
        female_close_family = (
            is_female
            & ~female_alone
            & (
                df["Parch"].gt(0)
                | (df["SibSp"].gt(0) & df["Title"].isin(["Miss", "Mrs"]))
            )
        )
        female_male_relatives = (
            is_female
            & ~female_alone
            & ~female_close_family
        )

        group = pd.Series("Unknown", index=df.index, dtype="string")
        group.loc[is_adult_male] = "A_AdultMale"
        group.loc[is_boy] = "B_Boy"
        group.loc[female_close_family] = "C_FemaleCloseFamily"
        group.loc[female_male_relatives] = "D_FemaleMaleRelatives"
        group.loc[female_alone] = "E_FemaleAlone"

        df["PassengerGroup"] = group
        df["IsAdultMale"] = is_adult_male.astype("int8")
        df["IsBoy"] = is_boy.astype("int8")
        df["IsFemaleAlone"] = female_alone.astype("int8")
        return df

    def _add_interactions(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        df = self._add_female_first_class(df, fitting=fitting)
        df = self._add_child(df, fitting=fitting)
        df = self._add_mother(df, fitting=fitting)
        df = self._add_fare_per_person(df, fitting=fitting)
        return df

    # ------------------------------------------------------------------
    # Leakage-safe target encoding
    # ------------------------------------------------------------------
    @staticmethod
    def _sex_class_keys(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            list(zip(df["Sex"], df["Pclass"])),
            index=df.index,
        )

    def _build_target_maps(
        self,
        df: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        self._require_columns(
            df,
            [
                "Ticket",
                "FamilyID",
                "Sex",
                "Pclass",
                "PassengerGroup",
            ],
            self.TARGET_FEATURE,
        )

        target = pd.Series(np.asarray(y), index=df.index, dtype=float)
        global_rate = float(target.mean())

        labeled = df[
            ["Ticket", "FamilyID", "Sex", "Pclass", "PassengerGroup"]
        ].copy()
        labeled["_target"] = target

        fallback_sex_class = (
            labeled.groupby(["Sex", "Pclass"])["_target"].mean().to_dict()
        )
        fallback_passenger_group = (
            labeled.groupby("PassengerGroup")["_target"].mean().to_dict()
        )

        ticket_stats = (
            labeled.groupby("Ticket", dropna=False)["_target"]
            .agg(["sum", "count"])
        )

        valid_family = labeled[labeled["FamilyID"] != "__NO_FAMILY__"]
        family_stats = (
            valid_family.groupby("FamilyID", dropna=False)["_target"]
            .agg(["sum", "count"])
        )

        return {
            "global_rate": global_rate,
            "fallback_sex_class": fallback_sex_class,
            "fallback_passenger_group": fallback_passenger_group,
            "ticket_sum": ticket_stats["sum"].to_dict(),
            "ticket_count": ticket_stats["count"].to_dict(),
            "family_sum": family_stats["sum"].to_dict(),
            "family_count": family_stats["count"].to_dict(),
        }

    def _fallback_rate(self, df: pd.DataFrame, maps: dict) -> pd.Series:
        sex_class = self._sex_class_keys(df).map(maps["fallback_sex_class"])
        passenger_group = df["PassengerGroup"].map(
            maps["fallback_passenger_group"]
        )
        return (
            sex_class
            .fillna(passenger_group)
            .fillna(maps["global_rate"])
            .astype(float)
        )

    def _smoothed_group_rate(
        self,
        keys: pd.Series,
        sums: dict,
        counts: dict,
        fallback: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        count = keys.map(counts).fillna(0).astype(float)
        total = keys.map(sums).fillna(0).astype(float)

        smoothing = float(self.target_smoothing)
        denominator = count + smoothing

        if smoothing == 0:
            rate = total.div(count.replace(0, np.nan))
        else:
            rate = (total + smoothing * fallback) / denominator

        rate = rate.where(count > 0, fallback)
        return rate.astype(float), count

    def _apply_target_maps(
        self,
        df: pd.DataFrame,
        maps: dict,
    ) -> pd.DataFrame:
        fallback = self._fallback_rate(df, maps)

        ticket_rate, ticket_count = self._smoothed_group_rate(
            df["Ticket"],
            maps["ticket_sum"],
            maps["ticket_count"],
            fallback,
        )

        valid_family_key = df["FamilyID"].where(
            df["FamilyID"] != "__NO_FAMILY__"
        )
        family_rate, family_count = self._smoothed_group_rate(
            valid_family_key,
            maps["family_sum"],
            maps["family_count"],
            fallback,
        )

        evidence = ticket_count + family_count
        combined = (
            ticket_rate * ticket_count + family_rate * family_count
        ).div(evidence.replace(0, np.nan)).fillna(fallback)

        df["FallbackSurvivalRate"] = fallback
        df["TicketSurvivalRate"] = ticket_rate
        df["FamilySurvivalRate"] = family_rate
        df["CombinedGroupSurvivalRate"] = combined
        df["GroupSurvivalSignal"] = combined - fallback
        df["TicketOutcomeCount"] = ticket_count
        df["FamilyOutcomeCount"] = family_count
        df["HasTicketOutcome"] = (ticket_count > 0).astype("int8")
        df["HasFamilyOutcome"] = (family_count > 0).astype("int8")
        return df

    def _add_oof_group_survival(
        self,
        df: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        result = df.copy()
        for column in self.TARGET_OUTPUT_COLUMNS:
            result[column] = np.nan

        class_counts = y.value_counts()
        max_splits = int(class_counts.min()) if not class_counts.empty else 0
        n_splits = min(int(self.target_encoding_folds), max_splits)

        if n_splits < 2:
            # This should only occur in tiny synthetic/debug datasets.
            fallback_maps = self._build_target_maps(df, y)
            return self._apply_target_maps(result, fallback_maps)

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        y_array = y.to_numpy()
        for train_positions, valid_positions in splitter.split(df, y_array):
            train_index = df.index[train_positions]
            valid_index = df.index[valid_positions]

            fold_maps = self._build_target_maps(
                df.loc[train_index],
                y.loc[train_index],
            )
            encoded = self._apply_target_maps(
                df.loc[valid_index].copy(),
                fold_maps,
            )
            result.loc[valid_index, self.TARGET_OUTPUT_COLUMNS] = encoded[
                list(self.TARGET_OUTPUT_COLUMNS)
            ].to_numpy()

        return result


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop configured columns while preserving a pandas DataFrame."""

    def __init__(self, columns: tuple[str, ...] = ()):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        del y
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnDropper requires a pandas DataFrame.")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnDropper requires a pandas DataFrame.")
        return X.drop(columns=list(self.columns), errors="ignore")


def build_feature_engineer(cfg: dict) -> TitanicFeatureEngineer:
    preprocessing_cfg = cfg["preprocessing"]
    engineer_cfg = preprocessing_cfg.get("engineer_features", {})
    enabled = engineer_cfg.get("enabled", False)
    features = tuple(engineer_cfg.get("features", [])) if enabled else ()

    target_cfg = preprocessing_cfg.get("target_encoding", {})
    return TitanicFeatureEngineer(
        features=features,
        target_encoding_folds=target_cfg.get("folds", 5),
        target_smoothing=target_cfg.get("smoothing", 2.0),
        random_state=cfg["experiment"].get("random_state", 42),
    )


def build_column_dropper(cfg: dict) -> ColumnDropper:
    columns = tuple(cfg["preprocessing"].get("drop_columns", []))
    return ColumnDropper(columns=columns)


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    preprocessing_cfg = cfg["preprocessing"]

    numerical_columns = preprocessing_cfg.get("numerical", [])
    boolean_columns = preprocessing_cfg.get("boolean", [])
    categorical_columns = preprocessing_cfg.get("categorical", [])

    configured = numerical_columns + boolean_columns + categorical_columns
    duplicates = sorted(
        column
        for column in set(configured)
        if configured.count(column) > 1
    )
    if duplicates:
        raise ValueError(
            "Columns cannot appear in multiple preprocessing groups: "
            + ", ".join(duplicates)
        )

    impute_cfg = preprocessing_cfg.get("impute", {})
    numerical_strategy = impute_cfg.get("numerical", "median")
    boolean_strategy = impute_cfg.get("boolean", "most_frequent")
    categorical_strategy = impute_cfg.get("categorical", "most_frequent")

    scale_enabled = preprocessing_cfg.get("scale", {}).get("enabled", True)
    encode_enabled = preprocessing_cfg.get("encode", {}).get("enabled", True)

    numerical_steps = [
        ("imputer", SimpleImputer(strategy=numerical_strategy))
    ]
    if scale_enabled:
        numerical_steps.append(("scaler", StandardScaler()))

    boolean_steps = [
        ("imputer", SimpleImputer(strategy=boolean_strategy))
    ]

    categorical_steps = [
        ("imputer", SimpleImputer(strategy=categorical_strategy))
    ]
    if encode_enabled:
        categorical_steps.append(
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        )

    transformers = []
    if numerical_columns:
        transformers.append(
            ("num", Pipeline(numerical_steps), numerical_columns)
        )
    if boolean_columns:
        transformers.append(
            ("bool", Pipeline(boolean_steps), boolean_columns)
        )
    if categorical_columns:
        transformers.append(
            ("cat", Pipeline(categorical_steps), categorical_columns)
        )

    if not transformers:
        raise ValueError("No preprocessing columns were configured.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def prepare_features(
    df: pd.DataFrame,
    cfg: dict,
    y: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Exploratory helper.

    Evaluation and model training should use the feature engineer inside the
    sklearn Pipeline. When target encoding is enabled, pass ``y``.
    """
    engineer = build_feature_engineer(cfg)
    dropper = build_column_dropper(cfg)
    engineered = engineer.fit_transform(df, y)
    return dropper.fit_transform(engineered)
