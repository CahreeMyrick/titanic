#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


FORWARD_ORDER = [
    "a00_core_baseline",
    "a01_title_age",
    "a02_family",
    "a03_ticket_fare",
    "a04_cabin",
    "a05_group_frequency",
    "a06_full_interactions",
    "a07_full_with_bins",
]

LEAVE_ONE_OUT_ORDER = [
    "a10_full_minus_family",
    "a11_full_minus_ticket",
    "a12_full_minus_cabin",
    "a13_full_minus_group_frequency",
    "a14_full_minus_interactions",
    "a15_full_minus_fare_derived",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Titanic ablation cross-validation results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("sols/ablations"),
        help="Root directory containing experiment CV JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sols/ablation_summary"),
        help="Directory in which summary files will be written.",
    )
    parser.add_argument(
        "--reference",
        default="a06_full_interactions",
        help="Full-system experiment used for leave-one-group-out comparisons.",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> pd.DataFrame:
    json_paths = sorted(results_dir.rglob("*_cv.json"))

    if not json_paths:
        raise FileNotFoundError(
            f"No '*_cv.json' files were found under: {results_dir}"
        )

    records: list[dict] = []

    for path in json_paths:
        try:
            with path.open("r", encoding="utf-8") as file:
                result = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: skipping unreadable result {path}: {exc}", file=sys.stderr)
            continue

        required = {"experiment", "model", "scoring", "folds", "scores", "mean", "std"}
        missing = required.difference(result)

        if missing:
            print(
                f"Warning: skipping {path}; missing fields: {sorted(missing)}",
                file=sys.stderr,
            )
            continue

        scores = result["scores"]
        record = {
            "experiment": result["experiment"],
            "model": result["model"],
            "scoring": result["scoring"],
            "folds": int(result["folds"]),
            "mean": float(result["mean"]),
            "std": float(result["std"]),
            "min": float(result.get("min", min(scores))),
            "max": float(result.get("max", max(scores))),
            "range": float(result.get("max", max(scores)))
            - float(result.get("min", min(scores))),
            "result_file": str(path),
        }

        for index, score in enumerate(scores, start=1):
            record[f"fold_{index}"] = float(score)

        records.append(record)

    if not records:
        raise RuntimeError("Result files were found, but none contained valid CV results.")

    frame = pd.DataFrame(records)

    duplicates = frame.duplicated(subset=["experiment", "model"], keep=False)
    if duplicates.any():
        duplicate_names = frame.loc[
            duplicates, ["experiment", "model", "result_file"]
        ]
        raise ValueError(
            "Duplicate experiment/model results were found:\n"
            + duplicate_names.to_string(index=False)
        )

    return frame


def build_forward_summary(results: pd.DataFrame) -> pd.DataFrame:
    available = results[results["experiment"].isin(FORWARD_ORDER)].copy()

    order = {name: index for index, name in enumerate(FORWARD_ORDER)}
    available["order"] = available["experiment"].map(order)
    available = available.sort_values(["model", "order"]).reset_index(drop=True)

    available["previous_experiment"] = available.groupby("model")[
        "experiment"
    ].shift(1)
    available["incremental_delta"] = available.groupby("model")["mean"].diff()

    return available


def build_leave_one_out_summary(
    results: pd.DataFrame,
    reference_name: str,
) -> pd.DataFrame:
    reference_rows = results[results["experiment"] == reference_name]

    if reference_rows.empty:
        raise ValueError(
            f"Reference experiment '{reference_name}' was not found. "
            "Run it before summarizing leave-one-out ablations."
        )

    available = results[results["experiment"].isin(LEAVE_ONE_OUT_ORDER)].copy()
    order = {name: index for index, name in enumerate(LEAVE_ONE_OUT_ORDER)}
    available["order"] = available["experiment"].map(order)

    reference_means = reference_rows.set_index("model")["mean"]
    available["reference_experiment"] = reference_name
    available["reference_mean"] = available["model"].map(reference_means)

    missing_models = available["reference_mean"].isna()
    if missing_models.any():
        models = sorted(available.loc[missing_models, "model"].unique())
        raise ValueError(
            f"No '{reference_name}' result exists for model(s): {models}"
        )

    # Positive means removing the feature group reduced performance.
    available["ablation_delta"] = (
        available["reference_mean"] - available["mean"]
    )

    def interpretation(delta: float) -> str:
        tolerance = 1e-12
        if delta > tolerance:
            return "removal_hurt"
        if delta < -tolerance:
            return "removal_helped"
        return "no_change"

    available["interpretation"] = available["ablation_delta"].map(interpretation)
    return available.sort_values(["model", "order"]).reset_index(drop=True)


def format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()

    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(format_number)

    headers = [str(column) for column in selected.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")

    return "\n".join(lines)


def write_markdown_report(
    all_results: pd.DataFrame,
    forward: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    output_path: Path,
    reference_name: str,
) -> None:
    ranked = all_results.sort_values(
        ["model", "mean", "std"],
        ascending=[True, False, True],
    ).copy()
    ranked["rank"] = ranked.groupby("model")["mean"].rank(
        method="dense",
        ascending=False,
    ).astype(int)

    sections = [
        "# Titanic Ablation Summary",
        "",
        "## Overall ranking",
        "",
        markdown_table(
            ranked,
            ["rank", "experiment", "model", "mean", "std", "min", "max"],
        ),
        "",
        "## Forward feature ladder",
        "",
        "`incremental_delta = current mean - previous mean`",
        "",
        markdown_table(
            forward,
            [
                "experiment",
                "previous_experiment",
                "model",
                "mean",
                "std",
                "incremental_delta",
            ],
        ),
        "",
        "## Leave-one-group-out ablations",
        "",
        (
            f"Reference: `{reference_name}`. "
            "`ablation_delta = reference mean - ablated mean`."
        ),
        "",
        (
            "A positive delta means removing the feature group hurt accuracy; "
            "a negative delta means removal improved accuracy."
        ),
        "",
        markdown_table(
            leave_one_out,
            [
                "experiment",
                "model",
                "mean",
                "std",
                "reference_mean",
                "ablation_delta",
                "interpretation",
            ],
        ),
        "",
    ]

    output_path.write_text("\n".join(sections), encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        results = load_results(args.results_dir)
        forward = build_forward_summary(results)
        leave_one_out = build_leave_one_out_summary(
            results,
            reference_name=args.reference,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results_path = args.output_dir / "all_results.csv"
    forward_path = args.output_dir / "forward_ladder.csv"
    leave_one_out_path = args.output_dir / "leave_one_out.csv"
    report_path = args.output_dir / "summary.md"

    results.sort_values(
        ["model", "mean"],
        ascending=[True, False],
    ).to_csv(all_results_path, index=False)

    forward.to_csv(forward_path, index=False)
    leave_one_out.to_csv(leave_one_out_path, index=False)

    write_markdown_report(
        all_results=results,
        forward=forward,
        leave_one_out=leave_one_out,
        output_path=report_path,
        reference_name=args.reference,
    )

    print()
    print("=" * 76)
    print("Overall ranking")
    print("=" * 76)
    print(
        results.sort_values("mean", ascending=False)[
            ["experiment", "model", "mean", "std", "min", "max"]
        ].to_string(
            index=False,
            formatters={
                "mean": lambda x: f"{x:.4f}",
                "std": lambda x: f"{x:.4f}",
                "min": lambda x: f"{x:.4f}",
                "max": lambda x: f"{x:.4f}",
            },
        )
    )

    print()
    print("=" * 76)
    print("Forward ladder")
    print("=" * 76)
    print(
        forward[
            ["experiment", "mean", "std", "incremental_delta"]
        ].to_string(
            index=False,
            formatters={
                "mean": lambda x: f"{x:.4f}",
                "std": lambda x: f"{x:.4f}",
                "incremental_delta": format_number,
            },
        )
    )

    print()
    print("=" * 76)
    print(f"Leave-one-out results (reference: {args.reference})")
    print("=" * 76)
    print(
        leave_one_out[
            ["experiment", "mean", "std", "ablation_delta", "interpretation"]
        ].to_string(
            index=False,
            formatters={
                "mean": lambda x: f"{x:.4f}",
                "std": lambda x: f"{x:.4f}",
                "ablation_delta": lambda x: f"{x:+.4f}",
            },
        )
    )

    print()
    print("Saved:")
    for path in (
        all_results_path,
        forward_path,
        leave_one_out_path,
        report_path,
    ):
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
