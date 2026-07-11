#!/usr/bin/env bash

set -uo pipefail

CONFIG_DIR="titanic_ablation_configs"
LOG_DIR="sols/ablation_logs"
CONTROLLER="controller.py"

mkdir -p "$LOG_DIR"

configs=(
    "$CONFIG_DIR/a00_core_baseline.yaml"
    "$CONFIG_DIR/a01_title_age.yaml"
    "$CONFIG_DIR/a02_family.yaml"
    "$CONFIG_DIR/a03_ticket_fare.yaml"
    "$CONFIG_DIR/a04_cabin.yaml"
    "$CONFIG_DIR/a05_group_frequency.yaml"
    "$CONFIG_DIR/a06_full_interactions.yaml"
    "$CONFIG_DIR/a07_full_with_bins.yaml"
    "$CONFIG_DIR/a10_full_minus_family.yaml"
    "$CONFIG_DIR/a11_full_minus_ticket.yaml"
    "$CONFIG_DIR/a12_full_minus_cabin.yaml"
    "$CONFIG_DIR/a13_full_minus_group_frequency.yaml"
    "$CONFIG_DIR/a14_full_minus_interactions.yaml"
    "$CONFIG_DIR/a15_full_minus_fare_derived.yaml"
)

passed=()
failed=()

for config in "${configs[@]}"; do
    experiment_name="$(basename "$config" .yaml)"
    log_file="$LOG_DIR/${experiment_name}.log"

    echo
    echo "============================================================"
    echo "Running: $experiment_name"
    echo "Config:  $config"
    echo "Log:     $log_file"
    echo "============================================================"

    start_time=$(date +%s)

    if python "$CONTROLLER" --config "$config" 2>&1 | tee "$log_file"; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))

        passed+=("$experiment_name")
        echo "Completed $experiment_name in ${elapsed}s"
    else
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))

        failed+=("$experiment_name")
        echo "FAILED: $experiment_name after ${elapsed}s"
    fi
done

echo
echo "============================================================"
echo "Ablation run summary"
echo "============================================================"
echo "Passed: ${#passed[@]}"
echo "Failed: ${#failed[@]}"

if ((${#passed[@]} > 0)); then
    echo
    echo "Successful experiments:"
    printf "  - %s\n" "${passed[@]}"
fi

if ((${#failed[@]} > 0)); then
    echo
    echo "Failed experiments:"
    printf "  - %s\n" "${failed[@]}"
    exit 1
fi

echo
echo "All ablation experiments completed successfully."
