from __future__ import annotations

import yaml
import pandas as pd

from pathlib import Path
from experiment import Experiment


def main():
    config_path = Path("v1_config.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    experiment = Experiment(cfg)

    train_path = Path(cfg["paths"]["train_data"])
    test_path = Path(cfg["paths"]["test_data"])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    passenger_ids = test_df[cfg["experiment"]["id_column"]]

    X, y = experiment.prepare_train_data(train_df)
    X_test = experiment.prepare_test_data(test_df)

    model = experiment.train(X, y)

    preds = experiment.get_predictions(model, X_test)

    if cfg["outputs"].get("save_model", False):
        experiment.save_model(model)

    if cfg["outputs"].get("save_predictions", True):
        experiment.save_predictions(passenger_ids, preds)


if __name__ == "__main__":
    main()
