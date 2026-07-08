import argparse
import yaml

from experiment import Experiment


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a machine learning experiment."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the experiment configuration file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    experiment = Experiment(cfg)
    experiment.run()


if __name__ == "__main__":
    main()
