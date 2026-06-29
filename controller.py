import yaml

from experiment import Experiment


def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    experiment = Experiment(cfg)
    experiment.run()


if __name__ == "__main__":
    main()