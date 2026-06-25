import yaml
from pathlib import Path
from experiment import Experiment
import preprocess

def main():

    # --- Load Config --- #
    CONFIG_PATH = Path("v1_config.yaml")
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    # --- Initialize Experiment --- #
    
    # --- Load Data --- #


    if "drop_columns" in cfg["preprocessing"]:


if __name__ == "__main__":
    main()

