
import hydra
from omegaconf import DictConfig

import os
import wandb
from utils.run_experiment import run_experiment


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = cfg.experiment.wandb_path
    run_experiment(cfg)


if __name__ == "__main__":
    main()


