import torch.backends.cudnn as cudnn
import wandb

cudnn.benchmark = True

import sys
sys.path.append('..')

from src.engine.cfg import CFG as cfg
from src.tools import main_train

wandb.login()

def main():
    wandb.init(project="cifar100_baselines")
    cfg.CD.LR = wandb.config.lr
    cfg.CD.EPOCHS = wandb.config.epochs
    cfg.CD.PROB = wandb.config.prob
    main_train(cfg, False, None)

# Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "best_acc"},
    "parameters": {
        "lr": {"max": 0.1, "min": 0.001},
        "epochs": {"max": 3, "min":1},
        "prob": {"max": 1.0, "min":0.1},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 100,
        "eta": 50,
    },
}

# Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="cifar100_baselines")
cfg.merge_from_file("configs/cifar100/da_vgg13_vgg8.yaml")
wandb.agent(sweep_id, function=main, count=200)
