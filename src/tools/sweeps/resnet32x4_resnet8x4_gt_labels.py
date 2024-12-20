import torch.backends.cudnn as cudnn
import wandb

cudnn.benchmark = True

import sys
sys.path.append('..')
# sys.path.append('../..')


from src.engine.cfg import CFG as cfg
from src.tools import main_train

wandb.login()

def main():
    wandb.init(project="CDD-sweep")
    cfg.DA.LR = wandb.config.lr
    cfg.DA.PROB = wandb.config.prob
    main_train(cfg, False, None)

# Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "best_acc"},
    "parameters": {
        "lr": {"max": 0.1, "min": 0.001},
        "prob": {"max": 1.0, "min":0.3},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 100,
        "eta": 50,
    },
}

# Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CDD-sweep")
cfg.merge_from_file("configs/cifar100/da_resnet32x4_resnet8x4.yaml")
wandb.agent(sweep_id, function=main, count=200)
