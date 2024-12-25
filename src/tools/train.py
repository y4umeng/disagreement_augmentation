import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random

cudnn.benchmark = True

import sys
sys.path.append('..')

from src.models import cifar_model_dict, imagenet_model_dict, tiny_imagenet_model_dict
from src.distillers import distiller_dict
from src.dataset import get_dataset
from src.engine.utils import load_checkpoint, log_msg
from src.engine.cfg import CFG as cfg
from src.engine.cfg import show_cfg
from src.engine import trainer_dict

def ensure_reproducibility(seed=88):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main_train(cfg, resume, opts):
    ensure_reproducibility()
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif cfg.DATASET.TYPE == "tiny_imagenet":
            model_student, pretrain_model_path = tiny_imagenet_model_dict[cfg.DISTILLER.STUDENT]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for student {}".format(cfg.DISTILLER.STUDENT)
            model_student = model_student(num_classes=num_classes)
            model_student.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    elif cfg.DISTILLER.TYPE == "DA":
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_dict = tiny_imagenet_model_dict if cfg.DATASET.TYPE == "tiny_imagenet" else cifar_model_dict
            model_student = model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
            net, pretrain_model_path = model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student, model_teacher, cfg)
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_dict = tiny_imagenet_model_dict if cfg.DATASET.TYPE == "tiny_imagenet" else cifar_model_dict
            net, pretrain_model_path = model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main_train(cfg, args.resume, args.opts)
