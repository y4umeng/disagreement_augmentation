import torch
import torch.nn as nn
import torch.nn.functional as F

from ._augmented import AugmentedDistiller, kd_loss

class KD(AugmentedDistiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher, cfg)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        if self.cfg.EXPERIMENT.DA and torch.rand(1)[0] < self.cfg.DA.PROB:
            image = self.DA(image)

        if self.cfg.EXPERIMENT.FGSM and torch.rand(1)[0] < self.cfg.FGSM.PROB:
            image = self.FGSM2(image, target)

        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, self.logit_stand
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
