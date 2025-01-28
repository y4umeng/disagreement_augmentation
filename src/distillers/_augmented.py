import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class AugmentedDistiller(Distiller):
    """Base distiller with Disagreement Augmentation"""

    def __init__(self, student, teacher, cfg):
        super(AugmentedDistiller, self).__init__(student, teacher)
        self.cfg = cfg
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 

    def DA(self, images):
        images.requires_grad_(True)
        optimizer = torch.optim.Adam([images], lr=self.cfg.DA.LR)
        for _ in range(self.cfg.DA.EPOCHS):
            logits_student, _ = self.student(images)
            logits_teacher, _ = self.teacher(images)
            loss = -1 * kd_loss(
                        logits_student, logits_teacher, 1, self.logit_stand
                    )
            images.grad = torch.autograd.grad(loss, images)[0]
            optimizer.step()
            optimizer.zero_grad()
        images.requires_grad_(False)
        return images
    

    def _fgsm_attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image
    
    def _denorm(self, batch, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], device="cuda"):
        """
        Convert a batch of tensors to their original scale.

        Args:
            batch (torch.Tensor): Batch of normalized tensors.
            mean (torch.Tensor or list): Mean used for normalization.
            std (torch.Tensor or list): Standard deviation used for normalization.

        Returns:
            torch.Tensor: batch of tensors without normalization applied to them.
        """
        if isinstance(mean, list):
            mean = torch.tensor(mean).to(device)
        if isinstance(std, list):
            std = torch.tensor(std).to(device)

        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    
    def FGSM(self, images, targets):
        # modified from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
        images.requires_grad = True
        output, _ = self.student(images)
        
        # Calculate the loss
        loss = F.nll_loss(output, targets)
        
        # Zero all existing gradients
        self.student.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = images.grad.data

        # Restore the data to its original scale
        data_denorm = self._denorm(images)

        # Call FGSM Attack
        perturbed_data = self._fgsm_attack(data_denorm, self.cfg.FGSM.EPSILON, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))(perturbed_data)
        return perturbed_data_normalized.detach()
    
    def FGSM2(self, images, targets):
        images.requires_grad_(True)
        optimizer = torch.optim.Adam([images], lr=self.cfg.FGSM.EPSILON)
        logits_student, _ = self.student(images)
        loss = -1 * F.nll_loss(logits_student, targets)
        images.grad = torch.autograd.grad(loss, images)[0]
        optimizer.step()
        optimizer.zero_grad()
        images.requires_grad_(False)
        return images
    

    def forward_eval(self, image, lr, epochs):
        augmented_image = image.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([augmented_image], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        logits = {}
        
        for epoch in range(epochs[-1] + 1):
            logits_student, _ = self.student(augmented_image)
            logits_student = torch.nn.functional.normalize(logits_student, p=1.0, dim=-1)
            logits_teacher, _ = self.teacher(augmented_image)
            logits_teacher = torch.nn.functional.normalize(logits_teacher, p=1.0, dim=-1)

            self.student.zero_grad()
            self.teacher.zero_grad()
            loss = nn.MSELoss()(logits_student, logits_teacher)
            loss = -loss

            if epoch in epochs:
                logits[epoch] = {"student": logits_student.clone().detach(), 
                                 "teacher": logits_teacher.clone().detach(), 
                                 "image": augmented_image.clone().detach(),
                                 "loss": F.mse_loss(logits_student, logits_teacher, reduction='none').mean(dim=1)}

            augmented_image.grad = torch.autograd.grad(loss, augmented_image)[0]
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

        return logits
