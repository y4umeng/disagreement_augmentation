# Disagreement Augmentation for Distillation

Disagreement Augmentation (DA) introduces a novel method for improving knowledge distillation by optimizing input samples to emphasize areas of disagreement between a teacher model and a student model. Inspired by the Socratic method, DA leverages structured conflict to challenge the student, fostering better generalization and robustness.

The DA algorithm is implemented [here](https://github.com/y4umeng/disagreement_augmentation/blob/ae44d2078f93cba7cf2f0d67296644a295210dfd/src/distillers/DA.py#L18).

## Overview

Knowledge distillation traditionally minimizes divergence between teacher and student models. In contrast, DA introduces disagreement as a constructive signal, generating augmented training samples that maximize divergence between the models' predictions. This approach challenges the student to reconcile these differences, leading to improved learning.

### Key Features:
- **Improved Generalization**: Models trained with DA consistently outperform baseline methods in validation accuracy.
- **Robustness to Perturbations**: DA-trained models show greater resilience to adversarial-like samples.
- **Compatibility**: DA can integrate with existing knowledge distillation frameworks.

## CIFAR-100 Validation Results

| **Teacher**      | **Student**       | **Baseline Accuracy (\%)** | **DA Accuracy (\%)**      |
|-------------------|-------------------|----------------------------|----------------------------|
| Resnet32x4        | Resnet8x4         | $73.66 \pm 0.26$           | $74.59 \pm 0.24$          |
| VGG13             | VGG8              | $73.33 \pm 0.25$           | $73.76 \pm 0.29$          |
| Resnet32x4        | ShuffleNet-V2     | $71.67 \pm 0.34$           | $73.70 \pm 0.19$          |


## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/y4umeng/disagreement_augmentation.git
cd disagreement_augmentation
pip install -r requirements.txt
```

# Usage 

See <https://github.com/megvii-research/mdistiller?tab=readme-ov-file#getting-started> for setting up Weights & Biases logging.

Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

Train a Resnet8x4 student with a pretrained Resnet32x4 teacher.

```bash
cd src

# Baseline
python tools/train.py --cfg configs/cifar100/baseline_resnet32x4_resnet8x4.yaml

# Train with DA
python tools/train.py --cfg configs/cifar100/da_resnet32x4_resnet8x4.yaml
```

# Acknowledgements

This codebase is modified from <https://github.com/megvii-research/mdistiller>

