import os
from .resnet import (
    resnet34,
)
from .resnetv2 import ResNet50, ResNet18
from .ShuffleNetv2 import ShuffleV2
from .mv2_tinyimagenet import mobilenetv2_tinyimagenet

tiny_imagenet_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../download_ckpts/tiny_imagenet_teachers/"
)

tiny_imagenet_model_dict = {
    "ResNet18": (ResNet18, tiny_imagenet_model_prefix + "ResNet18_vanilla/student_best_62_4"),
    "MobileNetV2": (mobilenetv2_tinyimagenet, None),
    "ShuffleV2": (ShuffleV2, None),
    "resnet34": (resnet34, tiny_imagenet_model_prefix + "resnet34_vanilla/resnet34")
}
