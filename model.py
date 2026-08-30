"""
Model: pretrained ResNet backbone + binary classification head.

Start with resnet18 (fast, good for debugging your pipeline),
then switch to resnet50 in config.py once everything works end-to-end.
"""
import torch.nn as nn
from torchvision import models

import config


def build_model(backbone=None, freeze_backbone=None, num_classes=2):
    backbone = backbone or config.BACKBONE
    freeze_backbone = config.FREEZE_BACKBONE if freeze_backbone is None else freeze_backbone

    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=weights)
    elif backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if freeze_backbone:
        for param in net.parameters():
            param.requires_grad = False
    else:
        # Partial freeze: lock the early, generic feature extractors
        # (edges/textures transfer fine from ImageNet) and leave the later
        # layers trainable, where generator-fingerprint memorization was
        # happening under full fine-tuning. This is the middle ground
        # between "full fine-tune" (90% indist / 47% crossgen) and
        # "full freeze" (70% even in-dist).
        for layer_name in config.FREEZE_LAYERS:
            layer = getattr(net, layer_name, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = False

    # Replace the final FC layer -- this stays trainable even if backbone is frozen
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return net.to(config.DEVICE)
