"""
model.py

Defines transfer learning models for Food-101 classification.
Default model: ResNet50 (ImageNet pretrained)

Functions:
- get_resnet50()
- get_model()
"""

import torch.nn as nn
from torchvision.models import resnet50


def get_resnet50(num_classes=101, pretrained=True, fine_tune=False):
    """
    Returns a ResNet50 model with a modified final classification layer.
    """

    model = resnet50(weights="IMAGENET1K_V1" if pretrained else None)

    # Freeze backbone
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    # Replace FC layer for 101 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_model(name="resnet50", num_classes=101, pretrained=True, fine_tune=False):
    """
    Factory function to select different models.
    Currently supports:
        - resnet50
    """
    name = name.lower()

    if name == "resnet50":
        return get_resnet50(
            num_classes=num_classes, pretrained=pretrained, fine_tune=fine_tune
        )

    else:
        raise ValueError(f"Unsupported model: {name}")
