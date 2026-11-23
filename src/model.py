import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model(num_classes=5, pretrained=True, freeze_backbone=False):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model
