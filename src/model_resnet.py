
import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Transfer(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        # Load pretrained ImageNet weights
        weights = ResNet50_Weights.DEFAULT

        self.resnet = resnet50(weights=weights)

        # Freeze pretrained layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the original ImageNet classifier
        num_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(
            num_features,
            num_classes
        )

    def forward(self, x):

        return self.resnet(x)

    def train(self, mode=True):

        super().train(mode)

        # Keep frozen feature extractor and BatchNorm
        # layers in evaluation mode
        self.resnet.eval()

        # Keep the new classification layer trainable
        self.resnet.fc.train(mode)

        return self
