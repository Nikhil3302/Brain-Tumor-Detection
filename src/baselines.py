"""
Baseline Comparative Models: Standard CNN and VGG16
According to Section 4, Table 1, and Figure 7 of the paper:
- Standard CNN
- VGG16
Benchmarked against the proposed Hybrid YOLO-NAS + EfficientNet approach.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class StandardCNN(nn.Module):
    """
    Standard Convolutional Neural Network baseline.
    Represents conventional CNN architectures benchmarked in Table 1.
    """

    def __init__(self, num_classes=4, in_channels=3):
        super(StandardCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)


class VGG16Baseline(nn.Module):
    """
    VGG16 Transfer Learning baseline as benchmarked in Table 1 and Figure 7.
    """

    def __init__(self, num_classes=4, pretrained=True):
        super(VGG16Baseline, self).__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        self.vgg16 = models.vgg16(weights=weights)

        # Custom classification head
        in_features = self.vgg16.classifier[0].in_features
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.vgg16(x)
