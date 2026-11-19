"""
EfficientNet Classifier for Brain Tumor Classification
According to Section 3.3 and Figure 4:
"In this method, Regions Of Interest are used in Efficient Net to classify tumors
as glioma, meningioma, pituitary tumor, or non-tumor. Efficient Net is good at picking
out the important things, like tumor texture, shape and intensity."
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BrainTumorEfficientNet(nn.Module):
    """
    EfficientNet-based classifier fine-tuned for Brain MRI ROI classification.
    Classes:
    0: Glioma
    1: Meningioma
    2: Pituitary
    3: Non-Tumor
    """

    CLASSES = ["Glioma", "Meningioma", "Pituitary", "Non-Tumor"]
    NUM_CLASSES = len(CLASSES)

    def __init__(
        self,
        variant="b0",
        num_classes=4,
        pretrained=True,
        dropout_rate=0.3,
    ):
        super(BrainTumorEfficientNet, self).__init__()
        self.num_classes = num_classes
        self.variant = variant

        # Load backbone
        if variant == "b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
        elif variant == "b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
        else:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features

        # Replace classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate * 0.5, inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Forward pass returning raw class logits.
        :param x: Tensor of shape (Batch, 3, H, W)
        :return: Tensor of shape (Batch, num_classes)
        """
        return self.backbone(x)

    def extract_features(self, x):
        """
        Extracts high-level feature embedding before final classification layer.
        """
        return self.backbone.features(x)

    def predict_probabilities(self, x):
        """
        Computes normalized class probabilities via Softmax.
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities

    def classify_roi(self, roi_tensor, device="cpu"):
        """
        Classifies an extracted ROI tensor and returns class name, confidence,
        and all class probabilities.
        """
        self.eval()
        if len(roi_tensor.shape) == 3:
            roi_tensor = roi_tensor.unsqueeze(0)

        roi_tensor = roi_tensor.to(device)
        with torch.no_grad():
            probs = self.predict_probabilities(roi_tensor)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_label = self.CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Binary aggregation: Tumor vs Non-Tumor (matching Table 1 / Figure 6)
        # Non-Tumor is class index 3, Tumor is classes 0, 1, 2
        tumor_prob = float(probs[0] + probs[1] + probs[2])
        non_tumor_prob = float(probs[3])

        binary_label = "Non-Tumor" if pred_idx == 3 else "Tumor"
        binary_confidence = non_tumor_prob if pred_idx == 3 else tumor_prob

        prob_dict = {label: float(probs[i]) for i, label in enumerate(self.CLASSES)}

        return {
            "predicted_class": pred_label,
            "class_index": pred_idx,
            "confidence": confidence,
            "all_probabilities": prob_dict,
            "binary_label": binary_label,
            "binary_confidence": binary_confidence,
            "tumor_probability": tumor_prob,
            "non_tumor_probability": non_tumor_prob,
        }
