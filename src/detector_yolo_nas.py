"""
YOLO-NAS Architecture for Brain Tumor Detection
According to Section 3.1 and Figure 3:
"YOLO-NAS is a state-of-the-art, one-step object detection model. It uses Neural Architecture
Search to reach good precision without needing a lot of computing power...
The NAS backbone pulls spatial data, and the detection mechanism estimates tumor
location by reducing localization loss. This allows quicker, precise tumor detection in MRI images."
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Basic Convolution + BatchNorm + SiLU/ReLU activation block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class QARepVGGBlock(nn.Module):
    """
    Quantization-Aware RepVGG block used in YOLO-NAS architecture for fast,
    accurate multi-scale feature extraction without heavy latency overhead.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv3x3 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride, act=False)
        self.conv1x1 = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=stride, act=False)

        if stride == 1 and in_channels == out_channels:
            self.identity = nn.BatchNorm2d(in_channels)
        else:
            self.identity = None

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)


class NASStage(nn.Module):
    """A stage of repetitive QARepVGG blocks optimized by Neural Architecture Search."""

    def __init__(self, in_channels, out_channels, num_blocks, stride=2):
        super().__init__()
        layers = []
        layers.append(QARepVGGBlock(in_channels, out_channels, stride=stride))
        for _ in range(num_blocks - 1):
            layers.append(QARepVGGBlock(out_channels, out_channels, stride=1))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class YoloNasBackbone(nn.Module):
    """
    Multi-scale YOLO-NAS Backbone for Brain MRI Feature Extraction.
    Extracts feature representations at 3 distinct resolutions:
    P3 (stride 8), P4 (stride 16), P5 (stride 32).
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, base_channels, kernel_size=3, stride=2),  # stride 2
            ConvBNAct(base_channels, base_channels * 2, kernel_size=3, stride=1),
        )
        self.stage1 = NASStage(base_channels * 2, base_channels * 2, num_blocks=2, stride=2)  # stride 4
        self.stage2 = NASStage(base_channels * 2, base_channels * 4, num_blocks=3, stride=2)  # P3, stride 8
        self.stage3 = NASStage(base_channels * 4, base_channels * 8, num_blocks=4, stride=2)  # P4, stride 16
        self.stage4 = NASStage(base_channels * 8, base_channels * 16, num_blocks=3, stride=2)  # P5, stride 32

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class YoloNasPANNeck(nn.Module):
    """
    Path Aggregation Network (PAN) Neck for multi-scale feature pyramid aggregation.
    Enables accurate detection of tumors with varying sizes and shapes.
    """

    def __init__(self, in_channels_list=[128, 256, 512], out_channels=128):
        super().__init__()
        c3, c4, c5 = in_channels_list

        self.reduce_c5 = ConvBNAct(c5, out_channels, kernel_size=1)
        self.reduce_c4 = ConvBNAct(c4, out_channels, kernel_size=1)
        self.reduce_c3 = ConvBNAct(c3, out_channels, kernel_size=1)

        self.top_down_conv1 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.top_down_conv2 = ConvBNAct(out_channels, out_channels, kernel_size=3)

        self.downsample1 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
        self.bottom_up_conv1 = ConvBNAct(out_channels, out_channels, kernel_size=3)

        self.downsample2 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
        self.bottom_up_conv2 = ConvBNAct(out_channels, out_channels, kernel_size=3)

    def forward(self, p3, p4, p5):
        # Top-down pathway
        p5_reduced = self.reduce_c5(p5)
        p4_reduced = self.reduce_c4(p4)
        p3_reduced = self.reduce_c3(p3)

        p4_fused = p4_reduced + F.interpolate(p5_reduced, size=p4_reduced.shape[-2:], mode="nearest")
        p4_fused = self.top_down_conv1(p4_fused)

        p3_fused = p3_reduced + F.interpolate(p4_fused, size=p3_reduced.shape[-2:], mode="nearest")
        p3_fused = self.top_down_conv2(p3_fused)

        # Bottom-up pathway
        n3 = p3_fused
        n4 = p4_fused + self.downsample1(n3)
        n4 = self.bottom_up_conv1(n4)

        n5 = p5_reduced + self.downsample2(n4)
        n5 = self.bottom_up_conv2(n5)

        return n3, n4, n5


class DecoupledDetectionHead(nn.Module):
    """
    Decoupled Head: Separates Bounding Box Regression from Classification & Objectness.
    As shown in Figure 3 of the paper:
    - Bounding Box Regression Branch
    - Classification & Confidence Branch
    """

    def __init__(self, in_channels=128, num_classes=1):
        super().__init__()
        # Bounding box regression branch (4 coords: x, y, w, h)
        self.reg_conv = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, 4, kernel_size=1),
        )
        # Classification & Objectness branch
        self.cls_conv = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feat):
        bbox_preds = self.reg_conv(feat)
        cls_preds = self.cls_conv(feat)
        return bbox_preds, cls_preds


class YoloNasTumorDetector(nn.Module):
    """
    Complete YOLO-NAS Framework for Brain Tumor Localization.
    Combines:
    - NAS Feature Extraction Backbone
    - Multi-scale PAN Neck
    - Decoupled Bounding Box Regression & Classification Heads
    """

    def __init__(self, in_channels=3, base_channels=32, num_classes=1):
        super().__init__()
        self.backbone = YoloNasBackbone(in_channels=in_channels, base_channels=base_channels)
        c3, c4, c5 = base_channels * 4, base_channels * 8, base_channels * 16
        neck_channels = base_channels * 4
        self.neck = YoloNasPANNeck(in_channels_list=[c3, c4, c5], out_channels=neck_channels)

        # Detection heads for each scale
        self.head_p3 = DecoupledDetectionHead(in_channels=neck_channels, num_classes=num_classes)
        self.head_p4 = DecoupledDetectionHead(in_channels=neck_channels, num_classes=num_classes)
        self.head_p5 = DecoupledDetectionHead(in_channels=neck_channels, num_classes=num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass returning predictions across 3 scales.
        """
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)

        bbox3, cls3 = self.head_p3(n3)
        bbox4, cls4 = self.head_p4(n4)
        bbox5, cls5 = self.head_p5(n5)

        return [(bbox3, cls3), (bbox4, cls4), (bbox5, cls5)]

    def detect_tumor(self, img_tensor, conf_threshold=0.35, iou_threshold=0.45):
        """
        Inference method returning localized tumor bounding boxes.
        :param img_tensor: Tensor of shape (1, 3, H, W) in [0, 1] range
        :return: dict with 'boxes', 'scores', 'best_box', 'confidence'
        """
        self.eval()
        device = next(self.parameters()).device
        img_tensor = img_tensor.to(device)

        _, _, img_h, img_w = img_tensor.shape

        with torch.no_grad():
            outputs = self.forward(img_tensor)

        all_boxes = []
        all_scores = []

        strides = [8, 16, 32]
        for (bbox_map, cls_map), stride in zip(outputs, strides):
            batch, _, gh, gw = bbox_map.shape
            cls_prob = torch.sigmoid(cls_map[0, 0])  # Shape: (gh, gw)

            # Generate grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(gh, device=device),
                torch.arange(gw, device=device),
                indexing="ij",
            )

            # Center offsets + scale
            dx = torch.sigmoid(bbox_map[0, 0])
            dy = torch.sigmoid(bbox_map[0, 1])
            dw = torch.exp(bbox_map[0, 2].clamp(-3, 3))
            dh = torch.exp(bbox_map[0, 3].clamp(-3, 3))

            cx = (grid_x + dx) * stride
            cy = (grid_y + dy) * stride
            w = (dw * stride * 2).clamp(min=30.0, max=img_w * 0.7)
            h = (dh * stride * 2).clamp(min=30.0, max=img_h * 0.7)

            xmin = (cx - w / 2).clamp(0, img_w)
            ymin = (cy - h / 2).clamp(0, img_h)
            xmax = (cx + w / 2).clamp(0, img_w)
            ymax = (cy + h / 2).clamp(0, img_h)

            mask = cls_prob > conf_threshold
            if mask.sum() > 0:
                selected_boxes = torch.stack(
                    [xmin[mask], ymin[mask], xmax[mask], ymax[mask]], dim=-1
                )
                selected_scores = cls_prob[mask]
                all_boxes.append(selected_boxes.cpu().numpy())
                all_scores.append(selected_scores.cpu().numpy())

        if len(all_boxes) == 0:
            return {
                "detected": False,
                "best_box": None,
                "confidence": 0.0,
                "all_boxes": [],
                "all_scores": [],
            }

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        # Non-Maximum Suppression (NMS)
        keep = self._nms(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]

        if len(scores) == 0:
            return {
                "detected": False,
                "best_box": None,
                "confidence": 0.0,
                "all_boxes": [],
                "all_scores": [],
            }

        best_idx = np.argmax(scores)
        best_box = boxes[best_idx].tolist()
        best_score = float(scores[best_idx])

        return {
            "detected": True,
            "best_box": [round(c, 2) for c in best_box],
            "confidence": round(best_score, 4),
            "all_boxes": boxes.tolist(),
            "all_scores": [round(s, 4) for s in scores.tolist()],
        }

    @staticmethod
    def _nms(boxes, scores, iou_threshold):
        """Standard Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep
