"""
Model Training Script for Brain Tumor Detection and Classification
Trains:
1. BrainTumorEfficientNet on MRI Regions of Interest (ROIs)
2. YoloNasTumorDetector localization head on MRI bounding boxes
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Set cache
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib")

from src.classifier_efficientnet import BrainTumorEfficientNet
from src.detector_yolo_nas import YoloNasTumorDetector
from src.dataset import build_sample_dataset, SyntheticMRIGenerator
from src.roi_extractor import ROIExtractor
from src.preprocessing import MRIPreprocessor


class ROITrainingDataset(Dataset):
    """Dataset of extracted ROIs for fine-tuning EfficientNet."""

    def __init__(self, samples_meta):
        self.samples = samples_meta
        self.preprocessor = MRIPreprocessor(target_size=(224, 224), apply_clahe=True)
        self.roi_extractor = ROIExtractor(target_size=(224, 224), padding_ratio=0.08)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        img = cv2.imread(meta["path"])
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = meta["label"]
        bbox = meta["bbox"]

        # Extract ROI
        roi_res = self.roi_extractor.extract_roi(img, bbox, is_normalized=False)
        roi = roi_res["roi"]

        # Normalize
        norm_roi = self.preprocessor.normalize_intensity(roi)
        tensor_roi = torch.from_numpy(norm_roi).permute(2, 0, 1).float()

        return tensor_roi, torch.tensor(label, dtype=torch.long)


class DetectorTrainingDataset(Dataset):
    """Dataset for training YOLO-NAS tumor localization."""

    def __init__(self, samples_meta):
        self.samples = samples_meta
        self.preprocessor = MRIPreprocessor(target_size=(224, 224), apply_clahe=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        img = cv2.imread(meta["path"])
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        prep = self.preprocessor.preprocess(img)
        resized_img = prep["resized"]
        norm_img = prep["normalized"]

        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        has_tumor = 0.0 if meta["label"] == 3 else 1.0  # 3 is Non-Tumor

        # Normalize bbox to [0, 1] relative to resized image
        if meta["bbox"] is not None and has_tumor > 0:
            x1, y1, x2, y2 = meta["bbox"]
            norm_box = [
                x1 / float(orig_w),
                y1 / float(orig_h),
                x2 / float(orig_w),
                y2 / float(orig_h),
            ]
        else:
            norm_box = [0.0, 0.0, 0.0, 0.0]

        return tensor_img, torch.tensor(has_tumor, dtype=torch.float32), torch.tensor(norm_box, dtype=torch.float32)


def train_models(epochs=12, batch_size=16):
    os.makedirs("models_saved", exist_ok=True)
    os.makedirs("dataset/train_samples", exist_ok=True)

    print("\n[Step 1] Preparing Curated Brain MRI Training Data (N=160)...")
    meta = build_sample_dataset("dataset/train_samples", n_per_class=40)

    # 1. Train EfficientNet ROI Classifier
    print("\n[Step 2] Training EfficientNet ROI Classifier...")
    roi_dataset = ROITrainingDataset(meta)
    roi_loader = DataLoader(roi_dataset, batch_size=batch_size, shuffle=True)

    classifier = BrainTumorEfficientNet(variant="b0", num_classes=4, pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    classifier.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in roi_loader:
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        acc = 100.0 * correct / max(total, 1)
        print(f"  EfficientNet Epoch [{epoch:02d}/{epochs:02d}] - Loss: {total_loss/total:.4f} | Accuracy: {acc:.1f}%")

    classifier_path = "models_saved/efficientnet_b0_best.pth"
    torch.save(classifier.state_dict(), classifier_path)
    print(f"  [+] EfficientNet weights saved to {classifier_path}")

    # 2. Train YOLO-NAS Localization Head
    print("\n[Step 3] Training YOLO-NAS Tumor Localization Head...")
    det_dataset = DetectorTrainingDataset(meta)
    det_loader = DataLoader(det_dataset, batch_size=batch_size, shuffle=True)

    detector = YoloNasTumorDetector()
    cls_criterion = nn.BCEWithLogitsLoss()
    reg_criterion = nn.SmoothL1Loss()
    optimizer_det = optim.AdamW(detector.parameters(), lr=8e-4, weight_decay=1e-4)

    detector.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        count = 0
        for images, has_tumor, norm_boxes in det_loader:
            b_size, _, img_h, img_w = images.shape
            optimizer_det.zero_grad()
            outputs = detector(images)

            loss = torch.tensor(0.0)
            for (bbox_map, cls_map), stride in zip(outputs, [8, 16, 32]):
                gh, gw = cls_map.shape[-2:]
                target_cls = torch.zeros_like(cls_map)
                
                for b in range(b_size):
                    if has_tumor[b] > 0.5:
                        # Bounding box center in pixels
                        x1 = norm_boxes[b, 0] * img_w
                        y1 = norm_boxes[b, 1] * img_h
                        x2 = norm_boxes[b, 2] * img_w
                        y2 = norm_boxes[b, 3] * img_h
                        
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        
                        gx = int(cx / stride)
                        gy = int(cy / stride)
                        gx = min(max(0, gx), gw - 1)
                        gy = min(max(0, gy), gh - 1)
                        
                        target_cls[b, 0, gy, gx] = 1.0

                loss = loss + cls_criterion(cls_map, target_cls)

            loss.backward()
            optimizer_det.step()
            epoch_loss += loss.item()
            count += 1

        print(f"  YOLO-NAS Epoch [{epoch:02d}/{epochs:02d}] - Localization Loss: {epoch_loss/count:.4f}")

    detector_path = "models_saved/yolo_nas_best.pth"
    torch.save(detector.state_dict(), detector_path)
    print(f"  [+] YOLO-NAS weights saved to {detector_path}")

    print("\n[Complete] All models trained and saved successfully!")


if __name__ == "__main__":
    train_models(epochs=12, batch_size=16)
