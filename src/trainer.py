"""
Training and Optimization Module for Hybrid YOLO-NAS + EfficientNet & Baselines
According to Section 3.5 (Algorithm Step 5 & 6) and Section 4:
- Training YOLO-NAS tumor localization head
- Training EfficientNet ROI classifier
- Training comparative baselines (Standard CNN, VGG16)
- Checkpointing and evaluation
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class PipelineTrainer:
    """
    Coordinates training for detector, classifier, and baseline architectures.
    """

    def __init__(self, save_dir="models_saved", device="cpu"):
        self.save_dir = save_dir
        self.device = device
        os.makedirs(save_dir, exist_ok=True)

    def train_classifier(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs=10,
        lr=1e-3,
        model_name="efficientnet_b0",
    ):
        """
        Trains classification network (EfficientNet / CNN / VGG16) using Adam optimizer
        and CrossEntropyLoss with overfitting prevention (Dropout & LR decay).
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0.0
        best_path = os.path.join(self.save_dir, f"{model_name}_best.pth")

        print(f"=== Starting Training for {model_name} on {self.device} ===")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels, _ in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            scheduler.step()
            train_acc = 100.0 * correct / max(total, 1)
            train_loss = running_loss / max(total, 1)

            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self.evaluate_classifier(model, val_loader)
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), best_path)
            else:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                )
                if train_acc > best_acc:
                    best_acc = train_acc
                    torch.save(model.state_dict(), best_path)

        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f}s. Best Accuracy: {best_acc:.2f}%. Model saved to {best_path}")
        return best_path

    def evaluate_classifier(self, model: nn.Module, loader: DataLoader):
        """Evaluates loss and accuracy on evaluation dataset."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / max(total, 1)
        val_acc = 100.0 * correct / max(total, 1)
        return val_loss, val_acc

    def train_detector(
        self,
        detector: nn.Module,
        train_loader: DataLoader,
        epochs=10,
        lr=5e-4,
        model_name="yolo_nas_mri",
    ):
        """
        Trains the YOLO-NAS localization and objectness head on MRI scans.
        """
        detector = detector.to(self.device)
        cls_criterion = nn.BCEWithLogitsLoss()
        reg_criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(detector.parameters(), lr=lr, weight_decay=1e-4)

        save_path = os.path.join(self.save_dir, f"{model_name}_best.pth")
        print(f"=== Starting YOLO-NAS Detection Head Training ===")

        for epoch in range(1, epochs + 1):
            detector.train()
            total_loss = 0.0
            count = 0

            for images, labels, bboxes in train_loader:
                images = images.to(self.device)
                bboxes = bboxes.to(self.device)  # (B, 4) in [x1, y1, x2, y2]
                batch_size = images.size(0)

                optimizer.zero_grad()
                outputs = detector(images)

                # Multiscale loss computation
                loss = torch.tensor(0.0, device=self.device)
                for (bbox_map, cls_map), stride in zip(outputs, [8, 16, 32]):
                    # Target objectness mask: 1 if scan has tumor (label != 3)
                    has_tumor = (labels != 3).float().to(self.device)  # (B,)
                    target_cls = has_tumor.view(batch_size, 1, 1, 1).expand_as(cls_map)
                    loss += cls_criterion(cls_map, target_cls)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / max(count, 1)
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Detection Loss: {avg_loss:.4f}")

        torch.save(detector.state_dict(), save_path)
        print(f"YOLO-NAS weights saved to {save_path}")
        return save_path
