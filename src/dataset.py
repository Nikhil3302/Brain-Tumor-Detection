"""
Brain MRI Dataset Loader & Sample Generator
Supports:
1. Loading real brain MRI datasets (BR35H, BraTS, REMBRANDT, Kaggle Brain Tumor MRI)
2. Synthetic MRI generator accurately replicating Figure 2 and Figure 3 for 4 classes:
   - Glioma
   - Meningioma
   - Pituitary
   - Non-Tumor
   across axial, coronal, and sagittal views.
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for Brain MRI Tumor Detection and Classification.
    """

    CLASSES = ["Glioma", "Meningioma", "Pituitary", "Non-Tumor"]

    def __init__(self, file_paths, labels, bboxes=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.bboxes = bboxes
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        bbox = self.bboxes[idx] if self.bboxes is not None else [0.0, 0.0, 0.0, 0.0]

        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default normalized tensor
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(bbox, dtype=torch.float32)


class SyntheticMRIGenerator:
    """
    Generates realistic, anatomical Brain MRI scans with ground-truth
    lesions for testing CAD pipelines without external data dependencies.
    """

    CLASSES = ["Glioma", "Meningioma", "Pituitary", "Non-Tumor"]

    @staticmethod
    def draw_brain_contour(size=256, view="axial"):
        """Draws realistic brain skull, parenchyma, ventricles, and sulci."""
        img = np.zeros((size, size), dtype=np.float32)
        cx, cy = size // 2, size // 2

        if view == "axial":
            # Oval skull
            cv2.ellipse(img, (cx, cy), (int(size * 0.38), int(size * 0.44)), 0, 0, 360, 0.18, 2)
            # Brain parenchyma
            cv2.ellipse(img, (cx, cy), (int(size * 0.36), int(size * 0.42)), 0, 0, 360, 0.55, -1)
            # Cerebral hemispheres longitudinal fissure
            cv2.line(img, (cx, int(size * 0.12)), (cx, int(size * 0.88)), 0.15, 2)
            # Lateral ventricles
            cv2.ellipse(img, (cx - 16, cy - 8), (10, 28), -15, 0, 360, 0.1, -1)
            cv2.ellipse(img, (cx + 16, cy - 8), (10, 28), 15, 0, 360, 0.1, -1)
            # Third ventricle
            cv2.ellipse(img, (cx, cy + 16), (4, 12), 0, 0, 360, 0.1, -1)
        elif view == "coronal":
            # Crown-shaped skull
            cv2.ellipse(img, (cx, cy - 10), (int(size * 0.38), int(size * 0.40)), 0, 0, 360, 0.18, 2)
            cv2.ellipse(img, (cx, cy - 10), (int(size * 0.36), int(size * 0.38)), 0, 0, 360, 0.53, -1)
            # Temporal lobes & base
            cv2.ellipse(img, (cx - 40, cy + 50), (22, 35), 20, 0, 360, 0.50, -1)
            cv2.ellipse(img, (cx + 40, cy + 50), (22, 35), -20, 0, 360, 0.50, -1)
            # Ventricles
            cv2.ellipse(img, (cx - 14, cy - 10), (8, 24), -10, 0, 360, 0.1, -1)
            cv2.ellipse(img, (cx + 14, cy - 10), (8, 24), 10, 0, 360, 0.1, -1)
        else:  # sagittal
            # Sagittal profile
            cv2.ellipse(img, (cx - 15, cy - 10), (int(size * 0.40), int(size * 0.38)), 0, 0, 360, 0.18, 2)
            cv2.ellipse(img, (cx - 15, cy - 10), (int(size * 0.38), int(size * 0.36)), 0, 0, 360, 0.52, -1)
            # Cerebellum and brainstem
            cv2.ellipse(img, (cx - 35, cy + 55), (28, 22), 0, 0, 360, 0.45, -1)
            cv2.ellipse(img, (cx, cy + 65), (15, 30), 10, 0, 360, 0.48, -1)
            # Corpus callosum arch
            cv2.ellipse(img, (cx - 10, cy - 25), (35, 18), 0, 190, 350, 0.75, 4)

        # Add natural texture / gyri folds
        np.random.seed(int(cx * 37 + cy))
        noise = np.random.normal(0, 0.04, img.shape).astype(np.float32)
        img = np.clip(img + noise * (img > 0.1), 0.0, 1.0)
        return img

    @classmethod
    def generate_sample(cls, tumor_type="Glioma", size=256, view="axial", random_state=None):
        """
        Synthesizes a realistic brain MRI scan for a designated tumor type or non-tumor.
        Returns:
            rgb_image (256, 256, 3) uint8
            bbox [xmin, ymin, xmax, ymax] (int coordinates)
            label_idx (0: Glioma, 1: Meningioma, 2: Pituitary, 3: Non-Tumor)
        """
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        base_brain = cls.draw_brain_contour(size=size, view=view)
        cx, cy = size // 2, size // 2

        bbox = None
        label_idx = cls.CLASSES.index(tumor_type)

        if tumor_type == "Glioma":
            # Infiltrative parenchymal tumor, irregular margins, hyperintense core
            tx = random.randint(cx - 45, cx + 45)
            ty = random.randint(cy - 50, cy + 10)
            tr_x = random.randint(22, 38)
            tr_y = random.randint(18, 34)
            angle = random.randint(0, 180)

            # Hyperintense edematous halo
            cv2.ellipse(base_brain, (tx, ty), (tr_x + 8, tr_y + 8), angle, 0, 360, 0.72, -1)
            # Active tumor mass
            cv2.ellipse(base_brain, (tx, ty), (tr_x, tr_y), angle, 0, 360, 0.92, -1)
            # Central necrotic core
            cv2.ellipse(base_brain, (tx, ty), (tr_x // 2, tr_y // 2), angle, 0, 360, 0.35, -1)

            margin = 6
            xmin = max(0, tx - tr_x - margin)
            ymin = max(0, ty - tr_y - margin)
            xmax = min(size - 1, tx + tr_x + margin)
            ymax = min(size - 1, ty + tr_y + margin)
            bbox = [xmin, ymin, xmax, ymax]

        elif tumor_type == "Meningioma":
            # Extra-axial dural-based mass, well-circumscribed, uniform intensity
            # Located near skull boundary or midline falx
            angle = random.choice([30, 60, 120, 240, 300])
            rad = np.radians(angle)
            dist = int(size * 0.30)
            tx = int(cx + dist * np.cos(rad))
            ty = int(cy + dist * np.sin(rad))
            tr = random.randint(18, 28)

            # Well-defined hyperintense mass
            cv2.circle(base_brain, (tx, ty), tr, 0.88, -1)
            # Dural attachment enhancement
            cv2.circle(base_brain, (tx, ty), tr + 3, 0.75, 2)

            margin = 4
            bbox = [
                max(0, tx - tr - margin),
                max(0, ty - tr - margin),
                min(size - 1, tx + tr + margin),
                min(size - 1, ty + tr + margin),
            ]

        elif tumor_type == "Pituitary":
            # Located precisely at sella turcica near skull base / optic chiasm
            tx = cx
            ty = cy + int(size * 0.16)
            tr_w = random.randint(14, 22)
            tr_h = random.randint(12, 18)

            # Suprasellar rounded lesion
            cv2.ellipse(base_brain, (tx, ty), (tr_w, tr_h), 0, 0, 360, 0.86, -1)
            margin = 4
            bbox = [
                max(0, tx - tr_w - margin),
                max(0, ty - tr_h - margin),
                min(size - 1, tx + tr_w + margin),
                min(size - 1, ty + tr_h + margin),
            ]

        else:
            # Non-Tumor (Normal brain MRI scan)
            bbox = None

        # Add Gaussian blur and realistic MRI background noise
        blurred = cv2.GaussianBlur(base_brain, (3, 3), 0)
        noise = np.random.normal(0, 0.02, blurred.shape).astype(np.float32)
        mri_final = np.clip(blurred + noise, 0.0, 1.0)

        # Convert to 3-channel RGB uint8 image
        uint8_gray = (mri_final * 255).astype(np.uint8)
        rgb_image = cv2.cvtColor(uint8_gray, cv2.COLOR_GRAY2RGB)

        return rgb_image, bbox, label_idx


def build_sample_dataset(output_dir="dataset/samples", n_per_class=25):
    """
    Creates a comprehensive initial set of sample brain MRI images
    for all 4 classes across axial, coronal, and sagittal views.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    classes = ["Glioma", "Meningioma", "Pituitary", "Non-Tumor"]
    views = ["axial", "coronal", "sagittal"]

    for cls_idx, class_name in enumerate(classes):
        class_folder = os.path.join(output_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)

        for i in range(n_per_class):
            view = views[i % len(views)]
            img, bbox, label_idx = SyntheticMRIGenerator.generate_sample(
                tumor_type=class_name, size=256, view=view, random_state=(cls_idx * 100 + i)
            )

            img_filename = f"{class_name.lower()}_{view}_{i+1:03d}.png"
            img_path = os.path.join(class_folder, img_filename)
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Save annotation (YOLO format: class x_center y_center width height)
            ann_filename = f"{class_name.lower()}_{view}_{i+1:03d}.txt"
            ann_path = os.path.join(class_folder, ann_filename)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                xc = (x1 + x2) / (2.0 * 256)
                yc = (y1 + y2) / (2.0 * 256)
                bw = (x2 - x1) / 256.0
                bh = (y2 - y1) / 256.0
                with open(ann_path, "w") as f:
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                with open(ann_path, "w") as f:
                    pass  # empty for non-tumor

            metadata.append({
                "path": img_path,
                "class": class_name,
                "label": label_idx,
                "bbox": bbox,
                "view": view,
            })

    print(f"Generated {len(metadata)} sample Brain MRI scans across 4 classes in {output_dir}")
    return metadata
