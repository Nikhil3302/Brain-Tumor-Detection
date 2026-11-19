"""
Region of Interest (ROI) Extraction Module
According to Section 3.2:
"Following tumor region detection by YOLO-NAS, regions of interest (ROIs) are taken
from the original MRI images using the predicted bounding boxes. ROI extraction lessens
background noise, making sure the classification focuses on areas relevant to tumors."
"""

import cv2
import numpy as np


class ROIExtractor:
    """
    Extracts and standardizes the tumor Region of Interest (ROI) from brain MRI scans
    based on bounding box coordinates predicted by YOLO-NAS.
    """

    def __init__(self, target_size=(224, 224), padding_ratio=0.08):
        """
        :param target_size: Output resolution for EfficientNet classification (default: 224x224)
        :param padding_ratio: Marginal margin around tumor boundary to capture context
        """
        self.target_size = target_size
        self.padding_ratio = padding_ratio

    def extract_roi(self, image: np.ndarray, bbox, is_normalized=False) -> dict:
        """
        Extracts ROI from the image using bounding box [xmin, ymin, xmax, ymax].

        :param image: Input MRI image (H, W, C) or (H, W)
        :param bbox: [xmin, ymin, xmax, ymax] or None
        :param is_normalized: Whether bbox coords are in [0, 1]
        :return: dict with 'roi', 'crop_coords', and 'has_tumor_region'
        """
        h, w = image.shape[:2]

        if bbox is None or len(bbox) != 4:
            # If no tumor detected (Non-tumor scan), extract central brain tissue
            # as Region of Interest for Non-Tumor verification
            center_x, center_y = w // 2, h // 2
            half_w, half_h = int(w * 0.35), int(h * 0.35)
            xmin = max(0, center_x - half_w)
            ymin = max(0, center_y - half_h)
            xmax = min(w, center_x + half_w)
            ymax = min(h, center_y + half_h)
            has_tumor_region = False
        else:
            if is_normalized:
                xmin = int(bbox[0] * w)
                ymin = int(bbox[1] * h)
                xmax = int(bbox[2] * w)
                ymax = int(bbox[3] * h)
            else:
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Apply contextual padding around bounding box
            box_w = xmax - xmin
            box_h = ymax - ymin
            pad_x = int(box_w * self.padding_ratio)
            pad_y = int(box_h * self.padding_ratio)

            xmin = max(0, xmin - pad_x)
            ymin = max(0, ymin - pad_y)
            xmax = min(w, xmax + pad_x)
            ymax = min(h, ymax + pad_y)

            # Ensure non-empty crop
            if xmax <= xmin or ymax <= ymin:
                xmin, ymin, xmax, ymax = 0, 0, w, h
            has_tumor_region = True

        # Crop ROI
        roi_crop = image[ymin:ymax, xmin:xmax]

        # Resize to standard EfficientNet dimensions
        roi_resized = cv2.resize(
            roi_crop, self.target_size, interpolation=cv2.INTER_LINEAR
        )

        return {
            "roi": roi_resized,
            "raw_crop": roi_crop,
            "crop_coords": (xmin, ymin, xmax, ymax),
            "has_tumor_region": has_tumor_region,
        }
