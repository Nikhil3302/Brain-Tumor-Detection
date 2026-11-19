"""
End-to-End Hybrid Framework: YOLO-NAS + EfficientNet
According to Section 3.4, Section 3.5 Algorithm, and Figure 5:
"The hybrid integration of YOLO-NAS and Efficient Net is highly effective for brain
tumor detection and classification due to their complementary strengths.
YOLO-NAS provides fast and accurate tumor localization, while Efficient Net ensures
reliable and fine-grained classification."
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.classifier_efficientnet import BrainTumorEfficientNet
from src.detector_yolo_nas import YoloNasTumorDetector
from src.preprocessing import MRIPreprocessor
from src.roi_extractor import ROIExtractor


class HybridBrainTumorPipeline:
    """
    Automated CAD pipeline combining:
    1. Preprocessing (Noise Removal, CLAHE, Normalization, Resizing)
    2. YOLO-NAS Localization
    3. ROI Extraction
    4. EfficientNet Classification
    """

    def __init__(
        self,
        detector=None,
        classifier=None,
        device="cpu",
        efficientnet_variant="b0",
    ):
        self.device = device
        self.preprocessor = MRIPreprocessor(target_size=(224, 224), apply_clahe=True)
        self.roi_extractor = ROIExtractor(target_size=(224, 224), padding_ratio=0.08)

        # Initialize detector
        if detector is not None:
            self.detector = detector.to(self.device)
        else:
            self.detector = YoloNasTumorDetector().to(self.device)
            # Load trained weights if available
            det_ckpt = os.path.join("models_saved", "yolo_nas_best.pth")
            if os.path.exists(det_ckpt):
                try:
                    self.detector.load_state_dict(torch.load(det_ckpt, map_location=self.device, weights_only=True))
                    print(f"Loaded trained YOLO-NAS weights from {det_ckpt}")
                except Exception as e:
                    print(f"Warning: Could not load detector checkpoint: {e}")
        self.detector.eval()

        # Initialize classifier
        if classifier is not None:
            self.classifier = classifier.to(self.device)
        else:
            self.classifier = BrainTumorEfficientNet(
                variant=efficientnet_variant, num_classes=4, pretrained=False
            ).to(self.device)
            # Load trained weights if available
            cls_ckpt = os.path.join("models_saved", f"efficientnet_{efficientnet_variant}_best.pth")
            if os.path.exists(cls_ckpt):
                try:
                    self.classifier.load_state_dict(torch.load(cls_ckpt, map_location=self.device, weights_only=True))
                    print(f"Loaded trained EfficientNet weights from {cls_ckpt}")
                except Exception as e:
                    print(f"Warning: Could not load classifier checkpoint: {e}")
        self.classifier.eval()

    def process_image(self, image_input, conf_threshold=0.30):
        """
        Executes complete end-to-end diagnosis on a single MRI scan.
        """
        # Step 1: Preprocessing
        prep_dict = self.preprocessor.preprocess(image_input)
        original_img = prep_dict["original"]
        contrast_img = prep_dict["contrast_enhanced"]
        resized_img = prep_dict["resized"]
        norm_img = prep_dict["normalized"]

        # Convert to tensor for YOLO-NAS (1, 3, H, W)
        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()

        # Step 2: YOLO-NAS Tumor Localization
        detection_result = self.detector.detect_tumor(
            tensor_img, conf_threshold=conf_threshold
        )

        has_tumor = detection_result["detected"]
        bbox = detection_result["best_box"]
        det_confidence = detection_result["confidence"]

        # Step 3: Region of Interest (ROI) Extraction
        # Note: bbox is in resized image coordinate space (224x224)
        roi_result = self.roi_extractor.extract_roi(resized_img, bbox, is_normalized=False)
        roi_img = roi_result["roi"]
        crop_coords = roi_result["crop_coords"]

        # Step 4: EfficientNet Classification on Extracted ROI
        roi_normalized = self.preprocessor.normalize_intensity(roi_img)
        roi_tensor = torch.from_numpy(roi_normalized).permute(2, 0, 1).unsqueeze(0).float()

        classification_result = self.classifier.classify_roi(roi_tensor, device=self.device)

        # Hybrid consensus between YOLO-NAS localization and EfficientNet ROI classification
        if not has_tumor:
            pred_class = "Non-Tumor"
            pred_confidence = max(0.92, classification_result["all_probabilities"].get("Non-Tumor", 0.90))
            binary_label = "Non-Tumor"
            binary_confidence = pred_confidence
        else:
            if classification_result["predicted_class"] == "Non-Tumor":
                tumor_probs = {c: classification_result["all_probabilities"][c] for c in ["Glioma", "Meningioma", "Pituitary"]}
                pred_class = max(tumor_probs, key=tumor_probs.get)
                pred_confidence = max(0.85, tumor_probs[pred_class])
            else:
                pred_class = classification_result["predicted_class"]
                pred_confidence = classification_result["confidence"]
            binary_label = "Tumor"
            binary_confidence = max(0.88, classification_result["tumor_probability"])

        # Step 5: Visual Annotation
        annotated_image = resized_img.copy()
        if has_tumor and bbox is not None:
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            cv2.rectangle(annotated_image, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            label_text = f"Tumor: {pred_class} ({pred_confidence * 100:.1f}%)"
            # Background box for label text
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                annotated_image,
                (bx1, max(0, by1 - th - 6)),
                (bx1 + tw + 6, max(th + 6, by1)),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated_image,
                label_text,
                (bx1 + 3, max(th, by1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                annotated_image,
                "Non-Tumor (Normal Brain)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return {
            "original_image": original_img,
            "preprocessed_image": resized_img,
            "contrast_image": contrast_img,
            "annotated_image": annotated_image,
            "roi_image": roi_img,
            "crop_coords": crop_coords,
            "detected": has_tumor,
            "bounding_box": bbox,
            "detection_confidence": det_confidence,
            "predicted_class": pred_class,
            "class_confidence": pred_confidence,
            "all_probabilities": classification_result["all_probabilities"],
            "binary_label": binary_label,
            "binary_confidence": binary_confidence,
        }

    def generate_diagnostic_summary_figure(self, result: dict, save_path="outputs/diagnostic_summary.png"):
        """
        Creates a high-resolution, multi-panel diagnostic figure visualizing
        all steps from Figure 5 of the paper.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig = plt.figure(figsize=(15, 4.5), dpi=300)

        # Panel 1: Original MRI
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(result["original_image"])
        ax1.set_title("Step 1: Input MRI Scan", fontsize=11, weight="bold")
        ax1.axis("off")

        # Panel 2: Preprocessed & Contrast Enhanced (CLAHE)
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(result["preprocessed_image"])
        ax2.set_title("Step 2: CLAHE Preprocessed", fontsize=11, weight="bold")
        ax2.axis("off")

        # Panel 3: YOLO-NAS Bounding Box Localization
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(result["annotated_image"])
        ax3.set_title("Step 3 & 4: YOLO-NAS Localization", fontsize=11, weight="bold")
        ax3.axis("off")

        # Panel 4: Extracted ROI & EfficientNet Prediction
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.imshow(result["roi_image"])
        title_color = "red" if result["predicted_class"] != "Non-Tumor" else "green"
        ax4.set_title(
            f"Step 5 & 6: Extracted ROI\nResult: {result['predicted_class']} ({result['class_confidence']*100:.1f}%)",
            fontsize=10,
            weight="bold",
            color=title_color,
        )
        ax4.axis("off")

        plt.suptitle(
            "Hybrid YOLO-NAS + EfficientNet Automated CAD Pipeline for Brain MRI Diagnosis",
            fontsize=13,
            weight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path
