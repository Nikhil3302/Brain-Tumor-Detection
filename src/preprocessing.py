"""
Image Preprocessing Pipeline for Brain MRI Images
According to Section 3.5 (Algorithm Step 3) and Figure 5:
- Noise removal (Gaussian / Median filtering)
- Intensity Normalization (Min-Max normalization [0, 1])
- Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
- Image Resizing (Standard input dimensions)
- Data Augmentation (Rotation, Flipping)
"""

import cv2
import numpy as np
from PIL import Image


class MRIPreprocessor:
    """
    Standardized MRI Preprocessing Pipeline as outlined in the research paper.
    """

    def __init__(
        self,
        target_size=(224, 224),
        apply_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        noise_filter="median",
        median_kernel=3,
        gaussian_kernel=(5, 5),
    ):
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size
        )
        self.noise_filter = noise_filter
        self.median_kernel = median_kernel
        self.gaussian_kernel = gaussian_kernel

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Removes acquisition noise from MRI scans.
        """
        if self.noise_filter == "median":
            return cv2.medianBlur(image, self.median_kernel)
        elif self.noise_filter == "gaussian":
            return cv2.GaussianBlur(image, self.gaussian_kernel, 0)
        elif self.noise_filter == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        to improve visibility of tumor boundaries against soft tissues.
        """
        if not self.apply_clahe:
            return image

        if len(image.shape) == 2:
            return self.clahe.apply(image)
        elif len(image.shape) == 3:
            # Convert to LAB color space, apply CLAHE on L-channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = self.clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        return image

    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes pixel intensity to [0.0, 1.0] range.
        """
        image_float = image.astype(np.float32)
        min_val = np.min(image_float)
        max_val = np.max(image_float)
        if max_val - min_val > 1e-6:
            normalized = (image_float - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image_float)
        return normalized

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes MRI scan to standard input dimensions (e.g. 224x224).
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

    def preprocess(self, image_input) -> dict:
        """
        Executes complete multi-stage preprocessing pipeline.
        Returns a dictionary containing intermediate and final steps.
        """
        # Load / convert to numpy array
        if isinstance(image_input, str):
            # File path
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not load image from {image_input}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 3:
                img_rgb = image_input.copy()
            else:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            raise TypeError("Unsupported image input type")

        # Step 1: Denoise
        denoised = self.remove_noise(img_rgb)

        # Step 2: Contrast Enhancement (CLAHE)
        contrast_enhanced = self.enhance_contrast(denoised)

        # Step 3: Resize
        resized = self.resize_image(contrast_enhanced)

        # Step 4: Intensity Normalization
        normalized = self.normalize_intensity(resized)

        return {
            "original": img_rgb,
            "denoised": denoised,
            "contrast_enhanced": contrast_enhanced,
            "resized": resized,
            "normalized": normalized,
        }

    @staticmethod
    def augment(image: np.ndarray, bboxes=None):
        """
        Applies rotation and horizontal/vertical flipping as described in Section 3.6.
        """
        h, w = image.shape[:2]
        augmented_images = []

        # Original
        augmented_images.append(image)

        # Horizontal Flip
        h_flip = cv2.flip(image, 1)
        augmented_images.append(h_flip)

        # Vertical Flip
        v_flip = cv2.flip(image, 0)
        augmented_images.append(v_flip)

        # 90-degree Rotation
        rot90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(rot90)

        return augmented_images
