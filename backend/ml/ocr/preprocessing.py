from __future__ import annotations

import cv2
import numpy as np

from backend.ml.ocr.quality import ImageQualityAnalyzer, ImageQualityReport


class AdaptivePreprocessor:
    """Adaptive preprocessing that avoids destructive over-binarization."""

    def __init__(self) -> None:
        self.quality_analyzer = ImageQualityAnalyzer()

    def run(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, ImageQualityReport, list[dict]]:
        report = self.quality_analyzer.analyze(image)
        gray = self._gray(image)
        steps: list[dict] = [{"name": "image_quality_analysis", "status": "completed", "details": report.to_dict()}]

        normalized = self._illumination_normalize(gray, strength=report.strategy)
        steps.append({"name": "illumination_normalization", "status": "completed"})

        clip = {"mild": 1.6, "medium": 2.2, "aggressive": 2.8}[report.strategy]
        enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(normalized)
        steps.append({"name": "clahe", "status": "completed", "details": {"clip_limit": clip}})

        if report.strategy == "mild":
            denoised = cv2.bilateralFilter(enhanced, 5, 35, 35)
        elif report.strategy == "medium":
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 8, 7, 21)
            denoised = cv2.bilateralFilter(denoised, 7, 45, 45)
        else:
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 12, 7, 21)
            denoised = cv2.bilateralFilter(denoised, 9, 60, 60)
        steps.append({"name": "edge_preserving_denoising", "status": "completed", "details": {"mode": report.strategy}})

        sharpened = self._edge_preserving_sharpen(denoised, report.strategy)
        steps.append({"name": "edge_preserving_sharpening", "status": "completed"})

        soft_binary = self._soft_threshold(sharpened, report.strategy)
        cleaned_binary = self._morphological_cleanup(soft_binary, report.strategy)
        steps.append({"name": "adaptive_gaussian_threshold", "status": "completed", "details": {"mode": report.strategy}})
        steps.append({"name": "morphological_cleanup", "status": "completed"})

        # OCR engines receive a detail-preserving grayscale image; the binary image is
        # kept for segmentation and previews only.
        ocr_image = cv2.addWeighted(sharpened, 0.82, cleaned_binary, 0.18, 0)
        return ocr_image, cleaned_binary, report, steps

    def _gray(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)

    def _illumination_normalize(self, gray: np.ndarray, strength: str) -> np.ndarray:
        kernel_size = {"mild": 31, "medium": 45, "aggressive": 61}[strength]
        if kernel_size % 2 == 0:
            kernel_size += 1
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        normalized = cv2.divide(gray, background, scale=255)
        return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    def _edge_preserving_sharpen(self, image: np.ndarray, mode: str) -> np.ndarray:
        amount = {"mild": 0.45, "medium": 0.65, "aggressive": 0.85}[mode]
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.1)
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    def _soft_threshold(self, image: np.ndarray, mode: str) -> np.ndarray:
        block_size = {"mild": 41, "medium": 35, "aggressive": 29}[mode]
        c_value = {"mild": 13, "medium": 10, "aggressive": 7}[mode]
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value)

    def _morphological_cleanup(self, binary: np.ndarray, mode: str) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1 if mode != "aggressive" else 0)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
