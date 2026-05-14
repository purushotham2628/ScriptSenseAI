from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessingConfig:
    enable_super_resolution: bool = False
    enable_background_removal: bool = True
    enable_skew_correction: bool = True
    enable_stain_removal: bool = True
    enable_erosion_repair: bool = True
    enable_faded_text_enhancement: bool = True
    clahe_clip_limit: float = 2.0
    adaptive_block_size: int = 31
    adaptive_c: int = 11


class AdvancedPreprocessor:
    """Robust manuscript preprocessing for noisy, damaged, and unseen scripts."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()

    def run(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        metrics: Dict[str, float] = {}
        rgb = self._ensure_rgb(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        metrics["input_brightness"] = float(gray.mean())
        metrics["input_blur"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        normalized = self.illumination_normalization(gray)
        contrast = self.contrast_enhancement(normalized)
        denoised = self.denoise(contrast)

        if self.config.enable_skew_correction:
            denoised = self.skew_correction(denoised)

        if self.config.enable_stain_removal:
            denoised = self.stain_removal(denoised)

        enhanced = self.faded_text_enhancement(denoised) if self.config.enable_faded_text_enhancement else denoised
        sharpened = self.edge_sharpening(enhanced)
        thresholded = self.adaptive_threshold(sharpened)

        if self.config.enable_erosion_repair:
            thresholded = self.erosion_repair(thresholded)

        metrics["output_brightness"] = float(thresholded.mean())
        metrics["output_blur"] = float(cv2.Laplacian(thresholded, cv2.CV_64F).var())
        return thresholded, metrics

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image

    def illumination_normalization(self, gray: np.ndarray) -> np.ndarray:
        background = cv2.medianBlur(gray, 35)
        normalized = cv2.divide(gray, background, scale=255)
        return normalized

    def contrast_enhancement(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip_limit, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)

    def skew_correction(self, gray: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(gray < np.percentile(gray, 35)))
        if len(coords) < 20:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 15:
            return gray
        h, w = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def stain_removal(self, gray: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        return cv2.subtract(gray, background) + np.uint8(np.mean(background))

    def faded_text_enhancement(self, gray: np.ndarray) -> np.ndarray:
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        return cv2.add(cv2.subtract(gray, black_hat), top_hat)

    def edge_sharpening(self, gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
        return cv2.addWeighted(gray, 1.55, blur, -0.55, 0)

    def adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        block_size = self.config.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.config.adaptive_c,
        )

    def erosion_repair(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.medianBlur(closed, 3)

    def optional_super_resolution(self, image: np.ndarray) -> np.ndarray:
        # Hook for ESRGAN/Real-ESRGAN. Bicubic upscaling is a safe fallback.
        if not self.config.enable_super_resolution:
            return image
        return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    def optional_unet_cleanup(self, image: np.ndarray) -> np.ndarray:
        # Hook for a trained U-Net manuscript restoration model.
        return image
