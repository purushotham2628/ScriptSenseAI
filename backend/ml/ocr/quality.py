from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class ImageQualityReport:
    blur_score: float
    noise_level: float
    contrast_score: float
    skew_angle: float
    text_density: float
    readability: float
    strategy: str

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


class ImageQualityAnalyzer:
    """Estimate manuscript quality and choose a conservative preprocessing mode."""

    def analyze(self, image: np.ndarray) -> ImageQualityReport:
        gray = self._gray(image)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise_level = self._estimate_noise(gray)
        contrast_score = self._contrast_score(gray)
        skew_angle = self._estimate_skew(gray)
        text_density = self._estimate_text_density(gray)

        blur_norm = np.clip(blur_score / 650.0, 0.0, 1.0)
        noise_norm = 1.0 - np.clip(noise_level / 42.0, 0.0, 1.0)
        contrast_norm = np.clip(contrast_score / 72.0, 0.0, 1.0)
        density_norm = 1.0 - abs(float(text_density) - 0.18) / 0.18
        density_norm = float(np.clip(density_norm, 0.0, 1.0))
        skew_norm = 1.0 - np.clip(abs(skew_angle) / 12.0, 0.0, 1.0)
        readability = float(
            np.clip(
                0.30 * blur_norm
                + 0.24 * noise_norm
                + 0.24 * contrast_norm
                + 0.12 * density_norm
                + 0.10 * skew_norm,
                0.0,
                1.0,
            )
        )
        strategy = self._choose_strategy(readability, noise_level, contrast_score, blur_score)

        return ImageQualityReport(
            blur_score=blur_score,
            noise_level=float(noise_level),
            contrast_score=float(contrast_score),
            skew_angle=float(skew_angle),
            text_density=float(text_density),
            readability=readability,
            strategy=strategy,
        )

    def _gray(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)

    def _estimate_noise(self, gray: np.ndarray) -> float:
        median = cv2.medianBlur(gray, 3)
        residual = gray.astype("float32") - median.astype("float32")
        return float(np.std(residual))

    def _contrast_score(self, gray: np.ndarray) -> float:
        p5, p95 = np.percentile(gray, [5, 95])
        local_std = cv2.blur((gray.astype("float32") - cv2.blur(gray.astype("float32"), (9, 9))) ** 2, (9, 9))
        return float((p95 - p5) * 0.75 + np.sqrt(local_std).mean() * 0.25)

    def _estimate_skew(self, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=max(30, gray.shape[1] // 8), maxLineGap=12)
        if lines is None:
            return 0.0
        angles = []
        for line in lines[:80]:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -35 <= angle <= 35:
                angles.append(angle)
        return float(np.median(angles)) if angles else 0.0

    def _estimate_text_density(self, gray: np.ndarray) -> float:
        enhanced = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return float(np.count_nonzero(binary) / binary.size)

    def _choose_strategy(self, readability: float, noise: float, contrast: float, blur: float) -> str:
        if readability < 0.32 or noise > 34 or contrast < 24 or blur < 65:
            return "aggressive"
        if readability < 0.62 or noise > 18 or contrast < 45 or blur < 180:
            return "medium"
        return "mild"
