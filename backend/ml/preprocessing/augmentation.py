from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:  # pragma: no cover - optional in lightweight installs
    A = None


class RobustAugmentationFactory:
    """Augmentations that simulate real manuscript degradation for low-data scripts."""

    def build(self, image_size: int = 384):
        if A is None:
            return None
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=255),
            A.Rotate(limit=8, border_mode=cv2.BORDER_REPLICATE, p=0.7),
            A.Perspective(scale=(0.02, 0.08), p=0.35),
            A.ElasticTransform(alpha=18, sigma=5, alpha_affine=4, p=0.25),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
            ], p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.75),
            A.GaussNoise(var_limit=(8, 80), p=0.45),
            A.CoarseDropout(max_holes=8, max_height=42, max_width=42, min_holes=1, fill_value=235, p=0.35),
            A.Lambda(image=self._synthetic_cracks, p=0.35),
            A.Lambda(image=self._ink_degradation, p=0.45),
            A.Lambda(image=self._faded_character_simulation, p=0.45),
        ])

    def _synthetic_cracks(self, image: np.ndarray, **kwargs) -> np.ndarray:
        output = image.copy()
        h, w = output.shape[:2]
        for _ in range(np.random.randint(1, 5)):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2 = int(np.clip(x1 + np.random.normal(0, w * 0.25), 0, w - 1))
            y2 = int(np.clip(y1 + np.random.normal(0, h * 0.25), 0, h - 1))
            color = int(np.random.randint(20, 95))
            cv2.line(output, (x1, y1), (x2, y2), color=(color, color, color), thickness=np.random.randint(1, 3))
        return output

    def _ink_degradation(self, image: np.ndarray, **kwargs) -> np.ndarray:
        output = image.copy().astype(np.float32)
        mask = np.random.uniform(0.55, 1.0, size=output.shape[:2]).astype(np.float32)
        if output.ndim == 3:
            mask = mask[..., None]
        output = output * mask + 255 * (1 - mask) * 0.25
        return np.clip(output, 0, 255).astype(np.uint8)

    def _faded_character_simulation(self, image: np.ndarray, **kwargs) -> np.ndarray:
        alpha = np.random.uniform(0.58, 0.88)
        parchment = np.full_like(image, np.random.randint(218, 246))
        return cv2.addWeighted(image, alpha, parchment, 1 - alpha, 0)
