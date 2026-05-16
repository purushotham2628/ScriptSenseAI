from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class SuperResolutionService:
    """Real-ESRGAN integration with a deterministic OpenCV fallback."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path
        self._upsampler = None
        self._load_error: str | None = None
        self._tried_load = False

    def upscale(self, image: np.ndarray, quality_readability: float) -> tuple[np.ndarray, dict]:
        height, width = image.shape[:2]
        shortest = min(height, width)
        scale = 4 if shortest < 520 else 2 if shortest < 1200 or quality_readability < 0.55 else 1
        if scale == 1:
            return image, {"scale": 1, "engine": "none", "status": "skipped"}

        upscaled = self._real_esrgan_upscale(image, scale)
        if upscaled is not None:
            return upscaled, {"scale": scale, "engine": "Real-ESRGAN", "status": "completed"}

        interpolation = cv2.INTER_CUBIC if scale <= 2 else cv2.INTER_LANCZOS4
        fallback = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
        return fallback, {
            "scale": scale,
            "engine": "opencv_lanczos",
            "status": "fallback",
            "note": self._load_error or "Real-ESRGAN package/model unavailable",
        }

    def _real_esrgan_upscale(self, image: np.ndarray, scale: int) -> np.ndarray | None:
        if not self._tried_load:
            self._tried_load = True
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer

                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                self._upsampler = RealESRGANer(
                    scale=4,
                    model_path=str(self.model_path) if self.model_path else "RealESRGAN_x4plus.pth",
                    model=model,
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    gpu_id=None,
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                self._load_error = str(exc)
                self._upsampler = None
        if self._upsampler is None:
            return None
        try:  # pragma: no cover - optional dependency
            output, _ = self._upsampler.enhance(image, outscale=scale)
            return output
        except Exception as exc:
            self._load_error = str(exc)
            return None
