from __future__ import annotations

import re

import numpy as np

from backend.ml.ocr.quality import ImageQualityReport


class ConfidenceFusion:
    """Weighted confidence fusion across image, OCR, language, and token validity."""

    def fuse(
        self,
        ocr_confidence: float,
        quality: ImageQualityReport,
        language_confidence: float,
        corrected_text: str,
        unknown_ratio: float,
    ) -> dict:
        token_validity = self._token_validity(corrected_text, unknown_ratio)
        score = (
            0.42 * np.clip(ocr_confidence, 0.0, 1.0)
            + 0.22 * np.clip(quality.readability, 0.0, 1.0)
            + 0.18 * np.clip(language_confidence, 0.0, 1.0)
            + 0.18 * token_validity
        )
        score = float(np.clip(score, 0.0, 1.0))
        warnings = []
        if score < 0.50:
            warnings.append("Text too degraded for reliable translation.")
        if quality.strategy == "aggressive":
            warnings.append("Image required aggressive restoration; faint character readings may be unstable.")
        if unknown_ratio > 0.35:
            warnings.append("High unknown-token ratio after OCR correction.")
        return {
            "confidence": score,
            "token_validity": token_validity,
            "warnings": warnings,
            "components": {
                "ocr_confidence": float(ocr_confidence),
                "image_readability": float(quality.readability),
                "language_confidence": float(language_confidence),
                "token_validity": float(token_validity),
            },
        }

    def _token_validity(self, text: str, unknown_ratio: float) -> float:
        tokens = re.findall(r"\w+", text or "", flags=re.UNICODE)
        if not tokens:
            return 0.0
        useful = [token for token in tokens if len(token) > 1]
        length_score = min(1.0, len(useful) / max(4, len(tokens)))
        return float(np.clip(0.65 * length_score + 0.35 * (1.0 - unknown_ratio), 0.0, 1.0))
