from __future__ import annotations

import re
import unicodedata


class OCRCorrector:
    """Lightweight correction with optional SymSpell/transformer hooks."""

    COMMON_LATIN = {
        "et", "in", "est", "sunt", "erat", "deus", "dominus", "rex", "annus", "filius",
        "pater", "mater", "civitas", "populus", "vita", "mors", "terra", "sanctus",
    }

    def __init__(self) -> None:
        self._symspell = None
        self._symspell_loaded = False

    def correct(self, text: str, language: str = "auto") -> dict:
        original = text or ""
        corrected = unicodedata.normalize("NFKC", original)
        corrected = re.sub(r"([a-z])([A-Z])", r"\1 \2", corrected)
        corrected = re.sub(r"([A-Za-z])[-_~]+([A-Za-z])", r"\1\2", corrected)
        corrected = re.sub(r"\s+", " ", corrected).strip()
        corrected = self._fix_common_confusions(corrected, language)
        corrected = self._symspell_correct(corrected, language)
        unknown_ratio = self._unknown_ratio(corrected, language)
        delta = 0.04 if corrected and corrected != original else 0.0
        if unknown_ratio > 0.55:
            delta -= 0.10
        return {"text": corrected, "unknown_symbol_ratio": unknown_ratio, "confidence_delta": delta}

    def _fix_common_confusions(self, text: str, language: str) -> str:
        if language in {"la", "en", "auto"}:
            replacements = {"|": "I", "0": "O", "5": "S", " rn ": " m ", "vv": "w"}
            for old, new in replacements.items():
                text = text.replace(old, new)
        return text

    def _symspell_correct(self, text: str, language: str) -> str:
        if language not in {"la", "en"}:
            return text
        if not self._symspell_loaded:
            self._symspell_loaded = True
            try:
                from symspellpy import SymSpell

                self._symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                for word in self.COMMON_LATIN:
                    self._symspell.create_dictionary_entry(word, 50)
            except Exception:
                self._symspell = None
        if self._symspell is None:
            return text
        words = []
        for word in text.split():
            clean = re.sub(r"[^A-Za-z]", "", word).lower()
            if len(clean) < 4:
                words.append(word)
                continue
            suggestions = self._symspell.lookup(clean, 2, max_edit_distance=1)
            words.append(suggestions[0].term if suggestions else word)
        return " ".join(words)

    def _unknown_ratio(self, text: str, language: str) -> float:
        tokens = [token.lower() for token in re.findall(r"[A-Za-z]{2,}", text)]
        if not tokens:
            return 1.0
        if language == "la":
            known = sum(token in self.COMMON_LATIN or len(token) > 3 for token in tokens)
        else:
            known = sum(len(token) > 1 for token in tokens)
        return 1.0 - known / len(tokens)
