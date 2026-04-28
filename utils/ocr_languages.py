"""
EasyOCR language handling helpers.

Some advertised project languages are not available in every EasyOCR build, and
some supported languages cannot be loaded together in one reader.
"""

from typing import List, Sequence, Tuple

import easyocr
from easyocr.config import all_lang_list


def create_easyocr_reader(languages: Sequence[str]) -> Tuple[easyocr.Reader, List[str]]:
    """
    Create an EasyOCR reader with a compatible language list.

    Args:
        languages: Requested EasyOCR language codes.

    Returns:
        Tuple of (reader, resolved language list).
    """
    supported = set(all_lang_list)
    requested = list(dict.fromkeys(languages or ["en"]))
    filtered = [lang for lang in requested if lang in supported]

    if not filtered:
        filtered = ["en"]

    candidates = [filtered]

    if "en" in filtered:
        candidates.extend([["en", lang] for lang in filtered if lang != "en"])
        candidates.append(["en"])
    else:
        candidates.extend([[lang] for lang in filtered])
        candidates.append(["en"])

    last_error = None
    for candidate in candidates:
        try:
            reader = easyocr.Reader(candidate, gpu=False, download_enabled=False)
            skipped = [lang for lang in requested if lang not in candidate]
            if skipped:
                print(
                    "EasyOCR warning: using languages "
                    f"{candidate}; skipped unsupported/incompatible languages {skipped}"
                )
            return reader, candidate
        except (FileNotFoundError, ValueError) as exc:
            last_error = exc

    raise ValueError(f"Could not initialize EasyOCR reader: {last_error}")
