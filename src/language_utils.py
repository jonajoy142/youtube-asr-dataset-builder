"""
Language detection utilities for code-switching detection.
Handles Malayalam script detection via Unicode ranges + langdetect for English.
"""

import re
import unicodedata
from typing import List, Dict, Tuple

# Malayalam Unicode block: 0D00–0D7F
MALAYALAM_RANGE = (0x0D00, 0x0D7F)

# Other Indic scripts we want to detect (for future phases)
INDIC_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari (Hindi)
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "bn": (0x0980, 0x09FF),  # Bengali
}


def is_malayalam_char(char: str) -> bool:
    """Check if a character is in the Malayalam Unicode block."""
    code = ord(char)
    return MALAYALAM_RANGE[0] <= code <= MALAYALAM_RANGE[1]


def is_indic_char(char: str, lang_code: str = "ml") -> bool:
    """Check if a character belongs to a given Indic script."""
    if lang_code == "ml":
        return is_malayalam_char(char)
    if lang_code in INDIC_RANGES:
        code = ord(char)
        lo, hi = INDIC_RANGES[lang_code]
        return lo <= code <= hi
    return False


def detect_word_language(word: str, primary_lang: str = "ml") -> str:
    """
    Detect whether a word is in the primary language or English.
    Uses Unicode script ranges — no external API needed.

    Returns: 'ml' (or primary_lang code) | 'en' | 'mixed' | 'unknown'
    """
    if not word.strip():
        return "unknown"

    # Strip punctuation for analysis
    clean = re.sub(r'[^\w\s]', '', word, flags=re.UNICODE)
    if not clean:
        return "unknown"

    indic_chars = sum(1 for c in clean if is_indic_char(c, primary_lang))
    latin_chars = sum(1 for c in clean if c.isascii() and c.isalpha())
    total = indic_chars + latin_chars

    if total == 0:
        return "unknown"

    indic_ratio = indic_chars / total

    if indic_ratio >= 0.8:
        return primary_lang
    elif indic_ratio <= 0.1:
        return "en"
    else:
        return "mixed"


def get_language_spans(
    words: List[Dict],
    primary_lang: str = "ml"
) -> Tuple[List[Dict], List[float]]:
    """
    Given WhisperX word-level output, assign language tags and find switch points.

    Args:
        words: List of dicts with keys: word, start, end, score
        primary_lang: ISO 639-1 code of the primary non-English language

    Returns:
        (language_spans, switch_points)
        language_spans: list of {start, end, lang, text}
        switch_points: list of timestamps where language switches
    """
    if not words:
        return [], []

    language_spans = []
    switch_points = []
    prev_lang = None

    for word_data in words:
        word = word_data.get("word", "").strip()
        start = word_data.get("start", 0.0)
        end = word_data.get("end", 0.0)

        lang = detect_word_language(word, primary_lang)

        # Normalize mixed → best guess based on context
        if lang == "mixed":
            lang = primary_lang  # default mixed to primary lang

        if lang == "unknown":
            # Carry forward previous language
            lang = prev_lang if prev_lang else "en"

        # Detect switch point
        if prev_lang is not None and lang != prev_lang and lang in ("en", primary_lang):
            switch_points.append(round(start, 3))

        language_spans.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "lang": lang,
            "text": word
        })

        prev_lang = lang

    return language_spans, switch_points


def compute_language_ratio(language_spans: List[Dict], primary_lang: str = "ml") -> Dict:
    """
    Compute ratio of words in each language.

    Returns: {primary_lang: float, 'en': float}
    """
    total = len(language_spans)
    if total == 0:
        return {primary_lang: 0.0, "en": 0.0}

    primary_count = sum(1 for s in language_spans if s["lang"] == primary_lang)
    en_count = sum(1 for s in language_spans if s["lang"] == "en")

    return {
        primary_lang: round(primary_count / total, 3),
        "en": round(en_count / total, 3)
    }


def is_valid_codeswitched_clip(
    language_spans: List[Dict],
    switch_points: List[float],
    primary_lang: str = "ml",
    min_ratio: float = 0.15
) -> Tuple[bool, str]:
    """
    Determine if a clip is valid for the code-switching benchmark.

    Rules:
    - Must have at least 1 switch point
    - Minority language must be >= min_ratio of words
    - Must have both primary_lang and English words

    Returns: (is_valid, rejection_reason)
    """
    if not switch_points:
        return False, "no_switch_points"

    ratios = compute_language_ratio(language_spans, primary_lang)

    primary_ratio = ratios.get(primary_lang, 0)
    en_ratio = ratios.get("en", 0)

    minority_ratio = min(primary_ratio, en_ratio)

    if minority_ratio < min_ratio:
        return False, f"minority_lang_ratio_too_low ({minority_ratio:.2f} < {min_ratio})"

    if primary_ratio == 0:
        return False, "no_primary_language_detected"

    if en_ratio == 0:
        return False, "no_english_detected"

    return True, "ok"
