"""
WhisperX transcription module.
Handles device detection (MPS for M4 Mac, CUDA for cloud, CPU fallback).
Returns word-level timestamps for switch point detection.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_device() -> Tuple[str, str]:
    """
    Auto-detect best available device.
    Returns: (device_str, compute_type)

    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Using CUDA GPU")
            return "cuda", "float16"

        # Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple MPS (M-series chip)")
            # whisperx on MPS needs int8 — float16 not fully supported
            return "mps", "int8"

    except ImportError:
        pass

    logger.info("Using CPU (no GPU detected)")
    return "cpu", "int8"


def load_whisper_model(model_size: str = "small"):
    """
    Load WhisperX model with appropriate device settings.
    Call once and reuse — model loading is slow.
    """
    import whisperx

    device, compute_type = get_device()

    logger.info(f"Loading whisper-{model_size} on {device} ({compute_type})")

    # MPS workaround: whisperx uses faster-whisper which doesn't support MPS directly
    # Fall back to CPU for the actual whisper inference, MPS for alignment
    whisper_device = "cpu" if device == "mps" else device

    model = whisperx.load_model(
        model_size,
        device=whisper_device,
        compute_type=compute_type,
        language=None,  # auto-detect — important for code-switched speech
    )

    return model, device


def transcribe_audio(
    audio_path: str,
    model,
    device: str,
    batch_size: int = 16
) -> Optional[List[Dict]]:
    """
    Transcribe audio file with WhisperX, returning word-level timestamps.

    Returns list of word dicts: [{word, start, end, score}]
    Returns None on failure.
    """
    import whisperx

    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None

    try:
        # Step 1: Load audio
        audio = whisperx.load_audio(audio_path)

        # Step 2: Transcribe (get segments with approximate timestamps)
        # batch_size lower on CPU/MPS to avoid OOM
        actual_batch_size = batch_size if device == "cuda" else 4
        result = model.transcribe(audio, batch_size=actual_batch_size)

        if not result.get("segments"):
            logger.warning(f"No segments transcribed for {audio_path}")
            return []

        # Step 3: Align to get word-level timestamps
        # Use CPU for alignment device on MPS (more compatible)
        align_device = "cpu" if device == "mps" else device
        detected_lang = result.get("language", "en")

        # Only align if language is supported by whisperx alignment models
        supported_align_langs = {"en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"}

        if detected_lang in supported_align_langs:
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=align_device
                )
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    align_device,
                    return_char_alignments=False
                )
            except Exception as e:
                logger.warning(f"Alignment failed, using segment-level timestamps: {e}")

        # Extract all words with timestamps
        words = []
        for segment in result.get("segments", []):
            segment_words = segment.get("words", [])
            if segment_words:
                words.extend(segment_words)
            else:
                # Fallback: treat whole segment as one "word"
                words.append({
                    "word": segment.get("text", "").strip(),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "score": segment.get("avg_logprob", 0.0)
                })

        logger.info(f"Transcribed {len(words)} words from {audio_path}")
        return words

    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        return None


def segment_into_clips(
    words: List[Dict],
    min_duration: float = 4.0,
    max_duration: float = 15.0
) -> List[List[Dict]]:
    """
    Split word list into clips of appropriate duration.
    Tries to split at natural pauses (gaps between words > 0.5s).

    Returns: list of word-lists (each = one clip)
    """
    if not words:
        return []

    clips = []
    current_clip = []
    clip_start = words[0].get("start", 0.0)

    for i, word in enumerate(words):
        current_clip.append(word)
        current_duration = word.get("end", 0.0) - clip_start

        # Check if we should end the clip here
        is_last_word = (i == len(words) - 1)
        next_word = words[i + 1] if not is_last_word else None

        # Natural pause: gap > 0.5s between words
        gap_after = (next_word["start"] - word["end"]) if next_word else 999

        should_split = (
            is_last_word or
            (current_duration >= max_duration) or
            (current_duration >= min_duration and gap_after > 0.5)
        )

        if should_split and current_duration >= min_duration:
            clips.append(current_clip)
            current_clip = []
            if next_word:
                clip_start = next_word.get("start", 0.0)
        elif should_split and current_duration < min_duration:
            # Too short — merge into next clip instead
            pass

    # Don't discard remaining words if they're long enough
    if current_clip:
        duration = current_clip[-1].get("end", 0) - current_clip[0].get("start", 0)
        if duration >= min_duration:
            clips.append(current_clip)

    return clips
