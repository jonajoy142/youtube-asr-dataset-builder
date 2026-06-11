"""
Audio clip extraction — cuts full audio into timestamped segments.
Uses pydub (ffmpeg under the hood).
"""

import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def extract_clip(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str
) -> bool:
    """
    Extract a clip from a full audio file using ffmpeg via pydub.
    Returns True on success.
    """
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path)
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        clip = audio[start_ms:end_ms]

        # Normalize to 16kHz mono WAV — standard for ASR training
        clip = clip.set_frame_rate(16000).set_channels(1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        clip.export(output_path, format="wav")
        return True

    except Exception as e:
        logger.error(f"Failed to extract clip {start_sec}-{end_sec} from {audio_path}: {e}")
        return False


def extract_clips_batch(
    audio_path: str,
    clips_metadata: List[Dict],
    output_dir: str,
    save_audio: bool = True
) -> List[Dict]:
    """
    Extract multiple clips from one audio file.

    Args:
        audio_path: Source audio file
        clips_metadata: List of clip dicts with start_sec, end_sec, clip_id
        output_dir: Where to save .wav files
        save_audio: If False (research/publish mode), skip actual extraction

    Returns: clips_metadata with audio_file paths added (if save_audio=True)
    """
    results = []

    for clip in clips_metadata:
        clip_id = clip["clip_id"]
        start = clip["start_sec"]
        end = clip["end_sec"]

        if save_audio:
            output_path = os.path.join(output_dir, f"{clip_id}.wav")
            success = extract_clip(audio_path, start, end, output_path)
            clip["audio_file"] = output_path if success else None
        else:
            clip["audio_file"] = None  # ID-only mode for HuggingFace publishing

        results.append(clip)

    return results


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Return duration of audio file in seconds."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except Exception as e:
        logger.error(f"Could not get duration for {audio_path}: {e}")
        return None
