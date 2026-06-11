"""
Manglish Code-Switching Scraper — Main Actor
============================================
Purpose: Build a dataset for the Manglish ASR Benchmark project.

Pipeline:
1. Search YouTube for Manglish content (or process provided URLs)
2. Download audio
3. Transcribe with WhisperX (word-level timestamps)
4. Detect language spans + switch points
5. Filter clips by code-switching quality
6. Output: YouTube IDs + timestamps (for HuggingFace) + optional audio files
"""

import os
import json
import asyncio
import logging
import uuid
from typing import List, Dict

logger = logging.getLogger(__name__)

# ── Detect if running on Apify or locally ──────────────────────────────────
def is_apify_env() -> bool:
    return os.environ.get("APIFY_IS_AT_HOME") == "1"


async def get_input() -> Dict:
    """Get input from Apify or local INPUT.json."""
    if is_apify_env():
        from apify import Actor
        return await Actor.get_input() or {}
    else:
        input_path = "storage/key_value_stores/default/INPUT.json"
        if os.path.exists(input_path):
            with open(input_path) as f:
                return json.load(f)
        # Default local input for development
        return {
            "searchQueries": ["Malayalam English vlog", "Manglish interview"],
            "youtubeUrls": [],
            "primaryLanguage": "ml",
            "whisperModel": "small",
            "maxVideos": 3,
            "minClipDuration": 4,
            "maxClipDuration": 15,
            "minCodeSwitchRatio": 0.15,
            "publishIdsOnly": True,
        }


async def push_data(items: List[Dict]) -> None:
    """Push results to Apify dataset or local output file."""
    if is_apify_env():
        from apify import Actor
        await Actor.push_data(items)
    else:
        output_path = "output/results.json"
        os.makedirs("output", exist_ok=True)
        existing = []
        if os.path.exists(output_path):
            with open(output_path) as f:
                existing = json.load(f)
        existing.extend(items)
        with open(output_path, "w") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(items)} clips → {output_path}")


async def main() -> None:
    """Main pipeline."""

    if is_apify_env():
        from apify import Actor
        async with Actor:
            await _run_pipeline()
    else:
        await _run_pipeline()


async def _run_pipeline() -> None:
    from src.downloader import search_youtube, download_audio, get_video_metadata
    from src.transcriber import load_whisper_model, transcribe_audio, segment_into_clips
    from src.language_utils import get_language_spans, compute_language_ratio, is_valid_codeswitched_clip
    from src.audio_utils import extract_clips_batch

    # ── 1. Load input ──────────────────────────────────────────────────────
    actor_input = await get_input()

    search_queries = actor_input.get("searchQueries", [])
    youtube_urls = actor_input.get("youtubeUrls", [])
    primary_lang = actor_input.get("primaryLanguage", "ml")
    model_size = actor_input.get("whisperModel", "small")
    max_videos = actor_input.get("maxVideos", 10)
    min_clip_duration = actor_input.get("minClipDuration", 4)
    max_clip_duration = actor_input.get("maxClipDuration", 15)
    min_ratio = actor_input.get("minCodeSwitchRatio", 0.15)
    publish_ids_only = actor_input.get("publishIdsOnly", True)

    logger.info(f"Starting pipeline | lang={primary_lang} | model={model_size} | max_videos={max_videos}")

    # ── 2. Collect video URLs ──────────────────────────────────────────────
    videos_to_process = []

    # From direct URLs
    for url in youtube_urls:
        meta = get_video_metadata(url)
        if meta:
            videos_to_process.append(meta)

    # From search queries
    remaining = max_videos - len(videos_to_process)
    if remaining > 0 and search_queries:
        results_per_query = max(1, remaining // len(search_queries))
        for query in search_queries:
            if len(videos_to_process) >= max_videos:
                break
            results = search_youtube(query, max_results=results_per_query)
            # Deduplicate by video ID
            existing_ids = {v["id"] for v in videos_to_process}
            for r in results:
                if r["id"] not in existing_ids and len(videos_to_process) < max_videos:
                    videos_to_process.append(r)
                    existing_ids.add(r["id"])

    if not videos_to_process:
        logger.error("No videos found. Check your search queries or URLs.")
        return

    logger.info(f"Processing {len(videos_to_process)} videos")

    # ── 3. Load Whisper model (once, reuse across all videos) ─────────────
    model, device = load_whisper_model(model_size)

    # ── 4. Process each video ──────────────────────────────────────────────
    all_clips = []
    audio_dir = "output/audio"
    clips_dir = "output/clips"
    clip_counter = 0

    for video in videos_to_process:
        video_id = video["id"]
        video_url = video.get("url", f"https://youtube.com/watch?v={video_id}")
        logger.info(f"Processing: {video.get('title', video_id)}")

        # Download audio
        audio_path = download_audio(video_url, output_dir=audio_dir)
        if not audio_path:
            logger.warning(f"Skipping {video_id} — download failed")
            continue

        # Transcribe
        words = transcribe_audio(audio_path, model, device)
        if words is None:
            logger.warning(f"Skipping {video_id} — transcription failed")
            continue

        if not words:
            logger.warning(f"Skipping {video_id} — no words transcribed")
            continue

        # Segment into clips
        clip_word_groups = segment_into_clips(
            words,
            min_duration=min_clip_duration,
            max_duration=max_clip_duration
        )

        logger.info(f"Found {len(clip_word_groups)} segments in {video_id}")

        # Analyse each clip for code-switching
        valid_clips_metadata = []

        for word_group in clip_word_groups:
            if not word_group:
                continue

            clip_start = word_group[0].get("start", 0.0)
            clip_end = word_group[-1].get("end", 0.0)

            # Detect language spans + switch points
            language_spans, switch_points = get_language_spans(word_group, primary_lang)

            # Validate code-switching quality
            is_valid, reason = is_valid_codeswitched_clip(
                language_spans, switch_points, primary_lang, min_ratio
            )

            if not is_valid:
                logger.debug(f"Clip {clip_start:.1f}s rejected: {reason}")
                continue

            ratios = compute_language_ratio(language_spans, primary_lang)
            avg_confidence = _avg_confidence(word_group)
            transcript = " ".join(w.get("word", "") for w in word_group).strip()

            clip_id = f"{primary_lang}_en_{clip_counter:04d}"
            clip_counter += 1

            clip_meta = {
                "clip_id": clip_id,
                "youtube_id": video_id,
                "video_title": video.get("title", ""),
                "start_sec": round(clip_start, 3),
                "end_sec": round(clip_end, 3),
                "duration_sec": round(clip_end - clip_start, 3),
                "transcript": transcript,
                "language_spans": language_spans,
                "switch_points": switch_points,
                "switch_count": len(switch_points),
                "primary_lang_ratio": ratios.get(primary_lang, 0),
                "en_ratio": ratios.get("en", 0),
                "confidence": avg_confidence,
                "audio_file": None,  # set below if save_audio=True
            }

            valid_clips_metadata.append(clip_meta)

        if not valid_clips_metadata:
            logger.info(f"No valid code-switched clips in {video_id}")
            # Clean up audio if we're not keeping it
            if publish_ids_only and os.path.exists(audio_path):
                os.remove(audio_path)
            continue

        # Extract audio clips (or skip in ID-only mode)
        save_audio = not publish_ids_only
        valid_clips_metadata = extract_clips_batch(
            audio_path,
            valid_clips_metadata,
            output_dir=clips_dir,
            save_audio=save_audio
        )

        # In publish_ids_only mode, remove full video audio after clipping
        if publish_ids_only and os.path.exists(audio_path):
            os.remove(audio_path)

        all_clips.extend(valid_clips_metadata)
        logger.info(f"✓ {video_id}: {len(valid_clips_metadata)} valid clips extracted")

        # Push results incrementally (so Apify shows progress)
        await push_data(valid_clips_metadata)

    # ── 5. Final summary ───────────────────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info(f"DONE — {len(all_clips)} code-switched clips extracted")
    logger.info(f"Videos processed: {len(videos_to_process)}")
    logger.info(f"Mode: {'ID-only (research)' if publish_ids_only else 'Full audio saved'}")

    if all_clips:
        avg_switches = sum(c["switch_count"] for c in all_clips) / len(all_clips)
        logger.info(f"Avg switch points per clip: {avg_switches:.1f}")

    if not is_apify_env():
        logger.info(f"Results saved to: output/results.json")
        if not publish_ids_only:
            logger.info(f"Audio clips saved to: output/clips/")


def _avg_confidence(words: List[Dict]) -> float:
    """Average Whisper confidence score across words."""
    scores = [w.get("score", 0) for w in words if w.get("score") is not None]
    return round(sum(scores) / len(scores), 3) if scores else 0.0
