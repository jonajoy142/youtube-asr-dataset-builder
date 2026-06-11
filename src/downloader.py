"""
YouTube audio downloader using yt-dlp.
Downloads audio-only streams and returns local file paths.
"""

import os
import logging
import yt_dlp
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def search_youtube(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search YouTube and return video metadata (no download).

    Returns list of: {id, title, duration, url, channel}
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "default_search": "ytsearch",
        "noplaylist": True,
    }

    search_query = f"ytsearch{max_results}:{query}"
    results = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_query, download=False)
            entries = info.get("entries", [])
            for entry in entries:
                if not entry:
                    continue
                duration = entry.get("duration", 0)
                # Skip very short (<60s) or very long (>30min) videos
                if duration and (duration < 60 or duration > 1800):
                    continue
                results.append({
                    "id": entry.get("id"),
                    "title": entry.get("title"),
                    "duration": duration,
                    "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
                    "channel": entry.get("uploader", "unknown"),
                })
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")

    return results


def download_audio(
    youtube_url: str,
    output_dir: str = "./output/audio",
    audio_format: str = "wav"
) -> Optional[str]:
    """
    Download audio from a YouTube URL.
    Returns local file path or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract video ID for filename
    video_id = _extract_video_id(youtube_url)
    output_path = os.path.join(output_dir, f"{video_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
            "preferredquality": "192",
        }],
        # Respect rate limits to avoid blocks
        "sleep_interval": 2,
        "max_sleep_interval": 5,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        final_path = os.path.join(output_dir, f"{video_id}.{audio_format}")
        if os.path.exists(final_path):
            logger.info(f"Downloaded: {final_path}")
            return final_path
        else:
            logger.error(f"File not found after download: {final_path}")
            return None

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"Download failed for {youtube_url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading {youtube_url}: {e}")
        return None


def get_video_metadata(youtube_url: str) -> Optional[Dict]:
    """Fetch metadata for a specific video without downloading."""
    ydl_opts = {"quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return {
                "id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration"),
                "channel": info.get("uploader"),
                "upload_date": info.get("upload_date"),
                "url": youtube_url,
            }
    except Exception as e:
        logger.error(f"Failed to get metadata for {youtube_url}: {e}")
        return None


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    import re
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Fallback: use full URL hash
    return str(hash(url))[:11]
