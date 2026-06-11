# Multilingual Code-Switching Audio Scraper

Scrapes YouTube for **code-switched speech** (e.g. Malayalam+English / Manglish), transcribes with WhisperX, detects language switch points, and outputs annotated clips ready for ASR benchmark dataset creation.

Built as part of the **Manglish ASR Benchmark** project — a rigorous evaluation dataset and leaderboard for code-switched Indian English speech recognition.

---

## What It Does

1. **Searches YouTube** for content matching your queries (or processes URLs you provide)
2. **Downloads audio** with yt-dlp
3. **Transcribes** with WhisperX — word-level timestamps
4. **Detects language spans** — which words are Malayalam vs English, using Unicode script ranges
5. **Finds switch points** — timestamps where the speaker switches language mid-sentence
6. **Filters clips** by code-switching quality (configurable ratio threshold)
7. **Outputs** annotated JSON — YouTube IDs + timestamps for research publishing, or full audio clips for local training

---

## Output Format

Each result item:

```json
{
  "clip_id": "ml_en_0042",
  "youtube_id": "dQw4w9WgXcQ",
  "video_title": "My Day in Kerala - Manglish Vlog",
  "start_sec": 42.3,
  "end_sec": 55.1,
  "duration_sec": 12.8,
  "transcript": "Njan yesterday office-il പോയി, but the meeting was boring",
  "language_spans": [
    {"start": 0.0, "end": 0.4, "lang": "ml", "text": "Njan"},
    {"start": 0.4, "end": 1.1, "lang": "en", "text": "yesterday"},
    {"start": 1.1, "end": 2.0, "lang": "ml", "text": "office-il"},
    {"start": 2.0, "end": 4.5, "lang": "en", "text": "but the meeting was boring"}
  ],
  "switch_points": [0.4, 1.1, 2.0],
  "switch_count": 3,
  "primary_lang_ratio": 0.42,
  "en_ratio": 0.58,
  "confidence": 0.87
}
```

---

## Running Locally (M4 Mac / Any Machine)

### Prerequisites

```bash
# Install ffmpeg (required for audio processing)
brew install ffmpeg   # Mac
# sudo apt install ffmpeg  # Linux

# Install Python dependencies
pip install -r requirements.txt
```

### Run

```bash
# Edit input settings
nano storage/key_value_stores/default/INPUT.json

# Run
python -m src
```

Results → `output/results.json`

Audio clips (if publishIdsOnly=false) → `output/clips/`

---

## Running on Apify

Deploy via Apify CLI:

```bash
npm install -g apify-cli
apify login
apify push
```

Or drag the folder into Apify Console → Create Actor → Upload source.

---

## Research Mode vs Full Audio Mode

| Setting | `publishIdsOnly: true` | `publishIdsOnly: false` |
|---------|----------------------|------------------------|
| What's saved | YouTube ID + timestamps | Actual .wav clip files |
| For | HuggingFace dataset publishing | Local model training |
| Audio downloaded | Deleted after processing | Saved to output/clips/ |
| Legal | Follows academic dataset norms | For personal/research use only |

**For HuggingFace publishing:** Use `publishIdsOnly: true`. Publish your results.json as the dataset — users reconstruct audio from IDs themselves. This is the standard approach (AudioSet, VGGSound, etc.)

---

## Supported Languages (Phase 1)

| Code | Language | Script Detection |
|------|----------|-----------------|
| `ml` | Malayalam | Unicode 0D00–0D7F |
| `hi` | Hindi | Unicode 0900–097F |
| `ta` | Tamil | Unicode 0B80–0BFF |
| `te` | Telugu | Unicode 0C00–0C7F |
| `kn` | Kannada | Unicode 0C80–0CFF |
| `bn` | Bengali | Unicode 0980–09FF |

---

## The Switch-Point WER Metric

This scraper feeds the **Switch-Point WER** benchmark — a novel evaluation metric that measures ASR accuracy specifically in a ±2 word window around each language switch.

Standard WER misses that models fail *specifically at the moment of switching*. Switch-Point WER isolates this.

→ [Manglish ASR Benchmark on HuggingFace](#) *(link after publish)*

---

## Project Context

This is part of a larger research effort:
- **Phase 1:** Malayalam+English (Manglish) — this actor
- **Phase 2:** Tamil+English, Hindi+English
- **End goal:** Published HuggingFace dataset + leaderboard + fine-tuned Whisper checkpoint

---

*Built by Jona Joy | Kireap Technologies*
