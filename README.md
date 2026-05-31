# fluency

**Local speech-to-text dictation — fully offline, no cloud, one binary**

![Platform: Linux & Windows](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-blue)

Fluency is a standalone desktop application that turns microphone audio into text using OpenAI's Whisper model. Everything runs locally — no API keys, no internet required, no data leaves your machine.

## Features

- **Record & transcribe** from your microphone in real time
- **Transcribe audio files** (WAV, MP3) — drag and drop or file picker
- **Audio preprocessing** — normalization, high-pass filter, noise gate, pre-emphasis (all configurable)
- **LLM formatting** — optional local LLM (360M–1.7B) adds punctuation, capitalization, paragraph breaks, and removes filler words. Each sentence is processed independently in small chunks with a **diff validator** that guarantees no words are changed or rewritten
- **Bigram correction** — statistical model fixes common homophone errors (their/there, to/too)
- **Auto-correction** — re-runs Whisper on low-confidence segments for better accuracy
- **10 Whisper model sizes** — tiny (75 MB) to large-v3 (3 GB), auto-downloaded on first use
- **Full whisper.cpp control** — sampling strategy, temperature, VAD, CPU threads, and more
- **Transcription history** — auto-saved, browsable, loadable
- **Clipboard integration** — one-click copy (Ctrl+C)
- **Keyboard shortcuts** — Ctrl+Enter to record, Ctrl+C to copy, Ctrl+S to save, Ctrl+Shift+C to clear
- **Single binary** — no Python, no DLLs, no runtime dependencies

## Quick Start

```sh
# Run the GUI
fluency

# Record from microphone and print text
fluency dictate

# Transcribe an audio file
fluency transcribe recording.wav
```

On first run, the GUI prompts you to download a Whisper model. The `base` model (150 MB) is recommended for most users.

## GUI

Run without arguments to open the desktop window:

- **Record** / **Stop** — Ctrl+Enter to toggle
- **Copy** — Ctrl+C to copy transcription to clipboard
- **Clear** — Ctrl+Shift+C to clear
- **File menu** — open audio files, save transcriptions
- **History** — browse, load, and delete past auto-saved transcriptions

### Settings (4 tabs)

| Tab | What you can configure |
|-----|----------------------|
| **Audio** | Input device, normalize volume, high-pass filter (rumble removal), noise gate, pre-emphasis |
| **Model** | Whisper model size (auto-downloaded), custom model file, recognition language |
| **Processing** | Sampling strategy (greedy/beam search), temperature, VAD (silence skipping), thread count, context settings, and all whisper.cpp parameters |
| **LLM & Correction** | LLM model, temperature, chunk size, max tokens per chunk, 4 prompt presets (Clean paragraphs, Bullet points, Smart, Minimal), custom prompt override, bigram correction toggle, auto-correction thresholds |

Every control has a tooltip — hover over anything to see what it does.

## CLI Usage

```sh
fluency dictate [options]
fluency transcribe <file> [options]
```

| Argument | Description |
|----------|-------------|
| `-m, --model <path>` | Whisper model path (defaults to auto-downloaded) |
| `-l, --language <code>` | Language code or `auto` (default: `auto`) |
| `-c, --clipboard` | Copy result to clipboard (dictate only) |
| `--bigram` | Enable bigram correction |
| `--llm-format` | Enable LLM formatting |
| `--correct` | Enable auto-correction on low-confidence segments |

## LLM Formatting

Fluency can optionally run a small local LLM (360M–1.7B parameters) to clean up raw speech-to-text output. The pipeline works as follows:

1. **Smart chunking** — text is split on sentence boundaries into chunks of up to 500 characters (configurable). Each chunk starts at a fresh sentence, and the previous chunk's last sentence is included as overlapping context so the LLM sees continuity.
2. **Strict prompting** — the LLM is told exactly which changes are allowed (capitalization, punctuation, filler removal per style). It is explicitly forbidden from rephrasing or changing words.
3. **Diff validation** — after generation, the output is compared word-by-word against the original. If any words were added, removed, or reordered, the chunk is **rejected** and the original text is used instead. Only pure punctuation/capitalization/filler-removal changes are accepted.

This guarantees the LLM **never rewrites your text into a story** — it only formats what's already there.

### Prompt styles

| Style | Allowed changes |
|-------|----------------|
| **Clean paragraphs** | Capitals, periods, commas, paragraph breaks, filler removal |
| **Bullet points** | Each sentence on a new line with `- ` prefix, capitals, filler removal |
| **Smart** | Capitals, periods, commas, paragraph breaks, filler removal (subset — no sentence rewriting) |
| **Minimal** | Only capitalize first letter and add ending period |

### Supported models (auto-downloaded from Hugging Face)

| Model | File size | Quality |
|-------|-----------|---------|
| SmolLM2 360M | ~200 MB | Fast, good accuracy |
| Qwen2.5 0.5B | ~300 MB | Balanced |
| TinyLlama 1.1B | ~650 MB | High quality |
| SmolLM2 1.7B | ~950 MB | Best quality |

Inference uses [llama.cpp](https://github.com/ggerganov/llama.cpp) via the `llama-cpp-2` Rust crate (CPU-only, GGUF format).

## Building from Source

### Prerequisites

- Rust 1.80+ (latest stable)
- Linux: `libasound2-dev` (or `alsa-lib-devel`)
- Windows: no system dependencies

### Build

```sh
cargo build --release
```

The `fluency` binary is at `target/release/fluency`.

### ARM / Raspberry Pi

On ARM Linux (e.g. Raspberry Pi 5), `.cargo/config.toml` automatically passes
`-C target-cpu=cortex-a76` to work around a build issue with `gemm-f16`.
If building for a different ARM CPU, adjust or remove that file.

## Configuration

Settings are persisted to `$XDG_CONFIG_HOME/fluency/settings.json` (Linux) or the equivalent on Windows. All settings can be adjusted through the GUI and are saved automatically when the settings window is closed.

Model files are stored alongside the binary in `models/whisper/` and `models/llm/`.

## How It Works

1. **Record** — Audio captured at native sample rate, resampled to 16 kHz
2. **Filter** — Optional preprocessing: normalization, high-pass, noise gate, pre-emphasis
3. **Transcribe** — Audio fed to Whisper via `whisper.cpp` with your chosen parameters
4. **Correct** (optional) — Low-confidence segments re-transcribed with relaxed parameters
5. **Bigram** (optional) — Statistical correction of common homophone errors
6. **LLM Format** (optional) — Sentence-level chunked formatting with strict prompting and diff validation

All processing is local. No data leaves your machine.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Start / stop recording |
| `Ctrl+C` | Copy transcription to clipboard |
| `Ctrl+S` | Save transcription to file |
| `Ctrl+Shift+C` | Clear transcription |
| `Escape` | Stop recording / close settings |

## License

GNU General Public License v3.0
