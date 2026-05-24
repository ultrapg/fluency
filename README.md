# fluency

**Local speech-to-text dictation powered by Whisper**

Fluency is a standalone desktop application that turns microphone audio into text using OpenAI's Whisper model — fully offline, no cloud API required. It runs on Linux and Windows with a single binary that provides both a graphical interface and a command-line tool.

## Features

- **Record & transcribe** from your microphone in real time
- **Transcribe audio files** (WAV, MP3)
- **Audio preprocessing** — normalization, high-pass filter (rumble removal), noise gate, pre-emphasis
- **Text formatting** — auto-capitalization, punctuation, filler word removal, self-correction fixing
- **Bigram language model** — homophone correction and common speech-to-text error fixing
- **10 Whisper model sizes** — from tiny (75 MB, fastest) to large-v3 (3 GB, best accuracy), auto-downloaded
- **Full whisper.cpp control** — sampling strategy (greedy/beam search), temperature, VAD, thread count, and more
- **Transcription history** — auto-saved, browsable, searchable
- **Clipboard integration** — copy results with one click
- **Keyboard shortcuts** — Ctrl+Enter to record, Ctrl+C to copy, Ctrl+S to save
- **Single binary** — one executable, no DLLs, no Python, no runtime dependencies

## Quick Start

```
# Run the GUI
fluency

# Record from microphone and print text
fluency dictate

# Transcribe an audio file
fluency transcribe recording.wav
```

On first run, the app will prompt you to download a Whisper model through the GUI. The `base` model (150 MB) is recommended for most users.

## CLI Usage

```
fluency dictate [-m <model>] [-l <language>] [-c] [--format <bool>] [--lm <bool>]
fluency transcribe <file> [-m <model>] [-l <language>] [--format <bool>] [--lm <bool>]
```

| Argument | Description |
|---|---|
| `-m, --model` | Whisper model path (defaults to auto-downloaded model) |
| `-l, --language` | Language code or `auto` (default: `auto`) |
| `-c, --clipboard` | Copy result to clipboard (dictate only) |
| `--format` | Apply punctuation and capitalization (default: `true`) |
| `--lm` | Apply bigram language model correction (default: `false`) |

## GUI

Run without arguments to open the desktop window:

- **Record** button or Ctrl+Enter to start/stop microphone capture
- **Copy** button or Ctrl+C to copy transcription to clipboard
- **Save** button or Ctrl+S to write transcription to a file
- **Settings** window with four tabs:
  - **Audio** — input device, noise filtering, normalization
  - **Formatting** — capitalization, punctuation, filler words, self-corrections
  - **Model** — model size selection, custom model path, language
  - **Processing** — sampling strategy, temperature, VAD, thread count, and all whisper.cpp parameters
- **History** window to browse, load, and delete past transcriptions

## Configuration

Settings are persisted to `$XDG_CONFIG_HOME/fluency/settings.json` (Linux) or the equivalent config directory on Windows. All settings can be adjusted through the GUI and are saved automatically.

## Building from Source

### Prerequisites

- Rust (latest stable)
- Cargo
- Linux: `libasound2-dev` (or `alsa-lib-devel`)

### Build

```sh
cargo build --release
```

The single `fluency` binary will be at `target/release/fluency`.

## How It Works

1. **Record** — Audio is captured from your microphone at the native sample rate and resampled to 16 kHz
2. **Filter** — Optional preprocessing (normalization, high-pass, noise gate, pre-emphasis) cleans up the signal
3. **Transcribe** — The audio is fed to Whisper through `whisper.cpp` with your chosen settings
4. **Format** — Text is post-processed with capitalization, punctuation, filler removal, and self-correction logic
5. **Correct** — An optional bigram language model trained on English and German text fixes common homophone errors

All processing happens locally. No data leaves your machine.

## License

GNU General Public License v3.0
