# JustDictate

**Hold-to-dictate speech-to-text for macOS.** Hold a hotkey, speak, release — your words are transcribed and typed into the focused app. Runs entirely on-device using [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) via ONNX. No cloud, no API keys, no GPU required.

Inspired by [SuperWhisper](https://superwhisper.com/). Built with Python, rumps, and PyObjC.

## Features

- **Hold-to-dictate** — hold Right Command (configurable), speak, release
- **Escape to cancel** — press Escape while recording to discard audio and cancel
- **Completion sound** — plays a "Tink" sound when transcription finishes
- **Auto-type** — transcribed text is pasted into whatever app is focused
- **Floating overlay** — dark translucent window with waveform animation while recording
- **Menu bar app** — lives in the menu bar, no dock icon
- **Fully offline** — model runs locally on CPU via ONNX Runtime
- **Fast** — ~0.5s transcription for typical sentences
- **Configurable** — hotkey, trailing space toggle, all from the menu bar

## Requirements

- macOS 13+
- [uv](https://docs.astral.sh/uv/) — `brew install uv`

## Quick Start

```bash
git clone https://github.com/gowtham-ponnana/JustDictate.git
cd JustDictate
chmod +x run.sh
./run.sh
```

First run installs dependencies and downloads the model (~2.5 GB). Subsequent launches start in ~2 seconds.

## macOS Permissions

Grant these in **System Settings > Privacy & Security**:

| Permission | Why |
|---|---|
| **Microphone** | Record audio for transcription |
| **Accessibility** | Detect hotkey + paste text via CGEvent |
| **Input Monitoring** | Global hotkey listener (pynput) |

## Usage

1. **Launch** — a 🎙 icon appears in the menu bar
2. **Hold Right Command** — a floating overlay appears and recording starts
3. **Speak** — waveform animates in real time
4. **Release** — audio is transcribed, typed into the focused app, and a completion sound plays
5. **Cancel** — press **Escape** while recording to discard and cancel (no transcription)

### Menu Bar Options

- **Status** — shows current state (loading / ready / error)
- **Hotkey** — switch between Right Command, Right Alt, or Left Ctrl + Left Alt
- **Add Trailing Space** — toggle automatic space after each dictation

## Building a .app Bundle

To run as a standalone app (recommended for permissions):

```bash
uv pip install pyinstaller

# PyInstaller conflicts with pyproject.toml dependencies
mv pyproject.toml pyproject.toml.bak
uv run pyinstaller JustDictate.spec --clean
mv pyproject.toml.bak pyproject.toml

# Install to Applications
cp -R dist/JustDictate.app /Applications/
```

Then grant permissions to `JustDictate.app` instead of Terminal.

### Creating a DMG

```bash
hdiutil create -volname "JustDictate" \
  -srcfolder dist/JustDictate.app \
  -ov -format UDZO JustDictate.dmg
```

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  just_dictate.py (rumps menu bar app)               │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ model_manager│  │  dictation   │  │ floating  │ │
│  │   .py        │  │  _engine.py  │  │ _window.py│ │
│  │              │  │              │  │           │ │
│  │ onnx-asr     │  │ pynput       │  │ PyObjC    │ │
│  │ Parakeet     │  │ sounddevice  │  │ NSWindow  │ │
│  │ TDT 0.6B v3  │  │ CGEvent      │  │ waveform  │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│                                                     │
│  config_manager.py — ~/.config/just-dictate/        │
└─────────────────────────────────────────────────────┘
```

1. **model_manager.py** — Downloads and loads the NVIDIA Parakeet TDT 0.6B v3 model via `onnx-asr`. Uses `CPUExecutionProvider` (CoreML is buggy on macOS). Caches to `~/.cache/just-dictate/`.

2. **dictation_engine.py** — Listens for hotkey via `pynput` using macOS virtual key codes. Records audio at 16kHz mono via `sounddevice`. On release, transcribes and pastes via clipboard + CGEvent Cmd+V. Pressing Escape during recording cancels and discards audio.

3. **floating_window.py** — Native macOS overlay using PyObjC `NSWindow` with `NSVisualEffectView` blur. Custom `WaveformView` draws animated bars from real-time RMS levels.

4. **config_manager.py** — Simple JSON config at `~/.config/just-dictate/config.json`.

## Configuration

Config file: `~/.config/just-dictate/config.json`

```json
{
  "hotkey": "right_cmd",
  "auto_type_method": "clipboard_paste",
  "add_trailing_space": true
}
```

### Hotkey Options

| Preset | Keys | Config Value |
|---|---|---|
| Right Command | ⌘R | `right_cmd` |
| Right Alt/Option | ⌥R | `right_alt` |
| Left Ctrl + Left Alt | ⌃L + ⌥L | `left_ctrl_left_alt` |

## Tech Stack

| Component | Library | Purpose |
|---|---|---|
| ASR Model | [onnx-asr](https://github.com/istupakov/onnx-asr) | NVIDIA Parakeet TDT 0.6B v3 inference |
| Runtime | [onnxruntime](https://onnxruntime.ai/) | ONNX model execution (CPU) |
| Audio | [sounddevice](https://python-sounddevice.readthedocs.io/) | Microphone recording |
| Hotkey | [pynput](https://pynput.readthedocs.io/) | Global key listener |
| Menu Bar | [rumps](https://github.com/jaredks/rumps) | macOS status bar app |
| UI | [PyObjC](https://pyobjc.readthedocs.io/) | Native NSWindow overlay |
| Packaging | [uv](https://docs.astral.sh/uv/) | Python version + dependency management |

## Troubleshooting

### "This process is not trusted!" warning
Grant **Accessibility** and **Input Monitoring** permissions to the app (or Terminal if running from source).

### Model download fails
Set a HuggingFace token for faster downloads: `export HF_TOKEN=hf_...`

### No text appears after transcription
Check **Accessibility** permission — needed for CGEvent keystroke injection.

### App won't launch (Python 3.14)
This app requires Python 3.11–3.13. `uv` handles this automatically via `run.sh`. If running manually, ensure you're not using Python 3.14+.

## License

MIT
