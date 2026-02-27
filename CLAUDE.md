# JustDictate — Developer Guide

This file is for AI assistants (Claude Code, Copilot, etc.) working on this codebase.

## What is this?

JustDictate is a macOS menu bar speech-to-text app. Hold a hotkey, speak, release — your words are transcribed and auto-typed into the focused app. Think SuperWhisper but open source, using NVIDIA's Parakeet TDT 0.6B v3 model via ONNX (no GPU required).

## Architecture

```
just_dictate.py      ← Entry point. rumps menu bar app. Ties everything together.
├── model_manager.py ← Loads Parakeet model via onnx-asr, handles transcription.
├── dictation_engine.py ← pynput hotkey listener + sounddevice recording + clipboard paste + Escape-to-cancel + Escape-to-undo.
├── floating_window.py  ← PyObjC NSWindow overlay with waveform animation.
└── config_manager.py   ← JSON config at ~/.config/just-dictate/config.json
```

## Key Technical Decisions

### Why onnx-asr (not PyTorch/whisper.cpp)?
- Pure ONNX inference — no PyTorch dependency (saves ~2 GB)
- `onnx-asr` handles model download, preprocessing, and inference in one package
- CPU-only via `CPUExecutionProvider` (CoreML provider is buggy on macOS with external data files)

### Why rumps + PyObjC (not tkinter/Qt)?
- Homebrew Python 3.13+ does not ship tkinter
- rumps is purpose-built for macOS menu bar apps
- PyObjC gives native NSWindow for the floating overlay (rumps is too limited for custom windows)

### Why NSSound for completion feedback (not subprocess)?
- `NSSound.soundNamed_("Tink")` plays built-in macOS system sounds with zero latency
- No subprocess overhead, no file path needed — the sound name is resolved by AppKit
- Already have PyObjC as a dependency, so no new imports

### Why CGEvent for paste (not osascript)?
- `osascript` requires its own Accessibility permission and has timeout issues
- CGEvent works with the same Accessibility permission that pynput already needs

### Why uv?
- System Python 3.14 is incompatible with onnxruntime
- uv auto-manages Python version (pinned to >=3.11,<3.14) and dependencies

## Known Gotchas

These are bugs that took hours to solve. **Do not change** these patterns without understanding why:

1. **`providers=['CPUExecutionProvider']` is mandatory** in `load_model()`. Without it, onnxruntime defaults to CoreML on macOS which crashes with `"model_path must not be empty"` on models with external data files.

2. **`model.recognize(audio, sample_rate=16000)`** — the parameter is `sample_rate=`, NOT `sr=`. Wrong name silently produces garbage.

3. **Empty model directory breaks onnx-asr download**. If `MODEL_DIR` exists but is empty, `onnx_asr.load_model()` tries to find files locally and fails instead of downloading. The code explicitly removes empty dirs before calling load.

4. **`rumps.App.quit_button` is a property**, not a regular attribute. Assigning `self.quit_button = MenuItem(...)` calls the property setter. Don't add a custom quit button to `self.menu` — let rumps handle it via the `quit_button="Quit"` constructor arg.

5. **`rumps.Timer(0, callback)` crashes** with `ValueError: depythonifying 'double', got 'function'`. Use `PyObjCTools.AppHelper.callAfter()` for thread-safe UI updates instead.

6. **PyObjCTools is a namespace package** (no `__init__.py`). py2app's `imp.find_module` can't find it. If packaging with py2app, put it in `includes`, not `packages`.

7. **PyInstaller must bundle `onnx_asr` as data**, not just as a Python package. The `onnx_asr/preprocessors/*.onnx` files are loaded at runtime and won't be found otherwise.

8. **Escape cancel uses a `_cancelled` flag**, not just `_cancel_recording()`. The flag is needed because `_on_release` fires after `_cancel_recording` when the user releases the hotkey — without the flag, it would still try to transcribe. `_cancel_recording()` sets `_cancelled = True` and `_on_release()` checks/resets it.

9. **Escape has three behaviors** depending on state: (1) during recording → cancel, (2) within 5s after paste → undo via Cmd+Z, (3) otherwise → pass through. All routed through `_handle_escape()`. The undo sends CGEvent Cmd+Z (keycode 6), which works in GUI apps but not terminals (where Cmd+Z sends SIGTSTP).

## Running Locally

```bash
# Requires macOS and uv (brew install uv)
uv run python just_dictate.py
```

First run downloads the model (~2.5 GB) to `~/.cache/just-dictate/`.

## Building the .app Bundle

See the [PyInstaller spec generation](#) — create a spec file that:
- Includes `onnx_asr` as datas (for the `preprocessors/*.onnx` files)
- Includes `libonnxruntime.dylib` and `libportaudio.dylib` as binaries
- Sets `LSUIElement: True` in `info_plist` (hides dock icon)
- Sets `NSMicrophoneUsageDescription` in `info_plist`

```bash
uv pip install pyinstaller
mv pyproject.toml pyproject.toml.bak
uv run pyinstaller JustDictate.spec --clean
mv pyproject.toml.bak pyproject.toml
# Output: dist/JustDictate.app
```

## macOS Permissions

The app needs these in System Settings > Privacy & Security:
- **Microphone** — audio recording
- **Accessibility** — hotkey detection + CGEvent paste
- **Input Monitoring** — pynput global key listener

## Config

Stored at `~/.config/just-dictate/config.json`:

```json
{
  "hotkey": "right_cmd",
  "auto_type_method": "clipboard_paste",
  "add_trailing_space": true
}
```

Hotkey options: `right_cmd`, `right_alt`, `left_ctrl_left_alt`

## Model Cache

Downloaded to `~/.cache/just-dictate/`:
- `parakeet-v3/` — NVIDIA Parakeet TDT 0.6B v3 (~2.5 GB)
- `silero-vad/` — Silero VAD model (~2 MB)

## Dependencies

All managed via `pyproject.toml`:
- `onnx-asr[cpu,hub]` — ONNX ASR engine
- `onnxruntime>=1.24.2` — ONNX inference runtime
- `sounddevice` — microphone input (via PortAudio)
- `pynput` — global hotkey detection
- `numpy` — audio array processing
- `rumps` — macOS menu bar framework
- `pyobjc-framework-Cocoa` — NSWindow, NSView, etc.
- `pyobjc-framework-Quartz` — CGEvent for keystroke injection
