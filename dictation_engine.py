"""Hotkey detection, audio recording, and auto-typing."""

import logging
import subprocess
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd
from pynput import keyboard

import config_manager

log = logging.getLogger(__name__)

# Right Command vk code on macOS
VK_RIGHT_CMD = 0x36
VK_ESCAPE = 0x35

# How long after a paste the user can press Escape to undo it
UNDO_WINDOW = 5.0


class DictationEngine:
    def __init__(
        self,
        on_recording_start: Callable | None = None,
        on_recording_stop: Callable | None = None,
        on_recording_cancel: Callable | None = None,
        on_audio_level: Callable[[float], None] | None = None,
        on_transcription_start: Callable | None = None,
        on_transcription_done: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        transcribe_fn: Callable[[np.ndarray], str] | None = None,
        on_paste_undo: Callable | None = None,
        on_recording_duration: Callable[[float], None] | None = None,
    ):
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_recording_cancel = on_recording_cancel
        self.on_audio_level = on_audio_level
        self.on_transcription_start = on_transcription_start
        self.on_transcription_done = on_transcription_done
        self.on_error = on_error
        self.transcribe_fn = transcribe_fn
        self.on_paste_undo = on_paste_undo
        self.on_recording_duration = on_recording_duration

        self._recording = False
        self._recording_start_time: float | None = None
        self._cancelled = False
        self._last_paste_time: float | None = None
        self._audio_chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._held_vk: set[int] = set()
        self._listener: keyboard.Listener | None = None
        self._target_vk_codes: set[int] = set()

        self._load_hotkey_config()

    def _load_hotkey_config(self) -> None:
        cfg = config_manager.load()
        preset = config_manager.get_hotkey_preset(cfg["hotkey"])
        self._target_vk_codes = set(preset["vk_codes"])

    def start(self) -> None:
        """Start listening for the hotkey."""
        self._load_hotkey_config()
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        log.info("Hotkey listener started.")

    def stop(self) -> None:
        """Stop the hotkey listener and any active recording."""
        if self._recording:
            self._stop_recording()
        if self._listener:
            self._listener.stop()
            self._listener = None

    def _get_vk(self, key) -> int | None:
        """Extract vk code from a pynput key event."""
        vk = getattr(key, "vk", None)
        if vk is not None:
            return vk
        # Fallback for named keys
        vk_map = {
            keyboard.Key.cmd_r: VK_RIGHT_CMD,
            keyboard.Key.alt_r: 0x3D,
            keyboard.Key.ctrl_l: 0x3B,
            keyboard.Key.alt_l: 0x3A,
            keyboard.Key.ctrl_r: 0x3E,
            keyboard.Key.shift_r: 0x3C,
        }
        return vk_map.get(key)

    def _on_press(self, key) -> None:
        vk = self._get_vk(key)
        if vk is None:
            # Check for Escape via named key (pynput may not give vk for it)
            if key == keyboard.Key.esc:
                self._handle_escape()
            return
        if vk == VK_ESCAPE:
            self._handle_escape()
            return
        self._held_vk.add(vk)

        if not self._recording and self._target_vk_codes.issubset(self._held_vk):
            self._start_recording()

    def _on_release(self, key) -> None:
        vk = self._get_vk(key)
        if vk is None:
            return
        self._held_vk.discard(vk)

        if self._recording and not self._target_vk_codes.issubset(self._held_vk):
            self._stop_recording()
            if self._cancelled:
                self._cancelled = False
                return
            threading.Thread(target=self._transcribe_and_type, daemon=True).start()

    def _start_recording(self) -> None:
        self._recording = True
        self._recording_start_time = time.time()
        self._audio_chunks = []

        def audio_callback(indata, frames, time_info, status):
            if status:
                log.warning("Audio status: %s", status)
            chunk = indata[:, 0].copy()
            self._audio_chunks.append(chunk)
            if self.on_audio_level:
                rms = float(np.sqrt(np.mean(chunk**2)))
                self.on_audio_level(rms)

        self._stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="float32",
            blocksize=1600,  # 100ms chunks
            callback=audio_callback,
        )
        self._stream.start()
        log.info("Recording started.")
        if self.on_recording_start:
            self.on_recording_start()

    def _cancel_recording(self) -> None:
        """Cancel the current recording — discard audio, skip transcription."""
        self._cancelled = True
        self._recording = False
        self._recording_start_time = None
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._audio_chunks = []
        log.info("Recording cancelled.")
        if self.on_recording_cancel:
            self.on_recording_cancel()

    def _handle_escape(self) -> None:
        """Route Escape: cancel recording if active, else undo last paste."""
        if self._recording:
            self._cancel_recording()
        elif (
            self._last_paste_time is not None
            and (time.time() - self._last_paste_time) < UNDO_WINDOW
        ):
            self._undo_last_paste()

    def _undo_last_paste(self) -> None:
        """Send Cmd+Z to undo the last paste, then clear the undo window."""
        import Quartz

        src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        # keycode 6 = 'z'
        z_down = Quartz.CGEventCreateKeyboardEvent(src, 6, True)
        z_up = Quartz.CGEventCreateKeyboardEvent(src, 6, False)
        Quartz.CGEventSetFlags(z_down, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventSetFlags(z_up, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, z_down)
        Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, z_up)

        self._last_paste_time = None
        log.info("Undo last paste (Cmd+Z).")
        if self.on_paste_undo:
            self.on_paste_undo()

    def _stop_recording(self) -> None:
        self._recording = False
        duration = None
        if self._recording_start_time is not None:
            duration = time.time() - self._recording_start_time
            self._recording_start_time = None
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info("Recording stopped.")
        if self.on_recording_stop:
            self.on_recording_stop()
        if duration is not None and self.on_recording_duration:
            self.on_recording_duration(duration)

    def _transcribe_and_type(self) -> None:
        if not self._audio_chunks:
            return

        audio = np.concatenate(self._audio_chunks)
        duration = len(audio) / 16000
        if duration < 0.3:
            log.info("Audio too short (%.2fs), skipping.", duration)
            return

        if self.on_transcription_start:
            self.on_transcription_start()

        try:
            if self.transcribe_fn is None:
                raise RuntimeError("No transcribe function set.")
            text = self.transcribe_fn(audio)
            if not text:
                log.info("Empty transcription.")
                if self.on_transcription_done:
                    self.on_transcription_done("")
                return

            cfg = config_manager.load()
            if cfg.get("add_trailing_space", True):
                text += " "

            log.info("Transcribed: %s", text.strip())
            self._auto_type(text)
            self._last_paste_time = time.time()

            if self.on_transcription_done:
                self.on_transcription_done(text)

        except Exception as e:
            err_msg = str(e)
            log.error("Transcription error: %s", err_msg)
            if self.on_error:
                self.on_error(err_msg)

    def _auto_type(self, text: str) -> None:
        """Paste text into the focused app via clipboard + CGEvent Cmd+V."""
        import Quartz

        try:
            # Save current clipboard
            old_clip = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, timeout=2
            ).stdout
        except Exception:
            old_clip = ""

        # Set new clipboard content
        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        p.communicate(text.encode("utf-8"))
        time.sleep(0.05)

        # Cmd+V paste via CGEvent (no osascript needed)
        # keycode 9 = 'v'
        src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        cmd_down = Quartz.CGEventCreateKeyboardEvent(src, 9, True)
        cmd_up = Quartz.CGEventCreateKeyboardEvent(src, 9, False)
        Quartz.CGEventSetFlags(cmd_down, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventSetFlags(cmd_up, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, cmd_down)
        Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, cmd_up)
        time.sleep(0.15)

        # Restore old clipboard
        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        p.communicate(old_clip.encode("utf-8"))
