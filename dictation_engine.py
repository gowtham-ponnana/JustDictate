"""Hotkey detection, audio recording, and auto-typing."""

import logging
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
SAMPLE_RATE = 16000
LISTENER_START_TIMEOUT = 2.0
_clipboard_lock = threading.RLock()


def input_permissions() -> dict[str, bool | None]:
    """Read current permissions without requesting or changing macOS access."""
    result = {"accessibility": None, "input_monitoring": None, "event_posting": None}
    try:
        from ApplicationServices import AXIsProcessTrusted
        result["accessibility"] = bool(AXIsProcessTrusted())
    except (ImportError, AttributeError):
        pass
    try:
        from Quartz import CGPreflightPostEventAccess
        result["event_posting"] = bool(CGPreflightPostEventAccess())
    except (ImportError, AttributeError):
        pass
    try:
        from Quartz import CGPreflightListenEventAccess
        result["input_monitoring"] = bool(CGPreflightListenEventAccess())
    except (ImportError, AttributeError):
        pass
    return result


class _ReadyKeyboardListener(keyboard.Listener):
    """Distinguish pynput's failed-tap 'ready' signal from a working listener."""
    def __init__(self, *args, **kwargs):
        self._startup_event = threading.Event()
        self._tap_created = False
        self._startup_ok = False
        super().__init__(*args, **kwargs)

    def _create_event_tap(self):
        tap = super()._create_event_tap()
        self._tap_created = tap is not None
        return tap

    def _mark_ready(self):
        super()._mark_ready()
        # Installed pynput marks ready even when CGEventTapCreate returns None.
        # A real run loop is assigned only after it has a valid tap/source.
        self._startup_ok = self._tap_created and getattr(self, "_loop", None) is not None
        self._startup_event.set()

    def _run(self):
        try:
            super()._run()
        finally:
            # An exception before _mark_ready must wake the bounded startup wait.
            self._startup_event.set()

    def wait_until_ready(self, timeout=LISTENER_START_TIMEOUT) -> bool:
        return self._startup_event.wait(timeout) and self._startup_ok and self.is_alive()


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
        input_device: str | None = None,
        transform_text: Callable[[str], str] | None = None,
        on_result: Callable[[str, str, np.ndarray], None] | None = None,
        before_paste: Callable[[str], None] | None = None,
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
        self._input_device = input_device
        self.transform_text = transform_text
        self.on_result = on_result
        self.before_paste = before_paste
        self._state_lock = threading.RLock()
        self._processing = False
        self._starting = False
        self._suspended = False
        self._stopped = False
        self._generation = 0
        self._worker: threading.Thread | None = None
        self._restore_timer: threading.Timer | None = None
        self._pending_restore = None

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
        self.refresh_config()

    def refresh_config(self) -> None:
        """Refresh menu settings outside the transcription hot path."""
        cfg = config_manager.load()
        preset = config_manager.get_hotkey_preset(cfg["hotkey"])
        with self._state_lock:
            self._target_vk_codes = set(preset["vk_codes"])
            self._add_trailing_space = bool(cfg.get("add_trailing_space", True))

    @property
    def is_busy(self) -> bool:
        with self._state_lock:
            return self._recording or self._processing or self._starting

    @property
    def is_listening(self) -> bool:
        with self._state_lock:
            return (not self._stopped and self._listener is not None
                    and self._listener.is_alive())

    def set_suspended(self, suspended: bool) -> bool:
        """Suspend hotkeys while a modal review window owns keyboard input.

        Atomically refuse suspension when recording or processing. Callers must
        check the result before opening a modal window or reinitializing audio.
        Resume requires a fresh hotkey press.
        """
        with self._state_lock:
            if suspended and (self.is_busy or self._suspended):
                return False
            self._suspended = suspended
            if suspended:
                self._held_vk.clear()
            return True

    def start(self) -> None:
        """Start only after macOS has created a working global hotkey tap."""
        self._load_hotkey_config()
        with self._state_lock:
            if self._starting:
                raise RuntimeError("Hotkey setup is already in progress.")
            if self.is_listening:
                return
            self._starting = True
            self._stopped = True
            self._held_vk.clear()
            generation = self._generation
        listener = None
        try:
            permissions = input_permissions()
            if permissions["accessibility"] is False and permissions["input_monitoring"] is False:
                raise RuntimeError(
                    "Hotkeys are unavailable. Enable JustDictate in System Settings → "
                    "Privacy & Security → Accessibility (and Input Monitoring if required), "
                    "then retry hotkey setup."
                )
            if (permissions["event_posting"] is False or
                    (permissions["event_posting"] is None and permissions["accessibility"] is not True)):
                raise RuntimeError(
                    "Automatic paste is unavailable. Enable JustDictate in System Settings → "
                    "Privacy & Security → Accessibility, then retry hotkey setup."
                )
            listener = _ReadyKeyboardListener(on_press=self._on_press, on_release=self._on_release)
            listener.daemon = True
            with self._state_lock:
                if self._generation != generation:
                    raise RuntimeError("Hotkey setup was cancelled.")
                self._listener = listener
            listener.start()
            if not listener.wait_until_ready(LISTENER_START_TIMEOUT):
                raise RuntimeError(
                    "macOS could not start the global hotkey listener. Check Accessibility "
                    "and Input Monitoring permissions for JustDictate, then retry hotkey setup."
                )
            # Native device/permission waits must not hold the state lock. The
            # starting flag prevents capture and device reinitialization meanwhile.
            if not self._warm_up_device():
                raise RuntimeError(
                    "Microphone is unavailable. Enable JustDictate in System Settings → "
                    "Privacy & Security → Microphone, check the selected input device, "
                    "then retry hotkey setup."
                )
            with self._state_lock:
                if self._generation != generation:
                    raise RuntimeError("Hotkey setup was cancelled.")
                if not listener.is_alive():
                    raise RuntimeError("The hotkey listener stopped during setup. Retry hotkey setup.")
                self._stopped = False
        except Exception:
            if listener is not None:
                listener.stop()
            with self._state_lock:
                if self._listener is listener:
                    self._listener = None
            raise
        finally:
            with self._state_lock:
                self._starting = False
        log.info("Hotkey listener started.")

    def stop(self) -> None:
        """Stop the hotkey listener and any active recording."""
        with self._state_lock:
            self._stopped = True
            self._generation += 1
            self._held_vk.clear()
            if self._recording:
                self._cancel_recording()
            if self._listener:
                self._listener.stop()
                self._listener = None
        # Let an already-posted Cmd+V consume its clipboard before shutdown.
        # Joining outside the state lock also lets its restore callback finish.
        timer = self._restore_timer
        if timer is not None and timer is not threading.current_thread():
            timer.join(timeout=0.25)

    def _warm_up_device(self) -> bool:
        """Briefly open and close the audio device to pre-initialize the OS driver.

        After this, subsequent InputStream opens are near-instant because the
        device driver is already loaded. The mic is NOT left open.
        """
        stream = None
        succeeded = False
        try:
            stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                device=self._input_device,
            )
            stream.start()
            stream.stop()
            succeeded = True
        except Exception as e:
            log.warning("Failed to warm up audio device: %s", e)
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception as e:
                    log.warning("Failed to close warm-up stream: %s", e)
                    succeeded = False
        if succeeded:
            log.info("Warmed up audio device: %s", self._input_device or "system default")
        return succeeded

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
        with self._state_lock:
            was_down = vk in self._held_vk
            self._held_vk.add(vk)
            # A press during processing is ignored until released and pressed
            # again. Key repeats must not start a recording halfway through a hold.
            if (
                not was_down
                and vk in self._target_vk_codes
                and not self.is_busy
                and not self._suspended
                and not self._stopped
                and self._target_vk_codes.issubset(self._held_vk)
            ):
                self._start_recording()

    def _on_release(self, key) -> None:
        vk = self._get_vk(key)
        if vk is None:
            return
        with self._state_lock:
            self._held_vk.discard(vk)
            if not self._recording or self._target_vk_codes.issubset(self._held_vk):
                return
            # Claim processing before closing the stream. A worker gets its own
            # detached recording, never a mutable reference to the next capture.
            self._processing = True
            chunks = self._stop_recording()
            generation = self._generation
            self._worker = threading.Thread(
                target=self._transcribe_and_type, args=(chunks, generation), daemon=True
            )
            self._worker.start()

    def _start_recording(self) -> None:
        self._recording = True
        self._cancelled = False
        self._recording_start_time = time.monotonic()
        chunks: list[np.ndarray] = []
        self._audio_chunks = chunks
        self._last_paste_time = None

        def audio_callback(indata, frames, time_info, status):
            if status:
                log.warning("Audio status: %s", status)
            chunk = indata[:, 0].copy()
            chunks.append(chunk)
            if self.on_audio_level:
                rms = float(np.sqrt(np.mean(chunk**2)))
                self._notify(self.on_audio_level, rms)

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=1600,  # 100ms chunks
                callback=audio_callback,
                device=self._input_device,
            )
            self._stream.start()
        except Exception as e:
            self._cancel_recording()
            self._notify(self.on_error, str(e))
            return
        log.info("Recording started.")
        self._notify(self.on_recording_start)

    def _cancel_recording(self) -> None:
        """Cancel the current recording — discard audio, skip transcription."""
        self._cancelled = True
        self._recording = False
        self._recording_start_time = None
        self._close_stream()
        self._audio_chunks = []
        log.info("Recording cancelled.")
        self._notify(self.on_recording_cancel)

    def _handle_escape(self) -> None:
        """Route Escape: cancel recording if active, else undo last paste."""
        with self._state_lock:
            if self._suspended or self._stopped:
                return
            if self._recording:
                self._cancel_recording()
            elif (
                not self._processing
                and self._last_paste_time is not None
                and (time.monotonic() - self._last_paste_time) < UNDO_WINDOW
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

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception as e:
            log.warning("Failed to stop audio stream: %s", e)
        finally:
            try:
                stream.close()
            except Exception as e:
                log.warning("Failed to close audio stream: %s", e)

    def _stop_recording(self) -> list[np.ndarray]:
        self._recording = False
        self._recording_start_time = None
        self._close_stream()
        chunks, self._audio_chunks = self._audio_chunks, []
        # Use actual audio duration instead of wall-clock time.
        # Wall-clock inflates massively if a key release is missed by pynput
        # (common with Cmd keys during app switching on macOS).
        duration = None
        if chunks:
            duration = sum(len(c) for c in chunks) / SAMPLE_RATE
        log.info("Recording stopped.")
        self._notify(self.on_recording_stop)
        if duration is not None:
            self._notify(self.on_recording_duration, duration)
        return chunks

    def _transcribe_and_type(self, chunks: list[np.ndarray], generation: int) -> None:
        try:
            with self._state_lock:
                if self._stopped or self._suspended or generation != self._generation:
                    self._notify(self.on_transcription_done, "")
                    return
            if not chunks:
                self._notify(self.on_transcription_done, "")
                return
            audio = np.concatenate(chunks)
            if len(audio) / SAMPLE_RATE < 0.3:
                self._notify(self.on_transcription_done, "")
                return
            self._notify(self.on_transcription_start)
            if self.transcribe_fn is None:
                raise RuntimeError("No transcribe function set.")
            raw = self.transcribe_fn(audio)
            if not raw:
                self._notify(self.on_transcription_done, "")
                return
            delivered = raw
            if self.transform_text is not None:
                try:
                    candidate = self.transform_text(raw)
                    if not isinstance(candidate, str) or not candidate.strip():
                        raise ValueError("Personalization returned empty or non-text output")
                    delivered = candidate
                except Exception:
                    log.warning("Personalization failed; using original transcript.")
            with self._state_lock:
                if self._stopped or self._suspended or generation != self._generation:
                    self._notify(self.on_transcription_done, "")
                    return
                text = delivered + (" " if self._add_trailing_space else "")
                self._auto_type(text)
                self._last_paste_time = time.monotonic()
            self._notify(self.on_transcription_done, text)
            # Consumers enqueue persistence so no disk IO delays the paste or
            # completion sound. Unreviewed model output is not a training label.
            self._notify(self.on_result, raw, delivered, audio)
        except Exception as e:
            err_msg = str(e)
            log.error("Transcription error: %s", err_msg)
            self._notify(self.on_error, err_msg)
        finally:
            with self._state_lock:
                self._processing = False

    @staticmethod
    def _notify(callback, *args) -> None:
        if callback is not None:
            try:
                callback(*args)
            except Exception:
                log.warning("Dictation callback failed (%s).", getattr(callback, "__name__", "callback"))

    def _auto_type(self, text: str) -> None:
        """Paste text into the focused app via clipboard + CGEvent Cmd+V."""
        import Quartz

        with _clipboard_lock:
            if self._restore_timer is not None:
                self._restore_timer.cancel()
            self._restore_clipboard()
            # Fail before modifying the clipboard if its contents cannot be
            # preserved, rather than silently deleting a copied image/file.
            snapshot = _clipboard_snapshot()
            count = _clipboard_set(text)
            pending = (snapshot, count)
            self._pending_restore = pending
        try:
            if self.before_paste is None:
                time.sleep(0.05)
            else:
                # Optional AX context capture shares the existing clipboard
                # settle interval rather than adding another wait to each paste.
                settle_started = time.monotonic()
                self._notify(self.before_paste, text)
                remaining = 0.05 - (time.monotonic() - settle_started)
                if remaining > 0:
                    time.sleep(remaining)
            # Cmd+V paste via CGEvent (no osascript needed)
            # keycode 9 = 'v'
            src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
            cmd_down = Quartz.CGEventCreateKeyboardEvent(src, 9, True)
            cmd_up = Quartz.CGEventCreateKeyboardEvent(src, 9, False)
            Quartz.CGEventSetFlags(cmd_down, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventSetFlags(cmd_up, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, cmd_down)
            Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, cmd_up)
        finally:
            # Restoration need not hold up completion feedback for 150ms. Guard
            # it with changeCount so a user copy/new paste is not overwritten.
            with _clipboard_lock:
                self._restore_timer = threading.Timer(0.15, self._restore_clipboard, args=(pending,))
                self._restore_timer.daemon = True
                self._restore_timer.start()

    def _restore_clipboard(self, expected=None) -> None:
        with _clipboard_lock:
            pending = self._pending_restore
            if pending is None or (expected is not None and pending is not expected):
                return
            self._pending_restore = None
            try:
                _clipboard_restore(*pending)
            except Exception as e:
                log.warning("Failed to restore clipboard: %s", e)


def _clipboard_get() -> str:
    """Read the clipboard as Unicode text (NSPasteboard, not pbcopy)."""
    from AppKit import NSPasteboard, NSPasteboardTypeString

    value = NSPasteboard.generalPasteboard().stringForType_(NSPasteboardTypeString)
    return str(value) if value else ""


def _clipboard_set(text: str) -> int:
    """Write Unicode text to the clipboard.

    pbcopy without a UTF-8 locale tags the pasteboard as MacRoman, which
    turns UTF-8 umlauts into mojibake (ä → √§). NSPasteboard stores UTF-8
    natively.
    """
    from AppKit import NSPasteboard, NSPasteboardTypeString

    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    if not pb.setString_forType_(text, NSPasteboardTypeString):
        raise RuntimeError("NSPasteboard rejected the clipboard string")
    return int(pb.changeCount())


def _clipboard_snapshot() -> list[list[tuple[str, bytes]]]:
    """Copy all pasteboard item types, including rich text, images, and files."""
    from AppKit import NSPasteboard

    result = []
    for item in NSPasteboard.generalPasteboard().pasteboardItems() or []:
        types = []
        for kind in item.types():
            data = item.dataForType_(kind)
            if data is not None:
                types.append((str(kind), bytes(data)))
        result.append(types)
    return result


def _clipboard_restore(snapshot: list[list[tuple[str, bytes]]], expected_count: int) -> bool:
    from AppKit import NSPasteboard, NSPasteboardItem
    from Foundation import NSData

    pb = NSPasteboard.generalPasteboard()
    if int(pb.changeCount()) != expected_count:
        return False
    items = []
    for saved in snapshot:
        item = NSPasteboardItem.alloc().init()
        for kind, payload in saved:
            data = NSData.dataWithBytes_length_(payload, len(payload))
            if not item.setData_forType_(data, kind):
                raise RuntimeError("Could not reconstruct the previous clipboard")
        items.append(item)
    # Constructing provider-backed data can take time; check ownership again.
    if int(pb.changeCount()) != expected_count:
        return False
    pb.clearContents()
    if items and not pb.writeObjects_(items):
        raise RuntimeError("Could not restore the previous clipboard")
    return True
