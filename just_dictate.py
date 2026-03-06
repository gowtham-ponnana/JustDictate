"""JustDictate — macOS menu bar speech-to-text app."""

import logging
import threading

import rumps
import sounddevice as sd

from config_manager import (
    HOTKEY_PRESETS,
    load as load_config,
    load_stats,
    save as save_config,
    save_stats,
)
from dictation_engine import DictationEngine
from floating_window import FloatingOverlay
from model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("justdictate")


class JustDictateApp(rumps.App):
    def __init__(self):
        super().__init__("JustDictate", title="🎙", quit_button="Quit")

        self.model = ModelManager()
        self.overlay = FloatingOverlay()
        self.engine = None

        # Menu items
        self.status_item = rumps.MenuItem("Status: Loading model...")
        stats = load_stats()
        self.time_item = rumps.MenuItem(
            self._format_time(stats["total_recording_seconds"])
        )
        self.mic_menu = rumps.MenuItem("Microphone")
        self.hotkey_menu = rumps.MenuItem("Hotkey")
        self.trailing_space_item = rumps.MenuItem(
            "Add Trailing Space", callback=self._toggle_trailing_space
        )

        # Build hotkey submenu
        cfg = load_config()
        for key, preset in HOTKEY_PRESETS.items():
            item = rumps.MenuItem(preset["label"], callback=self._change_hotkey)
            item._hotkey_key = key
            if key == cfg["hotkey"]:
                item.state = True
            self.hotkey_menu.add(item)

        # Build microphone submenu
        self._populate_mic_menu(cfg)

        self.trailing_space_item.state = cfg.get("add_trailing_space", True)

        # rumps auto-appends the quit button — don't add it here
        self.menu = [
            self.status_item,
            self.time_item,
            None,  # separator
            self.mic_menu,
            self.hotkey_menu,
            self.trailing_space_item,
        ]

        # Load model in background
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            self.model.load()
            self._update_status("Status: Ready — hold hotkey to dictate")

            cfg = load_config()
            input_device = cfg.get("input_device")
            if input_device is not None:
                # Verify saved device still exists
                try:
                    names = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
                    if input_device not in names:
                        log.warning("Saved input device %r not found, falling back to system default.", input_device)
                        input_device = None
                except Exception:
                    input_device = None

            self.engine = DictationEngine(
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,
                on_recording_cancel=self._on_recording_cancel,
                on_audio_level=self._on_audio_level,
                on_transcription_start=self._on_transcription_start,
                on_transcription_done=self._on_transcription_done,
                on_error=self._on_error,
                transcribe_fn=self.model.transcribe,
                on_paste_undo=self._on_paste_undo,
                on_recording_duration=self._on_recording_duration,
                input_device=input_device,
            )
            self.engine.start()
            log.info("JustDictate ready.")
        except Exception as e:
            err_msg = str(e)
            log.error("Failed to load model: %s", err_msg)
            self._update_status(f"Status: Error — {err_msg[:60]}")

    def _update_status(self, text: str):
        """Thread-safe status update via PyObjC main thread dispatch."""
        from PyObjCTools import AppHelper
        AppHelper.callAfter(lambda: setattr(self.status_item, "title", text))

    # -- Engine callbacks --

    def _on_recording_start(self):
        self.overlay.show("Recording...")
        self.title = "🔴"

    def _on_recording_stop(self):
        self.title = "🎙"

    def _on_recording_cancel(self):
        self.overlay.hide()
        self.title = "🎙"

    def _on_audio_level(self, rms: float):
        self.overlay.update_levels(rms)

    def _on_transcription_start(self):
        self.overlay.set_status("Transcribing...")
        self.title = "⏳"

    def _on_transcription_done(self, text: str):
        self.overlay.hide()
        self.title = "🎙"
        if text.strip():
            log.info("Typed: %s", text.strip())
            self._play_completion_sound()

    def _play_completion_sound(self):
        """Play a short system sound to indicate transcription is done."""
        from AppKit import NSSound
        sound = NSSound.soundNamed_("Tink")
        if sound:
            sound.play()

    def _on_recording_duration(self, duration: float):
        stats = load_stats()
        stats["total_recording_seconds"] += duration
        stats["total_recordings"] += 1
        save_stats(stats)
        text = self._format_time(stats["total_recording_seconds"])
        from PyObjCTools import AppHelper
        AppHelper.callAfter(lambda: setattr(self.time_item, "title", text))

    @staticmethod
    def _format_time(seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"Total: {s}s"
        if s < 3600:
            return f"Total: {s // 60}m {s % 60}s"
        h = s // 3600
        m = (s % 3600) // 60
        return f"Total: {h}h {m}m"

    def _on_paste_undo(self):
        log.info("Undo last dictation paste.")

    def _on_error(self, msg: str):
        self.overlay.hide()
        self.title = "🎙"
        try:
            rumps.notification("JustDictate Error", "", msg[:200])
        except RuntimeError:
            log.error("Notification failed: %s", msg[:200])

    # -- Menu callbacks --

    def _populate_mic_menu(self, cfg: dict) -> None:
        """Populate the Microphone submenu with available input devices."""
        # clear() fails before the menu is attached to NSMenu (during __init__)
        if self.mic_menu._menu is not None:
            self.mic_menu.clear()

        saved_device = cfg.get("input_device")

        # System Default option
        default_item = rumps.MenuItem("System Default", callback=self._change_mic)
        default_item._device_name = None
        default_item.state = saved_device is None
        self.mic_menu.add(default_item)

        # Force PortAudio to re-query Core Audio for current device list
        # Without this, newly connected devices (AirPods, USB mics) won't appear
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass

        # List input devices
        try:
            devices = sd.query_devices()
            seen = set()
            for d in devices:
                if d["max_input_channels"] > 0 and d["name"] not in seen:
                    seen.add(d["name"])
                    item = rumps.MenuItem(d["name"], callback=self._change_mic)
                    item._device_name = d["name"]
                    item.state = d["name"] == saved_device
                    self.mic_menu.add(item)
        except Exception as e:
            log.warning("Failed to query audio devices: %s", e)

        # Separator + Refresh
        self.mic_menu.add(None)
        self.mic_menu.add(rumps.MenuItem("Refresh Devices", callback=self._refresh_devices))

    def _change_mic(self, sender) -> None:
        device_name = sender._device_name
        cfg = load_config()
        cfg["input_device"] = device_name
        save_config(cfg)

        # Update checkmarks
        for item in self.mic_menu.values():
            if hasattr(item, "_device_name"):
                item.state = item._device_name == device_name

        # Update device and warm it up (pre-initializes OS driver for fast recording start)
        if self.engine:
            self.engine._input_device = device_name
            threading.Thread(target=self.engine._warm_up_device, daemon=True).start()

        log.info("Input device changed to: %s", device_name or "System Default")

    def _refresh_devices(self, _sender) -> None:
        cfg = load_config()
        self._populate_mic_menu(cfg)
        log.info("Refreshed audio device list.")

    def _change_hotkey(self, sender):
        key = sender._hotkey_key
        cfg = load_config()
        cfg["hotkey"] = key
        save_config(cfg)

        # Update checkmarks
        for item in self.hotkey_menu.values():
            if hasattr(item, "_hotkey_key"):
                item.state = item._hotkey_key == key

        # Restart engine with new hotkey
        if self.engine:
            self.engine.stop()
            self.engine.start()

        log.info("Hotkey changed to: %s", key)

    def _toggle_trailing_space(self, sender):
        sender.state = not sender.state
        cfg = load_config()
        cfg["add_trailing_space"] = bool(sender.state)
        save_config(cfg)

def main():
    app = JustDictateApp()

    @rumps.events.before_quit
    def _cleanup():
        if app.engine:
            app.engine.stop()

    app.run()


if __name__ == "__main__":
    main()
