"""JustDictate — macOS menu bar speech-to-text app."""

import logging
import threading

import rumps

from config_manager import HOTKEY_PRESETS, load as load_config, save as save_config
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

        self.trailing_space_item.state = cfg.get("add_trailing_space", True)

        # rumps auto-appends the quit button — don't add it here
        self.menu = [
            self.status_item,
            None,  # separator
            self.hotkey_menu,
            self.trailing_space_item,
        ]

        # Load model in background
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            self.model.load()
            self._update_status("Status: Ready — hold hotkey to dictate")

            self.engine = DictationEngine(
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,
                on_recording_cancel=self._on_recording_cancel,
                on_audio_level=self._on_audio_level,
                on_transcription_start=self._on_transcription_start,
                on_transcription_done=self._on_transcription_done,
                on_error=self._on_error,
                transcribe_fn=self.model.transcribe,
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

    def _on_error(self, msg: str):
        self.overlay.hide()
        self.title = "🎙"
        try:
            rumps.notification("JustDictate Error", "", msg[:200])
        except RuntimeError:
            log.error("Notification failed: %s", msg[:200])

    # -- Menu callbacks --

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
