"""JustDictate — macOS menu bar speech-to-text app."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading

import rumps
import sounddevice as sd

from config_manager import (
    CONFIG_DIR,
    HOTKEY_PRESETS,
    load as load_config,
    load_stats,
    save as save_config,
    save_stats,
)
from dictation_engine import DictationEngine
from floating_window import FloatingOverlay
from model_manager import ModelManager
from learning_service import LearningService
from personalization import PersonalizationManager
from accessibility_reader import AccessibilityTextReader
from correction_monitor import CorrectionMonitor, STABLE_SECONDS

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
        self._startup_lock = threading.Lock()
        self._menu_active = False
        self.correction_monitor = None
        cfg = load_config()
        self.learn_from_edits = cfg["learn_from_edits"]
        self.personalization = PersonalizationManager(
            CONFIG_DIR / "personalization",
            enabled=cfg["personalization_enabled"],
            retain_audio=cfg["retain_personalization_audio"],
            retention_days=cfg["personalization_retention_days"],
        )
        self.learning = LearningService(
            self.personalization,
            enabled=cfg["personalization_enabled"],
            retain_audio=cfg["retain_personalization_audio"],
            is_idle=self._learning_is_idle,
            on_update=self._queue_learning_update,
        )
        self.correction_monitor = CorrectionMonitor(
            reader=AccessibilityTextReader(),
            record_fn=self.learning.record,
            correction_fn=self.learning.auto_correct,
            enabled_fn=lambda: self.learning.enabled and self.learn_from_edits,
            on_update=self._queue_learning_update,
        )

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
        for key, preset in HOTKEY_PRESETS.items():
            item = rumps.MenuItem(preset["label"], callback=self._change_hotkey)
            item._hotkey_key = key
            if key == cfg["hotkey"]:
                item.state = True
            self.hotkey_menu.add(item)

        # Build microphone submenu
        self._populate_mic_menu(cfg)

        self.trailing_space_item.state = cfg.get("add_trailing_space", True)
        self.learning_menu = rumps.MenuItem("Personalization")
        self.learning_enabled_item = rumps.MenuItem(
            "Save Dictations & Learn Locally", callback=self._toggle_learning
        )
        self.learning_enabled_item.state = self.learning.enabled
        self.learning_audio_item = rumps.MenuItem(
            "Keep Audio Locally", callback=self._toggle_learning_audio
        )
        self.learning_audio_item.state = self.learning.retain_audio
        self.learning_edits_item = rumps.MenuItem(
            "Learn from My Corrections", callback=self._toggle_learning_edits
        )
        self.learning_edits_item.state = self.learn_from_edits
        self.learning_status_item = rumps.MenuItem("Learning is off")
        self.learning_schedule_item = rumps.MenuItem("Monthly learning")
        self.learning_result_item = rumps.MenuItem("Learning needs 12 corrected or reviewed dictations")
        self.review_menu = rumps.MenuItem("Review Recent Dictations")
        for item in (
            self.learning_enabled_item,
            self.learning_audio_item,
            self.learning_edits_item,
            self.learning_status_item,
            self.learning_schedule_item,
            self.learning_result_item,
            self.review_menu,
            rumps.MenuItem("Check for Learning Update", callback=self._train_now),
            rumps.MenuItem("Restore Previous Learning", callback=self._rollback_learning),
            rumps.MenuItem("Learning Details…", callback=self._learning_details),
            rumps.MenuItem("Delete Learning Data…", callback=self._reset_learning),
        ):
            self.learning_menu.add(item)

        # rumps auto-appends the quit button — don't add it here
        self.menu = [
            self.status_item,
            self.time_item,
            None,  # separator
            self.mic_menu,
            self.hotkey_menu,
            self.trailing_space_item,
            self.learning_menu,
            rumps.MenuItem("Retry Hotkey Setup", callback=self._retry_hotkeys),
        ]

        self._refresh_learning_menu()
        # Calendar-month checks also catch up after sleep/restart. Training only
        # starts when the dictation engine is idle and never changes Parakeet.
        self._learning_timer = rumps.Timer(self._check_learning, 60)
        self._learning_timer.start()
        self.correction_monitor.start()

        # Load model in background
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        if not self._startup_lock.acquire(blocking=False):
            return
        try:
            self.model.load()

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

            if self.engine is not None:
                self.engine.stop()
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
                transform_text=self.learning.apply,
                on_result=self.correction_monitor.complete,
                before_paste=self.correction_monitor.prepare,
            )
            self.engine.start()
            self._update_status("Status: Ready — hold hotkey to dictate")
            self.learning.schedule_training()
            log.info("JustDictate ready.")
        except Exception as e:
            err_msg = str(e)
            log.error("Startup failed: %s", err_msg)
            self._update_status(f"Status: Error — {err_msg[:60]}")
        finally:
            self._startup_lock.release()

    def _retry_hotkeys(self, _sender):
        if self.engine and self.engine.is_busy:
            self._show_error("Finish the current dictation before restarting hotkeys.")
            return
        if self.engine and self.engine.is_listening:
            self._update_status("Status: Ready — hold hotkey to dictate")
            return
        self._update_status("Status: Checking hotkey setup…")
        threading.Thread(target=self._load_model, name="hotkey-setup", daemon=True).start()

    def _update_status(self, text: str):
        """Thread-safe status update via PyObjC main thread dispatch."""
        from PyObjCTools import AppHelper
        AppHelper.callAfter(lambda: setattr(self.status_item, "title", text))

    # -- Engine callbacks --

    def _on_recording_start(self):
        self.overlay.show("Recording...")
        self._set_title("🔴")

    def _on_recording_stop(self):
        self._set_title("🎙")

    def _on_recording_cancel(self):
        self.overlay.hide()
        self._set_title("🎙")

    def _on_audio_level(self, rms: float):
        self.overlay.update_levels(rms)

    def _on_transcription_start(self):
        self.overlay.set_status("Transcribing...")
        self._set_title("⏳")

    def _on_transcription_done(self, text: str):
        self.overlay.hide()
        self._set_title("🎙")
        if text.strip():
            from PyObjCTools import AppHelper
            AppHelper.callAfter(self._play_completion_sound)

    def _set_title(self, title):
        from PyObjCTools import AppHelper
        AppHelper.callAfter(lambda: setattr(self, "title", title))

    def _play_completion_sound(self):
        """Play a short system sound to indicate transcription is done."""
        from AppKit import NSSound
        sound = NSSound.soundNamed_("Tink")
        if sound:
            sound.play()

    def _on_recording_duration(self, duration: float):
        # Do not block the hotkey release with JSON reads/writes and fsync.
        from PyObjCTools import AppHelper
        AppHelper.callAfter(self._save_recording_duration, duration)

    def _save_recording_duration(self, duration: float):
        stats = load_stats()
        stats["total_recording_seconds"] += duration
        stats["total_recordings"] += 1
        save_stats(stats)
        text = self._format_time(stats["total_recording_seconds"])
        self.time_item.title = text

    @staticmethod
    def _format_time(seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"This month: {s}s"
        if s < 3600:
            return f"This month: {s // 60}m {s % 60}s"
        h = s // 3600
        m = (s % 3600) // 60
        return f"This month: {h}h {m}m"

    def _on_paste_undo(self):
        self.correction_monitor.cancel()
        log.info("Undo last dictation paste.")

    def _on_error(self, msg: str):
        self.correction_monitor.cancel()
        self.overlay.hide()
        self._set_title("🎙")
        from PyObjCTools import AppHelper
        AppHelper.callAfter(self._show_error, msg)

    def _show_error(self, msg):
        try:
            rumps.notification("JustDictate Error", "", msg[:200])
        except RuntimeError:
            log.error("Notification failed: %s", msg[:200])

    # -- Menu callbacks --

    def _queue_learning_update(self):
        from PyObjCTools import AppHelper
        AppHelper.callAfter(self._refresh_learning_menu)

    def _refresh_learning_menu(self):
        state = self.personalization.status()
        if state.get("unavailable"):
            self.learning_status_item.title = "Learning unavailable — dictation still works"
        elif self.learning.enabled:
            self.learning_status_item.title = (
                f"{state.get('reviewed_count', 0)} learning examples · "
                f"{state.get('rule_count', 0)} learned corrections"
            )
        else:
            self.learning_status_item.title = "Learning is off"
        self.learning_schedule_item.title = (
            f"Next monthly check: {state.get('next_training_month') or 'enable learning first'}"
        )
        attempt = self.learning.last_training_result or state.get("last_attempt") or {}
        self.learning_result_item.title = state.get("reason") or attempt.get(
            "reason", "Learning needs 12 corrected or reviewed dictations")
        if self.review_menu._menu is not None:
            self.review_menu.clear()
        recent = self.personalization.recent(limit=20)
        for record in recent:
            source = record.get("label_source")
            description = {"manual": "reviewed", "automatic": "edit captured"}.get(source, "unreviewed")
            label = f"Dictation {record['id']} — {description}"
            item = rumps.MenuItem(label, callback=self._review_dictation)
            item._record_id = record["id"]
            self.review_menu.add(item)
        if not recent:
            self.review_menu.add(rumps.MenuItem("No saved dictations yet"))

    def _check_learning(self, _timer):
        self.learning.schedule_training()

    def _learning_is_idle(self):
        return (self.engine is not None and not self.engine.is_busy and not self._menu_active
                and not (self.correction_monitor and self.correction_monitor.is_tracking))

    def _toggle_learning(self, sender):
        enabled = not self.learning.enabled
        self.correction_monitor.cancel()
        self.learning.set_enabled(enabled)
        sender.state = enabled
        cfg = load_config()
        cfg["personalization_enabled"] = enabled
        save_config(cfg)

    def _toggle_learning_audio(self, sender):
        enabled = not self.learning.retain_audio
        self.learning.set_retain_audio(enabled)
        sender.state = enabled
        cfg = load_config()
        cfg["retain_personalization_audio"] = enabled
        save_config(cfg)

    def _toggle_learning_edits(self, sender):
        self.correction_monitor.cancel()
        self.learn_from_edits = not self.learn_from_edits
        sender.state = self.learn_from_edits
        cfg = load_config()
        cfg["learn_from_edits"] = self.learn_from_edits
        save_config(cfg)

    def _suspend_for_menu(self):
        if self.engine is None or not self.engine.set_suspended(True):
            # A modal alert here could steal focus from an in-flight paste.
            self._show_error("Try again when JustDictate is ready and transcription has finished.")
            return False
        self._menu_active = True
        return True

    def _resume_after_menu(self):
        self._menu_active = False
        if self.engine:
            self.engine.set_suspended(False)

    def _review_dictation(self, sender):
        if not self.learning.enabled:
            self._show_error("Enable Save Dictations & Learn Locally before reviewing.")
            return
        if not self._suspend_for_menu():
            return
        sound = None
        try:
            record = next((row for row in self.personalization.recent(limit=20)
                           if row["id"] == sender._record_id), None)
            if record is None:
                return
            dialog = rumps.Window(
                title="Review dictation",
                message=("Enter exactly what you said. Keep the text unchanged to confirm it. "
                         "This teaches future dictations; it does not edit the text already pasted.\n\n"
                         f"Original recognition:\n{record['raw'][:1000]}"),
                default_text=record.get("corrected") or record["delivered"],
                ok="Confirm Transcript", cancel="Cancel", dimensions=(550, 180),
            )
            audio_name = record.get("audio_name")
            if audio_name and "/" not in audio_name and "\\" not in audio_name:
                from AppKit import NSSound
                sound = NSSound.alloc().initWithContentsOfFile_byReference_(
                    str(self.personalization.audio_directory / audio_name), True)
                if sound:
                    dialog.add_button("Play Recording")
            while True:
                response = dialog.run()
                if sound and response.clicked == 2:
                    sound.stop()
                    sound.play()
                    dialog.default_text = response.text
                    continue
                break
            if response.clicked == 1 and response.text.strip():
                if not self.personalization.review(record["id"], response.text.strip()):
                    self._show_error("This dictation is no longer available for review.")
                self._refresh_learning_menu()
        finally:
            if sound:
                sound.stop()
            self._resume_after_menu()

    def _learning_details(self, _sender):
        if not self._suspend_for_menu():
            return
        try:
            state = self.personalization.status()
            attempt = self.learning.last_training_result or state.get("last_attempt") or {}
            metrics = attempt.get("metrics") or state.get("metrics") or {}
            message = (
                f"Saved dictations: {state.get('record_count', 0)}\n"
                f"Edits learned automatically: {state.get('automatic_count', 0)}\n"
                f"Manually reviewed: {state.get('manual_count', 0)}\n"
                f"Learned corrections: {state.get('rule_count', 0)}\n"
                f"Last successful update: {state.get('last_trained') or 'none yet'}\n"
                f"Next monthly check: {state.get('next_training_month') or 'not scheduled'}\n"
                f"Last result: {state.get('reason') or attempt.get('reason', 'awaiting reviewed examples')}\n\n"
                "Fix a typo in text JustDictate inserted. In compatible apps, small edits are "
                f"learned automatically after {STABLE_SECONDS} seconds without further edits. "
                "Each edit restarts that wait. Edits are tracked for up to 15 minutes; "
                "unrelated writing and ambiguous rewrites are excluded. Manual review remains "
                "available for other apps. Learning needs 12 distinct corrected or reviewed "
                "dictations and must improve on examples excluded from training.\n\n"
                f"History is kept locally for up to {self.personalization.retention_days} days "
                "or 2,000 dictations. Examples used by a successful update are deleted immediately, "
                "including audio; learned corrections and validation totals remain. Parakeet stays unchanged; "
                "training learns text corrections. Delete Learning Data also removes learned corrections."
            )
            if metrics:
                message += (f"\n\nValidation token errors: {metrics.get('baseline_errors')} original, "
                            f"{metrics.get('candidate_errors')} candidate, across "
                            f"{metrics.get('validation_examples')} held-out examples.")
            rumps.alert("Local learning", message)
        finally:
            self._resume_after_menu()

    def _train_now(self, _sender):
        if not self.learning.enabled:
            self._show_error("Enable Save Dictations & Learn Locally first.")
            return
        if self.learning.schedule_training(force=True):
            self.learning_status_item.title = "Checking reviewed dictations…"
        else:
            self._show_error("A dictation or correction is still being tracked. Learning runs when idle.")

    def _rollback_learning(self, _sender):
        if not self._suspend_for_menu():
            return
        try:
            restored = self.personalization.rollback()
            self._refresh_learning_menu()
            if not restored:
                rumps.alert("No previous learning version", "No update is available to roll back.")
        finally:
            self._resume_after_menu()

    def _reset_learning(self, _sender):
        if not self._suspend_for_menu():
            return
        try:
            if rumps.alert(
                "Delete all learning data?",
                "This deletes saved transcripts, recordings and learned corrections. "
                "Future dictations can be saved again while learning is enabled.",
                ok="Delete", cancel="Cancel",
            ) == 1:
                self.correction_monitor.cancel()
                self.learning.reset()
                self._refresh_learning_menu()
        finally:
            self._resume_after_menu()

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
        if not self._suspend_for_menu():
            return
        try:
            self._select_mic(sender)
        finally:
            self._resume_after_menu()

    def _select_mic(self, sender) -> None:
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
            if not self.engine._warm_up_device():
                self._show_error("The selected microphone could not be opened. Check Microphone permission and device availability.")

        log.info("Input device changed to: %s", device_name or "System Default")

    def _refresh_devices(self, _sender) -> None:
        if not self._suspend_for_menu():
            return
        try:
            cfg = load_config()
            self._populate_mic_menu(cfg)
            log.info("Refreshed audio device list.")
        finally:
            self._resume_after_menu()

    def _change_hotkey(self, sender):
        if not self._suspend_for_menu():
            return
        try:
            self._select_hotkey(sender)
        finally:
            self._resume_after_menu()

    def _select_hotkey(self, sender):
        key = sender._hotkey_key
        cfg = load_config()
        cfg["hotkey"] = key
        save_config(cfg)

        # Update checkmarks
        for item in self.hotkey_menu.values():
            if hasattr(item, "_hotkey_key"):
                item.state = item._hotkey_key == key

        # No device warmup or listener restart is needed for a key preference.
        if self.engine:
            self.engine.refresh_config()

        log.info("Hotkey changed to: %s", key)

    def _toggle_trailing_space(self, sender):
        sender.state = not sender.state
        cfg = load_config()
        cfg["add_trailing_space"] = bool(sender.state)
        save_config(cfg)
        if self.engine:
            self.engine.refresh_config()

def main():
    if "--native-editor-probe" in sys.argv:
        import argparse
        import json
        from tools.native_editor_probe import run_probe
        parser = argparse.ArgumentParser(description="Probe JustDictate's own native test editor")
        parser.add_argument("--native-editor-probe", required=True, type=Path)
        args = parser.parse_args()
        report = run_probe()
        destination = args.native_editor_probe.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit({"passed": 0, "blocked": 2}.get(report["status"], 1))
    if "--self-test-report" in sys.argv:
        import argparse
        from app_selftest import run_selftest
        parser = argparse.ArgumentParser(description="Packaged JustDictate runtime verification")
        parser.add_argument("--self-test-report", required=True, type=Path)
        args = parser.parse_args()
        raise SystemExit(run_selftest(args.self_test_report))

    # A windowed bundle has no dependable terminal stream. Keep bounded local
    # diagnostics so startup/permission failures remain observable after launch.
    log_dir = CONFIG_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    file_handler = RotatingFileHandler(log_dir / "app.log", maxBytes=1_000_000,
                                      backupCount=2, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    log.info("Starting JustDictate (packaged=%s).", bool(getattr(sys, "frozen", False)))
    app = JustDictateApp()

    @rumps.events.before_quit
    def _cleanup():
        app._learning_timer.stop()
        if app.engine:
            app.engine.stop()
        app.correction_monitor.close()
        app.learning.close()

    app.run()


if __name__ == "__main__":
    main()
