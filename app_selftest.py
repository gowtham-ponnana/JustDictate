"""Isolated packaged-runtime checks; never record a microphone or send key events.

The ASR sample uses actual cached Parakeet weights and synthetic macOS speech.
Editor changes are explicit deterministic fixtures. This is not a test of live
microphone capture, global hotkeys, target-app paste, or real Accessibility edits.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import wave


class SelfTestFailure(RuntimeError):
    pass


def _require(report, name, condition, **details):
    report["checks"].append({"name": name, "passed": bool(condition), **details})
    if not condition:
        raise SelfTestFailure(name)


def _permissions():
    permissions = {"accessibility": None, "input_monitoring": None, "microphone": "unknown"}
    try:
        import ApplicationServices
        permissions["accessibility"] = bool(ApplicationServices.AXIsProcessTrusted())
    except Exception:
        pass
    try:
        import Quartz
        permissions["input_monitoring"] = bool(Quartz.CGPreflightListenEventAccess())
    except Exception:
        pass
    try:
        import AVFoundation
        status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeAudio)
        permissions["microphone"] = {
            0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized",
        }.get(int(status), "unknown")
    except ImportError:
        pass  # Optional framework; never install or request access during a test.
    except Exception:
        pass
    return permissions


def _drain(service):
    with service._queue.all_tasks_done:
        if not service._queue.all_tasks_done.wait_for(
            lambda: service._queue.unfinished_tasks == 0, timeout=5
        ):
            raise SelfTestFailure("learning_queue_timeout")


def _exercise_pipeline(report, audio, transcribe, clipboard, directory):
    """Real engine, queues, monitor, SQLite, WAV retention and training lifecycle."""
    import config_manager
    from correction_monitor import CorrectionMonitor, TextSnapshot, STABLE_SECONDS, TRACK_SECONDS
    from dictation_engine import DictationEngine
    from learning_service import LearningService
    from personalization import PersonalizationManager

    class Reader:
        snapshot = TextSnapshot("asr", "", (0, 0))

        def capture(self):
            return self.snapshot

    reader = Reader()
    clock = [0.0]
    manager = PersonalizationManager(directory, enabled=True, retain_audio=True)
    service = LearningService(manager, enabled=True, retain_audio=True)
    monitor = CorrectionMonitor(
        reader, service.record, service.auto_correct, lambda: True, clock=lambda: clock[0],
    )
    service._is_idle = lambda: not monitor.is_tracking
    errors, delivered = [], []
    original_load = config_manager.load
    try:
        # Constructing the engine must not initialize the user's config folder.
        config_manager.load = lambda: dict(config_manager.DEFAULTS)
        engine = DictationEngine(
            transcribe_fn=transcribe, transform_text=service.apply,
            on_result=monitor.complete, before_paste=monitor.prepare,
            on_error=lambda _message: errors.append(True),
            on_transcription_done=delivered.append,
        )
    finally:
        config_manager.load = original_load

    def private_sink(text):
        # Deliberately replace only external paste/HID. Exercise its capture
        # callback and a real private native pasteboard without touching general.
        engine.before_paste(text)
        clipboard.write(text)
        reader.snapshot = TextSnapshot(reader.snapshot.element, text, (len(text), 0))

    engine._auto_type = private_sink

    def dictate():
        engine._processing = True
        engine._transcribe_and_type([audio], engine._generation)
        _drain(service)
        monitor.poll()

    try:
        dictate()
        actual = delivered[0].strip() if delivered else ""
        _require(report, "actual_asr_through_engine", bool(actual) and not errors,
                 transcript=actual, actual_model_transcriptions=1)
        _require(report, "unmodified_output_not_labelled", manager.status()["automatic_count"] == 0)

        topics = (
            "the quarterly budget", "our next meeting", "the project deadline",
            "the security review", "the database backup", "the customer survey",
            "the product launch", "the server upgrade", "the invoice number",
            "the hiring plan", "the vacation schedule", "the network outage",
            "the design proposal", "the training session", "the presentation slides",
            "the delivery address", "the test results", "the release notes",
        )
        before_threshold_clear = True
        at_threshold_saved = True
        for index, topic in enumerate(topics):
            clock[0] += 10
            reader.snapshot = TextSnapshot(f"fixture-{index}", "", (0, 0))
            # These recognition strings are fixtures, not claimed ASR outputs.
            raw = f"Please ask Gotham tomorrow about {topic}."
            engine.transcribe_fn = lambda _audio, value=raw: value
            dictate()
            corrected = reader.snapshot.value.replace("Gotham", "Gowtham")
            reader.snapshot = TextSnapshot(reader.snapshot.element, corrected, (len(corrected), 0))
            clock[0] += 0.5
            monitor.poll()
            # Advance the injected clock; the test never sleeps for the real
            # settling period. Check just before and exactly at its boundary.
            epsilon = min(0.5, STABLE_SECONDS / 2)
            clock[0] += STABLE_SECONDS - epsilon
            monitor.poll()
            _drain(service)
            before_threshold_clear &= (manager.status()["automatic_count"] == index and
                                       manager.latest()["corrected"] is None)
            clock[0] += epsilon
            monitor.poll()
            _drain(service)
            at_threshold_saved &= (manager.status()["automatic_count"] == index + 1)

        _require(report, "correction_settle_threshold", before_threshold_clear and at_threshold_saved,
                 settle_seconds=STABLE_SECONDS, no_labels_before_threshold=before_threshold_clear,
                 labels_saved_at_threshold=at_threshold_saved)

        before = manager.status()
        before_audio = len(list(manager.audio_directory.glob("*.wav")))
        _require(report, "automatic_edit_capture", not errors and len(delivered) == 19 and
                 before["record_count"] == 19 and before["automatic_count"] == 18 and
                 before["manual_count"] == 0 and before_audio == 19,
                 records=before["record_count"], automatic_labels=before["automatic_count"],
                 manual_labels=before["manual_count"], wav_files=before_audio,
                 simulated_editor_cases=18)
        _require(report, "training_waits_for_live_edits", not service.schedule_training(force=True))
        clock[0] += TRACK_SECONDS + 1
        monitor.poll()
        _require(report, "tracking_expiry", not monitor.is_tracking)
        _require(report, "training_queued", service.schedule_training(force=True))
        _drain(service)
        result = service.last_training_result or {}
        metrics = result.get("metrics", {})
        _require(report, "validated_training", result.get("status") == "trained" and
                 metrics.get("candidate_errors", 1) == 0 and
                 metrics.get("baseline_errors", 0) > 0,
                 status=result.get("status"), metrics=metrics)
        after = manager.status()
        after_audio = len(list(manager.audio_directory.glob("*.wav")))
        _require(report, "used_history_and_audio_deleted", result.get("purged_records") == 18 and
                 after["record_count"] == 1 and after["automatic_count"] == 0 and after_audio == 1,
                 purged_records=result.get("purged_records"), remaining_records=after["record_count"],
                 remaining_wav_files=after_audio)
        _require(report, "unlabelled_history_preserved", manager.latest()["raw"] == actual)
        raw_probe = "Please ask Gotham tomorrow about the office keys."
        corrected_probe = "Please ask Gowtham tomorrow about the office keys."
        reopened = PersonalizationManager(directory, enabled=True)
        _require(report, "learned_rule_survives_reopen", reopened.apply(raw_probe) == corrected_probe and
                 reopened.status()["rule_count"] == 1,
                 rule_count=reopened.status()["rule_count"])
    finally:
        monitor.close()
        engine.stop()
        service.close()


def _perform_checks(report):
    import objc

    with objc.autorelease_pool():
        _perform_native_checks(report)


def _perform_native_checks(report):
    import numpy as np
    from Foundation import NSData
    from AppKit import NSPasteboard, NSPasteboardTypeString
    from model_manager import ModelManager, MODEL_ID

    report["permissions"] = _permissions()
    pasteboard = None
    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    previous_constant = None
    hub_constants = None
    try:
        pasteboard = NSPasteboard.pasteboardWithUniqueName()
        _require(report, "private_native_pasteboard_created", pasteboard is not None)
        text = "JustDictate self-test — café 日本語 😀"
        binary_type = "com.justdictate.selftest.binary"
        payload = b"\x00\x01justdictate\xff"
        pasteboard.clearContents()
        pasteboard.setString_forType_(text, NSPasteboardTypeString)
        pasteboard.setData_forType_(NSData.dataWithBytes_length_(payload, len(payload)), binary_type)
        _require(report, "native_clipboard_unicode_and_formats",
                 str(pasteboard.stringForType_(NSPasteboardTypeString)) == text and
                 bytes(pasteboard.dataForType_(binary_type)) == payload)

        class PrivateClipboard:
            def write(self, value):
                pasteboard.clearContents()
                if not pasteboard.setString_forType_(value, NSPasteboardTypeString):
                    raise SelfTestFailure("private_clipboard_write")
                if str(pasteboard.stringForType_(NSPasteboardTypeString)) != value:
                    raise SelfTestFailure("private_clipboard_roundtrip")

        os.environ["HF_HUB_OFFLINE"] = "1"
        import huggingface_hub.constants as hub_constants
        previous_constant = hub_constants.HF_HUB_OFFLINE
        hub_constants.HF_HUB_OFFLINE = True
        model = ModelManager()
        load_started = time.monotonic()
        model.load()
        _require(report, "cached_model_loaded", model.is_loaded, model_id=MODEL_ID,
                 fast_decode_enabled=bool(model._fast_decode),
                 load_seconds=round(time.monotonic() - load_started, 3))
        _require(report, "optimized_decoder_selected", model._fast_decode)
        with tempfile.TemporaryDirectory(prefix="justdictate-app-selftest-") as temporary:
            root = Path(temporary)
            wav_path = root / "speech.wav"
            subprocess.run(
                ["/usr/bin/say", "-o", str(wav_path), "--data-format=LEI16@16000",
                 "Please send the final report before tomorrow morning."],
                check=True, capture_output=True, timeout=15,
            )
            with wave.open(str(wav_path)) as wav:
                _require(report, "synthetic_audio_format", wav.getframerate() == 16000 and
                         wav.getnchannels() == 1 and wav.getsampwidth() == 2)
                audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
            _exercise_pipeline(report, audio, model.transcribe, PrivateClipboard(), root / "history")
    finally:
        if hub_constants is not None:
            hub_constants.HF_HUB_OFFLINE = previous_constant
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline
        if pasteboard is not None:
            pasteboard.releaseGlobally()


def run_selftest(report_path: Path) -> int:
    """Write a sanitized JSON report and return zero only if every check passed."""
    started = time.monotonic()
    report_path = Path(report_path).expanduser().resolve()
    report = {
        "kind": "packaged_runtime_self_test", "passed": False, "checks": [],
        "report_path": str(report_path),
        "scope": {
            "actual_cached_asr": True, "audio": "synthetic macOS speech",
            "editor_edits": "simulated field snapshots and recognition fixtures",
            "clipboard": "native private pasteboard only", "microphone_opened": False,
            "keyboard_events_sent": False, "user_history_accessed": False,
            "live_hardware_end_to_end": False,
        },
    }
    try:
        _perform_checks(report)
        report["passed"] = bool(report["checks"]) and all(check["passed"] for check in report["checks"])
    except Exception as error:
        # Arbitrary exceptions may contain filesystem paths or native values.
        report["error"] = {"type": type(error).__name__}
    permissions = report.get("permissions", {})
    permission_values = [permissions.get("accessibility"), permissions.get("input_monitoring")]
    microphone = permissions.get("microphone", "unknown")
    permission_values.append(True if microphone == "authorized" else
                             None if microphone == "unknown" else False)
    report["readiness"] = {
        "packaged_runtime_passed": report["passed"],
        "permissions_ready": False if False in permission_values else
                             True if all(value is True for value in permission_values) else None,
        "live_microphone_and_target_app_tested": False,
    }
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".justdictate-selftest-", dir=report_path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(report, stream, indent=2)
            stream.flush()
        os.replace(temporary, report_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return 0 if report["passed"] else 1
