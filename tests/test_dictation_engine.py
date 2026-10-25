"""Engine regressions without microphone access, input injection, or real clipboard IO."""

import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import dictation_engine as module
from dictation_engine import DictationEngine, SAMPLE_RATE, VK_RIGHT_CMD

ORIGINAL_CLIPBOARD_RESTORE = module._clipboard_restore


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self.closed = False

    def start(self):
        return self

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def feed(self, value=1.0, seconds=0.4):
        data = np.full((int(seconds * SAMPLE_RATE), 1), value, dtype=np.float32)
        self.callback(data, len(data), None, None)


class FakeTimer:
    def __init__(self, interval, callback, args=()):
        self.interval, self.callback, self.args = interval, callback, args
        self.cancelled = False
        self.daemon = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.callback(*self.args)

    def join(self, timeout=None):
        self.join_timeout = timeout
        if not self.cancelled:
            self.fire()


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.config = {"hotkey": "right_cmd", "add_trailing_space": True}
        self.config_patch = patch.object(module.config_manager, "load", return_value=self.config)
        self.config_mock = self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.stream_patch = patch.object(module.sd, "InputStream", side_effect=FakeStream)
        self.stream_patch.start()
        self.addCleanup(self.stream_patch.stop)
        self.key = SimpleNamespace(vk=VK_RIGHT_CMD)

    def engine(self, **kwargs):
        result = DictationEngine(**kwargs)
        result._auto_type = Mock()
        self.addCleanup(result.stop)
        return result

    def record(self, engine, *, value=1.0, seconds=0.4):
        engine._on_press(self.key)
        self.assertTrue(engine._recording)
        engine._stream.feed(value, seconds)
        engine._on_release(self.key)

    def join(self, engine):
        engine._worker.join(timeout=2)
        self.assertFalse(engine._worker.is_alive(), "Worker did not finish")

    def test_recording_is_detached_and_busy_holds_are_not_started_late(self):
        entered, resume = threading.Event(), threading.Event()
        observed = []

        def transcribe(audio):
            observed.append(audio.copy())
            entered.set()
            self.assertTrue(resume.wait(2))
            return "first"

        engine = self.engine(transcribe_fn=transcribe)
        self.record(engine, value=0.25)
        self.assertTrue(entered.wait(2))
        self.assertTrue(engine.is_busy)
        self.assertEqual(engine._audio_chunks, [])
        engine._on_press(self.key)
        self.assertFalse(engine._recording)
        # A new list cannot change the detached samples owned by the worker.
        engine._audio_chunks = [np.full(6400, 0.9, dtype=np.float32)]
        resume.set()
        self.join(engine)
        np.testing.assert_array_equal(observed[0], np.full(6400, 0.25, dtype=np.float32))
        engine._auto_type.assert_called_once_with("first ")
        engine._on_press(self.key)  # Key repeat from the ignored hold.
        self.assertFalse(engine._recording)
        engine._on_release(self.key)
        engine._on_press(self.key)
        self.assertTrue(engine._recording)

    def test_stop_during_inference_prevents_paste_and_learning(self):
        entered, resume = threading.Event(), threading.Event()

        def transcribe(audio):
            entered.set()
            self.assertTrue(resume.wait(2))
            return "obsolete"

        on_result = Mock()
        engine = self.engine(transcribe_fn=transcribe, on_result=on_result)
        self.record(engine)
        self.assertTrue(entered.wait(2))
        engine.stop()
        resume.set()
        self.join(engine)
        engine._auto_type.assert_not_called()
        on_result.assert_not_called()
        self.assertFalse(engine.is_busy)

    def test_suspend_refuses_inflight_work_and_blocks_fresh_holds_when_idle(self):
        entered, resume = threading.Event(), threading.Event()

        def transcribe(audio):
            entered.set()
            self.assertTrue(resume.wait(2))
            return "obsolete"

        engine = self.engine(transcribe_fn=transcribe)
        self.record(engine)
        self.assertTrue(entered.wait(2))
        self.assertFalse(engine.set_suspended(True))
        resume.set()
        self.join(engine)
        engine._auto_type.assert_called_once_with("obsolete ")
        self.assertTrue(engine.set_suspended(True))
        self.assertFalse(engine.set_suspended(True))  # Another modal cannot claim it.
        engine._on_press(self.key)
        self.assertFalse(engine._recording)
        self.assertTrue(engine.set_suspended(False))
        engine._on_press(self.key)
        self.assertFalse(engine._recording)  # Repeat is not a fresh press.
        engine._on_release(self.key)
        engine._on_press(self.key)
        self.assertTrue(engine._recording)
        self.assertFalse(engine.set_suspended(True))

    def test_escape_cancel_discards_audio_and_requires_new_press(self):
        transcribe = Mock(return_value="never")
        engine = self.engine(transcribe_fn=transcribe)
        engine._on_press(self.key)
        stream = engine._stream
        stream.feed()
        engine._handle_escape()
        engine._on_press(self.key)
        self.assertFalse(engine._recording)
        self.assertTrue(stream.closed)
        engine._on_release(self.key)
        transcribe.assert_not_called()
        self.assertEqual(engine._audio_chunks, [])
        self.assertFalse(engine.is_busy)

    def test_short_audio_finishes_overlay_without_inference(self):
        done, transcribe = Mock(), Mock()
        engine = self.engine(transcribe_fn=transcribe, on_transcription_done=done)
        self.record(engine, seconds=0.1)
        self.join(engine)
        transcribe.assert_not_called()
        done.assert_called_once_with("")
        self.assertFalse(engine.is_busy)

    def test_stream_open_failure_recovers_state(self):
        error = Mock()
        engine = self.engine(on_error=error)
        with patch.object(module.sd, "InputStream", side_effect=RuntimeError("device missing")):
            engine._on_press(self.key)
        self.assertFalse(engine.is_busy)
        self.assertIsNone(engine._stream)
        error.assert_called_once_with("device missing")

    def test_corrected_result_keeps_raw_audio_and_completes_before_persistence(self):
        events = []
        received = []
        engine = self.engine(
            transcribe_fn=lambda audio: "Gotham",
            transform_text=lambda text: "Gowtham",
            on_transcription_done=lambda text: events.append(("done", text)),
            on_result=lambda *args: (events.append(("result", args[1])), received.append(args)),
        )
        # Re-reading config during transcription would be a hot-path regression.
        self.config_mock.side_effect = AssertionError("Config read on hot path")
        self.record(engine)
        self.join(engine)
        engine._auto_type.assert_called_once_with("Gowtham ")
        self.assertEqual(events, [("done", "Gowtham "), ("result", "Gowtham")])
        self.assertEqual(received[0][:2], ("Gotham", "Gowtham"))
        self.assertEqual(received[0][2].shape, (6400,))

    def test_personalization_exception_and_invalid_output_fall_back(self):
        for transform in (Mock(side_effect=ValueError("failed")), lambda text: "", lambda text: None):
            with self.subTest(transform=transform):
                engine = self.engine(transcribe_fn=lambda audio: "original", transform_text=transform)
                self.record(engine)
                self.join(engine)
                engine._auto_type.assert_called_once_with("original ")

    def test_failed_paste_is_not_recorded_as_success(self):
        done, result, error = Mock(), Mock(), Mock()
        engine = self.engine(transcribe_fn=lambda audio: "original", on_transcription_done=done,
                             on_result=result, on_error=error)
        engine._auto_type.side_effect = RuntimeError("paste failed")
        self.record(engine)
        self.join(engine)
        done.assert_not_called()
        result.assert_not_called()
        error.assert_called_once_with("paste failed")
        self.assertIsNone(engine._last_paste_time)
        self.assertFalse(engine.is_busy)

    def test_refresh_config_updates_trailing_space(self):
        engine = self.engine(transcribe_fn=lambda audio: "words")
        self.config["add_trailing_space"] = False
        engine.refresh_config()
        self.record(engine)
        self.join(engine)
        engine._auto_type.assert_called_once_with("words")

    def test_callback_failure_does_not_drop_successful_result(self):
        result = Mock()
        engine = self.engine(transcribe_fn=lambda audio: "words", on_result=result,
                             on_transcription_done=Mock(side_effect=ValueError("UI failed")))
        self.record(engine)
        self.join(engine)
        result.assert_called_once()
        self.assertFalse(engine.is_busy)


class ListenerStartupTests(unittest.TestCase):
    def setUp(self):
        config = patch.object(module.config_manager, "load", return_value={"hotkey": "right_cmd"})
        config.start()
        self.addCleanup(config.stop)
        self.engine = DictationEngine()
        self.engine._warm_up_device = Mock(return_value=True)
        self.addCleanup(self.engine.stop)

    def test_denied_permissions_do_not_open_microphone_or_create_event_tap(self):
        permissions = {"accessibility": False, "input_monitoring": False, "event_posting": False}
        with patch.object(module, "input_permissions", return_value=permissions), \
                patch.object(module, "_ReadyKeyboardListener") as listener:
            with self.assertRaisesRegex(RuntimeError, "Privacy & Security"):
                self.engine.start()
            listener.assert_not_called()
        self.engine._warm_up_device.assert_not_called()
        self.assertFalse(self.engine.is_listening)
        self.assertTrue(self.engine._stopped)

    def test_accessibility_can_allow_tap_without_separate_input_monitoring_grant(self):
        permissions = {"accessibility": True, "input_monitoring": False, "event_posting": True}
        with patch.object(module, "input_permissions", return_value=permissions), \
                patch.object(module, "_ReadyKeyboardListener") as constructor:
            listener = constructor.return_value
            listener.wait_until_ready.return_value = True
            listener.is_alive.return_value = True
            self.engine.start()
            self.engine.start()  # A retry on an already-running engine is idempotent.
            listener.start.assert_called_once()
            listener.wait_until_ready.assert_called_once_with(module.LISTENER_START_TIMEOUT)
        self.engine._warm_up_device.assert_called_once()
        self.assertTrue(self.engine.is_listening)

    def test_input_monitoring_alone_does_not_claim_paste_permission(self):
        for posting in (False, None):
            with self.subTest(posting=posting), \
                    patch.object(module, "input_permissions", return_value={
                        "accessibility": False, "input_monitoring": True, "event_posting": posting}), \
                    patch.object(module, "_ReadyKeyboardListener") as constructor:
                with self.assertRaisesRegex(RuntimeError, "Automatic paste is unavailable"):
                    self.engine.start()
                constructor.assert_not_called()
        self.engine._warm_up_device.assert_not_called()

    def test_unknown_posting_api_is_supported_when_accessibility_is_granted(self):
        permissions = {"accessibility": True, "input_monitoring": None, "event_posting": None}
        with patch.object(module, "input_permissions", return_value=permissions), \
                patch.object(module, "_ReadyKeyboardListener") as constructor:
            constructor.return_value.wait_until_ready.return_value = True
            constructor.return_value.is_alive.return_value = True
            self.engine.start()
        self.assertTrue(self.engine.is_listening)

    def test_tap_failure_or_startup_timeout_never_reports_listening(self):
        permissions = {"accessibility": True, "input_monitoring": True, "event_posting": True}
        with patch.object(module, "input_permissions", return_value=permissions), \
                patch.object(module, "_ReadyKeyboardListener") as constructor:
            listener = constructor.return_value
            listener.wait_until_ready.return_value = False
            with self.assertRaisesRegex(RuntimeError, "global hotkey listener"):
                self.engine.start()
            listener.stop.assert_called_once()
        self.engine._warm_up_device.assert_not_called()
        self.assertIsNone(self.engine._listener)
        self.assertFalse(self.engine.is_listening)

    def test_microphone_failure_stops_listener_and_clears_starting(self):
        permissions = {"accessibility": True, "input_monitoring": True, "event_posting": True}
        self.engine._warm_up_device.return_value = False
        with patch.object(module, "input_permissions", return_value=permissions), \
                patch.object(module, "_ReadyKeyboardListener") as constructor:
            listener = constructor.return_value
            listener.wait_until_ready.return_value = True
            listener.is_alive.return_value = True
            with self.assertRaisesRegex(RuntimeError, "Microphone is unavailable"):
                self.engine.start()
            listener.stop.assert_called_once()
        self.assertFalse(self.engine.is_listening)
        self.assertFalse(self.engine.is_busy)
        self.assertFalse(self.engine._starting)
        self.assertIsNone(self.engine._listener)

    def test_blocked_microphone_setup_does_not_hold_state_lock_and_can_be_cancelled(self):
        entered, release = threading.Event(), threading.Event()
        failures = []

        def warmup():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("Test did not release warm-up")
            return True

        def start():
            try:
                self.engine.start()
            except Exception as error:
                failures.append(error)

        self.engine._warm_up_device = warmup
        permissions = {"accessibility": True, "input_monitoring": True, "event_posting": True}
        with patch.object(module, "input_permissions", return_value=permissions), \
                patch.object(module, "_ReadyKeyboardListener") as constructor:
            constructor.return_value.wait_until_ready.return_value = True
            constructor.return_value.is_alive.return_value = True
            worker = threading.Thread(target=start)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertTrue(self.engine._state_lock.acquire(timeout=0.1))
                self.engine._state_lock.release()
                self.assertTrue(self.engine.is_busy)
                self.assertFalse(self.engine.set_suspended(True))
                self.assertFalse(self.engine.is_listening)
                self.engine._on_press(SimpleNamespace(vk=VK_RIGHT_CMD))
                self.assertFalse(self.engine._recording)
                self.engine.stop()
            finally:
                release.set()
                worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIn("cancelled", str(failures[0]))
        self.assertFalse(self.engine._starting)
        self.assertFalse(self.engine.is_listening)

    def test_native_pynput_none_tap_thread_is_not_reported_ready(self):
        permissions = {"accessibility": True, "input_monitoring": True, "event_posting": True}
        with patch.object(module, "input_permissions", return_value=permissions), \
                patch.object(module.keyboard.Listener, "_create_event_tap", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "global hotkey listener"):
                self.engine.start()
        self.engine._warm_up_device.assert_not_called()
        self.assertFalse(self.engine.is_listening)

    def test_real_warmup_returns_false_on_device_error_and_closes_started_stream(self):
        stream = Mock()
        stream.start.side_effect = RuntimeError("microphone denied")
        with patch.object(module.sd, "InputStream", return_value=stream):
            self.assertFalse(DictationEngine._warm_up_device(self.engine))
        stream.close.assert_called_once()

    def test_pynput_failed_tap_mark_ready_is_rejected_even_before_thread_exits(self):
        listener = module._ReadyKeyboardListener()
        with patch.object(module.keyboard.Listener, "_create_event_tap", return_value=None):
            self.assertIsNone(listener._create_event_tap())
        listener._loop = None
        listener._mark_ready()
        with patch.object(listener, "is_alive", return_value=True):
            self.assertFalse(listener.wait_until_ready(timeout=0))

    def test_pynput_valid_tap_requires_runloop_and_live_thread(self):
        listener = module._ReadyKeyboardListener()
        with patch.object(module.keyboard.Listener, "_create_event_tap", return_value=object()):
            listener._create_event_tap()
        listener._loop = object()
        listener._mark_ready()
        with patch.object(listener, "is_alive", return_value=True):
            self.assertTrue(listener.wait_until_ready(timeout=0))
        with patch.object(listener, "is_alive", return_value=False):
            self.assertFalse(listener.wait_until_ready(timeout=0))

    def test_pynput_early_exception_signals_startup_wait(self):
        listener = module._ReadyKeyboardListener()
        with patch.object(module.keyboard.Listener, "_run", side_effect=RuntimeError("native failure")):
            with self.assertRaises(RuntimeError):
                listener._run()
        self.assertTrue(listener._startup_event.is_set())
        self.assertFalse(listener.wait_until_ready(timeout=0))

    def test_permission_helper_uses_only_readonly_preflight_calls(self):
        ax = SimpleNamespace(AXIsProcessTrusted=Mock(return_value=True))
        quartz = SimpleNamespace(CGPreflightListenEventAccess=Mock(return_value=False),
                                 CGPreflightPostEventAccess=Mock(return_value=True))
        with patch.dict(sys.modules, {"ApplicationServices": ax, "Quartz": quartz}):
            self.assertEqual(module.input_permissions(), {
                "accessibility": True, "input_monitoring": False, "event_posting": True})
        ax.AXIsProcessTrusted.assert_called_once_with()
        quartz.CGPreflightListenEventAccess.assert_called_once_with()
        quartz.CGPreflightPostEventAccess.assert_called_once_with()


class ClipboardTests(unittest.TestCase):
    def setUp(self):
        cfg = patch.object(module.config_manager, "load", return_value={"hotkey": "right_cmd"})
        cfg.start()
        self.addCleanup(cfg.stop)
        self.engine = DictationEngine()
        self.clipboard = [[("public.utf8-plain-text", b"original"), ("public.png", b"image")]]
        self.count = 10
        self.restore_calls = []

        def snapshot():
            return self.clipboard

        def write(text):
            self.count += 1
            self.clipboard = [[("public.utf8-plain-text", text.encode())]]
            return self.count

        def restore(saved, expected):
            self.restore_calls.append((saved, expected))
            if self.count != expected:
                return False
            self.count += 1
            self.clipboard = saved
            return True

        for target, replacement in (("_clipboard_snapshot", snapshot), ("_clipboard_set", write),
                                    ("_clipboard_restore", restore)):
            p = patch.object(module, target, replacement)
            p.start()
            self.addCleanup(p.stop)
        timer = patch.object(module.threading, "Timer", FakeTimer)
        timer.start()
        self.addCleanup(timer.stop)
        sleep = patch.object(module.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        self.quartz = Mock()
        quartz = patch.dict(sys.modules, {"Quartz": self.quartz})
        quartz.start()
        self.addCleanup(quartz.stop)

    def test_paste_defers_only_restore_wait_and_preserves_types(self):
        original = self.clipboard
        self.engine._auto_type("hello")
        self.sleep.assert_called_once_with(0.05)
        self.assertEqual(self.restore_calls, [])
        self.assertEqual(self.engine._restore_timer.interval, 0.15)
        self.engine._restore_timer.fire()
        self.assertEqual(self.clipboard, original)

    def test_before_paste_capture_shares_the_existing_settle_window(self):
        for capture_seconds in (0.020, 0.037, 0.045):
            with self.subTest(capture_seconds=capture_seconds):
                events = []
                self.sleep.reset_mock()
                self.quartz.CGEventPost.reset_mock()
                self.engine.before_paste = lambda text: events.append(("capture", text))
                self.sleep.side_effect = lambda duration: events.append(("settle", duration))
                self.quartz.CGEventPost.side_effect = lambda *args: events.append(("paste", None))
                with patch.object(module.time, "monotonic",
                                  side_effect=[10.0, 10.0 + capture_seconds]):
                    self.engine._auto_type("hello")
                self.assertEqual(self.sleep.call_count, 1)
                remaining = self.sleep.call_args.args[0]
                self.assertAlmostEqual(remaining, 0.05 - capture_seconds)
                self.assertAlmostEqual(capture_seconds + remaining, 0.05)
                self.assertEqual(events[0], ("capture", "hello"))
                self.assertEqual([event[0] for event in events],
                                 ["capture", "settle", "paste", "paste"])

    def test_before_paste_failure_cannot_prevent_paste(self):
        self.engine.before_paste = Mock(side_effect=ValueError("capture unavailable"))
        with patch.object(module.time, "monotonic", side_effect=[10.0, 10.01]):
            self.engine._auto_type("hello")
        self.assertAlmostEqual(self.sleep.call_args.args[0], 0.04)
        self.assertEqual(self.quartz.CGEventPost.call_count, 2)

    def test_slow_capture_does_not_add_another_settle_sleep(self):
        self.engine.before_paste = Mock()
        with patch.object(module.time, "monotonic", side_effect=[10.0, 10.08]):
            self.engine._auto_type("hello")
        self.sleep.assert_not_called()
        self.assertEqual(self.quartz.CGEventPost.call_count, 2)

    def test_user_copy_during_restore_delay_is_preserved(self):
        self.engine._auto_type("hello")
        self.count += 1
        self.clipboard = [[("public.utf8-plain-text", b"new user copy")]]
        self.engine._restore_timer.fire()
        self.assertEqual(self.clipboard[0][0][1], b"new user copy")

    def test_back_to_back_pastes_and_stale_timer_preserve_original_clipboard(self):
        original = self.clipboard
        self.engine._auto_type("first")
        stale = self.engine._restore_timer
        self.engine._auto_type("second")
        self.assertTrue(stale.cancelled)
        stale.fire()  # Cancellation may race a callback already dispatched.
        self.assertEqual(self.clipboard[0][0][1], b"second")
        self.engine._restore_timer.fire()
        self.assertEqual(self.clipboard, original)

    def test_injection_failure_still_schedules_restore(self):
        original = self.clipboard
        self.quartz.CGEventPost.side_effect = RuntimeError("injection failed")
        with self.assertRaises(RuntimeError):
            self.engine._auto_type("hello")
        self.engine._restore_timer.fire()
        self.assertEqual(self.clipboard, original)

    def test_stop_waits_for_delayed_restore(self):
        original = self.clipboard
        self.engine._auto_type("hello")
        timer = self.engine._restore_timer
        self.engine.stop()
        self.assertEqual(timer.join_timeout, 0.25)
        self.assertEqual(self.clipboard, original)

    def test_actual_restore_helper_does_not_touch_changed_pasteboard(self):
        # Exercise the helper itself with AppKit doubles, rather than its caller.
        pb = Mock()
        pb.changeCount.return_value = 99
        appkit = Mock()
        appkit.NSPasteboard.generalPasteboard.return_value = pb
        with patch.dict(sys.modules, {"AppKit": appkit, "Foundation": Mock()}):
            self.assertFalse(ORIGINAL_CLIPBOARD_RESTORE([], 10))
        pb.clearContents.assert_not_called()

    def test_restore_rechecks_ownership_after_rebuilding_items(self):
        pb = Mock()
        pb.changeCount.side_effect = [10, 11]
        appkit = Mock()
        appkit.NSPasteboard.generalPasteboard.return_value = pb
        with patch.dict(sys.modules, {"AppKit": appkit, "Foundation": Mock()}):
            self.assertFalse(ORIGINAL_CLIPBOARD_RESTORE([[("public.png", b"image")]], 10))
        pb.clearContents.assert_not_called()


if __name__ == "__main__":
    unittest.main()
