"""Diagnose native edits and correction capture in our own temporary window.

No prompts, global events, microphone, clipboard, other documents, or user history.
Explicit source-only own-process mode leaves production permissions BLOCKED.
Every AX target is checked against this PID; macOS still enforces native access.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ScopeViolation(RuntimeError):
    pass


def _summary(ax, attribute, value):
    """Metadata only: never serialize document text or selected-range payloads."""
    allowed = {"AXRole", "AXSubrole", "AXEnabled", "AXNumberOfCharacters", "AXProtectedContent"}
    if attribute not in allowed:
        return {"kind": "redacted"}
    if value is None:
        return {"kind": "absent"}
    if isinstance(value, bool):
        return {"kind": "boolean", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "value": int(value)}
    if isinstance(value, str):
        known = {"AXTextArea", "AXTextField", "AXSecureTextField", "AXUnknown"}
        return {"kind": "string", "known_value": value if value in known else "other"}
    try:
        if ax.AXValueGetType(value) == ax.kAXValueAXErrorType:
            valid, error = ax.AXValueGetValue(value, ax.kAXValueAXErrorType, None)
            if valid:
                return {"kind": "attribute_error", "code": int(error)}
    except Exception:
        pass
    return {"kind": "native_object", "type": type(value).__name__}


class _ScopedAX:
    """Constrain native requests to this process and record sanitized outcomes."""
    _TARGET_CALLS = {
        "AXUIElementSetMessagingTimeout", "AXUIElementCopyAttributeValue",
        "AXUIElementCopyMultipleAttributeValues", "AXUIElementIsAttributeSettable",
    }

    def __init__(self, ax, pid, *, allow_untrusted_self=False):
        if pid != os.getpid():
            raise ScopeViolation("Only the current process may be inspected")
        self._ax, self._pid = ax, pid
        self._allow_untrusted_self = bool(allow_untrusted_self)
        self.events = []
        self.root = self.AXUIElementCreateApplication(pid)

    def AXIsProcessTrusted(self):
        # This changes no OS permission. Explicit self-only diagnostics may
        # still receive AXErrorAPIDisabled from the native APIs below.
        return bool(self._ax.AXIsProcessTrusted()) or self._allow_untrusted_self

    def AXUIElementCreateApplication(self, pid):
        if pid != self._pid:
            raise ScopeViolation("Cross-process application root refused")
        return self._ax.AXUIElementCreateApplication(pid)

    def AXUIElementCreateSystemWide(self):
        raise ScopeViolation("System-wide roots are forbidden")

    def _guard(self, element):
        result = self._ax.AXUIElementGetPid(element, None)
        if (not isinstance(result, tuple) or len(result) != 2 or
                result[0] != self._ax.kAXErrorSuccess or result[1] != self._pid):
            raise ScopeViolation("AX target ownership could not be established")

    def __getattr__(self, name):
        function = getattr(self._ax, name)
        if name not in self._TARGET_CALLS:
            if callable(function) and name not in {"AXValueGetType", "AXValueGetValue"}:
                raise ScopeViolation("Unreviewed native operation refused")
            return function

        def call(element, *args):
            self._guard(element)
            started = time.monotonic()
            event = {"operation": name}
            try:
                result = function(element, *args)
                if name == "AXUIElementSetMessagingTimeout":
                    event["error"] = int(result)
                elif isinstance(result, tuple) and len(result) == 2:
                    event["error"] = int(result[0])
                    if name == "AXUIElementCopyMultipleAttributeValues":
                        attributes = list(args[0])
                        event["attributes"] = attributes
                        values = result[1]
                        if values is not None and len(values) == len(attributes):
                            event["metadata"] = {
                                attribute: _summary(self._ax, attribute, value)
                                for attribute, value in zip(attributes, values)
                            }
                    elif name == "AXUIElementCopyAttributeValue":
                        event["attribute"] = str(args[0])
                        event["metadata"] = _summary(self._ax, args[0], result[1])
                    elif name == "AXUIElementIsAttributeSettable":
                        event["attribute"] = str(args[0])
                        event["settable"] = result[1] if isinstance(result[1], bool) else None
                return result
            except Exception as error:
                event["exception_type"] = type(error).__name__
                raise
            finally:
                event["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
                if len(self.events) < 250:
                    self.events.append(event)

        return call


def run_probe(*, allow_own_process_without_trust=False):
    """Run on the main thread and return sanitized evidence for this executable."""
    import ApplicationServices as AX
    import AppKit
    import objc
    from Foundation import NSDate, NSDefaultRunLoopMode
    from accessibility_reader import AccessibilityTextReader
    from correction_monitor import CorrectionMonitor, STABLE_SECONDS
    from dictation_engine import input_permissions
    from learning_service import LearningService
    from personalization import PersonalizationManager

    permissions = input_permissions()
    trusted = permissions["accessibility"] is True
    evidence = {
        "permissions": permissions, "scope": "own temporary NSTextView only",
        "steps": [], "captures": [], "native_ax": [],
        "production_permissions_ready": trusted,
        "own_process_mode": bool(allow_own_process_without_trust and not trusted),
        "settle_seconds": STABLE_SECONDS,
    }
    if not trusted and not allow_own_process_without_trust:
        return evidence | {"status": "blocked", "reason": "Accessibility is not granted to this executable; no prompt shown."}
    if threading.current_thread() is not threading.main_thread():
        return evidence | {"status": "failed", "reason": "Native editor probe must run on the main thread."}

    with objc.autorelease_pool(), tempfile.TemporaryDirectory(prefix="justdictate-native-probe-") as temporary:
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        app.finishLaunching()
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(80, 80, 560, 240),
            AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable,
            AppKit.NSBackingStoreBuffered, False,
        )
        window.setReleasedWhenClosed_(False)
        window.setTitle_("JustDictate — Temporary Native Test")
        editor = AppKit.NSTextView.alloc().initWithFrame_(AppKit.NSMakeRect(10, 10, 540, 220))
        editor.setEditable_(True)
        editor.setSelectable_(True)
        window.contentView().addSubview_(editor)
        baseline = "Own editor 😃\n"
        raw = "Please ask Gotham about the Sonic release."
        pasted = raw + " "
        editor.setString_(baseline)
        baseline_utf16 = len(baseline.encode("utf-16-le")) // 2
        editor.setSelectedRange_((baseline_utf16, 0))
        window.makeFirstResponder_(editor)
        window.makeKeyAndOrderFront_(None)
        app.activateIgnoringOtherApps_(True)
        scoped = _ScopedAX(AX, os.getpid(), allow_untrusted_self=not trusted)
        evidence["native_ax"] = scoped.events
        reader = AccessibilityTextReader(ax_module=scoped)
        reader._system = scoped.root
        manager = PersonalizationManager(Path(temporary) / "history", enabled=True)
        service = LearningService(manager, enabled=True)
        clock = [0.0]
        monitor = CorrectionMonitor(reader, service.record, service.auto_correct,
                                    lambda: True, clock=lambda: clock[0])

        def pump_until(ready, timeout=2.0):
            deadline = time.monotonic() + timeout
            while not ready() and time.monotonic() < deadline:
                event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                    AppKit.NSEventMaskAny, NSDate.dateWithTimeIntervalSinceNow_(0.001),
                    NSDefaultRunLoopMode, True,
                )
                if event is not None:
                    app.sendEvent_(event)
                app.updateWindows()
            return ready()

        def worker_call(function, phase):
            done, result, failures = threading.Event(), [], []
            started = time.monotonic()

            def call():
                try:
                    result.append(function())
                except Exception as error:
                    failures.append(type(error).__name__)
                finally:
                    done.set()

            worker = threading.Thread(target=call, daemon=True)
            worker.start()
            completed = pump_until(done.is_set)
            evidence["captures"].append({
                "phase": phase, "completed": completed,
                "reader_rejection": reader.last_rejection,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                "exception_type": failures[0] if failures else None,
            })
            if not completed or failures:
                raise RuntimeError("Owned native operation did not complete")
            return result[0] if result else None

        def capture(phase):
            for attempt in range(3):
                snapshot = worker_call(reader.capture, f"{phase}_{attempt + 1}")
                if snapshot is not None:
                    return snapshot
            return None

        def drain():
            with service._queue.all_tasks_done:
                if not service._queue.all_tasks_done.wait_for(
                    lambda: service._queue.unfinished_tasks == 0, timeout=2
                ):
                    raise RuntimeError("Local learning queue did not drain")

        def fail(reason):
            return evidence | {"status": "failed" if trusted else "blocked",
                               "own_process_pipeline": "failed", "reason": reason}

        try:
            focus_ready = pump_until(lambda: bool(app.isActive()) and bool(window.isKeyWindow()) and
                                     window.firstResponder() == editor)
            evidence["focus_ready"] = focus_ready
            if not focus_ready:
                return fail("Owned test window did not become active with its editor focused.")
            before = capture("initial")
            if before is None:
                return fail("Native reader rejected the owned field; inspect native_ax and captures.")
            if before.value != baseline or before.selection != (len(baseline), 0):
                return fail("Initial native text/UTF16 selection did not round-trip.")
            evidence["steps"].append("native field and emoji selection captured")
            worker_call(lambda: monitor.prepare(pasted), "prepare")
            if monitor._prepared is None:
                return fail("Correction monitor could not prepare the native pasted span.")
            editor.insertText_replacementRange_(pasted, (baseline_utf16, 0))
            if not monitor.complete(raw, raw, None):
                return fail("Local history capture was not queued.")
            drain()
            inserted = capture("inserted")
            if inserted is None or inserted.value != baseline + pasted or inserted.element != before.element:
                return fail("Native insertion or element identity did not round-trip.")
            worker_call(monitor.poll, "verify_paste")
            if manager.status()["automatic_count"] != 0:
                return fail("Unchanged pasted output was incorrectly labelled.")
            evidence["steps"].append("exact native paste verified without inventing a label")

            position = len((baseline + "Please ask ").encode("utf-16-le")) // 2
            editor.insertText_replacementRange_("Gowtham", (position, len("Gotham")))
            clock[0] += 1.0
            worker_call(monitor.poll, "first_edit_observed")
            epsilon = min(0.5, STABLE_SECONDS / 2)
            clock[0] += STABLE_SECONDS - epsilon
            worker_call(monitor.poll, "first_edit_before_threshold")
            drain()
            if manager.latest()["corrected"] is not None:
                return fail("First native correction was labelled before its settling interval.")
            clock[0] += epsilon
            worker_call(monitor.poll, "first_edit_stable")
            drain()
            row = manager.latest()
            if row is None or row["corrected"] != "Please ask Gowtham about the Sonic release." or row["label_source"] != "automatic":
                return fail("First native correction was not captured as an automatic label.")
            evidence["steps"].append("first native correction automatically labelled in temporary SQLite")

            position = len((baseline + "Please ask Gowtham about the ").encode("utf-16-le")) // 2
            editor.insertText_replacementRange_("Sona", (position, len("Sonic")))
            clock[0] += 1.0
            worker_call(monitor.poll, "second_edit_observed")
            clock[0] += STABLE_SECONDS - epsilon
            worker_call(monitor.poll, "second_edit_before_threshold")
            drain()
            if manager.latest()["corrected"] is not None:
                return fail("Second native correction was labelled before its settling interval.")
            clock[0] += epsilon
            worker_call(monitor.poll, "second_edit_stable")
            drain()
            row = manager.latest()
            if row is None or row["corrected"] != "Please ask Gowtham about the Sona release." or row["label_source"] != "automatic":
                return fail("Second nonadjacent correction did not update the automatic label.")
            evidence["steps"].append("two separated native corrections retained in one automatic label")
            editor.setString_(baseline + pasted)
            editor.setSelectedRange_((len((baseline + pasted).encode("utf-16-le")) // 2, 0))
            clock[0] += 1.0
            worker_call(monitor.poll, "reverted")
            drain()
            state = manager.status()
            if state["record_count"] != 1 or state["automatic_count"] != 0 or state["manual_count"] != 0:
                return fail("Reverting the native edit did not revoke its automatic label.")
            evidence["steps"].append("native edit reversal revoked the automatic label")
            evidence["own_process_pipeline"] = "passed"
            if not trusted:
                return evidence | {"status": "blocked", "reason": "Owned native pipeline passed; normal capture still requires Accessibility permission."}
            return evidence | {"status": "passed"}
        except Exception as error:
            evidence["exception_type"] = type(error).__name__
            return fail("Owned native probe failed; inspect sanitized native diagnostics.")
        finally:
            monitor.close()
            service.close()
            window.orderOut_(None)
            window.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-own-process-without-trust", action="store_true")
    args = parser.parse_args()
    result = run_probe(allow_own_process_without_trust=args.allow_own_process_without_trust)
    print(json.dumps(result, indent=2))
    return {"passed": 0, "blocked": 2}.get(result["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
