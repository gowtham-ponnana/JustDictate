"""AX reader tests use synthetic fields; no user document is inspected."""

import os
import unittest

import ApplicationServices as NativeAX
from AppKit import NSTextView, NSTextField, NSSecureTextField
from Foundation import NSNumber, NSNull

from accessibility_reader import (
    AccessibilityTextReader, CAPTURE_BUDGET_SECONDS, MAX_TEXT_CHARACTERS, utf16_selection,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeAX:
    """Real constants/CFRange marshalling with fully synthetic remote IPC."""
    def __init__(self, clock):
        self.clock = clock
        self.system = object()
        self.element = object()
        self.focused = self.element
        self.trusted = True
        self.writable = True
        self.failure = 0
        self.timeout_failure = 0
        self.delay = 0.0
        self.calls = []
        self.timeouts = []
        self.text = "Hello 😃 world"
        self.selection = (9, 5)  # UTF16 offsets selecting "world".
        self.metadata = {
            "AXRole": "AXTextArea",
            "AXSubrole": "AXUnknown",
            "AXEnabled": True,
            "AXNumberOfCharacters": len(self.text.encode("utf-16-le")) // 2,
            "AXProtectedContent": NativeAX.AXValueCreate(
                NativeAX.kAXValueAXErrorType, NativeAX.kAXErrorAttributeUnsupported),
        }

    def __getattr__(self, name):
        return getattr(NativeAX, name)

    def AXIsProcessTrusted(self):
        return self.trusted

    def AXUIElementCreateSystemWide(self):
        return self.system

    def AXUIElementSetMessagingTimeout(self, element, timeout):
        self.timeouts.append(timeout)
        return self.timeout_failure

    def AXUIElementCopyAttributeValue(self, element, name, output):
        self.calls.append((name,))
        self.clock.now += self.delay
        return self.failure, self.focused

    def AXUIElementCopyMultipleAttributeValues(self, element, names, options, output):
        self.calls.append(tuple(names))
        self.clock.now += self.delay
        if "AXValue" in names:
            return self.failure, [self.text, NativeAX.AXValueCreate(
                NativeAX.kAXValueCFRangeType, self.selection)]
        return self.failure, [self.metadata[name] for name in names]

    def AXUIElementIsAttributeSettable(self, element, name, output):
        self.calls.append(("settable", name))
        self.clock.now += self.delay
        return self.failure, self.writable


class AccessibilityReaderTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.ax = FakeAX(self.clock)
        self.reader = AccessibilityTextReader(ax_module=self.ax, clock=self.clock)

    def content_reads(self):
        return [call for call in self.ax.calls if call[0] == "AXValue"]

    def test_captures_same_field_with_python_selection_and_no_permission_prompt(self):
        snapshot = self.reader.capture()
        self.assertIs(snapshot.element, self.ax.element)
        self.assertEqual(snapshot.value, "Hello 😃 world")
        self.assertEqual(snapshot.selection, (8, 5))
        self.assertEqual(len(self.content_reads()), 1)
        self.assertTrue(all(0 < timeout <= 0.01 for timeout in self.ax.timeouts))
        self.assertEqual(self.ax.calls[0], ("AXFocusedUIElement",))
        self.assertEqual(self.ax.calls[-1], ("AXFocusedUIElement",))

    def test_untrusted_process_returns_without_remote_read_or_permission_request(self):
        self.ax.trusted = False
        self.assertIsNone(self.reader.capture())
        self.assertEqual(self.ax.calls, [])

    def test_native_number_and_null_optional_metadata_are_supported(self):
        self.ax.metadata["AXNumberOfCharacters"] = NSNumber.numberWithInt_(14)
        self.ax.metadata["AXProtectedContent"] = NSNull.null()
        self.assertIsNotNone(self.reader.capture())

    def test_optional_absent_subrole_supports_normal_cocoa_text_view(self):
        # A real, unattached Cocoa editor supplies the role/subrole metadata;
        # synthetic IPC keeps the test independent of AX permission and focus.
        editor = NSTextView.alloc().initWithFrame_(((0, 0), (200, 100)))
        self.assertEqual(editor.accessibilityRole(), "AXTextArea")
        self.assertIsNone(editor.accessibilitySubrole())
        self.ax.metadata["AXRole"] = editor.accessibilityRole()
        absent_values = [
            editor.accessibilitySubrole(), NSNull.null(),
            NativeAX.AXValueCreate(NativeAX.kAXValueAXErrorType, NativeAX.kAXErrorAttributeUnsupported),
            NativeAX.AXValueCreate(NativeAX.kAXValueAXErrorType, NativeAX.kAXErrorNoValue),
        ]
        for subrole in absent_values:
            with self.subTest(subrole=subrole):
                self.ax.metadata["AXSubrole"] = subrole
                self.assertIsNotNone(self.reader.capture())

    def test_unreadable_optional_subrole_or_protection_marker_abstains(self):
        for attribute in ("AXSubrole", "AXProtectedContent"):
            for error in (NativeAX.kAXErrorCannotComplete, NativeAX.kAXErrorInvalidUIElement):
                with self.subTest(attribute=attribute, error=error):
                    ax = FakeAX(self.clock)
                    ax.metadata[attribute] = NativeAX.AXValueCreate(NativeAX.kAXValueAXErrorType, error)
                    self.assertIsNone(AccessibilityTextReader(ax_module=ax, clock=self.clock).capture())
                    self.assertFalse(any(call[0] == "AXValue" for call in ax.calls))

    def test_absent_enabled_attribute_requires_positive_writability(self):
        for enabled in (None, NSNull.null(),
                        NativeAX.AXValueCreate(NativeAX.kAXValueAXErrorType,
                                              NativeAX.kAXErrorAttributeUnsupported),
                        NativeAX.AXValueCreate(NativeAX.kAXValueAXErrorType,
                                              NativeAX.kAXErrorNoValue)):
            with self.subTest(enabled=enabled):
                self.ax.metadata["AXEnabled"] = enabled
                self.ax.writable = True
                self.assertIsNotNone(self.reader.capture())
                self.ax.calls.clear()
                self.ax.writable = False
                self.assertIsNone(self.reader.capture())
                self.assertEqual(self.reader.last_rejection, "value_not_writable")
                self.assertEqual(self.content_reads(), [])

    def test_explicit_disabled_malformed_and_unreadable_enabled_values_stay_rejected(self):
        values = [False, 0, 2, "true", 1.0, [],
                  NativeAX.AXValueCreate(NativeAX.kAXValueAXErrorType, NativeAX.kAXErrorCannotComplete),
                  NativeAX.AXValueCreate(NativeAX.kAXValueAXErrorType, NativeAX.kAXErrorInvalidUIElement)]
        for enabled in values:
            with self.subTest(enabled=enabled):
                self.ax.metadata["AXEnabled"] = enabled
                self.assertIsNone(self.reader.capture())
                self.assertEqual(self.reader.last_rejection, "metadata_enabled_rejected")
        self.assertEqual(self.content_reads(), [])

    def test_native_cocoa_textview_missing_enabled_is_supported_only_when_editable(self):
        editor = NSTextView.alloc().initWithFrame_(((0, 0), (200, 100)))
        editor.setString_(self.ax.text)
        self.assertNotIn("AXEnabled", editor.accessibilityAttributeNames())
        self.assertIsNone(editor.accessibilityAttributeValue_("AXEnabled"))
        for attribute in ("AXRole", "AXSubrole", "AXEnabled", "AXNumberOfCharacters"):
            self.ax.metadata[attribute] = editor.accessibilityAttributeValue_(attribute)
        self.ax.writable = bool(editor.accessibilityIsAttributeSettable_("AXValue"))
        self.assertTrue(self.ax.writable)
        snapshot = self.reader.capture()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.selection, (8, 5))
        editor.setEditable_(False)
        self.ax.writable = bool(editor.accessibilityIsAttributeSettable_("AXValue"))
        self.ax.calls.clear()
        self.assertFalse(self.ax.writable)
        self.assertIsNone(self.reader.capture())
        self.assertEqual(self.reader.last_rejection, "value_not_writable")
        self.assertEqual(self.content_reads(), [])

    def test_native_disabled_and_secure_cocoa_cells_remain_rejected(self):
        for control_class in (NSTextField, NSSecureTextField):
            control = control_class.alloc().initWithFrame_(((0, 0), (200, 100)))
            control.setStringValue_(self.ax.text)
            if control_class is NSTextField:
                control.setEnabled_(False)
            cell = control.cell()
            for attribute in ("AXRole", "AXSubrole", "AXEnabled", "AXNumberOfCharacters"):
                self.ax.metadata[attribute] = cell.accessibilityAttributeValue_(attribute)
            self.ax.writable = bool(cell.accessibilityIsAttributeSettable_("AXValue"))
            self.assertIsNone(self.reader.capture())
        self.assertEqual(self.content_reads(), [])

    def test_secure_password_protected_readonly_and_nontext_fields_never_read_content(self):
        variants = [
            {"AXSubrole": "AXSecureTextField"},
            {"AXSubrole": "AXPasswordField"},
            {"AXProtectedContent": True},
            {"AXProtectedContent": "unknown"},
            {"AXRole": "AXStaticText"},
            {"AXRole": "AXWebArea"},
            {"AXEnabled": False},
            {"AXSubrole": 123},
            {"AXNumberOfCharacters": MAX_TEXT_CHARACTERS + 1},
        ]
        for values in variants:
            with self.subTest(values=values):
                ax = FakeAX(self.clock)
                ax.metadata.update(values)
                self.assertIsNone(AccessibilityTextReader(ax_module=ax, clock=self.clock).capture())
                self.assertFalse(any(call[0] == "AXValue" for call in ax.calls))
        self.ax.writable = False
        self.assertIsNone(self.reader.capture())
        self.assertEqual(self.content_reads(), [])

    def test_read_error_and_timeout_configuration_failure_abstain(self):
        self.ax.failure = NativeAX.kAXErrorCannotComplete
        self.assertIsNone(self.reader.capture())
        self.assertEqual(self.content_reads(), [])
        self.assertEqual(self.ax.calls, [("AXFocusedUIElement",)] * 2)
        self.ax.failure = NativeAX.kAXErrorInvalidUIElement
        self.ax.calls.clear()
        self.assertIsNone(self.reader.capture())
        self.assertEqual(self.ax.calls, [("AXFocusedUIElement",)])
        self.ax.failure = 0
        self.ax.timeout_failure = NativeAX.kAXErrorIllegalArgument
        self.ax.calls.clear()
        self.assertIsNone(self.reader.capture())
        self.assertEqual(self.ax.calls, [])

    def test_deadline_prevents_further_reads_and_shrinks_last_timeout(self):
        # Cooperative IPC takes up to 9ms per request, respecting the shrinking
        # timeout. Both attempts exhaust their budgets without a third attempt.
        def timed(function):
            def invoke(*args):
                result = function(*args)
                self.clock.now += min(0.009, self.ax.timeouts[-1])
                return result
            return invoke

        for name in ("AXUIElementCopyAttributeValue", "AXUIElementCopyMultipleAttributeValues",
                     "AXUIElementIsAttributeSettable"):
            setattr(self.ax, name, timed(getattr(self.ax, name)))
        self.assertIsNone(self.reader.capture())
        self.assertEqual(len(self.ax.calls), 6)
        self.assertLess(self.ax.timeouts[-1], 0.01)
        self.assertAlmostEqual(self.clock.now, CAPTURE_BUDGET_SECONDS)
        self.assertEqual(self.reader.last_rejection, "metadata_deadline")

    def test_cold_writable_deadline_retries_entire_snapshot_with_remaining_budget(self):
        original = self.ax.AXUIElementIsAttributeSettable
        replacement = object()

        def cold_writable(*args):
            result = original(*args)
            if self.clock.now == 0:
                self.clock.now = 0.037  # Observed cold first-capture failure.
                self.ax.delay = 0.001
                self.ax.focused = replacement
                self.ax.text = "Fresh 😀 words"
                count = len(self.ax.text.encode("utf-16-le")) // 2
                self.ax.metadata["AXNumberOfCharacters"] = count
                self.ax.selection = (count, 0)
            return result

        self.ax.AXUIElementIsAttributeSettable = cold_writable
        snapshot = self.reader.capture()
        self.assertIsNotNone(snapshot)
        self.assertIs(snapshot.element, replacement)
        self.assertEqual(snapshot.value, self.ax.text)
        self.assertEqual(snapshot.selection, (len(self.ax.text), 0))
        self.assertEqual([call[0] for call in self.ax.calls], [
            "AXFocusedUIElement", "AXRole", "settable",
            "AXFocusedUIElement", "AXRole", "settable", "AXValue", "AXFocusedUIElement",
        ])
        self.assertTrue(all(timeout <= 0.008 for timeout in self.ax.timeouts[3:]))
        self.assertAlmostEqual(self.clock.now, 0.042)
        self.assertLess(self.clock.now, CAPTURE_BUDGET_SECONDS)
        self.assertIsNone(self.reader.last_rejection)

    def test_transient_error_at_each_native_stage_gets_only_one_fresh_retry(self):
        stages = [
            ("AXUIElementCopyAttributeValue", 1),
            ("AXUIElementCopyMultipleAttributeValues", 1),
            ("AXUIElementIsAttributeSettable", 1),
            ("AXUIElementCopyMultipleAttributeValues", 2),
            ("AXUIElementCopyAttributeValue", 2),
        ]
        for method, failure_call in stages:
            with self.subTest(method=method, failure_call=failure_call):
                ax = FakeAX(self.clock)
                original = getattr(ax, method)
                call_count = 0

                def fail_once(*args):
                    nonlocal call_count
                    call_count += 1
                    result = original(*args)
                    if call_count == failure_call:
                        return NativeAX.kAXErrorCannotComplete, None
                    return result

                setattr(ax, method, fail_once)
                reader = AccessibilityTextReader(ax_module=ax, clock=self.clock)
                self.assertIsNotNone(reader.capture())
                self.assertIsNone(reader.last_rejection)
                self.assertLessEqual(len(ax.calls), 10)
                self.assertEqual([call[0] for call in ax.calls[-5:]], [
                    "AXFocusedUIElement", "AXRole", "settable", "AXValue", "AXFocusedUIElement",
                ])

    def test_retry_rechecks_protection_before_reading_newly_focused_field(self):
        original = self.ax.AXUIElementIsAttributeSettable

        def cold_writable(*args):
            result = original(*args)
            self.clock.now = 0.037
            self.ax.focused = object()
            self.ax.metadata["AXSubrole"] = "AXSecureTextField"
            return result

        self.ax.AXUIElementIsAttributeSettable = cold_writable
        self.assertIsNone(self.reader.capture())
        self.assertEqual(self.reader.last_rejection, "metadata_subrole_rejected")
        self.assertEqual(len(self.ax.calls), 5)
        self.assertEqual(self.content_reads(), [])

    def test_retry_timeout_overrun_does_not_start_third_attempt(self):
        original = self.ax.AXUIElementIsAttributeSettable

        def cold_writable(*args):
            result = original(*args)
            self.clock.now = 0.037
            self.ax.delay = 0.2
            return result

        self.ax.AXUIElementIsAttributeSettable = cold_writable
        self.assertIsNone(self.reader.capture())
        self.assertEqual([call[0] for call in self.ax.calls], [
            "AXFocusedUIElement", "AXRole", "settable", "AXFocusedUIElement",
        ])
        self.assertEqual(self.reader.last_rejection, "focus_deadline")
        self.assertEqual(self.content_reads(), [])

    def test_native_timeout_overrun_is_not_followed_by_more_calls(self):
        self.ax.delay = 0.2  # An uncooperative native call can overrun its timeout.
        self.assertIsNone(self.reader.capture())
        self.assertEqual(self.ax.calls, [("AXFocusedUIElement",)])
        self.assertEqual(self.reader.last_rejection, "focus_deadline")

    def test_diagnostics_distinguish_ax_failure_and_never_expose_exception_contents(self):
        self.ax.failure = NativeAX.kAXErrorCannotComplete
        self.assertIsNone(self.reader.capture())
        self.assertEqual(self.reader.last_rejection, "focus_ax_error")
        self.ax.failure = 0
        self.assertIsNotNone(self.reader.capture())
        self.assertIsNone(self.reader.last_rejection)

        def bad_native_call(*args):
            raise RuntimeError("Private document content must not appear")

        self.ax.AXUIElementCopyMultipleAttributeValues = bad_native_call
        self.assertIsNone(self.reader.capture())
        self.assertEqual(self.reader.last_rejection, "unexpected_native_exception")

    def test_native_cfrange_binding_returns_boolean_and_two_integer_offsets(self):
        wrapped = NativeAX.AXValueCreate(NativeAX.kAXValueCFRangeType, (9, 5))
        success, decoded = NativeAX.AXValueGetValue(wrapped, NativeAX.kAXValueCFRangeType, None)
        self.assertIs(success, True)
        self.assertEqual(len(decoded), 2)
        self.assertTrue(all(isinstance(offset, int) for offset in decoded))
        self.assertEqual(tuple(decoded), (9, 5))
        metadata = NativeAX.AXUIElementIsAttributeSettable.__metadata__()
        self.assertEqual(metadata["arguments"][2]["type"], b"o^Z")

    def test_focus_change_between_metadata_and_value_invalidates_snapshot(self):
        original = self.ax.AXUIElementCopyMultipleAttributeValues

        def changed(element, names, options, output):
            result = original(element, names, options, output)
            if "AXValue" in names:
                self.ax.focused = object()
            return result

        self.ax.AXUIElementCopyMultipleAttributeValues = changed
        self.assertIsNone(self.reader.capture())

    def test_inconsistent_length_and_partial_surrogate_selection_abstain(self):
        self.ax.metadata["AXNumberOfCharacters"] += 1
        self.assertIsNone(self.reader.capture())
        self.ax.metadata["AXNumberOfCharacters"] -= 1
        self.ax.selection = (7, 0)  # Between the emoji's two UTF16 code units.
        self.assertIsNone(self.reader.capture())
        self.ax.selection = (100, 0)
        self.assertIsNone(self.reader.capture())

    def test_overlong_payload_is_rejected_even_if_character_count_lies(self):
        self.ax.text = "a" * (MAX_TEXT_CHARACTERS + 1)
        self.assertIsNone(self.reader.capture())

    def test_concurrent_capture_never_waits_for_reader_lock(self):
        self.reader._lock.acquire()
        try:
            self.assertIsNone(self.reader.capture())
        finally:
            self.reader._lock.release()
        self.assertEqual(self.ax.calls, [])

    def test_native_binding_smoke_uses_only_own_process_and_synthetic_attributes(self):
        # Exercise actual timeout binding and real AX identity/marshalling while
        # all document reads remain FakeAX methods above.
        self.ax.system = NativeAX.AXUIElementCreateSystemWide()
        self.ax.element = NativeAX.AXUIElementCreateApplication(os.getpid())
        self.ax.focused = NativeAX.AXUIElementCreateApplication(os.getpid())
        self.ax.AXUIElementSetMessagingTimeout = NativeAX.AXUIElementSetMessagingTimeout
        snapshot = self.reader.capture()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.element, self.ax.focused)
        self.assertEqual(snapshot.selection, (8, 5))


class UTF16SelectionTests(unittest.TestCase):
    def test_ascii_empty_emoji_and_combining_mark_offsets(self):
        self.assertEqual(utf16_selection("", 0, 0), (0, 0))
        self.assertEqual(utf16_selection("abc", 1, 2), (1, 2))
        self.assertEqual(utf16_selection("a😃b", 1, 2), (1, 1))
        self.assertEqual(utf16_selection("a😃b", 3, 1), (2, 1))
        self.assertEqual(utf16_selection("e\u0301😃", 0, 4), (0, 3))

    def test_bad_ranges_do_not_guess(self):
        for start, length in ((2, 0), (1, 1), (-1, 1), (1, -1), (4, 1), (True, 0), (1.0, 0)):
            with self.subTest(start=start, length=length):
                self.assertIsNone(utf16_selection("a😃b", start, length))
        self.assertIsNone(utf16_selection("\ud800", 0, 1))


if __name__ == "__main__":
    unittest.main()
