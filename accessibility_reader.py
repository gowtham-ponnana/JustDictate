"""Bounded, read-only snapshots of the focused editable macOS text field.

This reader never requests Accessibility access, subscribes to key events, or
walks other documents. Unsupported, protected, ambiguous, or slow fields simply
produce no learning example. The native timeout is best effort, not a hard
real-time deadline; the deadline below prevents starting further AX requests.
"""

import threading
import time


MAX_TEXT_CHARACTERS = 32768
AX_TIMEOUT_SECONDS = 0.01
FIRST_ATTEMPT_BUDGET_SECONDS = 0.035
CAPTURE_BUDGET_SECONDS = 0.045


def utf16_selection(text: str, location: int, length: int) -> tuple[int, int] | None:
    """Translate an NSString/CFRange into Python offsets, refusing split emoji."""
    if (not isinstance(location, int) or isinstance(location, bool) or
            not isinstance(length, int) or isinstance(length, bool) or
            location < 0 or length < 0):
        return None
    end = location + length
    offsets = {0: 0}
    units = 0
    for index, character in enumerate(text):
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            return None  # Malformed/unpaired UTF-16 must not be reconstructed.
        units += 2 if codepoint > 0xFFFF else 1
        offsets[units] = index + 1
    if location not in offsets or end not in offsets:
        return None
    return offsets[location], offsets[end] - offsets[location]


class AccessibilityTextReader:
    def __init__(self, *, ax_module=None, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._system = None
        self._ax = ax_module
        self.last_rejection: str | None = None
        self._retryable = False
        if self._ax is None:
            try:
                import ApplicationServices
                self._ax = ApplicationServices
            except ImportError:
                pass

    def capture(self):
        """Return TextSnapshot(element, value, (Python start, length)) or None.

        The AX element wrapper compares using Core Foundation equality, allowing
        the correction monitor to identify the same field across captures.
        A transient first failure gets one fresh attempt within a shared 45ms
        scheduling budget, which fits inside dictation's 50ms pre-paste wait.
        Native timeouts are best effort; no request starts after the total budget.
        """
        if self._ax is None:
            return self._reject("ax_unavailable")
        if not self._lock.acquire(blocking=False):
            return self._reject("capture_busy")
        try:
            started = self._clock()
            deadline = started + CAPTURE_BUDGET_SECONDS
            for attempt_deadline in (started + FIRST_ATTEMPT_BUDGET_SECONDS, deadline):
                self.last_rejection = None
                self._retryable = False
                snapshot = self._capture(attempt_deadline)
                if snapshot is not None:
                    return snapshot
                if not self._retryable or self._clock() >= deadline:
                    return None
            return None
        except Exception:
            # AX failures must not delay/fail dictation or log document contents.
            return self._reject("unexpected_native_exception")
        finally:
            self._lock.release()

    def _reject(self, reason, *, retryable=False):
        # Diagnostic codes are fixed strings, never native values, exception
        # messages, text, selections, identifiers, or other document metadata.
        self.last_rejection = reason
        self._retryable = retryable
        return None

    def _capture(self, deadline):
        ax = self._ax
        if not ax.AXIsProcessTrusted():
            return self._reject("accessibility_not_trusted")
        if self._system is None:
            self._system = ax.AXUIElementCreateSystemWide()
        if self._system is None:
            return self._reject("system_element_unavailable")

        def call(stage, element, function, *args):
            remaining = deadline - self._clock()
            if remaining <= 0:
                return self._reject(stage + "_deadline", retryable=True)
            timeout = min(AX_TIMEOUT_SECONDS, remaining)
            if ax.AXUIElementSetMessagingTimeout(element, timeout) != ax.kAXErrorSuccess:
                return self._reject(stage + "_timeout_setup_failed")
            if self._clock() >= deadline:
                return self._reject(stage + "_deadline", retryable=True)
            result = function(element, *args)
            if self._clock() >= deadline:
                return self._reject(stage + "_deadline", retryable=True)
            if result[0] != ax.kAXErrorSuccess:
                return self._reject(stage + "_ax_error",
                                    retryable=result[0] == ax.kAXErrorCannotComplete)
            return result

        result = call("focus", self._system, ax.AXUIElementCopyAttributeValue,
                      ax.kAXFocusedUIElementAttribute, None)
        if result is None:
            return None
        if result[1] is None:
            return self._reject("focused_element_unavailable")
        element = result[1]

        # Inspect eligibility before requesting AXValue: secure fields never
        # reach the content read. Character count bounds the expected payload.
        metadata_names = [ax.kAXRoleAttribute, ax.kAXSubroleAttribute,
                          ax.kAXEnabledAttribute, ax.kAXNumberOfCharactersAttribute,
                          "AXProtectedContent"]
        result = call("metadata", element, ax.AXUIElementCopyMultipleAttributeValues,
                      metadata_names, 0, None)
        if result is None:
            return None
        metadata = result[1]
        if metadata is None or len(metadata) != len(metadata_names):
            return self._reject("metadata_incomplete")
        role, subrole, enabled, count, protected = metadata
        if role not in (ax.kAXTextFieldRole, ax.kAXTextAreaRole):
            return self._reject("metadata_role_rejected")
        if not self._safe_subrole(subrole):
            return self._reject("metadata_subrole_rejected")
        # Cocoa NSTextView can omit AXEnabled while exposing a writable AXValue.
        # Accept only recognized absence; explicit false, malformed data and
        # messaging errors still reject the field. Writability below is required
        # regardless of whether AXEnabled is present.
        if isinstance(enabled, (bool, int)):
            if enabled != 1:
                return self._reject("metadata_enabled_rejected")
        elif not self._absent_attribute(enabled):
            return self._reject("metadata_enabled_rejected")
        if (not isinstance(count, int) or isinstance(count, bool) or
                not 0 <= count <= MAX_TEXT_CHARACTERS):
            return self._reject("metadata_length_rejected")
        if not self._unprotected(protected):
            return self._reject("metadata_protection_rejected")

        result = call("writable", element, ax.AXUIElementIsAttributeSettable,
                      ax.kAXValueAttribute, None)
        if result is None:
            return None
        if result[1] is not True:
            return self._reject("value_not_writable")

        result = call("content", element, ax.AXUIElementCopyMultipleAttributeValues,
                      [ax.kAXValueAttribute, ax.kAXSelectedTextRangeAttribute],
                      ax.kAXCopyMultipleAttributeOptionStopOnError, None)
        if result is None:
            return None
        values = result[1]
        if values is None or len(values) != 2:
            return self._reject("content_incomplete")
        value, selected_range = values
        if not isinstance(value, str) or len(value) > MAX_TEXT_CHARACTERS:
            return self._reject("content_text_rejected")
        if ax.AXValueGetType(selected_range) != ax.kAXValueCFRangeType:
            return self._reject("selection_type_rejected")
        valid, raw_range = ax.AXValueGetValue(selected_range, ax.kAXValueCFRangeType, None)
        if not valid or raw_range is None or len(raw_range) != 2:
            return self._reject("selection_range_rejected")
        selection = utf16_selection(value, raw_range[0], raw_range[1])
        if selection is None:
            return self._reject("selection_invalid")
        # A concurrent edit between metadata and content reads is ambiguous.
        if len(value.encode("utf-16-le")) // 2 != count:
            return self._reject("content_length_changed")

        result = call("focus_recheck", self._system, ax.AXUIElementCopyAttributeValue,
                      ax.kAXFocusedUIElementAttribute, None)
        if result is None:
            return None
        if result[1] != element:
            return self._reject("focus_changed")
        if self._clock() >= deadline:
            return self._reject("complete_deadline", retryable=True)
        from correction_monitor import TextSnapshot
        self.last_rejection = None
        return TextSnapshot(element, value, selection)

    def _unprotected(self, value):
        """Optional protection markers may be absent, but not unreadable."""
        if isinstance(value, (bool, int)) and value == 0:
            return True
        return self._absent_attribute(value)

    def _safe_subrole(self, value):
        # Ordinary Cocoa NSTextView has no subrole. Absence is different from
        # an AX messaging error, which cannot safely establish field eligibility.
        if isinstance(value, str):
            return not any(word in value.casefold() for word in ("secure", "password", "protected"))
        return self._absent_attribute(value)

    def _absent_attribute(self, value):
        if value is None:
            return True
        from Foundation import NSNull
        if isinstance(value, NSNull):
            return True
        ax = self._ax
        try:
            if ax.AXValueGetType(value) == ax.kAXValueAXErrorType:
                valid, error = ax.AXValueGetValue(value, ax.kAXValueAXErrorType, None)
                return bool(valid and error in (ax.kAXErrorAttributeUnsupported, ax.kAXErrorNoValue))
        except (TypeError, ValueError):
            pass
        return False
