import json
import os
import unittest

from tools.native_editor_probe import ScopeViolation, _ScopedAX, _summary


class AttributeErrorValue:
    def __init__(self, code):
        self.code = code


class FakeAX:
    kAXErrorSuccess = 0
    kAXValueAXErrorType = 5

    def __init__(self):
        self.calls = []
        self.trusted = False
        self.attributes = {
            "AXRole": "AXTextArea", "AXEnabled": False,
            "AXProtectedContent": AttributeErrorValue(-25205),
            "AXValue": "private document contents", "AXSelectedTextRange": "private selection",
        }

    def AXIsProcessTrusted(self):
        return self.trusted

    def AXIsProcessTrustedWithOptions(self, options):
        raise AssertionError("permission prompt must never be invoked")

    def AXUIElementCreateApplication(self, pid):
        self.calls.append(("create", pid))
        return "owned"

    def AXUIElementGetPid(self, element, placeholder):
        return (0, os.getpid()) if element == "owned" else (0, os.getpid() + 1)

    def AXUIElementCreateSystemWide(self):
        raise AssertionError("system-wide root must never be created")

    def AXUIElementSetMessagingTimeout(self, element, timeout):
        self.calls.append(("timeout", element))
        return 0

    def AXUIElementCopyAttributeValue(self, element, attribute, placeholder):
        self.calls.append(("single", element))
        return 0, self.attributes.get(attribute)

    def AXUIElementCopyMultipleAttributeValues(self, element, attributes, options, placeholder):
        self.calls.append(("batch", element))
        return 0, [self.attributes.get(attribute) for attribute in attributes]

    def AXUIElementIsAttributeSettable(self, element, attribute, placeholder):
        self.calls.append(("writable", element))
        return 0, True

    def AXValueGetType(self, value):
        return self.kAXValueAXErrorType if isinstance(value, AttributeErrorValue) else 0

    def AXValueGetValue(self, value, value_type, placeholder):
        return True, value.code


class NativeProbeScopeTests(unittest.TestCase):
    def setUp(self):
        self.ax = FakeAX()
        self.scoped = _ScopedAX(self.ax, os.getpid(), allow_untrusted_self=True)

    def test_explicit_self_mode_does_not_grant_native_permission(self):
        self.assertTrue(self.scoped.AXIsProcessTrusted())
        self.assertFalse(self.ax.AXIsProcessTrusted())
        self.assertFalse(_ScopedAX(self.ax, os.getpid()).AXIsProcessTrusted())

    def test_other_process_roots_and_systemwide_roots_are_refused(self):
        with self.assertRaises(ScopeViolation):
            _ScopedAX(self.ax, os.getpid() + 1, allow_untrusted_self=True)
        with self.assertRaises(ScopeViolation):
            self.scoped.AXUIElementCreateApplication(os.getpid() + 1)
        with self.assertRaises(ScopeViolation):
            self.scoped.AXUIElementCreateSystemWide()

    def test_every_target_operation_refuses_other_process_elements(self):
        calls = [
            lambda: self.scoped.AXUIElementSetMessagingTimeout("other", 0.01),
            lambda: self.scoped.AXUIElementCopyAttributeValue("other", "AXValue", None),
            lambda: self.scoped.AXUIElementCopyMultipleAttributeValues("other", ["AXValue"], 0, None),
            lambda: self.scoped.AXUIElementIsAttributeSettable("other", "AXValue", None),
        ]
        for call in calls:
            with self.assertRaises(ScopeViolation):
                call()
        self.assertEqual(self.ax.calls, [("create", os.getpid())])

    def test_permission_request_and_unreviewed_native_operations_are_refused(self):
        with self.assertRaises(ScopeViolation):
            self.scoped.AXIsProcessTrustedWithOptions({})
        with self.assertRaises(ScopeViolation):
            self.scoped.AXUIElementGetPid("owned", None)

    def test_batch_diagnostics_preserve_errors_and_false_enabled_without_text(self):
        names = ["AXRole", "AXEnabled", "AXProtectedContent", "AXValue", "AXSelectedTextRange"]
        result = self.scoped.AXUIElementCopyMultipleAttributeValues("owned", names, 0, None)
        self.assertEqual(result[0], 0)
        metadata = self.scoped.events[-1]["metadata"]
        self.assertEqual(metadata["AXEnabled"], {"kind": "boolean", "value": False})
        self.assertEqual(metadata["AXProtectedContent"], {"kind": "attribute_error", "code": -25205})
        serialized = json.dumps(self.scoped.events)
        self.assertNotIn("private document contents", serialized)
        self.assertNotIn("private selection", serialized)
        self.assertIn("elapsed_ms", serialized)

    def test_single_value_is_redacted_and_trace_is_bounded(self):
        for _ in range(260):
            self.scoped.AXUIElementCopyAttributeValue("owned", "AXValue", None)
        self.assertEqual(len(self.scoped.events), 250)
        self.assertNotIn("private document contents", json.dumps(self.scoped.events))

    def test_unknown_string_metadata_is_not_recorded_verbatim(self):
        result = _summary(self.ax, "AXSubrole", "untrusted private value")
        self.assertEqual(result, {"kind": "string", "known_value": "other"})


if __name__ == "__main__":
    unittest.main()
