"""PyObjC NSWindow dark floating overlay with waveform visualization."""

import math
import threading

import objc
from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSMakeRect,
    NSScreen,
    NSTextField,
    NSView,
    NSVisualEffectMaterial,
    NSVisualEffectView,
    NSWindow,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSObject


class WaveformView(NSView):
    """Custom view that draws animated waveform bars."""

    def initWithFrame_(self, frame):
        self = objc.super(WaveformView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._levels = [0.0] * 30  # 30 bars
        self._accent_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.31, 0.76, 0.97, 1.0  # #4FC3F7
        )
        return self

    def setLevels_(self, levels):
        self._levels = list(levels)
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height
        n = len(self._levels)
        bar_width = max(2, (w / n) * 0.6)
        gap = w / n

        for i, level in enumerate(self._levels):
            bar_h = max(2, level * h * 0.9)
            x = i * gap + (gap - bar_width) / 2
            y = (h - bar_h) / 2

            bar_rect = NSMakeRect(x, y, bar_width, bar_h)
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bar_rect, bar_width / 2, bar_width / 2
            )
            alpha = 0.5 + 0.5 * level
            self._accent_color.colorWithAlphaComponent_(alpha).set()
            path.fill()


class FloatingOverlay:
    """Manages a floating recording overlay window."""

    WIDTH = 400
    HEIGHT = 100
    BOTTOM_OFFSET = 100

    def __init__(self):
        self._window = None
        self._waveform_view = None
        self._status_label = None
        self._lock = threading.Lock()

    def _ensure_window(self):
        """Create the NSWindow if it doesn't exist (must be called on main thread)."""
        if self._window is not None:
            return

        screen = NSScreen.mainScreen()
        screen_frame = screen.frame()
        x = (screen_frame.size.width - self.WIDTH) / 2
        y = self.BOTTOM_OFFSET

        frame = NSMakeRect(x, y, self.WIDTH, self.HEIGHT)

        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        window.setLevel_(25)  # NSFloatingWindowLevel = 3, use 25 for above-all
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setHasShadow_(True)
        window.setIgnoresMouseEvents_(True)
        window.setCollectionBehavior_(1 << 0)  # canJoinAllSpaces

        # Visual effect (blur) background
        content_frame = NSMakeRect(0, 0, self.WIDTH, self.HEIGHT)
        effect_view = NSVisualEffectView.alloc().initWithFrame_(content_frame)
        effect_view.setMaterial_(NSVisualEffectMaterial.Dark if hasattr(NSVisualEffectMaterial, 'Dark') else 2)
        effect_view.setState_(1)  # active
        effect_view.setWantsLayer_(True)
        effect_view.layer().setCornerRadius_(16)
        effect_view.layer().setMasksToBounds_(True)

        # Waveform view
        waveform_frame = NSMakeRect(20, 30, self.WIDTH - 40, self.HEIGHT - 50)
        self._waveform_view = WaveformView.alloc().initWithFrame_(waveform_frame)

        # Status label
        label_frame = NSMakeRect(20, 5, self.WIDTH - 40, 25)
        self._status_label = NSTextField.alloc().initWithFrame_(label_frame)
        self._status_label.setStringValue_("Recording...")
        self._status_label.setBezeled_(False)
        self._status_label.setDrawsBackground_(False)
        self._status_label.setEditable_(False)
        self._status_label.setSelectable_(False)
        self._status_label.setTextColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.9)
        )
        self._status_label.setFont_(NSFont.systemFontOfSize_weight_(13, 0.3))
        self._status_label.setAlignment_(1)  # center

        effect_view.addSubview_(self._waveform_view)
        effect_view.addSubview_(self._status_label)
        window.contentView().addSubview_(effect_view)

        self._window = window

    def show(self, status: str = "Recording..."):
        """Show the overlay (call from any thread)."""
        def _do():
            self._ensure_window()
            self._status_label.setStringValue_(status)
            self._window.orderFront_(None)
        _run_on_main(_do)

    def hide(self):
        """Hide the overlay."""
        def _do():
            if self._window:
                self._window.orderOut_(None)
        _run_on_main(_do)

    def set_status(self, text: str):
        """Update the status label."""
        def _do():
            if self._status_label:
                self._status_label.setStringValue_(text)
        _run_on_main(_do)

    def update_levels(self, rms: float):
        """Update waveform with an RMS value (0.0–1.0+)."""
        def _do():
            if self._waveform_view is None:
                return
            # Generate animated bar levels from RMS
            import random
            base = min(rms * 5, 1.0)  # amplify for visibility
            levels = []
            for i in range(30):
                # Create a wave shape centered in the middle
                center_dist = abs(i - 15) / 15
                wave = base * (1 - center_dist * 0.5)
                jitter = random.uniform(0.8, 1.2)
                levels.append(max(0.05, min(1.0, wave * jitter)))
            self._waveform_view.setLevels_(levels)
        _run_on_main(_do)


def _run_on_main(fn):
    """Execute fn on the main thread via performSelectorOnMainThread."""
    from PyObjCTools import AppHelper
    AppHelper.callAfter(fn)
