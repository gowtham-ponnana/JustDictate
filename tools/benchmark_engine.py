"""Measure completion-feedback latency without using the microphone or clipboard.

Run: .venv/bin/python tools/benchmark_engine.py --iterations 10

The baseline is loaded from git (HEAD by default). Both implementations retain
their real sleep calls; inference, Quartz, and clipboard operations are doubles.
This isolates scheduling overhead, not model inference or application paste time.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config_manager
import dictation_engine


def measure(module, iterations):
    elapsed = []
    count = 0

    def clipboard_set(text):
        nonlocal count
        count += 1
        return count

    patches = [
        patch.object(module, "_clipboard_get", return_value="previous clipboard"),
        patch.object(module, "_clipboard_set", side_effect=clipboard_set),
    ]
    if hasattr(module, "_clipboard_snapshot"):
        patches.extend([
            patch.object(module, "_clipboard_snapshot", return_value=[]),
            patch.object(module, "_clipboard_restore", return_value=True),
        ])
    for replacement in patches:
        replacement.start()
    try:
        engine = module.DictationEngine(transcribe_fn=lambda audio: "Benchmark words")
        for index in range(iterations + 2):
            completed = []
            engine.on_transcription_done = lambda text: completed.append(time.perf_counter())
            chunks = [np.zeros(6400, dtype=np.float32)]
            started = time.perf_counter()
            if hasattr(engine, "_generation"):
                engine._processing = True
                engine._transcribe_and_type(chunks, engine._generation)
            else:
                engine._audio_chunks = chunks
                engine._transcribe_and_type()
            if not completed:
                raise RuntimeError("Transcription did not signal completion")
            if index >= 2:
                elapsed.append((completed[0] - started) * 1000)
            # This is deliberately outside the completion timing. Let every
            # deferred restore finish before the next independent trial.
            timer = getattr(engine, "_restore_timer", None)
            if timer is not None:
                timer.join(timeout=1)
                if timer.is_alive():
                    raise RuntimeError("Clipboard restore timer did not finish")
        engine.stop()
    finally:
        for replacement in reversed(patches):
            replacement.stop()
    return {"median_ms": round(statistics.median(elapsed), 3),
            "p95_ms": round(float(np.quantile(elapsed, 0.95)), 3),
            "samples_ms": [round(value, 3) for value in elapsed]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline_ref}:dictation_engine.py"], cwd=ROOT, text=True
    )
    baseline = types.ModuleType("baseline_dictation_engine")
    exec(compile(source, "baseline_dictation_engine.py", "exec"), baseline.__dict__)
    cfg = {"hotkey": "right_cmd", "add_trailing_space": True}
    with patch.object(config_manager, "load", return_value=cfg), patch.dict(sys.modules, {"Quartz": Mock()}):
        before = measure(baseline, args.iterations)
        after = measure(dictation_engine, args.iterations)
    print(json.dumps({
        "scope": "Completion feedback with fake inference/clipboard/input; real sleep calls",
        "baseline_ref": args.baseline_ref,
        "iterations": args.iterations,
        "before": before,
        "after": after,
        "median_saved_ms": round(before["median_ms"] - after["median_ms"], 3),
    }, indent=2))


if __name__ == "__main__":
    main()
