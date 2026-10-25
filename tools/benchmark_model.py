"""Reproducible local synthetic-speech comparison; never opens a microphone.

Run: .venv/bin/python tools/benchmark_model.py --rounds 3 --baseline-ref HEAD
Uses the already-cached model offline. Results are synthetic regression evidence,
not a real-speaker accuracy benchmark. No audio or transcripts leave the Mac.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import types
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_manager import ModelManager, _generate_greedy_text  # noqa: E402


def read_wave(path):
    with wave.open(str(path)) as source:
        assert source.getframerate() == 16000 and source.getsampwidth() == 2
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768


def signature(result):
    return [(token.id, token.text, token.start, token.duration) for token in result.tokens]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--baseline-ref", default="HEAD")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    baseline_code = subprocess.run(
        ["git", "show", f"{args.baseline_ref}:model_manager.py"],
        cwd=ROOT, capture_output=True, check=True, text=True,
    ).stdout
    baseline_module = types.ModuleType("baseline_model_manager")
    exec(compile(baseline_code, "baseline_model_manager.py", "exec"), baseline_module.__dict__)
    current = ModelManager()
    current.load()
    baseline = baseline_module.ModelManager()
    baseline._model, baseline._dtype = current._model, current._dtype

    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel

    sentences = {
        "short": "Please send the final report before tomorrow morning.",
        "numbers": "The meeting starts at three thirty. Please bring twenty five copies, not fifteen.",
        "technical": "The database connection timed out. Restart the background worker, then check the authentication token.",
    }
    report = {
        "platform": platform.platform(),
        "parakeet_mlx": importlib.metadata.version("parakeet-mlx"),
        "mlx": importlib.metadata.version("mlx"),
        "baseline_ref": args.baseline_ref,
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip(),
        "fast_decode_enabled": current._fast_decode,
        "rounds": args.rounds,
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="justdictate-benchmark-") as tmp:
        clips = {}
        for name, sentence in sentences.items():
            path = Path(tmp) / f"{name}.wav"
            subprocess.run(
                ["/usr/bin/say", "-o", str(path), "--data-format=LEI16@16000", sentence],
                check=True,
            )
            clips[name] = read_wave(path)
        clips["silence"] = np.zeros(4 * 16000, dtype=np.float32)
        speech = np.concatenate([clips["technical"], np.zeros(8000, dtype=np.float32)])
        clips["long_overlap_tail"] = np.tile(speech, int(np.ceil(114 * 16000 / len(speech))))[:114 * 16000]

        for name, audio in clips.items():
            # Compare each same-input greedy result independently of the fixed
            # chunk loop; token identity and exact timing must be unchanged.
            chunk_signatures_match = True
            for start in range(0, len(audio), 55 * 16000):
                segment = audio[start : start + 60 * 16000]
                if len(segment) < current._model.preprocessor_config.hop_length:
                    continue
                mel = get_logmel(mx.array(segment), current._model.preprocessor_config)
                old_result = current._model.generate(mel)[0]
                new_result = _generate_greedy_text(current._model, mel)
                chunk_signatures_match &= signature(old_result) == signature(new_result)
                if start + len(segment) == len(audio):
                    break
            if not chunk_signatures_match:
                raise AssertionError(f"Greedy token/timestamp mismatch for {name}")

            # Warm both paths, then alternate execution order to reduce bias.
            baseline.transcribe(audio)
            current.transcribe(audio)
            timings = {"baseline": [], "optimized": []}
            outputs = {}
            for trial in range(args.rounds):
                paths = [("baseline", baseline), ("optimized", current)]
                if trial % 2:
                    paths.reverse()
                for label, manager in paths:
                    t0 = time.perf_counter()
                    outputs[label] = manager.transcribe(audio)
                    mx.synchronize()
                    timings[label].append(time.perf_counter() - t0)
            old_median = statistics.median(timings["baseline"])
            new_median = statistics.median(timings["optimized"])
            if outputs["baseline"] != outputs["optimized"]:
                raise AssertionError(f"End-to-end text mismatch for {name}")
            report["cases"].append({
                "case": name,
                "audio_seconds": len(audio) / 16000,
                "token_ids_and_timestamps_match": chunk_signatures_match,
                "end_to_end_text_matches": outputs["baseline"] == outputs["optimized"],
                "baseline_median_seconds": round(old_median, 6),
                "optimized_median_seconds": round(new_median, 6),
                "speedup": round(old_median / new_median, 3),
                "baseline_seconds": timings["baseline"],
                "optimized_seconds": timings["optimized"],
                "baseline_text": outputs["baseline"],
                "optimized_text": outputs["optimized"],
            })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
