"""Reproducible local correction-learning benchmark using synthetic text only.

Run from the project root:
    .venv/bin/python tools/benchmark_personalization.py
    .venv/bin/python tools/benchmark_personalization.py --max-case

All history is created in a temporary directory and removed afterward. No audio,
microphone, real dictation history, network, or Parakeet inference is used. These
results measure the correction mechanism, not accuracy on a real speaker.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personalization import MAX_RECORDS, PersonalizationManager


def benchmark(iterations=2000, maximum=False):
    with tempfile.TemporaryDirectory(prefix="justdictate-learning-benchmark-") as directory:
        manager = PersonalizationManager(Path(directory) / "learning", enabled=True)
        count = MAX_RECORDS if maximum else 24
        for number in range(count):
            if maximum:
                group = number % 500
                raw = (
                    f"Please ask Name{group} tomorrow about task {number}. "
                    "We should check the release notes and review the changes together "
                    "before we share the final plan with the team later this week."
                )
                corrected = raw.replace(f"Name{group}", f"Person{group}")
            else:
                raw = f"Please ask Gotham tomorrow about task {number}."
                corrected = raw.replace("Gotham", "Gowtham")
            record_id = manager.record(raw, raw)
            manager.review(record_id, corrected)

        start = time.perf_counter()
        training = manager.train_if_due(force=True)
        training_ms = (time.perf_counter() - start) * 1000
        raw = ("Please ask Name0 tomorrow about the release." if maximum else
               "Please ask Gotham tomorrow about the release.")
        unrelated = ("Please visit Name0 tomorrow." if maximum else "Please visit Gotham tomorrow.")
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            output = manager.apply(raw)
            times.append((time.perf_counter_ns() - start) / 1000)
        times.sort()
        return {
            "scenario": "maximum_retained_history" if maximum else "small_personal_vocabulary",
            "synthetic_only": True,
            "reviewed_records": count,
            "training_status": training["status"],
            "training_ms": round(training_ms, 3),
            "rule_count": training.get("rule_count", 0),
            "metrics": training.get("metrics"),
            "apply_iterations": iterations,
            "apply_p50_us": round(statistics.median(times), 3),
            "apply_p95_us": round(times[int((len(times) - 1) * 0.95)], 3),
            "personalized_output": output,
            "unrelated_context_unchanged": manager.apply(unrelated) == unrelated,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--max-case", action="store_true", help="Also train on the full 2,000-record history limit")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    results = [benchmark(args.iterations)]
    if args.max_case:
        results.append(benchmark(args.iterations, maximum=True))
    print(json.dumps(results, indent=2))
    return 0 if all(result["training_status"] == "trained" and result["unrelated_context_unchanged"]
                    for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
