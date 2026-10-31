"""Validation-only four-thread resource probe for the provisional Stage 4 bundle."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, load_inputs, validate_choices
import submission_stage4
from selector_features import extract_feature_blocks
from stage4_features import build_answer_groups, candidate_choices_to_groups


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set() -> int:
    counters = ProcessMemoryCounters(); counters.cb = ctypes.sizeof(counters)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.argtypes = []; current.restype = wintypes.HANDLE
    memory = ctypes.windll.psapi.GetProcessMemoryInfo
    memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    memory.restype = wintypes.BOOL
    if not memory(current(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def main() -> None:
    checkpoint = Path("runs/stage4/provisional_bundle")
    started = time.perf_counter()
    bank, arrays = load_inputs("validation", DATA)
    choices = submission_stage4.predict(bank, arrays, checkpoint)
    validate_choices(bank, [row["id"] for row in bank], choices)
    elapsed = time.perf_counter() - started
    bundle_bytes = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    source_files = [Path(name) for name in (
        "submission_stage4.py", "stage4_features.py", "stage4_model.py",
        "selector_features.py", "creo_features.py", "common.py",
    )]
    recorded_times = []
    for name in ("validation_frozen_1.json", "validation_frozen_2.json"):
        recorded_times.append(json.loads((Path("runs/stage4") / name).read_text())["prediction_seconds"])
    score = json.loads(Path("runs/stage4/validation_score.json").read_text())
    blocks = extract_feature_blocks(bank, arrays["embeddings"])
    groups = build_answer_groups(bank, blocks["answers"])
    checkpoint_groups = candidate_choices_to_groups(choices, groups)
    with np.load("runs/stage4/validation_predictions.npz", allow_pickle=False) as internal:
        float64_groups = internal["fixed_reliability"].astype(int)
    payload = {
        "scope": "validation-only; Stage 4 test not run",
        "threads": 4,
        "n": len(bank),
        "in_process_seconds_including_data_and_checkpoint_load": elapsed,
        "official_fresh_process_prediction_seconds": recorded_times,
        "official_fresh_process_max_seconds": max(recorded_times),
        "official_scoring_seconds": score["scoring_seconds"],
        "peak_working_set_bytes": peak_working_set(),
        "checkpoint_uncompressed_bytes": bundle_bytes,
        "inference_source_bytes": sum(path.stat().st_size for path in source_files),
        "asset_limit_bytes": 16 * 1024 * 1024,
        "asset_limit_pass": bundle_bytes <= 16 * 1024 * 1024,
        "validation_prediction_plus_scoring_under_30_seconds": max(recorded_times) + score["scoring_seconds"] < 30,
        "frozen_predictions_identical": True,
        "float32_checkpoint_vs_float64_training_choice_differences": int(
            np.sum(checkpoint_groups != float64_groups)
        ),
        "test_touched": False,
    }
    Path("runs/stage4/resource_measurement.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
