"""Validation-only resource probe for a CREO checkpoint on Windows."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time

from common import DATA, load_inputs, validate_choices
import submission_stage3


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set() -> int:
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD
    ]
    get_process_memory_info.restype = wintypes.BOOL
    process = get_current_process()
    ok = get_process_memory_info(
        process, ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    bank, arrays = load_inputs("validation", DATA)
    choices = submission_stage3.predict(bank, arrays, args.checkpoint)
    validate_choices(bank, [row["id"] for row in bank], choices)
    elapsed = time.perf_counter() - started
    bundle_bytes = sum(
        path.stat().st_size for path in args.checkpoint.rglob("*") if path.is_file()
    )
    code_paths = [
        Path(name) for name in (
            "submission_stage3.py", "creo_features.py", "creo_model.py",
            "selector_features.py", "selector_model.py", "common.py",
        )
    ]
    payload = {
        "scope": "validation-only resource probe",
        "checkpoint": str(args.checkpoint),
        "n": len(bank),
        "prediction_seconds_including_load": elapsed,
        "peak_working_set_bytes": peak_working_set(),
        "bundle_bytes": bundle_bytes,
        "inference_code_bytes": sum(path.stat().st_size for path in code_paths),
        "test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
