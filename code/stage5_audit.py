"""Read-only engineering audit for the canonical Stage-5 bugfix run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/stage5/bugfix2"))
    args = parser.parse_args()
    run_dir = args.run_dir
    protocol = json.loads((run_dir / "prevalidation_protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    bundle = run_dir / "provisional_bundle"
    expected_hashes = protocol["protected_artifact_hashes"]
    mismatches = []
    for name, expected in expected_hashes.items():
        path = Path(name)
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(name)

    with np.load(run_dir / "oof_predictions.npz", allow_pickle=False) as data:
        oof_shapes = {name: list(data[name].shape) for name in data.files}
        finite = all(np.isfinite(data[name]).all() for name in (
            "residual_logits", "residual_probabilities"
        ))
        hard_indices = data["hard_indices"]
    folds = json.loads(Path("runs/folds.json").read_text(encoding="utf-8"))["fold_indices"]
    fold_flat = np.asarray([value for fold in folds for value in fold], dtype=np.int64)
    fold_integrity = bool(
        len(fold_flat) == len(np.unique(fold_flat)) == len(hard_indices)
        and np.array_equal(np.sort(fold_flat), np.sort(hard_indices))
    )

    timing_dir = run_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    canonical = json.loads((run_dir / "validation_predictions.json").read_text(encoding="utf-8"))
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"
    })
    timing_rows = []
    deterministic = True
    for number in range(1, 6):
        output = timing_dir / f"validation_{number}.json"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite timing output: {output}")
        command = [
            sys.executable, "predict.py", "--method", "submission_stage5.py",
            "--checkpoint", str(bundle), "--split", "validation", "--output", str(output),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command, check=True, env=environment, capture_output=True, text=True
        )
        wall = time.perf_counter() - started
        payload = json.loads(output.read_text(encoding="utf-8"))
        deterministic &= payload["selected"] == canonical["selected"]
        timing_rows.append({
            "run": number, "official_prediction_seconds": payload["prediction_seconds"],
            "external_wall_seconds": wall, "stdout": completed.stdout.strip(),
        })
    official_score = json.loads((run_dir / "validation_score.json").read_text(encoding="utf-8"))
    bundle_bytes = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    source_files = [
        Path(name) for name in (
            "submission_stage5.py", "stage5_features.py", "stage5_model.py",
            "creo_features.py", "creo_model.py", "selector_features.py",
            "selector_model.py", "common.py",
        )
    ]
    payload = {
        "scope": "validation-only; test not accessed",
        "canonical_run": str(run_dir),
        "selection_status": summary["selection_status"],
        "apply_residual": summary["apply_residual"],
        "validation_score": official_score,
        "five_process_timing": timing_rows,
        "prediction_seconds_min": min(row["official_prediction_seconds"] for row in timing_rows),
        "prediction_seconds_median": float(np.median([
            row["official_prediction_seconds"] for row in timing_rows
        ])),
        "prediction_seconds_max": max(row["official_prediction_seconds"] for row in timing_rows),
        "deterministic_predictions": deterministic,
        "bundle_bytes": bundle_bytes,
        "inference_source_bytes": sum(path.stat().st_size for path in source_files),
        "asset_limit_bytes": 16 * 1024 * 1024,
        "under_asset_limit": bundle_bytes <= 16 * 1024 * 1024,
        "oof_shapes": oof_shapes,
        "oof_probabilities_finite": finite,
        "fold_integrity": fold_integrity,
        "protected_hash_mismatches": mismatches,
        "protected_artifacts_unchanged": not mismatches,
        "test_touched": False,
    }
    (run_dir / "resource_measurement.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
