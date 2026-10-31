"""Verify the freeze and run exactly one official test prediction and score."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone


FINAL = Path("runs/final")
MANIFEST_PATH = FINAL / "freeze_manifest.json"
START_MARKER = FINAL / "test_run_started.json"
PREDICTIONS = FINAL / "test_predictions.json"
SCORE = FINAL / "test_score.json"
RESOURCE = FINAL / "formal_resource_measurement.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze(manifest: dict) -> None:
    for item in manifest["frozen_files"]:
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Frozen file missing: {path}")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen file changed after manifest creation: {path}")
    if manifest.get("status") != "frozen_before_test":
        raise ValueError("Manifest is not in frozen-before-test state")
    if manifest["checkpoint_config"].get("pairwise_weight") != 0.1:
        raise ValueError("Manifest checkpoint is not the approved Stage-2 hybrid")


def main() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Freeze manifest must exist before test")
    forbidden_existing = [path for path in (START_MARKER, PREDICTIONS, SCORE, RESOURCE) if path.exists()]
    if forbidden_existing:
        raise FileExistsError(
            "Formal test is single-use; refusing because final test artifacts already exist: "
            + ", ".join(str(path) for path in forbidden_existing)
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    verify_freeze(manifest)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONHASHSEED": "20261022",
    })
    predict_command = [
        sys.executable, "predict.py", "--method", "submission.py",
        "--checkpoint", "runs/final/frozen_checkpoint", "--split", "test",
        "--output", str(PREDICTIONS),
    ]
    score_command = [
        sys.executable, "score_predictions.py", "--predictions", str(PREDICTIONS),
        "--split", "test", "--output", str(SCORE),
    ]
    started_payload = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "predict_command": predict_command,
        "score_command": score_command,
        "single_use_guard": True,
    }
    START_MARKER.write_text(json.dumps(started_payload, indent=2) + "\n", encoding="utf-8")

    total_started = time.perf_counter()
    prediction_started = time.perf_counter()
    prediction_process = subprocess.run(
        predict_command, check=False, capture_output=True, text=True, env=env
    )
    prediction_wall = time.perf_counter() - prediction_started
    if prediction_process.returncode != 0:
        failure = {
            **started_payload,
            "stage": "prediction_failed",
            "returncode": prediction_process.returncode,
            "stdout": prediction_process.stdout,
            "stderr": prediction_process.stderr,
        }
        (FINAL / "test_run_failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("The one formal prediction process failed; no retry was attempted")

    scoring_started = time.perf_counter()
    scoring_process = subprocess.run(
        score_command, check=False, capture_output=True, text=True, env=env
    )
    scoring_wall = time.perf_counter() - scoring_started
    total_wall = time.perf_counter() - total_started
    if scoring_process.returncode != 0:
        failure = {
            **started_payload,
            "stage": "scoring_failed",
            "returncode": scoring_process.returncode,
            "stdout": scoring_process.stdout,
            "stderr": scoring_process.stderr,
            "prediction_wall_seconds": prediction_wall,
            "prediction_sha256": sha256(PREDICTIONS),
        }
        (FINAL / "test_run_failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("The one formal scoring process failed; no retry was attempted")

    prediction_payload = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    score_payload = json.loads(SCORE.read_text(encoding="utf-8"))
    resource = {
        "status": "formal_test_complete_no_further_tuning",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "frozen_checkpoint": manifest["frozen_checkpoint"],
        "test_questions": score_payload["n"],
        "prediction_process_wall_seconds": prediction_wall,
        "scoring_process_wall_seconds": scoring_wall,
        "complete_prediction_plus_scoring_wall_seconds": total_wall,
        "prediction_internal_seconds": prediction_payload.get("prediction_seconds"),
        "scoring_internal_seconds": score_payload.get("scoring_seconds"),
        "thread_limit": 4,
        "thread_environment": {
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONHASHSEED": "20261022",
        },
        "local_hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "inference_asset_bytes": manifest["resource_limits"]["inference_asset_bytes"],
        "asset_limit_bytes": manifest["resource_limits"]["max_inference_asset_bytes"],
        "asset_limit_passed": manifest["resource_limits"]["asset_limit_passed"],
        "time_limit_seconds": manifest["resource_limits"]["max_complete_cpu_seconds"],
        "local_time_limit_passed": total_wall <= manifest["resource_limits"]["max_complete_cpu_seconds"],
        "prediction_sha256": sha256(PREDICTIONS),
        "score_sha256": sha256(SCORE),
        "predict_stdout": prediction_process.stdout.strip(),
        "predict_stderr": prediction_process.stderr.strip(),
        "score_stdout": scoring_process.stdout.strip(),
        "score_stderr": scoring_process.stderr.strip(),
        "official_score": score_payload,
        "stage1_test_run": False,
        "post_test_training_or_tuning": False,
    }
    RESOURCE.write_text(json.dumps(resource, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resource, indent=2))


if __name__ == "__main__":
    main()
