"""Run the single frozen Stage 3 official test prediction and scoring process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


FINAL = Path("runs/stage3/final_stable")
PREDICTIONS = FINAL / "test_predictions.json"
SCORE = FINAL / "test_score.json"
RESOURCE = FINAL / "test_resource_measurement.json"


def main() -> None:
    for path in (PREDICTIONS, SCORE, RESOURCE):
        if path.exists():
            raise FileExistsError(f"Refusing to repeat or overwrite formal test artifact: {path}")
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
    })
    prediction_command = [
        sys.executable,
        "predict.py",
        "--method", str(FINAL / "frozen_code" / "submission_stage3.py"),
        "--checkpoint", str(FINAL / "frozen_checkpoint"),
        "--split", "test",
        "--output", str(PREDICTIONS),
    ]
    scoring_command = [
        sys.executable,
        "score_predictions.py",
        "--predictions", str(PREDICTIONS),
        "--split", "test",
        "--output", str(SCORE),
    ]
    total_started = time.perf_counter()
    prediction_started = time.perf_counter()
    prediction = subprocess.run(
        prediction_command, env=env, check=True, capture_output=True, text=True
    )
    prediction_wall = time.perf_counter() - prediction_started
    scoring_started = time.perf_counter()
    scoring = subprocess.run(
        scoring_command, env=env, check=True, capture_output=True, text=True
    )
    scoring_wall = time.perf_counter() - scoring_started
    total_wall = time.perf_counter() - total_started
    score_payload = json.loads(SCORE.read_text(encoding="utf-8"))
    checkpoint_bytes = sum(
        path.stat().st_size
        for path in (FINAL / "frozen_checkpoint").rglob("*")
        if path.is_file()
    )
    payload = {
        "formal_test_runs": 1,
        "method": "CREO stable-direct frozen before test",
        "threads": 4,
        "prediction_external_wall_seconds": prediction_wall,
        "scoring_external_wall_seconds": scoring_wall,
        "prediction_plus_scoring_external_wall_seconds": total_wall,
        "official_prediction_stdout": prediction.stdout.strip(),
        "official_scoring_stdout": scoring.stdout.strip(),
        "checkpoint_uncompressed_bytes": checkpoint_bytes,
        "asset_limit_bytes": 16 * 1024 * 1024,
        "asset_limit_pass": checkpoint_bytes <= 16 * 1024 * 1024,
        "time_limit_seconds": 30.0,
        "local_time_limit_pass": total_wall <= 30.0,
        "test_correct": score_payload["correct"],
        "test_n": score_payload["n"],
        "test_accuracy": score_payload["accuracy"],
        "post_test_model_changes_allowed": False,
    }
    RESOURCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
