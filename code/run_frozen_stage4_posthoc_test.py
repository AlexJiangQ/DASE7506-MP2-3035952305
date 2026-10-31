"""Run exactly one official test prediction/scoring for the frozen Stage 4 model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


FINAL = Path("runs/stage4/final_posthoc")
PREDICTIONS = FINAL / "test_predictions.json"
SCORE = FINAL / "test_score.json"
RESOURCE = FINAL / "test_resource_measurement.json"


def main() -> None:
    for path in (PREDICTIONS, SCORE, RESOURCE):
        if path.exists():
            raise FileExistsError(f"Refusing to repeat or overwrite Stage 4 test: {path}")
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
    })
    predict_command = [
        sys.executable, "predict.py",
        "--method", str(FINAL / "frozen_code" / "submission_stage4.py"),
        "--checkpoint", str(FINAL / "frozen_checkpoint"),
        "--split", "test", "--output", str(PREDICTIONS),
    ]
    score_command = [
        sys.executable, "score_predictions.py",
        "--predictions", str(PREDICTIONS),
        "--split", "test", "--output", str(SCORE),
    ]
    total_started = time.perf_counter()
    prediction_started = time.perf_counter()
    prediction = subprocess.run(
        predict_command, env=environment, check=True, capture_output=True, text=True
    )
    prediction_wall = time.perf_counter() - prediction_started
    scoring_started = time.perf_counter()
    scoring = subprocess.run(
        score_command, env=environment, check=True, capture_output=True, text=True
    )
    scoring_wall = time.perf_counter() - scoring_started
    total_wall = time.perf_counter() - total_started
    result = json.loads(SCORE.read_text(encoding="utf-8"))
    checkpoint_bytes = sum(
        path.stat().st_size
        for path in (FINAL / "frozen_checkpoint").rglob("*") if path.is_file()
    )
    payload = {
        "formal_stage4_test_runs": 1,
        "posthoc_nonblind": True,
        "replacement_pass": False,
        "method": "CREO-3W fixed_reliability",
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
        "test_correct": result["correct"], "test_n": result["n"],
        "test_accuracy": result["accuracy"],
        "post_test_training_tuning_or_switching_allowed": False,
    }
    RESOURCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
