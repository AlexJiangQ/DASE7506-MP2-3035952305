"""Pre-test engineering audit for the frozen stage-2 checkpoint."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

import submission


STAGE2 = Path("runs/stage2")
CHECKPOINT = STAGE2 / "provisional_model_bundle"


def synthetic_bank(n: int = 7) -> list[dict]:
    rows = []
    templates = [
        "1 + 1 = 2. Final Answer: \\boxed{2}",
        "We compute carefully. #### 2",
        "1 + 1 = 3. Final Answer: \\boxed{3}",
        "No numeric answer is available.",
    ]
    for i in range(n):
        rows.append({
            "id": f"arbitrary-stage2-id::{i * 101 + 9}",
            "question": f"What is 1 + 1? Wording variant {i}",
            "candidates": [
                {"index": j, "text": text + (" detail" * i)}
                for j, text in enumerate(templates)
            ],
        })
    return rows


def main() -> None:
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    timing_dir = STAGE2 / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    selected_payloads = []
    for run_no in range(1, 6):
        prediction_path = timing_dir / f"audit_predictions_{run_no}.json"
        started = time.perf_counter()
        prediction_process = subprocess.run(
            [
                sys.executable, "synthetic_audit_worker.py",
                "--checkpoint", str(CHECKPOINT),
                "--output", str(prediction_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        elapsed = time.perf_counter() - started
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        selected_payloads.append(prediction["selected"])
        runs.append({
            "run": run_no,
            "fresh_process_total_seconds": elapsed,
            "prediction_seconds": prediction["prediction_seconds"],
            "scoring_seconds": prediction["scoring_seconds"],
            "accuracy": prediction["accuracy"],
            "selected_sha256": hashlib.sha256(
                json.dumps(prediction["selected"], separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "predict_stdout": prediction_process.stdout.strip(),
        })

    rng = np.random.default_rng(20261022)
    bank = synthetic_bank()
    embeddings = rng.normal(size=(len(bank), 4, 1024)).astype(np.float32)
    choices_a = submission.predict(bank, {"embeddings": embeddings}, CHECKPOINT)
    choices_b = submission.predict(bank, {"embeddings": embeddings}, CHECKPOINT)
    if choices_a != choices_b:
        raise AssertionError("Frozen checkpoint is not deterministic on synthetic bank")
    if not all(type(choice) is int and 0 <= choice < 4 for choice in choices_a):
        raise AssertionError("Submission must return Python ints in range")

    permutation = np.asarray([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    permuted_bank = copy.deepcopy(bank)
    for row in permuted_bank:
        row["candidates"] = [row["candidates"][int(j)] for j in permutation]
        for j, candidate in enumerate(row["candidates"]):
            candidate["index"] = j
    permuted = submission.predict(
        permuted_bank, {"embeddings": embeddings[:, permutation]}, CHECKPOINT
    )
    if permuted != [int(inverse[choice]) for choice in choices_a]:
        raise AssertionError("Frozen checkpoint is not candidate-permutation equivariant")

    empty_bank = synthetic_bank()
    for row in empty_bank:
        for candidate in row["candidates"]:
            candidate["text"] = ""
    empty_choices = submission.predict(
        empty_bank,
        {"embeddings": np.zeros((len(empty_bank), 4, 1024), dtype=np.float32)},
        CHECKPOINT,
    )
    if not all(type(choice) is int and 0 <= choice < 4 for choice in empty_choices):
        raise AssertionError("All-unparseable fallback failed")

    checkpoint_files = [path for path in CHECKPOINT.rglob("*") if path.is_file()]
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoint_files)
    inference_sources = [Path("submission.py"), Path("selector_features.py"), Path("selector_model.py")]
    inference_asset_bytes = checkpoint_bytes + sum(path.stat().st_size for path in inference_sources)
    output = {
        "thread_limit": 4,
        "fresh_process_runs": runs,
        "mean_total_seconds": statistics.mean(row["fresh_process_total_seconds"] for row in runs),
        "max_total_seconds": max(row["fresh_process_total_seconds"] for row in runs),
        "audit_predictions_identical": all(payload == selected_payloads[0] for payload in selected_payloads),
        "synthetic_predictions_identical": choices_a == choices_b,
        "candidate_permutation_equivariance": True,
        "arbitrary_bank_length_and_ids": True,
        "empty_and_unparseable_safe": True,
        "python_int_outputs": True,
        "checkpoint_bytes": checkpoint_bytes,
        "inference_asset_bytes_including_python_sources": inference_asset_bytes,
        "asset_limit_bytes": 16 * 1024 * 1024,
        "asset_limit_passed": inference_asset_bytes <= 16 * 1024 * 1024,
        "test_touched": False,
    }
    (STAGE2 / "engineering_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
