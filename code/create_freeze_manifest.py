"""Create the immutable final Stage-2 checkpoint copy and freeze manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from datetime import datetime, timezone

import numpy as np


SEED = 20261022
FINAL = Path("runs/final")
SOURCE_CHECKPOINT = Path("runs/stage2/provisional_model_bundle")
FROZEN_CHECKPOINT = FINAL / "frozen_checkpoint"
MANIFEST = FINAL / "freeze_manifest.json"
MANIFEST_HASH = FINAL / "freeze_manifest.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, category: str) -> dict:
    return {
        "path": path.as_posix(),
        "category": category,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() or MANIFEST_HASH.exists() or FROZEN_CHECKPOINT.exists():
        raise FileExistsError("Freeze outputs already exist; refusing to overwrite an existing freeze")

    required_checkpoint_files = ["config.json", "model.npz", "loss_history.json"]
    for name in required_checkpoint_files:
        source = SOURCE_CHECKPOINT / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing Stage-2 checkpoint file: {source}")
    config = json.loads((SOURCE_CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    expected = {
        "variant": "hand_centered",
        "loss_kind": "listwise",
        "pairwise_weight": 0.1,
        "weight_scheme": "uniform",
        "decision_rule": "candidate",
        "hand_l2": 0.01,
        "embedding_l2": 1.0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Stage-2 checkpoint mismatch for {key}: {config.get(key)!r} != {value!r}")

    FROZEN_CHECKPOINT.mkdir(parents=True, exist_ok=False)
    for name in required_checkpoint_files:
        shutil.copy2(SOURCE_CHECKPOINT / name, FROZEN_CHECKPOINT / name)
        if sha256(SOURCE_CHECKPOINT / name) != sha256(FROZEN_CHECKPOINT / name):
            raise IOError(f"Frozen checkpoint copy mismatch: {name}")

    python_files = sorted(Path(".").glob("*.py")) + sorted(Path("tests").rglob("*.py"))
    code_files = python_files + [Path("requirements.txt")]
    frozen_records = [record(path, "code") for path in code_files if path.is_file()]
    frozen_records += [
        record(FROZEN_CHECKPOINT / name, "inference_asset")
        for name in required_checkpoint_files
    ]
    evidence_paths = [
        Path("runs/stage2/winner_pre_validation.json"),
        Path("runs/stage2/experiment_summary.json"),
        Path("runs/stage2/validation_predictions.jsonl"),
        Path("runs/stage2/validation_score.json"),
        FINAL / "paired_audit.json",
        FINAL / "paired_audit.csv",
        FINAL / "PAIRED_AUDIT.md",
    ]
    evidence_records = [record(path, "evidence") for path in evidence_paths]
    inference_asset_bytes = sum(
        item["bytes"] for item in frozen_records if item["category"] == "inference_asset"
    )
    manifest = {
        "protocol": "dase7506-mp2-open-hard-verifier-v5",
        "status": "frozen_before_test",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "selected_candidate": "Stage-2 hand+relative+centered linear scorer with listwise + 0.1 pairwise loss",
        "selection_evidence": {
            "train_hard_cv_mean_percent": 81.64498672313536,
            "train_hard_cv_se_percent": 1.0217200749367124,
            "fold_accuracy_percent": [
                82.74231678486997,
                79.19621749408984,
                83.68794326241135,
                83.451536643026,
                79.14691943127961,
            ],
            "stage1_cv_gain_points": 0.993131883522139,
            "improved_folds": 5,
        },
        "checkpoint_config": config,
        "frozen_checkpoint": FROZEN_CHECKPOINT.as_posix(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment_required": {
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "PYTHONHASHSEED": str(SEED),
            },
        },
        "resource_limits": {
            "max_inference_asset_bytes": 16 * 1024 * 1024,
            "max_complete_cpu_seconds": 30,
            "thread_limit": 4,
            "inference_asset_bytes": inference_asset_bytes,
            "asset_limit_passed": inference_asset_bytes <= 16 * 1024 * 1024,
        },
        "frozen_files": frozen_records,
        "evidence_files": evidence_records,
        "official_test_commands": [
            ".venv/Scripts/python.exe predict.py --method submission.py --checkpoint runs/final/frozen_checkpoint --split test --output runs/final/test_predictions.json",
            ".venv/Scripts/python.exe score_predictions.py --predictions runs/final/test_predictions.json --split test --output runs/final/test_score.json",
        ],
        "constraints": {
            "test_models": ["Stage-2 hybrid only"],
            "stage1_test_forbidden": True,
            "post_test_training_or_tuning_forbidden": True,
            "online_submission_performed": False,
            "github_issue_created": False,
            "final_report_created": False,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    MANIFEST_HASH.write_text(f"{sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "frozen_checkpoint": str(FROZEN_CHECKPOINT),
        "inference_asset_bytes": inference_asset_bytes,
        "frozen_file_count": len(frozen_records),
    }, indent=2))


if __name__ == "__main__":
    main()
