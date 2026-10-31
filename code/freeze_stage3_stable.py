"""Create the immutable Stage 3 stable-direct freeze and SHA-256 manifest."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


SOURCE_BUNDLE = Path("runs/stage3/provisional_creo_stable_bundle")
FINAL_ROOT = Path("runs/stage3/final_stable")
FROZEN_CHECKPOINT = FINAL_ROOT / "frozen_checkpoint"
FROZEN_CODE = FINAL_ROOT / "frozen_code"
CODE_FILES = (
    "submission_stage3.py",
    "creo_features.py",
    "creo_model.py",
    "selector_features.py",
    "selector_model.py",
    "common.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if FINAL_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze: {FINAL_ROOT}")
    config = json.loads((SOURCE_BUNDLE / "config.json").read_text(encoding="utf-8"))
    selected = config.get("selected_direct_gate", {})
    expected = {
        "challenger_source": "stage2",
        "gate_l2": 0.01,
        "class_balance": False,
        "threshold": 0.4,
        "reliability_l2": 0.01,
    }
    for key, value in expected.items():
        if selected.get(key) != value:
            raise ValueError(f"Stable-direct config mismatch for {key}")

    FINAL_ROOT.mkdir(parents=True)
    shutil.copytree(SOURCE_BUNDLE, FROZEN_CHECKPOINT)
    FROZEN_CODE.mkdir()
    for name in CODE_FILES:
        shutil.copy2(Path(name), FROZEN_CODE / name)

    selection_record = {
        "stage": "Stage 3 CREO stable-direct",
        "seed": 20261022,
        "selection_locked_before_test": True,
        "challenger_source": "Stage 2 frozen comparator",
        "train_hard_oof": {
            "correct": 1736,
            "n": 2114,
            "accuracy": 82.11920529801324,
            "delta_correct_vs_stage2": 10,
            "fold_deltas_vs_stage2": [1, 5, 5, 1, -2],
        },
        "validation": {
            "correct": 748,
            "n": 798,
            "accuracy": 93.734335839599,
            "hard_correct": 305,
            "hard_n": 355,
        },
        "test_status_at_freeze": "not_run",
        "reason": (
            "Chosen before Stage 3 test because it improved train-hard OOF by 10 "
            "questions and was non-decreasing in four of five grouped folds."
        ),
    }
    (FINAL_ROOT / "selection_record.json").write_text(
        json.dumps(selection_record, indent=2) + "\n", encoding="utf-8"
    )

    frozen_files = sorted(
        [path for path in FROZEN_CHECKPOINT.rglob("*") if path.is_file()]
        + [path for path in FROZEN_CODE.rglob("*") if path.is_file()]
        + [FINAL_ROOT / "selection_record.json"]
    )
    entries = [
        {
            "path": path.relative_to(FINAL_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in frozen_files
    ]
    checkpoint_bytes = sum(
        path.stat().st_size for path in FROZEN_CHECKPOINT.rglob("*") if path.is_file()
    )
    manifest = {
        "format": "dase7506-mp2-stage3-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "CREO stable-direct",
        "seed": 20261022,
        "test_status_at_manifest_creation": "not_run",
        "method_entrypoint": "frozen_code/submission_stage3.py",
        "checkpoint": "frozen_checkpoint",
        "checkpoint_uncompressed_bytes": checkpoint_bytes,
        "asset_limit_bytes": 16 * 1024 * 1024,
        "asset_limit_pass": checkpoint_bytes <= 16 * 1024 * 1024,
        "frozen_file_count": len(entries),
        "frozen_total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    (FINAL_ROOT / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
