"""Freeze the precommitted Stage 4 research leader for one post-hoc test."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


SOURCE = Path("runs/stage4/provisional_bundle")
FINAL = Path("runs/stage4/final_posthoc")
CHECKPOINT = FINAL / "frozen_checkpoint"
CODE = FINAL / "frozen_code"
CODE_FILES = (
    "submission_stage4.py", "stage4_features.py", "stage4_model.py",
    "selector_features.py", "creo_features.py", "common.py",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if FINAL.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 4 freeze: {FINAL}")
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    if config.get("selected_method") != "fixed_reliability":
        raise ValueError("Unexpected Stage 4 method")
    if config.get("replacement_pass") is not False:
        raise ValueError("Expected replacement_pass=false")
    FINAL.mkdir(parents=True)
    shutil.copytree(SOURCE, CHECKPOINT)
    CODE.mkdir()
    for name in CODE_FILES:
        shutil.copy2(name, CODE / name)
    record = {
        "stage": "Stage 4 CREO-3W",
        "method": "fixed_reliability",
        "seed": 20261022,
        "selection_locked_before_this_test": True,
        "replacement_pass": False,
        "formal_pre_test_decision": "retain_frozen_pre_stage4_model",
        "train_hard_oof": {"correct": 1700, "n": 2114, "accuracy": 80.4148880149687},
        "validation": {"correct": 737, "n": 798, "accuracy": 92.35588972431077},
        "test_status_at_freeze": "not_run",
        "interpretation": (
            "Post-hoc/non-blind evaluation requested after earlier public test results were known. "
            "The score cannot establish blind generalization and cannot be used for further tuning."
        ),
    }
    (FINAL / "selection_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    frozen = sorted(
        [path for path in CHECKPOINT.rglob("*") if path.is_file()]
        + [path for path in CODE.rglob("*") if path.is_file()]
        + [FINAL / "selection_record.json"]
    )
    entries = [{
        "path": path.relative_to(FINAL).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    } for path in frozen]
    checkpoint_bytes = sum(
        path.stat().st_size for path in CHECKPOINT.rglob("*") if path.is_file()
    )
    manifest = {
        "format": "dase7506-mp2-stage4-posthoc-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "CREO-3W fixed_reliability",
        "seed": 20261022,
        "test_status_at_manifest_creation": "not_run",
        "posthoc_nonblind": True,
        "replacement_pass": False,
        "entrypoint": "frozen_code/submission_stage4.py",
        "checkpoint": "frozen_checkpoint",
        "checkpoint_uncompressed_bytes": checkpoint_bytes,
        "asset_limit_bytes": 16 * 1024 * 1024,
        "asset_limit_pass": checkpoint_bytes <= 16 * 1024 * 1024,
        "frozen_file_count": len(entries),
        "frozen_total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    (FINAL / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
