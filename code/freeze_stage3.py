"""Freeze the precommitted Stage 3 high-validation CREO candidate."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil


SOURCE_CHECKPOINT = Path("runs/stage3/provisional_creo_bundle")
TARGET = Path("runs/stage3/final_high_validation")
CODE_FILES = (
    "submission_stage3.py",
    "creo_features.py",
    "creo_model.py",
    "selector_features.py",
    "selector_model.py",
    "common.py",
    "predict.py",
    "score_predictions.py",
    "requirements.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if TARGET.exists():
        if (TARGET / "freeze_manifest.json").exists():
            raise FileExistsError(f"Refusing to overwrite existing freeze: {TARGET}")
        checkpoint = TARGET / "checkpoint"
        code_snapshot = TARGET / "code_snapshot"
        expected = [code_snapshot / name for name in CODE_FILES]
        if not checkpoint.is_dir() or not code_snapshot.is_dir() or not all(
            path.is_file() for path in expected
        ):
            raise RuntimeError("Incomplete freeze cannot be safely resumed")
        print("Resuming verified pre-manifest freeze directory")
    else:
        checkpoint = TARGET / "checkpoint"
        code_snapshot = TARGET / "code_snapshot"
        shutil.copytree(SOURCE_CHECKPOINT, checkpoint)
        code_snapshot.mkdir(parents=True)
        for name in CODE_FILES:
            source = Path(name)
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, code_snapshot / name)
    if not SOURCE_CHECKPOINT.is_dir():
        raise FileNotFoundError(SOURCE_CHECKPOINT)

    records = []
    for path in sorted(TARGET.rglob("*")):
        if path.is_file():
            records.append({
                "path": path.relative_to(TARGET).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    checkpoint_bytes = sum(
        row["bytes"] for row in records if row["path"].startswith("checkpoint/")
    )
    manifest = {
        "format": "dase7506-mp2-stage3-freeze-v1",
        "frozen_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "precommitted_candidate": "CREO high-validation gate",
        "seed": 20261022,
        "method": "code_snapshot/submission_stage3.py",
        "checkpoint": "checkpoint",
        "selection_data": ["train-hard grouped OOF", "validation"],
        "test_used_for_selection": False,
        "selection_evidence": {
            "train_hard_oof_correct": 1720,
            "train_hard_oof_n": 2114,
            "train_hard_oof_accuracy": 81.3623462630085,
            "validation_correct": 755,
            "validation_n": 798,
            "validation_accuracy": 94.61152882205513,
            "validation_hard_correct": 312,
            "validation_hard_n": 355,
        },
        "checkpoint_bytes": checkpoint_bytes,
        "asset_limit_bytes": 16 * 1024 * 1024,
        "files": records,
    }
    manifest_path = TARGET / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_hash = sha256(manifest_path)
    (TARGET / "freeze_manifest.sha256").write_text(
        f"{manifest_hash}  freeze_manifest.json\n", encoding="ascii"
    )
    print(json.dumps({
        "target": str(TARGET.resolve()),
        "files": len(records),
        "checkpoint_bytes": checkpoint_bytes,
        "manifest_sha256": manifest_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
