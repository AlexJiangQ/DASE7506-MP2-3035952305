"""Freeze the canonical anchor-retained Stage-5 bundle for reproduction."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil


SOURCE = Path("runs/stage5/bugfix2/provisional_bundle")
TARGET = Path("runs/stage5/final_canonical_anchor_retained")
CODE_FILES = (
    "submission_stage5.py", "stage5_features.py", "stage5_model.py",
    "creo_features.py", "creo_model.py", "selector_features.py",
    "selector_model.py", "common.py", "predict.py", "score_predictions.py",
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
        raise FileExistsError(f"Refusing to overwrite existing freeze: {TARGET}")
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    if config.get("apply_residual") is not False:
        raise ValueError("Canonical Stage-5 reproduction must retain the anchor")
    checkpoint = TARGET / "checkpoint"
    code_snapshot = TARGET / "code_snapshot"
    shutil.copytree(SOURCE, checkpoint)
    code_snapshot.mkdir(parents=True)
    for name in CODE_FILES:
        source = Path(name)
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, code_snapshot / name)

    records = []
    for path in sorted(TARGET.rglob("*")):
        if path.is_file():
            records.append({
                "path": path.relative_to(TARGET).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    checkpoint_bytes = sum(
        item["bytes"] for item in records if item["path"].startswith("checkpoint/")
    )
    manifest = {
        "format": "dase7506-mp2-stage5-freeze-v1",
        "frozen_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "candidate": "Stage-5 canonical anchor-retained bundle",
        "purpose": "reproduce the already-tested Stage-3 anchor; not a residual improvement claim",
        "seed": 20261022,
        "method": "code_snapshot/submission_stage5.py",
        "checkpoint": "checkpoint",
        "apply_residual": False,
        "selection_status": config["selection_status"],
        "stage5_residual_oof": {
            "correct": 1723, "n": 2114, "accuracy": 81.5042573320719,
            "beneficial": 4, "harmful": 1, "neutral": 1,
        },
        "frozen_output_validation": {
            "correct": 755, "n": 798, "accuracy": 94.61152882205513,
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
