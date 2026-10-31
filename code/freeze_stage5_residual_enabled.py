"""Freeze the previously trained Stage-5 residual model with revisions enabled.

This is an explicitly requested exploratory evaluation of an unpromoted model.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil


SOURCE = Path("runs/stage5/bugfix2/provisional_bundle")
TARGET = Path("runs/stage5/final_residual_enabled")
CODE_FILES = (
    "submission_stage5.py", "stage5_features.py", "stage5_model.py",
    "creo_features.py", "creo_model.py", "selector_features.py",
    "selector_model.py", "common.py", "predict.py", "score_predictions.py",
    "requirements.txt",
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite: {TARGET}")
    original = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    if original["apply_residual"] is not False:
        raise ValueError("Expected the canonical anchor-retained source")
    checkpoint = TARGET / "checkpoint"
    shutil.copytree(SOURCE, checkpoint)
    config = dict(original)
    config["apply_residual"] = True
    config["selection_status"] = "exploratory_residual_enabled_not_promoted"
    (checkpoint / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    snapshot = TARGET / "code_snapshot"
    snapshot.mkdir()
    for name in CODE_FILES:
        shutil.copy2(Path(name), snapshot / name)
    files = [
        {"path": path.relative_to(TARGET).as_posix(), "bytes": path.stat().st_size,
         "sha256": digest(path)}
        for path in sorted(TARGET.rglob("*")) if path.is_file()
    ]
    manifest = {
        "format": "dase7506-mp2-stage5-exploratory-freeze-v1",
        "frozen_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "purpose": "exploratory test of the already trained residual model",
        "not_promoted_by_prevalidation_rule": True,
        "test_score_must_not_be_used_as_unbiased_model_selection": True,
        "seed": 20261022,
        "method": "code_snapshot/submission_stage5.py",
        "checkpoint": "checkpoint",
        "apply_residual": True,
        "source_config_sha256": digest(SOURCE / "config.json"),
        "checkpoint_bytes": sum(item["bytes"] for item in files if item["path"].startswith("checkpoint/")),
        "files": files,
    }
    path = TARGET / "freeze_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (TARGET / "freeze_manifest.sha256").write_text(
        f"{digest(path)}  freeze_manifest.json\n", encoding="ascii"
    )
    print(json.dumps({"manifest": str(path.resolve()),
                      "manifest_sha256": digest(path),
                      "checkpoint_bytes": manifest["checkpoint_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
