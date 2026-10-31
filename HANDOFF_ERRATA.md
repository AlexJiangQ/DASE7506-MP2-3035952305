# Handoff errata and historical reproduction notes

This file preserves the original research handoff and stage summaries. It records later facts without silently rewriting the earlier artifacts.

## Stage 3 stable-direct test status

The earlier DeepSeek handoff README and `STAGE3_SEARCH_SUMMARY.md`/`STAGE3_REPRODUCTION_NOTES.md` say that the Stage 3 stable-direct candidate was "not tested." That statement described its status when those documents were written. The stable-direct configuration was independently frozen and **later** evaluated once: its saved official result is **383/467 = 82.01284796573876%** at `code/runs/stage3/final_stable/test_score.json`. The selected high-validation CREO candidate had already been frozen before its first test and scored **387/467 = 82.86937901498929%**. The stable-direct result was a subsequent comparison after the high-validation test aggregate was known; it was not the basis for selecting the final method.

This correction changes no checkpoint, prediction, selection decision, or reported score for the selected high-validation method. `REPORT.md` now describes the later stable-direct test explicitly. The original handoff ZIP/README and stage-specific summaries remain as dated historical records and should be accompanied by this erratum when shared.

## Legacy Stage 2 test runner

`code/run_frozen_test_once.py` was built for the historical Stage 2 checkpoint but hard-codes `--method submission.py`. The current `code/submission.py` delegates to Stage 3, so that script is no longer an appropriate Stage 2 reproduction command. Its existing single-use guard also refuses to overwrite the old run. Use the current Stage 3 commands in `code/README.md` and the selected public checkpoint; do not run the legacy Stage 2 test runner as part of this submission.

## Published layout and local logs

The original Stage 3 freeze manifest records paths under its historical `code_snapshot/` and `checkpoint/` directory. The curated public assets put the checkpoint under `submission_assets/checkpoint/` and keep matching source files under `code/`. `SUBMISSION_MANIFEST.json` maps and hashes the published paths. The original `code/PACKAGE_MANIFEST.json` is a course-release manifest and predates our edited entry.

The A1 `audit_log.json` and historical `code/runs/final/test_run_started.json` contain local Windows user-profile paths. Both remain in ignored `code/runs/` and are excluded from the curated public assets. If an audit log must be shared later, create a separate sanitized copy that replaces the repository prefix with `<REPO_ROOT>` and verify it contains no local username; do not alter or delete the originals.
