# Pre-submission audit (2026-10-31, UTC+8)

This is a publication check for the selected Stage 3 CREO high-validation checkpoint. It does not record a new model-selection experiment or a new test run.

## Resolved checks

- `REPORT.md` and `REPORT.pdf` agree. The PDF is six A4 pages. All six pages were converted to PNG and visually inspected. Page 2 has the requested plain-text listwise formula and a complete Section 3 heading; page 3 has a real Section 4.1 table with unwrapped numbers; page 5 has a legible monospaced validation command block; page 6 has the correctly spaced Section 7 heading and `AI coding-assistance` disclosure. PDF text extraction found no literal `<table>`.
- From `code/`, `python -m venv .venv` and the report's pip command succeeded (NumPy was already installed). The three four-thread environment assignments and the exact validation prediction/scoring block ran with `--method .\submission.py --checkpoint ..\submission_assets\checkpoint --split validation`. The score is **755/798 = 94.61152882205513%**. All 798 IDs and choices match the earlier frozen validation predictions. The final external prediction-plus-scoring wall time was 3.692 s; official inner times were 2.719 s and 0.052 s. Outputs: `code/runs/reproduced_validation_predictions.json` and `code/runs/reproduced_validation_score.json`. This phase did not run the test split. The 31 unit/regression tests were passed in the preceding A1 audit and were not repeated here.
- `code/submission.py` imports `submission_stage3.predict`; its usage comment now points from `code/` to the public checkpoint. The frozen Stage 3 implementation and its four dependencies match all corresponding historical freeze hashes. The original 18-file freeze manifest entries and its SHA-256 sidecar all verify.
- `submission_assets/` contains only nine checkpoint files, the official saved test score JSON, the historical freeze manifest JSON, and its SHA-256 sidecar. Total: **170,734 bytes** (0.163 MiB); checkpoint: **166,594 bytes**, both well below 16 MiB. `.gitignore` excludes local `.venv/`, `runs/`, Python caches, and the temporary `REPORT_pre_submission.pdf`, while explicitly allowing `submission_assets/`.
- `SUBMISSION_MANIFEST.json` is the current publication file list with byte sizes and SHA-256 hashes. `code/PACKAGE_MANIFEST.json` describes the original course release and is not the submission manifest. The saved official test result is **387/467 = 82.86937901498929%**; it was not reproduced in this phase.
- The four public asset paths in the report match the directory: `submission_assets/checkpoint`, `submission_assets/final_test_score.json`, `submission_assets/freeze_manifest.json`, and `submission_assets/freeze_manifest.sha256`.

## Local-only material and publication scope

The publishable text/code scan found no machine-specific absolute paths or local usernames outside ignored experiment folders. Historical local logs such as `code/runs/a1_submission_audit_20261031_134006/audit_log.json` and `code/runs/final/test_run_started.json` do contain machine-specific absolute paths and must stay unpublished. The local `code/.venv/` is about 69 MB and `code/runs/` about 129 MB; both are ignored. Do not upload ignored folders manually. The older Stage 2/4/5 summaries and runner scripts are research history, not the current submission entry; follow `code/submission.py`, `REPORT.md`, and `SUBMISSION_MANIFEST.json` for reproduction. No local files were deleted.

## Manual publication steps

1. Publish the repository with `REPORT.md`, `REPORT.pdf`, `SUBMISSION_MANIFEST.json`, `PRE_SUBMISSION_AUDIT.md`, `code/`, `guide/`, data/licence files, and `submission_assets/`; keep ignored local folders out of the upload. Check the public file list once published.
2. Replace `<OWNER>/<REPO>/<COMMIT>` in the report's public URL template with the actual public repository owner, name, and immutable commit. Check that the repository and every asset URL are accessible without a local login or machine path.
3. Enter the correct student ID and the saved official accuracy in the course website's submission form, inspect the prepared GitHub issue, and create that issue to complete submission. This audit has not published, pushed, or created an issue.

## Optional one-time test reproduction

The selected Stage 3 checkpoint already has an official saved test score. A single independent test reproduction is reasonable only after the public version, paths, and hashes are fixed, if the author wants a final packaging check. Use exactly the published entry and checkpoint, record its resource measurement and output, and do not tune or switch models in response. No test reproduction was run during this audit.
