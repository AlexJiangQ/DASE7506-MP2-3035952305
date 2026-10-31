# Stage 3 Reproduction Notes

Run all commands from `code` with the existing Python 3.12 virtual environment. No git command is required.

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'

.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe .\run_stage3.py
.\.venv\Scripts\python.exe .\audit_stage3.py
.\.venv\Scripts\python.exe .\audit_gate_stability.py
.\.venv\Scripts\python.exe .\run_stage3_direct_gate.py
```

Reproduce the high-validation CREO validation result:

```powershell
.\.venv\Scripts\python.exe .\predict.py `
  --method .\submission_stage3.py `
  --checkpoint .\runs\stage3\provisional_creo_bundle `
  --split validation `
  --output .\runs\stage3\validation_predictions.json

.\.venv\Scripts\python.exe .\score_predictions.py `
  --predictions .\runs\stage3\validation_predictions.json `
  --split validation `
  --output .\runs\stage3\validation_score.json
```

Reproduce the stable-direct CREO validation result by substituting checkpoint `runs\stage3\provisional_creo_stable_bundle` and using the already-generated `validation_predictions_stable.json` / `validation_score_stable.json` output names.

Important state:

- Seed: `20261022`.
- Frozen grouped folds: `code\runs\folds.json`.
- High-validation result: 755/798 (`94.61152882205513%`).
- Stable-direct result: 748/798 (`93.734335839599%`).
- The precommitted high-validation Stage 3 candidate has been tested once: 387/467 (`82.86937901498929%`).
- Do not rerun test or test another Stage 3 model. The stable-direct candidate remains validation/OOF-only.
