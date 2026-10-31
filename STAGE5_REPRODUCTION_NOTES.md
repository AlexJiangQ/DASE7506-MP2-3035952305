# Stage 5 Reproduction Notes

Stage 5 uses Python 3.12, NumPy only, seed `20261022`, and the existing `code/.venv`. Run from `code/` without git commands.

The canonical current-workspace result is `runs/stage5/bugfix2`. Earlier `runs/stage5` and `runs/stage5/bugfix1` directories are intentionally preserved as bug records and must not be treated as canonical.

For a new non-overwriting reproduction directory:

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:STAGE5_RUN_DIR='runs/stage5/reproduction'

.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe .\stage5_train.py
```

Score the first frozen validation prediction:

```powershell
.\.venv\Scripts\python.exe .\score_predictions.py `
  --predictions .\runs\stage5\reproduction\validation_predictions.json `
  --split validation `
  --output .\runs\stage5\reproduction\validation_score.json
```

Run the independent frozen inference path and confirm identical choices:

```powershell
.\.venv\Scripts\python.exe .\predict.py `
  --method .\submission_stage5.py `
  --checkpoint .\runs\stage5\reproduction\provisional_bundle `
  --split validation `
  --output .\runs\stage5\reproduction\validation_predictions_official.json
```

The expected selection status is `no_stage5_model_met_replacement_rule_anchor_retained`, and the official validation score is 755/798 (`94.61152882205513%`). The bundle deliberately sets `apply_residual=false` because no residual model passed both OOF and validation replacement rules.

Canonical evidence files:

- `runs/stage5/CANONICAL_RUN.json`;
- `runs/stage5/bugfix2/prevalidation_protocol.json`;
- `runs/stage5/bugfix2/experiment_summary.json`;
- `runs/stage5/bugfix2/oof_predictions.npz`;
- `runs/stage5/bugfix2/reporting_audit.json`;
- `runs/stage5/bugfix2/calibration.json`;
- `runs/stage5/bugfix2/validation_score.json`;
- `runs/stage5/bugfix2/resource_measurement.json`;
- `runs/stage5/bugfix2/provisional_bundle`.

The canonical bundle was subsequently frozen with `apply_residual=false` and used for one authorised test reproduction: 387/467 (`82.86937901498929%`). This is the existing Stage 3 anchor score, not evidence that the residual model improved test accuracy. Do not rerun test or enable the non-promoted residual candidate.
