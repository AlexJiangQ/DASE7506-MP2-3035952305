# Stage 4 Reproduction Notes

Stage 4 is a train/CV/validation-only experiment. Do not run test.

Environment:

- Python 3.12.3;
- NumPy 2.5.3;
- seed `20261022`;
- four CPU threads;
- existing virtual environment `code\.venv`.

Run from `code`:

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'

.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe .\stage4_train.py
```

`stage4_train.py` performs the bounded Phase A–D search, nested candidate cross-fitting for group training, OOF calibration/gating, final train fitting, and the single validation pass for the OOF top three.

The current run used `stage4_finalize.py` only to align the saved provisional bundle and audit files with the already-evaluated raw OOF research leader. It did not retrain models or rerun validation:

```powershell
.\.venv\Scripts\python.exe .\stage4_finalize.py
```

Reproduce the saved provisional research leader on validation:

```powershell
.\.venv\Scripts\python.exe .\predict.py `
  --method .\submission_stage4.py `
  --checkpoint .\runs\stage4\provisional_bundle `
  --split validation `
  --output .\runs\stage4\validation-reproduction.json

.\.venv\Scripts\python.exe .\score_predictions.py `
  --predictions .\runs\stage4\validation-reproduction.json `
  --split validation `
  --output .\runs\stage4\validation-reproduction-score.json
```

Expected validation result: `737/798 = 92.35588972431077%`.

Resource probe:

```powershell
.\.venv\Scripts\python.exe .\measure_stage4.py
```

Important interpretation:

- Stage 4 OOF research leader: fixed three-way reliability, `80.4149% ± 0.8659%`.
- Stage 2 OOF baseline: `81.6450% ± 1.0217%`.
- Stage 3 stable-direct OOF: `82.1192%`.
- Replacement conditions are not met.
- The provisional bundle is a research artifact, not a recommendation to replace the frozen Stage 3 model.
- **Stage 4 has not run test.**
