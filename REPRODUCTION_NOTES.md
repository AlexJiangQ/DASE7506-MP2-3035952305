# MP2 Pre-Test Reproduction Notes

This document reproduces the provisional pre-test stage only. It does not authorize or run the formal test split.

## Environment

- Windows x64 (`Microsoft Windows NT 10.0.26200.0`).
- Python 3.12.3.
- NumPy 2.5.3, installed from the supplied `code/requirements.txt`.
- Primary seed: `20261022`.
- Local CPU identifier: `AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD`, 16 logical processors.
- Physical memory: 16,556,589,056 bytes.
- Four-thread variables are set for OMP, OpenBLAS and MKL during experiments and evaluation.

## Files added or modified

- `code/submission.py`: frozen checkpoint inference interface.
- `code/selector_features.py`: answer, control, grounding, conservative arithmetic, and relative embedding features.
- `code/selector_model.py`: listwise loss, scaling, deterministic NumPy optimizer and checkpoint I/O.
- `code/run_experiments.py`: unified bounded CV/search/validation entry point.
- `code/gate_experiment.py`: strict OOF gate experiment.
- `code/refresh_uncertainty.py`: fixed-model calibration and risk-coverage metrics.
- `code/baseline_audit.py`: baseline timing/config records.
- `code/tests/test_selector.py`: automated correctness and robustness tests.

No official data, manifest, `common.py`, `predict.py`, `score_predictions.py`, or `check_data.py` was modified.

## Environment setup

Run from `code/` in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The project uses NumPy only. No SciPy, scikit-learn, PyTorch, Transformers, external data, API, or additional model weight is required.

## Pre-test data integrity

The official `check_data.py` has no split filter and would open test inputs and references. During this stage it was run against `runs/pretest_data_view`, a train/validation-only view containing hard links to the unmodified official train/validation files and a matching restricted manifest:

```powershell
.\.venv\Scripts\python.exe check_data.py --data-dir runs\pretest_data_view
```

Observed result: passed, with 4,755 train and 798 validation questions and embedding shapes `[4755,4,1024]` and `[798,4,1024]`. The complete default data check remains deferred until formal test access is approved.

## Automated tests

```powershell
.\.venv\Scripts\python.exe -m compileall -q submission.py selector_features.py selector_model.py run_experiments.py gate_experiment.py refresh_uncertainty.py baseline_audit.py tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Six tests pass. They cover:

- direct use of `common.parse_answer`;
- multi-positive listwise loss and finite-difference gradient agreement;
- candidate permutation equivariance;
- arbitrary bank length and arbitrary IDs;
- empty/all-unparseable candidate safety;
- explicit checkpoint-missing failures;
- checkpoint save/load prediction consistency;
- exact Python `int` return type.

## Bounded search

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:PYTHONHASHSEED='20261022'
.\.venv\Scripts\python.exe run_experiments.py --runs-dir runs
.\.venv\Scripts\python.exe gate_experiment.py
.\.venv\Scripts\python.exe refresh_uncertainty.py
.\.venv\Scripts\python.exe baseline_audit.py
```

`run_experiments.py` performs:

1. train-hard extraction;
2. deterministic grouped 5-fold construction;
3. four rule baselines;
4. five declared information-block models at default regularization;
5. the finite L2 grid only for the top two block variants;
6. a uniform-positive loss ablation;
7. one validation pass for the CV top three;
8. provisional full-train-hard fitting and checkpoint export;
9. OOF temperature fitting and validation uncertainty analysis.

If a clean independent rerun is desired, point `--runs-dir` to a new directory so cached experiment artifacts cannot be reused.

## Official validation commands

```powershell
.\.venv\Scripts\python.exe predict.py `
  --method submission.py `
  --checkpoint runs\provisional_model_bundle `
  --split validation `
  --output runs\validation-predictions-1.json

.\.venv\Scripts\python.exe score_predictions.py `
  --split validation `
  --predictions runs\validation-predictions-1.json `
  --output runs\validation-score.json
```

Observed official result: 749/798, accuracy 93.85964912280701. A second fresh prediction produced identical IDs and choices.

## Provisional checkpoint

Directory: `code/runs/provisional_model_bundle`.

- Format: linear-v1, hand + relative + centered embedding.
- Loss: multi-positive listwise.
- Hand L2: 0.01.
- Embedding L2: 1.0.
- Uncompressed size: 23,386 bytes.
- Inference fails explicitly if the checkpoint is absent or malformed.

## Resource preflight

Five fresh validation prediction plus independent scoring processes were measured with four thread variables:

- mean 1.511 seconds;
- sample standard deviation 0.025 seconds;
- range 1.477–1.538 seconds.

This is a local preflight, not a claim about the reference Xeon or the still-unrun formal test. Detailed timings and asset hashes are under `code/runs/`.

## AI/reuse disclosure for this stage

Codex assisted with implementation, test construction, experiment orchestration and documentation. The method uses only general course concepts and the supplied project data/interfaces. The two optional downloaded report PDFs were not opened or copied. No other student's code, feature list, report wording, table structure or hyperparameters were used.

## Course design sources

The actual design sources were Lecture 1 pages 38–40; Lecture 2 pages 13–21; Lecture 3 pages 30–42; Lecture 4 pages 12–18; Lecture 5 pages 40–59 and 62–68; Lecture 6 pages 18–23 and 27–43; and Lecture 7 pages 42–63. Page numbers are PDF-reader page numbers.

## Deferred next stage

Only after explicit user approval: choose the final frozen code/checkpoint, create a freeze manifest, run the one formal test, and perform the final resource measurement. No test command is included here to prevent accidental execution during the paused stage.

## Stage 2 reproduction

Stage 2 reuses `runs/folds.json` exactly and writes only new experiment assets under `code/runs/stage2/`. Run from `code/` with the existing `.venv`:

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
.\.venv\Scripts\python.exe stage2_oof_analysis.py
.\.venv\Scripts\python.exe run_stage2.py
```

`run_stage2.py` performs the bounded train-only search and stops with a fixed winner before validation. It compares fixed answer-cluster aggregations, one 32-dimensional lexical replacement block, seven pre-declared weighting schemes, and pairwise weights `{0.1, 0.25, 0.5, 1.0}` against the multi-positive listwise reference. Each run stores config, seed, per-fold results, OOF scores/choices, loss history and elapsed time.

After the winner is fixed, the single official Stage 2 validation artifact was generated and scored with:

```powershell
.\.venv\Scripts\python.exe predict.py `
  --method submission.py `
  --checkpoint runs\stage2\provisional_model_bundle `
  --split validation `
  --output runs\stage2\validation_predictions.jsonl

.\.venv\Scripts\python.exe score_predictions.py `
  --predictions runs\stage2\validation_predictions.jsonl `
  --split validation `
  --output runs\stage2\validation_score.json
```

Observed result: 747/798, accuracy 93.60902255639098; validation-hard 85.63380281690141. `finalize_stage2_validation.py` verifies that the frozen official choices match an independent checkpoint reconstruction, computes the hard/answer strata, and fits temperature using train OOF scores only. It does not select a new model.

Run post-selection diagnostics and engineering checks with:

```powershell
.\.venv\Scripts\python.exe stage2_strata_analysis.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe check_data.py --data-dir runs\pretest_data_view
.\.venv\Scripts\python.exe stage2_engineering_audit.py
```

Eight unit tests pass, including a finite-difference gradient check for the weighted hybrid loss. Because the supplied manifest has no `audit` split, `stage2_engineering_audit.py` launches `synthetic_audit_worker.py` in five fresh processes on a fixed 798-question synthetic bank. Prediction and scoring average 0.748 seconds end to end; all five selected-index arrays are identical. The official validation prediction plus scoring took 1.092 seconds. The Stage 2 checkpoint is 23,486 bytes, and checkpoint plus the required inference Python sources is 52,503 bytes.

Stage 2 adds or modifies `selector_features.py`, `selector_model.py`, `submission.py`, `tests/test_selector.py`, `stage2_oof_analysis.py`, `run_stage2.py`, `finalize_stage2_validation.py`, `stage2_strata_analysis.py`, `synthetic_audit_worker.py`, `stage2_engineering_audit.py`, and `complete_stage2.py`. Official project files and data were not modified. No external package, data, API, pretrained weight, extra LLM generation, git operation, deletion, or D: drive storage was used.

The Stage 2 design uses the same course sources already listed: verifier and answer-equivalence concepts from Lecture 5 PDF-reader pages 40–59; self-consistency/answer voting from Lecture 5 pages 62–68 and Lecture 6 pages 18–23; loss/logit concepts from Lecture 3 pages 30–42 and Lecture 4 pages 12–18; scorer confidence cautions from Lecture 6 pages 27–43; and calibration/proper scoring rules from Lecture 7 pages 42–63. Lexical overlap is treated only as a surface control. These sources motivate the design and do not establish the observed empirical result.

The reproduction boundary remains unchanged: **the formal test has not been run**. Do not create a freeze manifest or execute a test command until the user chooses the final checkpoint and explicitly approves the final stage.

