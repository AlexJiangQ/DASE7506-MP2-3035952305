# MP2 — Code and Running Instructions

Start with the [Project Guide](../guide/GUIDE.md) for the task description, example, research questions, assessment, deadline and submission rules.

This student working copy contains the complete supplied dataset, official evaluation tools, and our frozen Stage 3 CREO selector. The upstream course release supplied an empty prediction interface; our current `submission.py` and `../submission_assets/checkpoint` provide the selected method. Run the commands below from **`code/`**.

## 1. Install and check the data

Use the published student repository containing `code/` and `submission_assets/`, with **Python 3.12**. From `code/`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py
```

The [upstream course repository](https://github.com/xudongwu-0/DASE7506-MP2) supplies the original data and evaluator, but **not** this trained selector or checkpoint. A submitted copy must include this working code and the public assets directory.

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. NumPy is sufficient for the supplied tools. Add and pin your own method's dependencies. After installation, prediction and scoring must work offline.

## 2. Method interface and evaluator

The current `submission.py` exposes `predict(bank, arrays, checkpoint)` by importing `submission_stage3.predict`. The signature also permits another Python method file for independent research.

- `bank` is a list in fixed question order. Each row has `id`, `question` and four `candidates`; each candidate has `index` and `text`.
- `arrays['embeddings']` is a float32 NumPy array of shape **[N, 4, 1024]**, aligned with the candidate order. Candidate text and these fixed representations are available to every method.
- `checkpoint` is a `pathlib.Path` or `None`. Save every learned weight, transformation, statistic and configuration needed for inference; return exactly one **Python integer from 0 to 3** per question, in bank order.
- Write any training code you need. Training-free methods are equally eligible. Fit using training inputs/references only; use validation for development. Question IDs are for alignment only.

Run official validation with the published checkpoint:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint ../submission_assets/checkpoint \
  --split validation --output runs/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/validation-predictions.json --output runs/validation-score.json
```

The Stage 3 checkpoint is supplied in this student publication. On Windows, set the thread environment variables before running the same Python commands. The runner requires `--method` and never opens reference answers.

The selected Stage 3 method was frozen before its original official test. For independent reproduction, the official test interface is:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint ../submission_assets/checkpoint \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

Submit the scorer's **`accuracy`** field, in percent. It checks the selected candidate's final numeric answer using `common.parse_answer`, including equivalent fractions/decimals. It grades final answers, not reasoning validity. A score claimed in a predictions file is ignored; missing/reordered IDs, invalid indices and a mismatched split/protocol are rejected.

## 3. Data and files

| Files | Purpose |
|---|---|
| `submission.py` | Current Stage 3 CREO submission entry; imports `submission_stage3.predict`. |
| `predict.py` | Runs your frozen method; writes predictions without reading references. |
| `score_predictions.py`, `common.py` | Fixed independent scorer, numeric parser and data loading. |
| `check_data.py` | Checks hashes, split disjointness and feature/reference alignment. |
| `data/candidates/gsm8k_{split}.jsonl` | Questions and four candidate traces. |
| `data/candidates/gsm8k_{split}.npz` | Fixed whole-trace embeddings; no model weights or correctness labels. |
| `data/gsm8k_{split}.jsonl` | Aligned references: training supervision, validation selection, test scoring only. |
| `data/manifest.json` | Protocol, counts, provenance and data hashes. |
| `data/LICENSE.txt`, `data/NOTICE.txt` | Source license and attribution. |
| `PACKAGE_MANIFEST.json` | Original course release file hashes; **not** the current submission manifest. The final file hashes are in `../SUBMISSION_MANIFEST.json`. |

Protocol: **`dase7506-mp2-open-hard-verifier-v5`**. Splits contain **4,755 / 798 / 467** train/validation/test questions and do not overlap. Every group has a correct final answer. Training and validation include easier groups; test questions have at least two different valid parsed answers. Candidate order is fixed. Test references are public for self-reporting and peer reproduction; they must not enter training, lookup tables or method selection.

The instructor may supply a separate audit bank without references:

```bash
python predict.py --method submission.py --checkpoint ../submission_assets/checkpoint \
  --split audit --data-dir /path/to/audit_inputs --output runs/audit-predictions.json
```

The same frozen method must accept that bank without retraining or hard-coded sample counts. It should use the bank/arrays passed to `predict()`, rather than opening reference files or assuming that every input is the public test set.

## 4. Current frozen Stage 3 submission

From `code/`, the matching published checkpoint is `../submission_assets/checkpoint`. The curated `submission_assets/` directory contains the checkpoint, official Stage 3 test-score JSON, and the original freeze manifest and hash sidecar. Source code stays under `code/`; `../SUBMISSION_MANIFEST.json` hashes the current entry, its dependencies, report, and published assets. `PACKAGE_MANIFEST.json` records the **original release**, so its `submission.py` hash is expected to differ after implementation.

```powershell
$env:OMP_NUM_THREADS='4'; $env:OPENBLAS_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
.\.venv\Scripts\python.exe .\predict.py --method .\submission.py --checkpoint ..\submission_assets\checkpoint --split validation --output .\runs\published_validation_predictions.json
.\.venv\Scripts\python.exe .\score_predictions.py --split validation --predictions .\runs\published_validation_predictions.json --output .\runs\published_validation_score.json
```

The expected validation result is **755/798 = 94.61152882205513%**. The saved official full-test result is **387/467 = 82.86937901498929%**; it was **not rerun** during the A1/A2 publication preparation. The original freeze manifest refers to `checkpoint/` and `code_snapshot/` inside its historical freeze directory. In the public layout, `checkpoint/` is under `submission_assets/`, while the source files are under `code/`; use `SUBMISSION_MANIFEST.json` to verify public paths and hashes. Do not use the legacy Stage 2 `run_frozen_test_once.py` with this Stage 3 entry.

Public directory URL template for the report or submission issue: `https://github.com/<OWNER>/<REPO>/tree/<COMMIT>/submission_assets/` (replace placeholders only after publishing and checking access).

## 5. Measurement and reproduction

The [Guide](../guide/GUIDE.md#3-constraints) defines the resource limits and measurement rules: **30 seconds** for complete CPU prediction plus scoring with **four threads**, and **16 MiB** of additional inference assets. The runner's JSON timing fields do not include all process startup; measure externally and record your hardware.

Pin your method's dependencies and provide exact installation/training/evaluation commands, seeds and matching inference assets. A peer should be able to use your submitted code version and checkpoint to run the official commands above **without retraining**. See the [submission and peer-review rules](../guide/GUIDE.md#4-submission-and-peer-review) for the full submission requirements and deadline.

## 6. Attribution

Questions and references derive from [GSM8K](https://github.com/openai/grade-school-math), distributed under its retained MIT license. Candidate solutions and fixed representations were prepared with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B); the model revision is recorded in the data manifest. No model weights are distributed. Keep `data/LICENSE.txt` and `data/NOTICE.txt` when redistributing this package.
