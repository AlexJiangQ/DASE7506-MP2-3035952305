# Stage 4 Post-hoc Test Record

The Stage 4 CREO-3W `fixed_reliability` research leader was frozen before this run and evaluated once at the user's explicit request.

## Result

- Correct: `379/467`
- Accuracy: `81.15631691648822%`
- Official prediction time: `1.7210` seconds
- Official scoring time: `0.0483` seconds
- External prediction plus scoring wall time: `2.6849` seconds
- Checkpoint assets: `215,846` bytes
- Formal Stage 4 test runs: `1`

## Interpretation constraint

This is a **post-hoc/non-blind** score because public test results from earlier stages were already known before Stage 4 began. It cannot establish blind generalization and must not be used for further training, tuning, threshold selection, feature selection, or checkpoint switching.

The pre-test Stage 4 decision was `replacement_pass=false`, based only on train grouped OOF and validation. The test result is consistent with that prior decision but did not cause it.

The frozen code/checkpoint and their pre-test hashes are recorded under `code/runs/stage4/final_posthoc`.
