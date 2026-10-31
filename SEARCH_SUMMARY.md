# MP2 Pre-Test Search Summary

Status: **paused before formal test**. The formal test split has not been read or run.

## 1. Protocol and data

- Primary seed: `20261022`.
- Training/model selection: 2,114 train-hard groups, using fixed grouped 5-fold CV.
- Confirmation: 798 validation groups, of which 355 are answer-diverse (`validation-hard`).
- Labels use `common.parse_answer`; a group can have multiple positive candidates.
- Fold scalers are fitted on the training portion of each fold only.
- Search was bounded: compare information blocks at default L2, search the declared L2 grid only for the top two information-block variants, and run one loss ablation.
- Search/modeling elapsed time recorded by the unified runner: 166.96 seconds. The 90-minute ceiling was not exhausted because the complete pre-declared finite search finished earlier.

## 2. Methods tried

The plurality tie rule was fixed before validation: choose among tied answer clusters and candidates with a SHA-256 content hash of the question and candidate text. It does not use IDs or candidate order.

### Baselines and initial block ablations

| Method | Train-hard CV mean | SE | Validation-all | Validation-hard |
|---|---:|---:|---:|---:|
| Fixed candidate 0 | 50.99 | 1.04 | 78.20 | 54.93 |
| Deterministic content-hash random | 48.91 | 0.52 | 76.94 | 52.96 |
| Longest trace | 55.06 | 0.97 | 78.07 | 56.06 |
| Parsed-answer plurality | 65.99 | 0.98 | 86.22 | 69.01 |
| Structure-only listwise | 74.41 | 1.50 | not run (outside CV top three) | not run |
| Raw-embedding-only listwise | 78.38 | 1.04 | not run (outside CV top three) | not run |
| Centered-embedding-only listwise | 78.90 | 1.32 | 91.73 | 81.41 |
| Compact hand + centered, default L2 | 80.23 | 1.26 | superseded by selected L2 | superseded |
| Compact hand + raw + centered, default L2 | 80.37 | 1.17 | superseded by selected L2 | superseded |

### CV finalists

| Model | Fold accuracies | Mean ± SE | Validation-all | Validation-hard |
|---|---|---:|---:|---:|
| Hand + raw + centered, hand L2=.01, embedding L2=1 | 80.61, 79.67, 82.03, 83.69, 78.67 | 80.94 ± 0.88 | 93.23 | 84.79 |
| **Hand + centered, hand L2=.01, embedding L2=1** | 82.03, 78.49, 82.03, 82.03, 78.67 | **80.65 ± 0.85** | **93.86** | **86.20** |
| Centered only, embedding L2=.1 | 78.96, 76.36, 82.98, 80.38, 75.83 | 78.90 ± 1.32 | 91.73 | 81.41 |

Only the CV top three learned configurations were evaluated on validation. Rule baselines were evaluated for context. No new feature was invented after validation.

## 3. Provisional best and selection reason

The provisional method is a multi-positive linear listwise scorer using:

- 32 compact hand/control features;
- 6 relative embedding features;
- 1,024-dimensional within-question centered embeddings;
- invalid-candidate masking with a learned-score fallback when every candidate is unparseable.

The full raw+centered model achieved the highest CV mean, but its gain over the simpler centered-only fusion was only **0.28 percentage point**, below the pre-declared 0.75-point promotion threshold. Its fold gains were also mixed, and validation-hard was 1.41 points lower. The simpler hand+centered model was therefore selected.

The official validation scorer reports **749/798 = 93.859649%**. On validation-hard the selected model scores **86.197183%**, versus 69.014085% for plurality.

## 4. Core ablations and conclusions

- Plurality is strong on validation-all but weak on answer-diverse groups. It scores only 20.51% when exactly one candidate is correct, but 100% when three or four are correct.
- Structure-only improves train-hard CV by 8.42 points over plurality, showing that simple consensus/style/grounding controls matter. These are surface controls, not proof of valid reasoning.
- Centered embeddings add another 4.49 points over structure-only and substantially improve validation-hard. Relative representation information therefore contributes beyond plurality and trace length.
- Raw embeddings add no stable benefit once centered embeddings and compact controls are present.
- Uniform-positive cross-entropy scored 80.75 ± 1.46, only 0.09 point above the default multi-positive listwise loss and with substantially higher fold variance. It was not promoted under the simplicity rule.

## 5. Gate and calibration failures

The OOF logistic gate was trained only from comparator OOF predictions and permitted gate features. It learned to choose the comparator on every effective conflict:

- CV mean improvement: 0.00 point;
- non-declining folds: 5/5, but no fold improved;
- validation-hard: unchanged at 86.20%.

It failed the required +1.0-point criterion and was rejected. The checkpoint contains no gate.

Temperature fitting on train OOF selected `T=0.4642`, but validation calibration remained poor:

- NLL: 0.5641;
- ECE (five bins): 0.3034;
- AURC: 0.0646;
- comparator/plurality parsed-answer agreement: 87.84%.

Accuracy across confidence bins is not monotonic, so maximum softmax is **not** presented as a calibrated probability. Calibration is recorded as a failed/descriptive experiment and does not change predictions.

## 6. Resources and engineering status

- Checkpoint assets: 23,386 bytes uncompressed (well below 16 MiB).
- Five fresh local validation prediction+scoring processes: mean 1.511 s, range 1.477–1.538 s, four-thread environment.
- Official validation prediction and scoring passed.
- Two frozen validation predictions produced identical IDs and selected indices.
- Six automated tests passed: official parser binding, multi-positive loss, finite-difference gradient, permutation equivariance, arbitrary length/IDs and all-unparseable inputs, explicit missing-checkpoint error, Python-int outputs, and checkpoint prediction roundtrip.
- The official `check_data.py` passed on an intentionally train/validation-only hard-link view. The full default invocation is deferred because it necessarily opens test inputs and references.
- No D: drive space was used. Approximately 4.8 GB remained on C: during preflight; D: was available if needed.

The implementation is ready for a formal freeze/test stage subject to user approval, but a final freeze manifest and reference test measurement do not yet exist.

## 7. Course knowledge actually used

- Lecture 1, PDF reader pages 38–40: embeddings.
- Lecture 2, pages 13–21: similarity, dot products and softmax.
- Lecture 3, pages 30–42: train/validation separation, loss and overfitting.
- Lecture 4, pages 12–18: classification logits, softmax, argmax and cross-entropy.
- Lecture 5, pages 40–59: verifier design, answer extraction and equivalence scoring.
- Lecture 5, pages 62–68: reasoning ablations and self-consistency.
- Lecture 6, pages 18–23: self-consistency voting and its systematic-error failure mode.
- Lecture 6, pages 27–43: answer scoring and confidence not implying correctness.
- Lecture 7, pages 42–63: calibration, selective prediction and proper scoring rules.

These sources motivated the design; they do not prove this project's empirical conclusions.

## 8. Potential next improvements, in priority order

| Priority | Direction | Expected benefit | Implementation risk | Extra compute |
|---:|---|---|---|---|
| 1 | Learned answer-cluster pooling | High: score answer hypotheses rather than individual traces | Medium | Low |
| 2 | Targeted training for single-correct and 3-to-1 wrong-consensus cases | High on hard bank | Medium; must avoid sampling bias | Low |
| 3 | Pairwise/listwise hybrid objective | Medium-high | Medium | Low-medium |
| 4 | Question-candidate lexical interaction | Medium | Medium; may add sparse assets | Low-medium |
| 5 | Supervised low-rank embedding projection | Medium | Medium-high; regularization sensitive | Medium |
| 6 | More robust consensus/comparator gate | Medium | High; current gate collapsed | Low |
| 7 | Stronger conservative mathematical expression parsing | Medium on arithmetic-error cases | High; false positives are dangerous | Low |
| 8 | Hard-example curriculum | Medium | Medium | Medium |
| 9 | Improved calibration mapping | Low for accuracy, useful for reliability analysis | Medium | Low |
| 10 | Seed/model averaging | Low-medium | Low, but must count every asset/member | Medium |

These are future candidates only. No further search will start without user approval.

## 9. Explicit pause condition

**Formal test has not been run.** No `freeze.json`, final report, GitHub issue, online submission, repository publication, or final link has been created.

## 10. Stage 2 targeted improvement (supersedes the earlier future-work status)

Stage 2 used the same seed (`20261022`), the exact same grouped folds, train OOF only for selection, and an independent `code/runs/stage2/` directory. It completed the four approved core directions in 837.64 seconds wall-clock, below the 90-minute ceiling. The winner was fixed before the only official Stage 2 validation prediction.

The initial OOF audit found 691 comparator/plurality answer conflicts: 462 good comparator overturns and 152 bad overturns. On 143 wrong 3-to-1 consensus groups, the Stage 1 comparator recovered 62.94%; on 684 single-correct groups it scored 60.09%.

| Stage 2 method | Train-hard CV mean ± SE | Fold accuracies | Single-correct | Wrong 3-to-1 |
|---|---:|---|---:|---:|
| Stage 1 listwise reference | 80.65 ± 0.85 | 82.03, 78.49, 82.03, 82.03, 78.67 | 60.09 | 62.94 |
| Best cluster pooling, max + .25 log count | 80.98 ± 0.80 | 81.09, 79.20, 82.98, 82.51, 79.15 | 56.87 | 54.55 |
| 32-d lexical replacement | 80.37 ± 0.86 | 80.85, 78.49, 82.03, 82.27, 78.20 | 60.23 | 65.03 |
| Best weighting, conflict ×1.5 | 80.65 ± 0.87 | 80.85, 79.20, 82.27, 82.74, 78.20 | 60.38 | 65.73 |
| **Listwise + 0.1 pairwise** | **81.64 ± 1.02** | **82.74, 79.20, 83.69, 83.45, 79.15** | **60.23** | **60.84** |

The hybrid improved the fold mean by 0.99 point and all 5/5 folds, so it passed the declared overall advancement rule. Its wrong-3-to-1 result fell by 2.10 points (three cases); this trade-off is retained in the evidence and must be considered before final model choice. Cluster pooling gained only 0.33 point and materially hurt both target strata; lexical interaction reduced overall CV; targeted weighting did not improve overall CV. These three additions were rejected. No gate or ensemble was retained or attempted after the bounded core search.

The frozen Stage 2 checkpoint obtained **747/798 = 93.609023%** on the single official validation scoring run and **85.633803%** on validation-hard. This is below Stage 1 by 0.2506 point overall and 0.5634 point on hard groups, so the CV improvement was not confirmed by validation. No feature, loss weight, or model choice was changed after seeing this result. Validation is explicitly a previously used development set, not an untouched test set.

Train-OOF temperature scaling selected `T=0.4467`. On validation it reduced descriptive NLL from 0.7712 to 0.5538 and five-bin ECE from 0.4489 to 0.2914. This is reported as score-confidence calibration, not as a probability of correctness or an accuracy gain.

Engineering status: eight tests passed; official train/validation-only `check_data.py`, official validation prediction/scoring, checkpoint reconstruction, permutation equivariance, all-unparseable fallback, Python-int outputs and the 16 MiB asset limit passed. The checkpoint is 23,486 bytes; checkpoint plus inference Python sources is 52,503 bytes. Official validation prediction plus scoring took 1.092 seconds. Five fresh synthetic-audit processes averaged 0.748 seconds (maximum 0.773 seconds) with four thread variables. The supplied manifest has no audit split, so the permitted synthetic audit bank was used for repeated timing. D: was not used.

**Formal test is still unrun.** No freeze manifest, final report/PDF, online submission, GitHub issue, repository publication, or checkpoint link was created. The next decision is either controlled additional research or choosing a checkpoint for final freeze and the one formal test.

