# Stage 3 CREO Search Summary

## Scope and research question

Stage 3 is an original **Consensus Reliability and Evidence-Based Overturning (CREO)** study. It does not reproduce another report's set-attention/pooling architecture. The central question is:

> When is answer plurality reliable, and what observable evidence is strong enough to justify overturning it without creating more bad overturns?

Only train references, frozen train-hard OOF predictions, and validation references were used for development and selection. The historical Stage 2 test result was known but was not used for any Stage 3 choice. After precommitting and freezing the high-validation candidate, one official Stage 3 test prediction and scoring run was completed.

## Implemented components

1. An answer-group evidence scorer aggregates candidate evidence by parsed answer.
2. A plurality reliability model estimates whether plurality is likely to be correct. This output is treated as a model score, not as a guaranteed correctness probability.
3. An evidence-based gate decides between plurality and a challenger.
4. Two challenger definitions were audited:
   - the best non-plurality answer group under the learned group scorer;
   - the answer group selected by a frozen Stage 1 or Stage 2 comparator.
5. Fixed within-answer score shrinkage was retained only as a baseline.

All labels use `common.parse_answer`. Features use supplied text, embeddings, frozen base scores, conservative bounded arithmetic parsing, and answer-group relationships. No IDs, fixed sample order, external data, pretrained weights, APIs, or extra LLM generations are used.

## Main results

| Method | Train-hard OOF | Fold deltas vs Stage 2 | Validation all | Validation hard | Wrong-plurality traps | Correct-plurality nontraps |
|---|---:|---|---:|---:|---:|---:|
| Stage 2 frozen comparator | 1726/2114 (81.646%) | baseline | 747/798 (93.609%) | 304/355 (85.634%) | 69/110 | 235/245 |
| Answer-group scorer | 1721/2114 (81.410%) | not retained | 751/798 (94.110%) | 308/355 (86.761%) | 71/110 | 237/245 |
| CREO high-validation gate | 1720/2114 (81.362%) | `[-2,-4,+6,-4,-2]` | **755/798 (94.612%)** | **312/355 (87.887%)** | **77/110** | 235/245 |
| CREO stable direct gate | **1736/2114 (82.119%)** | **`[+1,+5,+5,+1,-2]`** | 748/798 (93.734%) | 305/355 (85.915%) | 68/110 | **237/245** |
| Direct-gate validation leader | 1712/2114 (80.984%) | `[-5,-2,-4,0,-3]` | 750/798 (93.985%) | 307/355 (86.479%) | 70/110 | 237/245 |

The high-validation gate gains eight validation-hard questions over Stage 2 without reducing the correct-plurality stratum on validation. Its OOF evidence is weak: it loses six questions overall and improves only one fold. The stable direct gate gains ten OOF questions and is non-decreasing in four folds, but gains only one validation-hard question; it therefore misses the predeclared `validation-hard +2` replacement condition by one question.

No direct gate met the full conservative replacement rule simultaneously:

- OOF delta versus Stage 2 at least zero;
- at least 4/5 folds non-decreasing;
- validation-hard at least two additional correct questions;
- good overturns at least bad overturns.

Consequently Stage 3 has two meaningful provisional candidates, but no automatically justified final replacement:

- **High-validation candidate:** strongest observed validation result, higher distribution-shift/selection risk.
- **Stable-direct candidate:** stronger grouped-OOF evidence, smaller validation improvement.

## Ablations and findings

- Fixed answer pooling changed which consensus traps were recovered but did not improve OOF over Stage 2. The best validation pooling variants reached 306/355 hard, below CREO.
- The answer-group scorer preferred scalar L2 `0.01`, embedding L2 `1.0`, and no pairwise term. It improved validation but was not consistently better across folds.
- Reliability L2 `0.01` had the best validation-hard NLL (`0.28523`); OOF NLL was `0.35548`. This gap is another reason to avoid calling its output a correctness probability.
- The high-validation gate uses L2 `0.1`, class balancing, and threshold `0.6` with the learned non-plurality challenger.
- The stable-direct gate uses the frozen Stage 2 answer group as challenger, L2 `0.01`, no class balancing, and threshold `0.4`.
- The direct challenger experiment clarified the error trade-off: conservative gating can protect correct consensus and improve OOF, but may miss validation traps that the more aggressive learned challenger recovers.

## Engineering status

- Official validation prediction/scoring passed for all three saved Stage 3 bundles.
- Thirteen unit/regression tests pass, including parser binding, conservative arithmetic safety, permutation equivariance, arbitrary IDs, all-unparseable candidates, deterministic training, multi-positive loss gradient checks, and checkpoint behavior.
- Five fresh official validation prediction processes for the high-validation candidate produced identical choices. Times were `3.0073, 3.1604, 3.1822, 2.9973, 2.8802` seconds (median `3.0073`, maximum `3.1822`).
- High-validation bundle: 166,594 bytes; stable-direct bundle: 162,381 bytes. Required inference Python sources total 71,225 bytes. Both are far below the 16 MiB asset budget.
- Validation-only in-process resource probes measured approximately 207 MiB peak working set, including Python, NumPy, loaded validation embeddings, extracted features, and models.
- The official prediction plus scoring path is comfortably below 30 seconds on the current four-thread CPU setting.

## Current decision state

The high-validation CREO candidate was precommitted and frozen before test. Its official full-test result is **387/467 = 82.86937901498929%**. The stable-direct candidate was not tested. Test results were not used to retrain, tune, or switch the checkpoint.

The frozen test prediction took 1.8439 seconds according to the official runner; scoring took 0.0361 seconds. The externally measured complete prediction-plus-scoring wall time was 2.7747 seconds. No online submission, GitHub Issue, or final report has been created.

## Course knowledge used

The page references below use PDF-reader page numbers.

- Lecture 1, pp. 38–40: embedding representations.
- Lecture 2, pp. 13–21: similarity and softmax scoring.
- Lecture 3, pp. 30–42: train/validation separation, loss design, and overfitting risk.
- Lecture 4, pp. 12–18: logits and cross-entropy classification.
- Lecture 5, pp. 40–59: verifier framing, answer extraction, and equivalence scoring.
- Lecture 5, pp. 62–68: chain-of-thought ablations and self-consistency.
- Lecture 6, pp. 18–23: self-consistency voting.
- Lecture 6, pp. 27–43: scorers and the distinction between confidence and correctness.
- Lecture 7, pp. 42–63: calibration and proper scoring rules.

These lectures motivated the design and diagnostics; they do not prove the empirical conclusions above.
