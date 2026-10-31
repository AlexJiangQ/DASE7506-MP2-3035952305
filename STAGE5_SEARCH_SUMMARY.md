# Stage 5 CREO-RR Search Summary

## Research position

Stage 5 studies **CREO-RR: Regime-Conditioned Residual Revision**, a selective residual revision layer over the frozen Stage 3 high-validation CREO anchor. It is not a new general-purpose ensemble and does not reproduce set-attention or large fusion searches.

The research question is whether a proposal from an existing branch can correct a remaining anchor error while an explicit harmful-revision cost protects answers the anchor already gets right.

Stage 4 artifacts were unavailable. With explicit user approval, every Stage 4 source was removed before implementation. Stage 5 development and selection use only train references, strict train-hard OOF predictions, and validation. After the study concluded and the canonical bundle was frozen with `apply_residual=false`, an explicitly authorised full-test reproduction was run. It evaluates the unchanged Stage 3 anchor, not the residual candidate.

## Final method

The frozen anchor is the Stage 3 high-validation CREO. Train uses its strict OOF answer-group choices, group scores, reliability scores, and gate scores; validation uses its full-train frozen checkpoint.

Initial proposal sources were:

1. plurality answer group;
2. frozen Stage 2 candidate comparator;
3. Stage 3 learned answer-group scorer.

Proposal coverage showed that the Stage 3 group source contributed no uniquely correctable anchor errors, so the predeclared independent-coverage rule removed it. The canonical residual study therefore uses deduplicated plurality and Stage 2 proposals.

Each anchor–proposal pair is labelled beneficial, harmful, or neutral using answer groups derived with `common.parse_answer`. The selected model is a shared multinomial logistic model with 32 consensus-and-score-difference features, balanced-capped class weights, no harm auxiliary term, harm cost `2.0`, revision threshold `0.10`, and OOF-fitted temperature calibration. The hidden-dimension-16 MLP was not used: the limiting issue was revision safety/generalization rather than clear logistic under-capacity.

## Proposal coverage

The Stage 3 anchor is wrong on 394/2,114 train-hard questions. Across the initial sources, at least one proposal can correct 173 of those questions, giving an oracle upper bound of 1,893/2,114 (`89.55%`).

| Source | Different from anchor | Beneficial upper bound | Harmful upper bound | Unique beneficial |
|---|---:|---:|---:|---:|
| Plurality | 721 | 159 | 484 | 105 |
| Stage 2 | 109 | 52 | 46 | 11 |
| Stage 3 group | 96 | 44 | 43 | 0 — removed |

After deduplication, the retained plurality/Stage 2 set yields 2,782 residual examples: 173 beneficial, 502 harmful, and 2,107 neutral.

## Grouped OOF result

| Method | Correct | Accuracy / fold mean ± SE | Fold delta vs Stage 3 | Fold delta vs Stage 2 |
|---|---:|---:|---|---|
| Stage 3 anchor | 1,720/2,114 | 81.3623%; 81.3611 ± 1.2858 | baseline | `[-2,-4,+6,-4,-2]` relative history |
| CREO-RR safe candidate | **1,723/2,114** | **81.5043%; 81.5030 ± 1.2413** | **`[+1,+1,0,0,+1]`** | `[-1,-3,+6,-4,-1]` |

CREO-RR makes six OOF revisions: four beneficial, one harmful, and one neutral. Net correction is `+3`; revision precision is `66.67%` and coverage is `0.2838%`. It passes all predeclared OOF safety conditions and is non-decreasing in 5/5 folds.

Correct-plurality retention rises from 1,236/1,395 (`88.60%`) to 1,239/1,395 (`88.82%`). Wrong-plurality recovery stays 484/719. On single-correct-candidate groups, the safe candidate reaches 425/684 versus the anchor's 423/684.

All six selected revisions occur in the `other` vote regime; the 3-1, 2-2, 2-1-1, and 1-1-1-1 regimes are unchanged under the final safety threshold. This concentration is evidence of very selective coverage, not a claim that the regime is intrinsically reliable.

Stage 4 fold differences are unavailable because Stage 4 did not exist and was removed with user approval.

## Feature ablation

The following results use the Phase B fixed utility (`c_harm=2`, `tau=0.05`) before final safety selection.

| Feature set | Features | OOF correct | Beneficial / harmful | Revision precision |
|---|---:|---:|---:|---:|
| Consensus + score differences | 32 | **1,723** | 4 / 1 | **66.67%** |
| + reliability differences | 40 | 1,723 | 4 / 1 | 66.67% |
| + embedding group differences | 52 | 1,719 | 4 / 5 | 33.33% |
| + arithmetic/grounding | 68 | 1,722 | 5 / 3 | 50.00% |
| Full residual features | 80 | 1,722 | 7 / 5 | 50.00% |

Reliability features added no accuracy, so the simpler 32-feature set was retained under the `<0.25 point` simplicity rule. Embedding and arithmetic blocks increased harmful revisions instead of providing safer corrections.

## Safety utility and model ablation

Lower harm cost and threshold produced higher raw OOF accuracy but failed the `65%` precision rule. For example, `c_harm=1.5, tau=0` reaches 1,728 correct but revises 43 questions with only `55.81%` precision. The selected `c_harm=2, tau=0.10` makes six revisions at `66.67%` precision.

At the aggressive Phase D setting, uniform/balanced weighting and the harm auxiliary reached 1,726–1,728 correct but only `53.5–55.8%` precision. The harm auxiliary did not establish a stable safety advantage, so it was not retained.

The diagnostic precision–coverage curve illustrates the trade-off: at threshold `0.025`, coverage is `0.615%` with `61.54%` precision; at `0.10`, coverage is `0.284%` with `66.67%` precision; at `0.20`, only one revision remains and is beneficial.

## Calibration

Cross-fitted temperature calibration reduces multiclass OOF NLL from `0.26201` to `0.25729`. The fixed validation temperature is `0.85067`. OOF ECE is `0.02430`; beneficial binary NLL is `0.13763`, harmful binary NLL is `0.18143`, and the reported AURC is `0.96198`. The high AURC reflects the severe rarity of beneficial residual examples and is not presented as a correctness probability.

## Validation

Only the three OOF-shortlisted configurations were evaluated once. All three produce zero validation revisions. Therefore the canonical output is exactly the frozen anchor:

- validation-all: **755/798 = 94.6115%**;
- validation-hard: **312/355 = 87.8873%**;
- wrong-plurality traps: **77/110**;
- correct-plurality nontraps: **235/245**;
- single-correct candidates: **81/117**;
- beneficial/harmful/neutral revisions: **0/0/0**;
- paired accuracy difference: `0.00` points, bootstrap 95% interval `[0.00, 0.00]`.

No Stage 5 model satisfies the validation requirement of at least two additional correct questions. The provisional bundle is consequently configured with `apply_residual=false`, retaining Stage 3.

## Engineering and protocol audit

- Canonical run: `code/runs/stage5/bugfix2`.
- 29/29 existing and Stage 5 tests pass.
- Five fresh four-thread validation processes are deterministic; prediction median is `3.3213 s`, maximum `3.6211 s`.
- Canonical bundle size is 201,156 bytes, below 16 MiB.
- Stage 1–3 protected artifacts have zero hash mismatches.
- Fold coverage is exact and OOF probabilities are finite.
- Canonical training/search wall time is about 38 seconds; two earlier full runs are preserved for protocol transparency.

Two bugs were recorded and fully rerun in new, non-overwriting directories:

1. `bugfix1` corrected the single-correct reporting definition; it did not affect training or selection.
2. `bugfix2` enforced the predeclared independent proposal-coverage pruning and is the canonical run.

## Decision

Stage 5 does **not** replace Stage 3. It provides evidence that residual revisions can be precise under strict OOF, but the usable coverage is extremely small and the signal does not produce a validation improvement. The correct action under the frozen selection rule is to retain the Stage 3 high-validation anchor.

The frozen anchor-retained wrapper reproduces **387/467 = 82.86937901498929%**, exactly the existing Stage 3 anchor result. Official prediction took `2.1873 s`, scoring took `0.0426 s`, and complete external wall time was `3.1134 s`. The experimental residual revision remained disabled and was not evaluated on test.

**Stage 5 residual revision 未运行 test；canonical anchor-retained wrapper 仅复现既有 Stage 3 结果，不能声称提高正式 test accuracy。**

## Course knowledge used

PDF-reader page numbers:

- Lecture 1, pp. 38–40: embedding representations.
- Lecture 2, pp. 13–21: similarity and softmax.
- Lecture 3, pp. 30–42: train/validation separation, loss, and overfitting.
- Lecture 4, pp. 12–18: logits and cross-entropy.
- Lecture 5, pp. 40–59: verifier framing, answer extraction, and equivalence scoring.
- Lecture 5, pp. 62–68 and Lecture 6, pp. 18–23: self-consistency.
- Lecture 6, pp. 27–43: scorers and confidence not implying correctness.
- Lecture 7, pp. 42–63: calibration and proper scoring rules.

These sources motivated the method and diagnostics; they do not prove the empirical findings.
