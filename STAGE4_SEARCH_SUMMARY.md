# Stage 4 CREO-3W Search Summary

## Scope

Stage 4 investigates **Improving challenger coverage in consensus reliability and evidence-based overturning**. The method is named **CREO-3W** because it reasons over three distinct answer-group proposals:

1. plurality consensus;
2. a learned candidate comparator;
3. a learned answer-group scorer.

Branches that select the same parsed answer group are merged before the final decision. The work is an original extension of CREO rather than a reproduction of another report's complete set-attention, pooling, or ensemble system.

Stage 4 used train references, frozen grouped folds, train OOF predictions, and one validation pass for the OOF top three. Public test inputs/references and per-question test outcomes were not read or used. Historical test aggregates were treated only as background and did not affect any choice.

## Architecture

### Candidate comparator

The candidate model projects within-question centred 1024-dimensional embeddings into 32 or 48 dimensions. Its interaction features include the candidate projection, four-candidate mean, signed and absolute deviations, multiplicative interactions, same-answer and different-answer projected means, cosine relations, missing-peer masks, and the 32 Stage 2 hand/grounding controls.

The nonlinear variant uses a tanh MLP with hidden dimension 16 or 32. Projection and MLP parameters are jointly trained in float64 and saved in float32. The objective combines hard-group multi-positive listwise loss, optional hard-group pairwise loss, and an all-group BCE term whose hard/easy contributions are balanced.

A linear low-rank model was retained only as an ablation. Its projected linear score does not add expressive power over a raw linear embedding scorer; its purpose is to test whether the nonlinear interactions actually help.

### Answer-group scorer

Candidates are grouped by `common.parse_answer`. Group features include support, plurality membership, candidate score statistics, projected group mean and dispersion, cosine cohesion/separation, arithmetic and grounding summaries, answer numeric features, and singleton/missing masks. A group tanh MLP is trained with multi-positive group-listwise loss and an optional pairwise term.

Candidate scores used to train each outer-fold group model are generated with nested cross-fitting. Consequently, the group scorer does not train on in-sample candidate scores from the outer evaluation fold.

### Reliability and three-way gate

All branch reliabilities are learned from OOF predictions with hard/easy-balanced logistic calibration. The final experiments compare each individual branch, a fixed reliability rule, and three multinomial logistic gates. Duplicate branch answer groups have their probability mass merged.

## Candidate ablation

| Candidate model | Train-hard OOF accuracy | Correct / 2114 |
|---|---:|---:|
| Stage 2 linear baseline | **81.6450%** | **1726** |
| Linear low-rank, d=32 | 79.1382% | 1673 |
| Interaction MLP | 78.4288% | 1658 |
| Interaction + pairwise | 78.9017% | 1668 |
| Interaction + pairwise + BCE 0.05 | 79.5164% | 1681 |
| Interaction + pairwise + BCE 0.10 | 78.9494% | 1669 |
| Interaction + BCE 0.05, hidden=32 | 79.5640% | 1682 |
| Linear low-rank, d=48, hidden setting 32 | **79.6586%** | **1684** |

Pairwise loss improved the initial interaction MLP by about 0.47 point. BCE 0.05 added a further 0.61 point, while BCE 0.10 was worse. Increasing projection dimension to 48 harmed the nonlinear interaction models. The best Stage 4 candidate remained 1.99 points below Stage 2, so it failed the candidate replacement rule.

The fact that the linear ablation remained the strongest Stage 4 candidate is negative evidence for the proposed nonlinear interaction representation under this training budget. It is not interpreted as evidence that the linear projection itself adds new expressiveness.

## Group and gate ablation

| Method | OOF mean ± SE | Correct / 2114 | Trap correct | Correct-consensus retention |
|---|---:|---:|---:|---:|
| Plurality | 65.9869% ± 0.9838 | 1395 | 0/719 | 100.00% |
| Candidate branch | 79.6586% ± 0.9706 | 1684 | 457/719 | 87.96% |
| Scalar group MLP | 78.1450% ± 0.9716 | 1652 | 422/719 | 88.17% |
| Projected group MLP | 78.1440% ± 1.1370 | 1652 | 416/719 | 88.60% |
| Combined group MLP | 78.3337% ± 1.1847 | 1656 | 427/719 | 88.10% |
| Combined group MLP + pairwise | 78.4756% ± 1.0954 | 1659 | 429/719 | 88.17% |
| Fixed three-way reliability | **80.4149% ± 0.8659** | **1700** | 400/719 | **93.19%** |
| Logistic gate, L2=0.01 | 80.1789% ± 0.8556 | 1695 | 411/719 | 92.04% |
| Logistic gate, L2=0.1 | 79.8950% ± 0.8216 | 1689 | 408/719 | 91.83% |
| Logistic gate, L2=1.0 | 80.3202% ± 0.9287 | 1698 | 400/719 | 93.05% |

The fixed reliability rule is the Stage 4 OOF research leader. Its five fold correct counts are `[345, 337, 349, 342, 327]`.

- Relative to Stage 2: `[-5, +2, -5, -11, -7]`.
- Relative to Stage 3 stable-direct: `[-6, -3, -10, -12, -5]`.

It improves only one fold over Stage 2 and no folds over stable-direct. It therefore fails both the 0.75-point improvement requirement and the 4/5-fold stability requirement.

## Paired OOF interpretation

Against Stage 2 on train-hard:

- both correct: 1606;
- Stage-2-only: 120;
- Stage-4-only: 94;
- both wrong: 294;
- net Stage 4 difference: -26 questions.

CREO-3W does improve consensus protection: on correct-plurality groups it gains 61 Stage-4-only cases against 24 Stage-2-only cases. However, it loses heavily on the exact strata Stage 4 intended to improve:

- wrong-plurality traps: 33 Stage-4-only versus 96 Stage-2-only;
- single-correct groups: 45 Stage-4-only versus 88 Stage-2-only.

This supports the central failure diagnosis: reliability fusion can reduce unsafe overturns, but it cannot recover coverage when both learned challengers are weaker than the frozen Stage 2 comparator.

## Validation consistency check

Validation had already been observed in earlier stages, so it was not used to reverse the OOF conclusion. Only the OOF top three Stage 4 methods were evaluated once.

| Method | Validation all | Validation hard | Wrong-plurality traps | Single-correct | Correct consensus |
|---|---:|---:|---:|---:|---:|
| Fixed reliability | 737/798 (92.3559%) | 294/355 (82.8169%) | 62/110 | 67/114 | 232/245 |
| Gate L2=1.0 | 736/798 (92.2306%) | 293/355 (82.5352%) | 61/110 | 66/114 | 232/245 |
| Gate L2=0.01 | 739/798 (92.6065%) | 296/355 (83.3803%) | 62/110 | 64/114 | 234/245 |

The minor validation ordering differences do not change the OOF-primary conclusion. No feature, threshold, or model was added after viewing these results.

## Calibration

For the fixed-reliability OOF research leader:

| Scope | NLL | ECE | AURC |
|---|---:|---:|---:|
| All train OOF | 0.19772 | 0.02719 | 0.01187 |
| Train-hard OOF | 0.41711 | 0.03537 | 0.07696 |

Five probability-ordered accuracy buckets and ten-bin calibration details are stored in `code/runs/stage4/calibration.json`. These are calibrated model scores, not guaranteed probabilities of correctness.

## Engineering checks

- 30/30 Stage 1–4 unit and regression tests pass.
- Candidate and answer-group permutation equivariance pass.
- Multi-positive listwise, pairwise, BCE, MLP, and interaction-projection finite-difference checks pass.
- Arbitrary IDs, arbitrary bank length, empty traces, and all-unparseable candidates pass.
- Missing checkpoint errors are explicit.
- Two frozen validation prediction processes produced identical choices.
- Float32 checkpoint and float64 training-time validation choices differ on 0/798 questions.
- Maximum recorded official validation prediction time: 2.5764 seconds; scoring: 0.0576 seconds.
- Provisional checkpoint assets: 215,846 bytes, below 16 MiB.
- Validation-only peak working set: approximately 154.3 MiB.
- Total bounded search runtime: approximately 503 seconds.

## Selection decision

Stage 4 **does not satisfy the predeclared replacement conditions**. The saved `provisional_bundle` reproduces the raw OOF Stage 4 research leader for analysis, but the formal decision is:

> retain the frozen pre-Stage-4 model.

Stage 4 has not run test and cannot be claimed to improve formal test accuracy. A genuinely new blind audit bank would be required to confirm any generalization claim.

## Course knowledge used

PDF-reader page numbers:

- Lecture 1, pp. 38–40: embedding representations and projected features.
- Lecture 2, pp. 13–21: similarity, cosine relations, and softmax-normalized comparisons.
- Lecture 3, pp. 30–42: grouped train/validation separation, loss design, and overfitting control.
- Lecture 4, pp. 12–18: logits, BCE/cross-entropy, and classification objectives.
- Lecture 5, pp. 40–59: verifier framing, answer extraction, and answer equivalence.
- Lecture 5, pp. 62–68 and Lecture 6, pp. 18–23: self-consistency and voting.
- Lecture 6, pp. 27–43: scorer outputs and confidence not implying correctness.
- Lecture 7, pp. 42–63: calibration, NLL, ECE, and proper scoring rules.

The lectures motivate the methodology; they do not prove the experimental conclusions.
