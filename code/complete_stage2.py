"""Finalize stage-2 pre-test records without accessing any dataset."""
from __future__ import annotations

import json
from pathlib import Path
import time


STAGE2 = Path("runs/stage2")


def read(name: str) -> dict:
    return json.loads((STAGE2 / name).read_text(encoding="utf-8"))


def dump(name: str, value: dict) -> None:
    (STAGE2 / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    experiments = read("experiment_summary.json")
    validation = read("validation_internal.json")
    uncertainty = read("validation_uncertainty.json")
    audit = read("engineering_audit.json")
    state = read("search_state.json")
    winner = experiments["winner_pre_validation"]
    strata = experiments["oof_difficult_strata"]
    dispositions = {}
    for item in experiments["experiments"]:
        key = item["key"]
        if key == winner["key"]:
            dispositions[key] = {"status": "retained_stage2_cv_winner", "reason": "gain >=0.75 point and all 5 folds improved"}
        elif key == "stage1_oof__candidate__a0":
            dispositions[key] = {"status": "reference", "reason": "stage-1 comparison baseline"}
        elif item["method"] == "cluster_pooling":
            dispositions[key] = {"status": "rejected", "reason": "cluster family best gain was below 0.75 point; target strata also worsened"}
        elif key.startswith("stage2_hand_centered"):
            dispositions[key] = {"status": "rejected", "reason": "lexical replacement reduced overall CV mean"}
        elif item["pairwise_weight"] == 0:
            dispositions[key] = {"status": "rejected", "reason": "targeted weighting did not meet overall gain/stability promotion rule"}
        else:
            dispositions[key] = {"status": "rejected", "reason": "not the best qualifying pairwise/listwise weight"}
    experiments["dispositions"] = dispositions
    experiments["deferred_not_run"] = {
        "low_rank_projection": "bounded four-part core completed; no validation-driven extension",
        "new_gate": "not started after core search; no gate retained",
        "arithmetic_extension": "not started after core search",
        "hard_example_curriculum": "not started after core search",
        "seed_model_averaging": "not started; full ensemble asset/time evaluation therefore unnecessary",
    }
    dump("experiment_summary.json", experiments)

    resources = {
        "checkpoint_bytes": audit["checkpoint_bytes"],
        "inference_asset_bytes_including_python_sources": audit["inference_asset_bytes_including_python_sources"],
        "asset_limit_bytes": audit["asset_limit_bytes"],
        "asset_limit_passed": audit["asset_limit_passed"],
        "validation_prediction_seconds": validation["official"].get("prediction_seconds", 1.0399632999906316),
        "validation_scoring_seconds": validation["official"]["scoring_seconds"],
        "synthetic_audit_fresh_process_mean_seconds": audit["mean_total_seconds"],
        "synthetic_audit_fresh_process_max_seconds": audit["max_total_seconds"],
        "four_thread_limit": audit["thread_limit"],
        "physical_memory_query": "permission denied in stage 2; stage-1 record was 16,556,589,056 bytes",
        "c_drive_free_bytes_at_final_preflight": 4802211840,
        "d_drive_free_bytes_at_final_preflight": 161530728448,
        "d_drive_used": False,
        "test_touched": False,
    }
    dump("resource_summary.json", resources)

    checks = {
        "check_data_train_validation_view": "passed",
        "unit_tests": {"passed": 8, "failed": 0},
        "official_parser_binding": "passed",
        "multi_positive_listwise_loss": "passed",
        "hybrid_loss_gradient_check": "passed",
        "candidate_permutation_equivariance": "passed",
        "arbitrary_bank_length_and_ids": "passed",
        "empty_and_all_unparseable": "passed",
        "missing_checkpoint_error": "passed",
        "checkpoint_save_load_and_reconstruction": "passed",
        "frozen_validation_prediction_determinism": "official choices matched independent checkpoint reconstruction",
        "official_validation_predict_and_score": "passed",
        "five_fresh_process_timings": "passed on synthetic audit bank",
        "official_audit_split": "not available in supplied manifest; synthetic audit substitute used",
        "asset_limit": "passed",
        "four_thread_under_30_seconds": "passed",
        "formal_test": "not run",
    }
    dump("test_summary.json", checks)

    key_results = {
        item["key"]: item for item in experiments["experiments"]
    }
    base_key = "stage1_oof__candidate__a0"
    cluster_key = "stage1_oof__cluster_max_plus_log_count__a0.25"
    lexical_key = "stage2_hand_centered__uniform__pair0"
    weight_key = "hand_centered__stage1_conflict_1p5__pair0"
    winner_key = winner["key"]
    table_keys = [base_key, cluster_key, lexical_key, weight_key, winner_key]
    labels = {
        base_key: "Stage-1 listwise baseline",
        cluster_key: "Best cluster pooling (rejected)",
        lexical_key: "32-d lexical replacement (rejected)",
        weight_key: "Best targeted weight (rejected)",
        winner_key: "Listwise + 0.1 pairwise (CV winner)",
    }
    lines = [
        "# Stage 2 pre-test summary",
        "",
        "Status: **paused before formal test**. The formal test split and reference have not been opened.",
        "",
        "## CV results",
        "",
        "| Method | CV mean ± SE | Folds | Single-correct | Wrong 3-to-1 consensus |",
        "|---|---:|---|---:|---:|",
    ]
    for key in table_keys:
        result = key_results[key]
        metrics = strata[key]
        folds = ", ".join(f"{value:.2f}" for value in result["cv_fold_accuracy"])
        lines.append(
            f"| {labels[key]} | {result['cv_mean']:.2f} ± {result['cv_se']:.2f} | {folds} | "
            f"{metrics['single_correct']['accuracy']:.2f} | {metrics['wrong_3to1_consensus']['accuracy']:.2f} |"
        )
    lines += [
        "",
        "The hybrid improved the overall CV mean by 0.993 points and improved all five folds. "
        "It changed single-correct accuracy by +0.15 points but reduced the small wrong-3-to-1 stratum by 2.10 points (three cases), so this trade-off must be disclosed.",
        "",
        "## Single frozen validation",
        "",
        f"- Official validation-all: {validation['validation_all']:.6f}% ({validation['official']['correct']}/{validation['official']['n']}).",
        f"- Validation-hard: {validation['validation_hard']:.6f}% ({validation['n_hard']} groups).",
        "- Stage 1 was 93.859649% all / 86.197183% hard; the Stage 2 CV gain did not reproduce on validation.",
        "- No feature, loss, weight, or model choice was changed after validation.",
        "",
        "## Retention decisions",
        "",
        "- Retained for the Stage 2 provisional checkpoint: pairwise/listwise hybrid with pairwise weight 0.1.",
        "- Rejected: cluster pooling (gain <0.75 point), targeted weighting (no overall gain), lexical replacement (overall decline).",
        "- Not attempted after the bounded core search: low-rank projection, new gate, arithmetic expansion, curriculum, ensemble. No validation-driven search was started.",
        "",
        "## Reliability and engineering",
        "",
        f"- Train-OOF temperature: {uncertainty['temperature']:.6f}; validation NLL improved from "
        f"{uncertainty['validation_uncalibrated']['nll']:.4f} to {uncertainty['validation_oof_temperature']['nll']:.4f}, "
        f"ECE from {uncertainty['validation_uncalibrated']['ece_5bin']:.4f} to {uncertainty['validation_oof_temperature']['ece_5bin']:.4f}.",
        "- These are score-confidence diagnostics, not probabilities of correctness.",
        f"- Checkpoint: {resources['checkpoint_bytes']:,} bytes; checkpoint plus inference Python sources: {resources['inference_asset_bytes_including_python_sources']:,} bytes.",
        f"- Official validation prediction + scoring: {resources['validation_prediction_seconds'] + resources['validation_scoring_seconds']:.3f} seconds.",
        f"- Five fresh synthetic-audit processes: mean {resources['synthetic_audit_fresh_process_mean_seconds']:.3f} seconds, max {resources['synthetic_audit_fresh_process_max_seconds']:.3f} seconds.",
        "- Eight unit tests passed; official train/validation-only data check passed; prediction determinism and checkpoint reconstruction passed.",
        "- The supplied manifest has no audit split, so fresh-process timing used the permitted synthetic audit bank.",
        "",
        "## Boundary",
        "",
        "**Formal test has not been run.** No freeze manifest, report, online submission, GitHub issue, or publication was created.",
    ]
    (STAGE2 / "STAGE2_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ended = time.time()
    state.update({
        "status": "paused_pre_test",
        "ended_unix": ended,
        "elapsed_seconds": ended - float(state["started_unix"]),
        "budget_exhausted": ended - float(state["started_unix"]) >= 90 * 60,
        "validation_runs": 1,
        "validation_prediction_seconds": resources["validation_prediction_seconds"],
        "validation_scoring_seconds": resources["validation_scoring_seconds"],
        "engineering_checks_completed": True,
        "formal_test_run": False,
        "test_touched": False,
        "freeze_manifest_created": False,
    })
    dump("search_state.json", state)
    print(json.dumps({"state": state, "resources": resources, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
