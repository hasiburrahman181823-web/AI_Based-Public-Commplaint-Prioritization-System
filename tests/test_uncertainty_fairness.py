"""Validation-only review selection and descriptive subgroup audits."""

from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ml.uncertainty import CONFIDENCE_DEFINITION, coverage_error_curve, load_review_policy, select_review_threshold
from ml_training.scripts.check_priority_fairness import audit_priority_fairness, export_fairness_report
from ml_training.scripts.select_review_threshold import build_review_policy, main as threshold_main


def test_curve_keeps_tied_confidence_rows_atomic_and_has_zero_and_reject_all():
    curve = coverage_error_curve([0.9, 0.9, 0.5, 0.1], [0, 1, 0, 1])
    assert [row["n_accepted"] for row in curve] == [0, 2, 3, 4, 4]
    assert curve[0]["threshold"] > 0.9
    assert curve[0]["error_rate"] is None
    assert curve[-1]["threshold"] == 0
    assert curve[1]["error_rate"] == 0.5
    assert curve[2]["coverage"] == 0.75


def test_maximum_feasible_coverage_is_selected_without_assuming_error_is_monotone():
    # The error rate first rises, then falls as more correct low-confidence
    # observations enter; every actual operating point must be considered.
    policy = select_review_threshold([0.9, 0.8, 0.7, 0.6], [0, 1, 0, 0], max_error_rate=0.3)
    assert policy["coverage"] == 1
    assert policy["threshold"] == 0.6  # higher threshold wins the threshold-zero tie
    assert policy["error_rate"] == 0.25
    assert policy["status"] == "selected"


def test_policy_rejects_all_future_confidences_when_no_feasible_cohort_exists():
    policy = select_review_threshold([0.8, 0.6], [1, 1], max_error_rate=0, min_accepted=1)
    assert policy["status"] == "reject_all"
    assert policy["threshold"] > 1
    assert policy["coverage"] == 0 and policy["error_rate"] is None
    assert not 1.0 >= policy["threshold"]


def test_insufficient_minimum_support_rejects_all_and_acceptance_is_greater_equal():
    assert select_review_threshold([1, 0], [0, 0], min_accepted=3)["status"] == "reject_all"
    curve = coverage_error_curve([1, 0], [0, 1])
    assert curve[1]["threshold"] == 1 and curve[1]["n_accepted"] == 1
    assert curve[-1]["threshold"] == 0 and curve[-1]["n_accepted"] == 2


@pytest.mark.parametrize("confidence,errors", [([], []), ([0.5], []), ([float("nan")], [0]), ([1.1], [0]), ([-0.1], [0]), ([0.5], [2]), ([[0.5]], [0])])
def test_invalid_curve_inputs_rejected(confidence, errors):
    with pytest.raises(ValueError):
        coverage_error_curve(confidence, errors)


@pytest.mark.parametrize("kwargs", [{"max_error_rate": -1}, {"max_error_rate": 1.1}, {"max_error_rate": float("nan")}, {"max_error_rate": True}, {"min_accepted": 0}, {"min_accepted": True}, {"min_accepted": 1.5}])
def test_invalid_selection_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        select_review_threshold([0.5], [0], **kwargs)


def write_csv(path, rows, columns=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def validation_files(tmp_path):
    category, severity = tmp_path / "category.joblib", tmp_path / "severity.joblib"
    category.write_bytes(b"trusted category test fixture")
    severity.write_bytes(b"trusted severity test fixture")
    fingerprints = {"category": hashlib.sha256(category.read_bytes()).hexdigest(),
                    "severity": hashlib.sha256(severity.read_bytes()).hexdigest()}
    rows = []
    for index, (confidence, wrong) in enumerate(((0.95, False), (0.8, False), (0.5, True))):
        rows.append({"id": f"v{index}", "original_id": f"g{index}", "split": "val",
                     "category_true": "water", "category_pred": "road" if wrong else "water",
                     "severity_true": "High", "severity_pred": "High",
                     "category_confidence": confidence, "severity_confidence": 0.99,
                     "category_model_sha256": fingerprints["category"], "severity_model_sha256": fingerprints["severity"]})
    return {"rows": rows, "csv": write_csv(tmp_path / "validation.csv", rows), "category": category,
            "severity": severity, "fingerprints": fingerprints, "root": tmp_path}


def test_policy_export_binds_models_and_writes_real_curve_png(validation_files):
    fixture = validation_files
    report = fixture["root"] / "results"
    path = fixture["root"] / "policy.json"
    policy = build_review_policy(fixture["csv"], fixture["category"], fixture["severity"], report, path,
                                 max_error_rate=0, min_accepted=1)
    assert policy["threshold"] == 0.8
    assert policy["n_accepted"] == 2
    assert policy["confidence_definition"] == CONFIDENCE_DEFINITION
    assert policy["classifier_fingerprints"] == fixture["fingerprints"]
    assert load_review_policy(path, fixture["fingerprints"]) == policy
    assert (report / "coverage_error_curve.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads((report / "selection.json").read_text()) == policy
    with pytest.raises(ValueError, match="fingerprints"):
        load_review_policy(path, {"category": "0" * 64, "severity": "1" * 64})
    with pytest.raises(ValueError, match="exists"):
        build_review_policy(fixture["csv"], fixture["category"], fixture["severity"], report, path)


@pytest.mark.parametrize("mutation,fragment", [
    (lambda rows: rows[0].update(split="test"), "split=val"),
    (lambda rows: rows[1].update(id="v0"), "unique"),
    (lambda rows: rows[0].update(category_confidence="1.2"), "between 0 and 1"),
    (lambda rows: rows[0].update(severity_confidence="nan"), "finite"),
    (lambda rows: rows[0].update(category_model_sha256="0" * 64), "does not match"),
])
def test_threshold_input_validation_and_model_provenance(validation_files, mutation, fragment):
    fixture = validation_files
    mutation(fixture["rows"])
    write_csv(fixture["csv"], fixture["rows"])
    with pytest.raises(ValueError, match=fragment):
        build_review_policy(fixture["csv"], fixture["category"], fixture["severity"],
                            fixture["root"] / "results", fixture["root"] / "policy.json")


def test_reject_all_saved_policy_is_loadable(validation_files):
    fixture = validation_files
    report = fixture["root"] / "results"
    path = fixture["root"] / "policy.json"
    policy = build_review_policy(fixture["csv"], fixture["category"], fixture["severity"], report, path,
                                 min_accepted=100)
    assert policy["status"] == "reject_all" and policy["threshold"] > 1
    assert load_review_policy(path)["coverage"] == 0


def test_threshold_cli_errors_are_clear_and_do_not_tune_on_test(validation_files, capsys):
    fixture = validation_files
    fixture["rows"][0]["split"] = "test"
    write_csv(fixture["csv"], fixture["rows"])
    result = threshold_main(["--predictions", str(fixture["csv"]), "--category-model", str(fixture["category"]),
                             "--severity-model", str(fixture["severity"]), "--output-dir", str(fixture["root"] / "results"),
                             "--policy-output", str(fixture["root"] / "policy.json")])
    assert result == 2
    assert "split=val" in capsys.readouterr().err


@pytest.mark.parametrize("change", [
    {"threshold": float("nan")}, {"threshold": -0.1}, {"threshold": True},
    {"threshold": 1.1, "status": "selected"}, {"threshold": 0.8, "status": "reject_all"},
    {"confidence_definition": "priority_probability"}, {"classifier_fingerprints": {}},
])
def test_saved_review_policy_rejects_incompatible_or_invalid_semantics(tmp_path, change):
    policy = {"kind": "priority_review_policy", "schema_version": 1, "status": "selected", "threshold": 0.8,
              "confidence_definition": CONFIDENCE_DEFINITION,
              "classifier_fingerprints": {"category": "0" * 64, "severity": "1" * 64}}
    policy.update(change)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError):
        load_review_policy(path)


def fairness_frame():
    return pd.DataFrame({
        "priority_score": [2, 4, 6, 8], "lang": ["bn", "bn", "banglish", "banglish"],
        "area_id": ["a", "a", "b", "b"], "category_pred": ["water"] * 4,
        "severity_pred": ["High"] * 4, "review_required": [False, True, True, True],
    })


def test_unlabeled_fairness_distributions_have_no_fabricated_error_rates():
    report = audit_priority_fairness(fairness_frame(), min_group_size=2)
    assert report["overall"]["error_rate"] is None
    assert report["overall"]["priority_mae"] is None
    language = report["disparities"]["language"]
    assert language["mean_score_gap"] == 4
    assert language["error_rate_gap"] is None
    assert language["review_rate_gap"] == 0.5
    pair = language["pairwise"][0]
    assert pair["group_a"] == "banglish" and pair["group_b"] == "bn"
    assert pair["mean_score_difference_a_minus_b"] == 4
    assert all(row["n_labeled_error"] == 0 for row in report["groups"])
    assert report["conditional_groups"]


def test_available_classification_truth_produces_supported_group_error_disparity():
    frame = fairness_frame()
    frame["category_true"] = ["water", "water", "road", "road"]
    frame["severity_true"] = ["High"] * 4
    report = audit_priority_fairness(frame, min_group_size=2)
    groups = {row["group"]: row for row in report["groups"] if row["dimension"] == "language"}
    assert groups["bn"]["error_rate"] == 0
    assert groups["banglish"]["error_rate"] == 1
    assert report["disparities"]["language"]["error_rate_gap"] == 1
    assert report["conditioning_columns"] == {"category": "category_true", "severity": "severity_true"}


def test_small_groups_and_unknown_areas_remain_descriptive_but_not_comparisons():
    frame = fairness_frame()
    frame.loc[3, "area_id"] = ""
    report = audit_priority_fairness(frame, min_group_size=3)
    assert report["disparities"]["language"]["status"] == "insufficient_groups"
    assert report["disparities"]["language"]["mean_score_gap"] is None
    assert any(row["group"] == "__unknown__" for row in report["groups"])
    assert report["overall"]["n"] == len(frame)


def test_aligned_priority_judgment_errors_and_under_prioritization():
    frame = fairness_frame()
    frame["priority_true"], frame["priority_pred"] = [5, 4, 1, 2], [3, 4, 1, 3]
    report = audit_priority_fairness(frame, min_group_size=2)
    assert report["overall"]["priority_mae"] == 0.75
    assert report["overall"]["under_prioritization_rate"] == 0.25
    assert report["disparities"]["language"]["priority_mae_gap"] == 0.5


def test_composition_conditional_summary_does_not_erase_raw_gap():
    frame = fairness_frame()
    frame["severity_pred"] = ["Low", "Low", "High", "High"]
    report = audit_priority_fairness(frame, min_group_size=2)
    assert report["disparities"]["language"]["mean_score_gap"] == 4
    assert all(row["status"] == "insufficient_groups" for row in report["conditional_disparities"])


@pytest.mark.parametrize("change", [
    lambda frame: frame.assign(priority_score=[1, 2, float("inf"), 4]),
    lambda frame: frame.assign(priority_score=[1, 2, "", 4]),
    lambda frame: frame.assign(review_required=[True, "maybe", True, False]),
    lambda frame: frame.assign(classification_error=[0, 2, 0, 1]),
    lambda frame: frame.assign(priority_true=[1, 2, 3, 4]),
    lambda frame: frame.assign(priority_true=[1, 2, 3, 4], priority_pred=[1, 2, 3, 8]),
    lambda frame: frame.assign(priority_true=[1, 2, 3, 4], priority_pred=[1, 2.5, 3, 4]),
])
def test_invalid_fairness_data_is_not_silently_dropped(change):
    with pytest.raises(ValueError):
        audit_priority_fairness(change(fairness_frame()), min_group_size=1)


def test_fairness_exports_are_json_safe_and_preserve_group_support(tmp_path):
    path = tmp_path / "predictions.csv"
    fairness_frame().to_csv(path, index=False)
    result = export_fairness_report(path, tmp_path / "audit", min_group_size=2)
    loaded = json.loads((tmp_path / "audit" / "fairness.json").read_text(encoding="utf-8"))
    assert result == loaded
    rows = list(csv.DictReader((tmp_path / "audit" / "groups.csv").open(encoding="utf-8", newline="")))
    assert len(rows) == 4
    assert (tmp_path / "audit" / "conditional_groups.csv").is_file()
    assert "Descriptive" in (tmp_path / "audit" / "fairness.md").read_text()
    with pytest.raises(ValueError, match="exists"):
        export_fairness_report(path, tmp_path / "audit", min_group_size=2)
