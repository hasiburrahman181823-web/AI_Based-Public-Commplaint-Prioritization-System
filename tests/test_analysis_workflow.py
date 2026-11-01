"""Synthetic end-to-end validation-policy-analysis-fairness workflow."""

import csv
import json
from pathlib import Path

import pytest

from ml.analysis import PriorityAnalyzer
from ml.classifier_runtime import ClassifierRuntime, file_sha256
from ml_training.scripts import analyze_complaints as batch
from ml_training.scripts.check_priority_fairness import export_fairness_report
from ml_training.scripts.export_validation_predictions import export_validation_predictions
from ml_training.scripts.select_review_threshold import build_review_policy
from test_complaint_analysis import make_classifier


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "artifacts"
    category = make_classifier(root, "category")
    severity = make_classifier(root, "severity")
    val = tmp_path / "val.csv"
    rows = [{"id": str(index), "original_id": f"group-{index}", "split": "val", "text": text,
             "lang": lang, "category": target, "severity": level,
             "location": place, "timestamp": ""}
            for index, (text, lang, target, level, place) in enumerate([
                ("water tap pipe dry", "en", "Water", "Low", "Mirpur 10"),
                ("medical clinic care", "en", "Healthcare", "High", "Mohammadpur"),
                ("pani nai ekhane", "banglish", "Water", "Low", "Mirpur 10"),
                ("হাসপাতালে সমস্যা", "bn", "Healthcare", "High", "Mohammadpur"),
            ])]
    write_csv(val, rows)
    (root / "experiment.json").write_text(json.dumps({"experiment_kind": "calibrated_classifiers",
        "inputs": [{"path": str(val), "sha256": file_sha256(val)}]}), encoding="utf-8")
    return {"root": root, "category": category, "severity": severity, "val": val, "rows": rows}


def test_complete_pipeline_binds_resources_and_produces_auditable_outputs(source, tmp_path):
    joint = tmp_path / "uncertainty" / "validation_joint.csv"
    export_validation_predictions(source["val"], source["category"], source["severity"], joint)
    policy_path = source["root"] / "review_policy.json"
    policy = build_review_policy(joint, source["category"], source["severity"], joint.parent,
                                 policy_path, max_error_rate=1, min_accepted=1)
    assert policy["status"] == "selected"
    assert set(policy["classifier_resource_fingerprints"]) == {"category", "severity"}
    assert (joint.parent / "coverage_error_curve.png").read_bytes().startswith(b"\x89PNG")
    analyzer = PriorityAnalyzer(category_model=source["category"], severity_model=source["severity"],
                                review_policy=policy_path, duplicate_model=None, ranker=None)
    output, tabular = tmp_path / "analysis.jsonl", tmp_path / "analysis.csv"
    summary = batch.analyze_batch(analyzer, source["val"], output, tabular)
    results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["n"] == len(source["rows"]) == len(results)
    assert all(set(result) >= {"category", "severity", "priority_score", "priority_label", "confidence", "explanation", "review_required", "duplicate_candidates"} for result in results)
    assert all(result["provenance"]["classifier_resource_fingerprints"] == policy["classifier_resource_fingerprints"] for result in results)
    report = export_fairness_report(tabular, tmp_path / "fairness", min_group_size=1)
    assert report["n"] == len(results)
    assert (tmp_path / "fairness" / "fairness.json").is_file()
    assert (tmp_path / "fairness" / "groups.csv").is_file()
    with tabular.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["category_true"] for row in rows] == [row["category"] for row in source["rows"]]
    assert all("priority_pred" not in row for row in rows)


def test_modified_validation_resource_hash_cannot_create_an_automatic_policy(source, tmp_path):
    joint = tmp_path / "joint.csv"
    export_validation_predictions(source["val"], source["category"], source["severity"], joint)
    with joint.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["category_resources_sha256"] = "0" * 64
    write_csv(joint, rows)
    with pytest.raises(ValueError, match="resources"):
        build_review_policy(joint, source["category"], source["severity"], tmp_path / "curve", tmp_path / "policy.json", 1, 1)
    assert not (tmp_path / "policy.json").exists()


def test_batch_no_models_is_explicit_review_and_no_ground_truth_leak(source, tmp_path):
    analyzer = PriorityAnalyzer(category_model=None, severity_model=None, review_policy=None,
                                duplicate_model=None, ranker=None)
    source["rows"][0]["text"] = "fire beside school for three days"
    write_csv(source["val"], source["rows"])
    output = tmp_path / "fallback.jsonl"
    batch.analyze_batch(analyzer, source["val"], output)
    first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first["category"] is None and first["severity"] is None and first["confidence"] is None
    assert first["review_status"] == "needs_human_review"
    assert first["explanation"]["matched_keywords"]["life_risk"]
    with pytest.raises(ValueError, match="exists"):
        batch.analyze_batch(analyzer, source["val"], output)
    with pytest.raises(ValueError, match="distinct"):
        batch.analyze_batch(analyzer, source["val"], source["val"], overwrite=True)


def test_single_and_batch_cli_preserve_json_and_fail_cleanly(source, tmp_path, capsys):
    assert batch.main(["--text", "পানি নেই", "--category-model", str(tmp_path / "missing-category"),
                       "--severity-model", str(tmp_path / "missing-severity")]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["review_required"] and response["confidence"] is None
    assert batch.main(["--input", str(source["val"]), "--output", str(source["val"])]) == 2
    assert "ERROR:" in capsys.readouterr().err


def test_reference_validation_export_rejects_test_input_by_hash_and_split(source, tmp_path):
    source["rows"][0]["split"] = "test"
    write_csv(source["val"], source["rows"])
    with pytest.raises(ValueError, match="val"):
        export_validation_predictions(source["val"], source["category"], source["severity"], tmp_path / "bad.csv", True)
