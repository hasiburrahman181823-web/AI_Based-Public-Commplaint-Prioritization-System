"""Validation-only classifier exports with exact-model/input provenance."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import pytest

from ml.normalize import DEFAULT_VARIANTS_PATH
from ml_training.modeling import build_text_estimator, calibrate_prefit
from ml_training.scripts import export_validation_predictions as exporting


COLUMNS = ("id", "original_id", "split", "text", "lang", "category", "severity")


def write_csv(path, records, columns=COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def experiment(tmp_path):
    artifacts = tmp_path / "artifacts"
    resources = artifacts / "resources"
    resources.mkdir(parents=True)
    variants = resources / "banglish_variants.json"
    shutil.copyfile(DEFAULT_VARIANTS_PATH, variants)
    texts = ["water minor interruption", "water small interruption", "water serious interruption", "road major damage", "road severe damage", "road minor damage"]
    targets = {
        "category": ["Water", "Water", "Water", "Road & Transport", "Road & Transport", "Road & Transport"],
        "severity": ["Low", "Low", "High", "High", "High", "Low"],
    }
    models = {}
    for target in ("category", "severity"):
        base = build_text_estimator("word_lr", seed=12).fit(texts, targets[target])
        calibrator = calibrate_prefit(base, ["water problem", "road problem"], ["Water", "Road & Transport"] if target == "category" else ["Low", "High"])
        path = artifacts / target / "word_lr.joblib"
        path.parent.mkdir()
        labels = ["Road & Transport", "Water"] if target == "category" else ["Low", "High"]
        joblib.dump({
            "kind": "banglapsg_classifier", "schema_version": 1, "target": target,
            "model_name": "word_lr", "feature_type": "tfidf", "estimator": calibrator,
            "labels": labels, "artifact_root_relative": "..",
            "variants_file": "resources/banglish_variants.json",
            "normalization_dictionary_sha256": sha(variants),
            "calibration": "sigmoid on held-out synthetic validation",
        }, path)
        models[target] = path
    rows = [
        {"id": "0001", "original_id": "g1", "split": "val", "text": "water minor interruption", "lang": "en", "category": "Water", "severity": "Low"},
        {"id": "0002", "original_id": "g2", "split": "val", "text": "road severe damage", "lang": "en", "category": "Road & Transport", "severity": "High"},
        {"id": "0003", "original_id": "g3", "split": "val", "text": "পানি নেই", "lang": "bn", "category": "Water", "severity": "High"},
        {"id": "0004", "original_id": "g4", "split": "val", "text": "pany nei", "lang": "banglish", "category": "Water", "severity": "Low"},
    ]
    source = write_csv(tmp_path / "data" / "val.csv", rows)
    manifest = artifacts / "experiment.json"
    payload = {"schema_version": 1, "experiment_kind": "calibrated_classifiers", "inputs": [{"path": str(source), "sha256": sha(source)}]}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return {"source": source, "rows": rows, "models": models, "manifest": manifest,
            "manifest_payload": payload, "output": tmp_path / "results" / "validation_joint.csv"}


def export(experiment, **kwargs):
    return exporting.export_validation_predictions(
        experiment["source"], experiment["models"]["category"], experiment["models"]["severity"],
        experiment["output"], **kwargs,
    )


def refresh_manifest(experiment):
    payload = experiment["manifest_payload"]
    payload["inputs"][0]["sha256"] = sha(experiment["source"])
    experiment["manifest"].write_text(json.dumps(payload), encoding="utf-8")


def test_exports_real_calibrated_models_with_validation_ids_truth_and_probabilities(experiment):
    result = export(experiment)
    assert isinstance(result, dict)
    assert result["rows"] == len(experiment["rows"])
    assert result["source_validation_sha256"] == sha(experiment["source"])
    assert result["validation_provenance"] == {"category": "saved_validation_input_hash_match", "severity": "saved_validation_input_hash_match"}
    assert result["warnings"] == []
    records = read_csv(experiment["output"])
    assert len(records) == len(experiment["rows"])
    assert [row["id"] for row in records] == [row["id"] for row in experiment["rows"]]
    assert all(row["split"] == "val" for row in records)
    for actual, source in zip(records, experiment["rows"]):
        assert actual["original_id"] == source["original_id"]
        for target in ("category", "severity"):
            assert actual[target + "_true"] == source[target]
            assert actual[target + "_pred"]
            assert actual[target + "_model_sha256"] == sha(experiment["models"][target])
            assert len(actual[target + "_resources_sha256"]) == 64
            assert actual[target + "_validation_provenance"] == "saved_validation_input_hash_match"
            probabilities = json.loads(actual[target + "_probabilities"])
            assert sum(probabilities.values()) == pytest.approx(1)
            assert 0 <= float(actual[target + "_confidence"]) <= 1
            assert float(actual[target + "_confidence"]) == pytest.approx(max(probabilities.values()))


def test_path_named_val_accepts_source_without_an_explicit_split_column(experiment):
    write_csv(experiment["source"], experiment["rows"], [name for name in COLUMNS if name != "split"])
    refresh_manifest(experiment)
    export(experiment)
    assert all(row["split"] == "val" for row in read_csv(experiment["output"]))


@pytest.mark.parametrize("split", ["train", "test", "natural_test"])
def test_nonvalidation_rows_are_rejected_even_with_provenance_bypass(experiment, split):
    experiment["rows"][0]["split"] = split
    write_csv(experiment["source"], experiment["rows"])
    refresh_manifest(experiment)
    with pytest.raises(ValueError, match="val|validation"):
        export(experiment, allow_unverified_validation=True)
    assert not experiment["output"].exists()


@pytest.mark.parametrize("field", ["id", "original_id", "text", "lang", "category", "severity"])
def test_blank_required_input_values_fail_before_export(experiment, field):
    experiment["rows"][0][field] = " "
    write_csv(experiment["source"], experiment["rows"])
    refresh_manifest(experiment)
    with pytest.raises(ValueError):
        export(experiment)
    assert not experiment["output"].exists()


def test_duplicate_ids_are_rejected_and_no_input_rows_are_silently_dropped(experiment):
    experiment["rows"][1]["id"] = experiment["rows"][0]["id"]
    write_csv(experiment["source"], experiment["rows"])
    refresh_manifest(experiment)
    with pytest.raises(ValueError, match="unique|duplicate"):
        export(experiment)


@pytest.mark.parametrize("field,value", [("category", "Unseen category"), ("severity", "Critical"), ("lang", "guessed")])
def test_unknown_target_labels_and_languages_fail(experiment, field, value):
    experiment["rows"][0][field] = value
    write_csv(experiment["source"], experiment["rows"])
    refresh_manifest(experiment)
    with pytest.raises(ValueError):
        export(experiment)


def test_mismatched_validation_file_fingerprint_needs_explicit_override(experiment):
    experiment["rows"][0]["text"] += " changed"
    write_csv(experiment["source"], experiment["rows"])
    with pytest.raises(ValueError, match="hash|fingerprint|validation|match"):
        export(experiment)
    report = export(experiment, allow_unverified_validation=True)
    assert isinstance(report, dict)
    assert report["warnings"]
    assert set(report["validation_provenance"].values()) == {"caller_declared_validation_only"}
    assert len(read_csv(experiment["output"])) == len(experiment["rows"])


def test_absent_experiment_manifest_needs_explicit_unverified_validation_flag(experiment):
    experiment["manifest"].unlink()
    with pytest.raises(ValueError, match="manifest|experiment|validation|provenance"):
        export(experiment)
    report = export(experiment, allow_unverified_validation=True)
    assert isinstance(report, dict)
    assert report["warnings"]
    assert set(report["validation_provenance"].values()) == {"caller_declared_validation_only"}
    assert experiment["output"].is_file()


def test_wrong_model_target_cannot_be_used_in_export(experiment):
    with pytest.raises(ValueError, match="target"):
        exporting.export_validation_predictions(
            experiment["source"], experiment["models"]["severity"], experiment["models"]["category"], experiment["output"],
        )


def test_existing_output_requires_overwrite_and_inputs_cannot_be_outputs(experiment):
    export(experiment)
    original = experiment["output"].read_bytes()
    with pytest.raises(ValueError, match="exist|overwrite"):
        export(experiment)
    assert experiment["output"].read_bytes() == original
    export(experiment, overwrite=True)
    for path in (experiment["source"], experiment["models"]["category"], experiment["models"]["severity"]):
        with pytest.raises(ValueError):
            exporting.export_validation_predictions(experiment["source"], experiment["models"]["category"], experiment["models"]["severity"], path, overwrite=True)


def test_empty_validation_data_is_rejected(experiment):
    write_csv(experiment["source"], [])
    refresh_manifest(experiment)
    with pytest.raises(ValueError, match="empty|row"):
        export(experiment)


def test_malformed_csv_is_rejected(experiment):
    experiment["source"].write_text('id,original_id,text,lang,category,severity\n"unterminated', encoding="utf-8")
    with pytest.raises((ValueError, csv.Error)):
        export(experiment, allow_unverified_validation=True)


def test_cli_prints_actionable_errors_without_traceback(experiment, capsys):
    experiment["manifest"].unlink()
    code = exporting.main([
        "--input", str(experiment["source"]), "--category-model", str(experiment["models"]["category"]),
        "--severity-model", str(experiment["models"]["severity"]), "--output", str(experiment["output"]),
    ])
    assert code == 2
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err and "Traceback" not in captured.err
