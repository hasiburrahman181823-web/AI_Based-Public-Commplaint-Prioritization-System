"""Synthetic experiment integration tests; scores here are not research results."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pytest

from ml_training import evaluation
from ml_training.config import SCHEMA_COLUMNS
from ml_training.scripts import predict_classifiers as prediction
from ml_training.scripts import train_classifiers as training


def write_rows(path, rows, columns=SCHEMA_COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_experiment(directory):
    data = directory / "data"
    phrases = {
        "water": {"bn": "পানি নেই ২ দিন", "en": "water unavailable for two days", "banglish": "PANY NEI ২ din ekhane"},
        "roads": {"bn": "রাস্তা ভাঙা অনেক দিন", "en": "road broken for many days", "banglish": "rasta bhanga onek din"},
    }
    rows_by_split = {}
    for split, groups in (("train", 8), ("val", 4), ("test", 2)):
        rows = []
        for group in range(groups):
            category = "water" if group % 2 == 0 else "roads"
            for lang in ("bn", "en", "banglish"):
                rows.append({
                    "id": f"{split}-group-{group}-{lang}", "original_id": f"{split}-group-{group}",
                    "text": f"{phrases[category][lang]} {split}marker {group}", "lang": lang,
                    "category": category, "severity": "high" if group % 3 == 0 else "low",
                    "source_type": "synthetic_test", "location": f"Road {group}",
                    "timestamp": "2026-01-01T06:00:00Z",
                })
        write_rows(data / f"{split}.csv", rows)
        rows_by_split[split] = rows
    natural = write_rows(directory / "natural.csv", [
        {"text": "Naturally authored water issue naturalmarker", "lang": "en", "category": "water", "location": "Road X", "timestamp": ""},
        {"text": "ekhane rasta khub kharap naturalmarker", "lang": "banglish", "category": "roads", "location": "Road Y", "timestamp": ""},
    ], columns=("text", "lang", "category", "location", "timestamp"))
    return {"root": directory, "input": data, "natural": natural, "rows": rows_by_split}


def args_for(dataset, *, models=("char_svm", "word_lr"), targets=("category", "severity"), suffix="", noise_rate=0):
    return training.build_parser().parse_args([
        "--input-dir", str(dataset["input"]), "--natural-test", str(dataset["natural"]),
        "--models", *models, "--targets", *targets, "--severity-order", "low", "high",
        "--results-dir", str(dataset["root"] / f"results{suffix}"),
        "--artifacts-dir", str(dataset["root"] / f"artifacts{suffix}"),
        "--noise-rate", str(noise_rate), "--ece-bins", "5", "--seed", "18",
    ])


@pytest.fixture
def dataset(tmp_path):
    return make_experiment(tmp_path)


@pytest.fixture
def fast_plots(monkeypatch):
    # Numerical/report behavior still uses the real evaluation code. Valid tiny
    # PNGs replace rendering only in additional pipeline tests; the main smoke
    # below and evaluation unit tests exercise real Matplotlib output.
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aA8sAAAAASUVORK5CYII=")

    def render(matrix, labels, calibration, output_dir):
        for name in ("confusion_matrix.png", "reliability_diagram.png"):
            (output_dir / name).write_bytes(png)

    monkeypatch.setattr(evaluation, "_plots", render)


@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory):
    dataset = make_experiment(tmp_path_factory.mktemp("baseline-integration"))
    args = args_for(dataset)
    result = training.run_experiment(args)
    return {**dataset, "args": args, "manifest": result,
            "results": Path(args.results_dir), "artifacts": Path(args.artifacts_dir)}


def test_baseline_end_to_end_writes_all_cohorts_and_real_reports(baseline_run):
    run = baseline_run
    saved = json.loads((run["results"] / "metrics.json").read_text(encoding="utf-8"))
    assert saved == run["manifest"]
    assert len(saved["runs"]) == 4
    assert saved["labels"]["severity"] == ["low", "high"]
    assert saved["decision_rule"] == "argmax calibrated probability in explicit label order"
    cohorts = {"bangla_test", "english_test", "banglish_test", "natural_test", "banglish_noisy"}
    for experiment in saved["runs"]:
        assert set(experiment["cohorts"]) == cohorts
        assert (run["artifacts"] / experiment["artifact"]).is_file()
        for cohort, metrics in experiment["cohorts"].items():
            if experiment["target"] == "severity" and cohort == "natural_test":
                assert metrics["status"] == "skipped_empty"
                assert metrics["n"] == 0
                assert metrics["n_total"] == metrics["n_missing_target"] == 2
                assert metrics["macro_f1"] is None
                assert metrics["ece"] is None
                assert "No human severity labels" in metrics["skip_reason"]
                assert metrics["files"] == {}
                continue
            assert metrics["status"] == "ok"
            assert metrics["n"] == 2
            assert metrics["n_missing_target"] == 0
            assert 0 <= metrics["macro_f1"] <= 1
            assert 0 <= metrics["ece"] <= 1
            assert metrics["quadratic_weighted_kappa"] is None or -1 <= metrics["quadratic_weighted_kappa"] <= 1
            for filename, relative in metrics["files"].items():
                assert not Path(relative).is_absolute()
                destination = run["results"] / relative
                assert destination.is_file(), destination
                if filename.endswith(".png"):
                    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                    assert destination.stat().st_size > 100
            report = json.loads((run["results"] / metrics["files"]["classification_report.json"]).read_text(encoding="utf-8"))
            assert set(saved["labels"][experiment["target"]]).issubset(report)
    table = list(csv.DictReader((run["results"] / "results_table.csv").open(encoding="utf-8", newline="")))
    assert len(table) == 4 * 5
    assert {row["target"] for row in table} == {"category", "severity"}
    assert {row["model"] for row in table} == {"char_svm", "word_lr"}
    assert {row["cohort"] for row in table} == cohorts
    assert all("macro_f1" in row and "ece" in row and "quadratic_weighted_kappa" in row for row in table)


def test_zero_noise_is_a_paired_identity_test_not_a_new_sample(baseline_run):
    for run in baseline_run["manifest"]["runs"]:
        clean, noisy = run["cohorts"]["banglish_test"], run["cohorts"]["banglish_noisy"]
        assert clean["n"] == noisy["n"]
        assert clean["macro_f1"] == noisy["macro_f1"]
        assert noisy["macro_f1_drop"] == 0
        assert noisy["changed_rows_raw"] == noisy["changed_rows_after_preprocessing"] == 0
        assert noisy["noise_rate"] == 0


def test_joblib_base_vocabulary_excludes_calibration_test_and_natural_text(baseline_run):
    for target in ("category", "severity"):
        bundle = joblib.load(baseline_run["artifacts"] / target / "word_lr.joblib")
        frozen = bundle["estimator"].calibrated_classifiers_[0].estimator
        vocabulary = frozen.estimator["tfidf"].vocabulary_
        assert "trainmarker" in vocabulary
        assert not {"valmarker", "testmarker", "naturalmarker"}.intersection(vocabulary)
        assert bundle["feature_type"] == "tfidf"
        assert bundle["encoder_dir"] is None
        assert bundle["labels"] == baseline_run["manifest"]["labels"][target]


def test_model_predictions_survive_moving_entire_artifacts_tree(baseline_run, tmp_path):
    moved = tmp_path / "moved-artifacts"
    shutil.copytree(baseline_run["artifacts"], moved)
    texts, langs = ["PANY NEI ২ din", "রাস্তা ভাঙা"], ["banglish", "bn"]
    for target in ("category", "severity"):
        original = prediction.predict_bundle(baseline_run["artifacts"] / target / "word_lr.joblib", texts, langs)
        restored = prediction.predict_bundle(moved / target / "word_lr.joblib", texts, langs)
        assert original == restored
        assert restored[0]["text_normalized"] == "pani nai 2 din"
        labels = baseline_run["manifest"]["labels"][target]
        for row in restored:
            assert list(row["probabilities"]) == labels
            assert row["prediction"] == max(labels, key=lambda label: row["probabilities"][label])
            assert sum(row["probabilities"].values()) == pytest.approx(1)


def test_changed_inference_dictionary_is_rejected(baseline_run, tmp_path):
    moved = tmp_path / "changed-artifacts"
    shutil.copytree(baseline_run["artifacts"], moved)
    (moved / "resources" / "banglish_variants.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="dictionary fingerprint changed"):
        prediction.predict_bundle(moved / "category" / "word_lr.joblib", ["pani nai"], ["banglish"])


def test_same_seed_reproduces_robustness_perturbations_and_predictions(dataset, fast_plots):
    first = training.run_experiment(args_for(dataset, models=("char_svm",), targets=("category",), suffix="-one", noise_rate=0.7))
    second = training.run_experiment(args_for(dataset, models=("char_svm",), targets=("category",), suffix="-two", noise_rate=0.7))
    left, right = first["runs"][0]["cohorts"], second["runs"][0]["cohorts"]
    for cohort in left:
        for name in ("n", "macro_f1", "ece", "changed_rows_raw", "changed_rows_after_preprocessing", "macro_f1_drop"):
            assert left[cohort].get(name) == right[cohort].get(name)
    assert left["banglish_noisy"]["macro_f1_drop"] == pytest.approx(left["banglish_test"]["macro_f1"] - left["banglish_noisy"]["macro_f1"])
    path = Path("category/char_svm/banglish_noisy/predictions.csv")
    with (dataset["root"] / "results-one" / path).open(encoding="utf-8", newline="") as stream:
        rows_a = list(csv.DictReader(stream))
    with (dataset["root"] / "results-two" / path).open(encoding="utf-8", newline="") as stream:
        rows_b = list(csv.DictReader(stream))
    assert rows_a == rows_b


def test_mocked_minilm_pipeline_reuses_encoder_exports_resources_and_moves(dataset, fast_plots, monkeypatch, tmp_path):
    instances, encoded, saves = [], [], []

    class FakeEncoder:
        def __init__(self, model_name, *args, **kwargs):
            self.model_name = model_name
            instances.append((model_name, args, kwargs))

        def encode(self, texts):
            encoded.append(tuple(texts))
            rows = [np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest()[:8], dtype=np.uint8).astype(np.float32) + 1 for text in texts]
            matrix = np.asarray(rows)
            return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

        def save(self, path):
            path = Path(path)
            path.mkdir(parents=True)
            (path / "dummy-encoder.json").write_text('{"synthetic_test_double":true}', encoding="utf-8")
            saves.append(path)

    monkeypatch.setattr(training, "SentenceEncoder", FakeEncoder)
    monkeypatch.setattr(prediction, "SentenceEncoder", FakeEncoder)
    args = args_for(dataset, models=("minilm_lr", "minilm_svm"), suffix="-embedding")
    manifest = training.run_experiment(args)
    assert len(instances) == len(saves) == 1
    assert len(encoded) == len(set(encoded))
    assert instances[0][1][1] is False  # default allow_download positional argument
    artifacts = Path(args.artifacts_dir)
    assert (artifacts / "minilm_encoder" / "dummy-encoder.json").is_file()
    assert len(manifest["runs"]) == 4
    for target in ("category", "severity"):
        for model in ("minilm_lr", "minilm_svm"):
            bundle = joblib.load(artifacts / target / f"{model}.joblib")
            assert bundle["feature_type"] == "embedding"
            assert bundle["encoder_dir"] == "minilm_encoder"
            assert bundle["artifact_root_relative"] == ".."
            assert bundle["variants_file"] == "resources/banglish_variants.json"
    relocated = tmp_path / "relocated-embedding"
    shutil.copytree(artifacts, relocated)
    original = prediction.predict_bundle(artifacts / "severity" / "minilm_lr.joblib", ["PANY NEI"], ["banglish"])
    moved = prediction.predict_bundle(relocated / "severity" / "minilm_lr.joblib", ["PANY NEI"], ["banglish"])
    assert original == moved
    assert instances[-1][2]["allow_download"] is False
    assert str(relocated / "minilm_encoder") == instances[-1][0]


def test_missing_severity_order_fails_before_creating_output_directories(dataset):
    args = args_for(dataset)
    args.severity_order = None
    with pytest.raises(ValueError, match="Provide --severity-order"):
        training.run_experiment(args)
    assert not Path(args.results_dir).exists()
    assert not Path(args.artifacts_dir).exists()


@pytest.mark.parametrize("directory_name", ["results", "artifacts"])
def test_existing_outputs_are_protected(dataset, directory_name):
    args = args_for(dataset)
    output = Path(getattr(args, f"{directory_name}_dir"))
    output.mkdir()
    marker = output / "do-not-replace.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="Output already exists"):
        training.run_experiment(args)
    assert marker.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("overlap", ["results_input", "artifacts_input", "nested_outputs", "same_outputs", "natural_inside_results"])
def test_overlapping_input_and_output_paths_are_rejected(dataset, overlap):
    args = args_for(dataset)
    if overlap == "results_input":
        args.results_dir = str(dataset["input"] / "results")
    elif overlap == "artifacts_input":
        args.artifacts_dir = str(dataset["input"])
    elif overlap == "nested_outputs":
        args.artifacts_dir = str(Path(args.results_dir) / "artifacts")
    elif overlap == "same_outputs":
        args.artifacts_dir = args.results_dir
    else:
        args.results_dir = str(dataset["root"])
    with pytest.raises(ValueError, match="separate, non-nested|Natural input must be outside"):
        training.run_experiment(args)


def test_missing_validation_class_fails_before_fitting_or_publishing(dataset):
    for row in dataset["rows"]["val"]:
        row["category"] = "water"
    write_rows(dataset["input"] / "val.csv", dataset["rows"]["val"])
    args = args_for(dataset)
    with pytest.raises(ValueError, match="validation must contain exactly the training classes"):
        training.run_experiment(args)
    assert not Path(args.results_dir).exists()
    assert not Path(args.artifacts_dir).exists()


def test_missing_encoder_does_not_publish_partial_experiment(dataset, monkeypatch):
    class UnavailableEncoder:
        def __init__(self, *args):
            raise RuntimeError("Synthetic missing local encoder")

    monkeypatch.setattr(training, "SentenceEncoder", UnavailableEncoder)
    args = args_for(dataset, models=("minilm_lr",))
    with pytest.raises(RuntimeError, match="missing local encoder"):
        training.run_experiment(args)
    assert not Path(args.results_dir).exists()
    assert not Path(args.artifacts_dir).exists()


def test_cli_main_prints_clear_config_errors_without_claimed_metrics(dataset, capsys):
    code = training.main([
        "--input-dir", str(dataset["input"]), "--models", "char_svm",
        "--results-dir", str(dataset["root"] / "results"),
        "--artifacts-dir", str(dataset["root"] / "artifacts"),
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "ERROR: Provide --severity-order" in captured.err
    assert not captured.out


@pytest.mark.parametrize("field,value", [("noise_rate", -0.1), ("noise_rate", 1.1), ("ece_bins", 1)])
def test_invalid_noise_and_calibration_bin_configuration_fails(dataset, field, value):
    args = args_for(dataset)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="--noise-rate must be in"):
        training.run_experiment(args)


def test_prediction_main_reports_missing_model_and_input_validation(tmp_path, capsys):
    code = prediction.main(["--model", str(tmp_path / "missing.joblib"), "--text", "pani nai", "--lang", "banglish"])
    assert code == 2
    assert "ERROR:" in capsys.readouterr().err
    with pytest.raises(ValueError, match="matching, nonempty"):
        prediction.predict_bundle(tmp_path / "missing.joblib", [], [])
