"""Real calibrated linear classifiers with a synthetic embedding test double.

No pretrained checkpoint, neural performance or term attribution is simulated.
Only encoder vectors are substituted so runtime explanation and resource-binding
mechanics can be tested offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ml import classifier_runtime as runtime_module
from ml.classifier_runtime import ClassifierRuntime
from ml.normalize import DEFAULT_VARIANTS_PATH
from ml.uncertainty import CONFIDENCE_DEFINITION, load_review_policy
from ml_training.modeling import calibrate_prefit


class ThreeDimensionalEncoder:
    """Deterministic lexical vectorizer, deliberately not a MiniLM imitation."""

    def __init__(self, model_name="test-encoder", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            words = text.lower().split()
            water = sum(word in {"water", "pani", "pany", "পানি"} for word in words)
            electric = sum(word in {"fire", "electric", "electricity", "shock", "agun", "বিদ্যুৎ"} for word in words)
            value = np.asarray([0.1 + water, 0.1 + electric, 1.0], dtype=np.float32)
            rows.append(value / np.linalg.norm(value))
        return np.asarray(rows, dtype=np.float32).reshape(len(rows), 3)


def make_embedding_artifact(tmp_path: Path, kind="lr", target="category", multiclass=False):
    root = tmp_path / "artifacts"
    (root / "resources").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEFAULT_VARIANTS_PATH, root / "resources" / "banglish_variants.json")
    (root / "minilm_encoder").mkdir(exist_ok=True)
    (root / "minilm_encoder" / "synthetic_weights.json").write_text(
        '{"fixture":"offline three-dimensional encoder"}\n', encoding="utf-8"
    )
    train = ["water unavailable", "water missing", "pani nai", "পানি নেই",
             "electric fire", "electricity lost", "fire shock", "বিদ্যুৎ নেই"]
    validation = ["water supply lost", "pani unavailable", "electricity unavailable", "electric shock"]
    if target == "category":
        truth = ["water"] * 4 + ["electricity"] * 4
        val_truth = ["water", "water", "electricity", "electricity"]
        labels = ["water", "electricity"]  # Deliberately unlike sklearn's lexical order.
    else:
        truth = ["Low"] * 4 + ["High"] * 4
        val_truth = ["Low", "Low", "High", "High"]
        labels = ["Low", "High"]
    if multiclass:
        if target != "category":
            raise ValueError("This fixture's multiclass case is category only")
        train += ["road broken", "road closed", "rasta bhanga", "surface damaged"]
        truth += ["roads"] * 4
        validation += ["road surface bad", "road potholes"]
        val_truth += ["roads"] * 2
        labels = ["water", "roads", "electricity"]
    encoder = ThreeDimensionalEncoder()
    classifier = LogisticRegression(max_iter=2000, random_state=42) if kind == "lr" else LinearSVC(max_iter=10000, random_state=42)
    classifier.fit(encoder.encode(train), truth)
    calibrated = calibrate_prefit(classifier, encoder.encode(validation), val_truth)
    bundle = {
        "kind": "banglapsg_classifier", "schema_version": 1, "target": target,
        "model_name": f"minilm_{kind}", "feature_type": "embedding", "estimator": calibrated,
        "labels": labels, "artifact_root_relative": "..", "encoder_dir": "minilm_encoder",
        "variants_file": "resources/banglish_variants.json",
        "normalization_dictionary_sha256": hashlib.sha256(DEFAULT_VARIANTS_PATH.read_bytes()).hexdigest(),
        "calibration": "sigmoid on held-out validation with frozen fitted linear estimator",
    }
    (root / target).mkdir(exist_ok=True)
    path = root / target / f"minilm_{kind}.joblib"
    joblib.dump(bundle, path, compress=3)
    return path, root, encoder


@pytest.mark.parametrize("kind", ["lr", "svm"])
def test_embedding_runtime_probabilities_follow_explicit_label_order(kind, tmp_path):
    path, _, encoder = make_embedding_artifact(tmp_path, kind)
    runtime = ClassifierRuntime(path, expected_target="category", encoder=encoder)
    results = runtime.predict_many(["water unavailable", "electric fire"], ["en", "en"])
    assert {row["prediction"] for row in results} == {"water", "electricity"}
    for result in results:
        assert list(result["probabilities"]) == ["water", "electricity"]
        assert sum(result["probabilities"].values()) == pytest.approx(1)
        assert result["prediction"] == max(result["probabilities"], key=result["probabilities"].get)
        assert result["confidence"] == max(result["probabilities"].values())


@pytest.mark.parametrize("kind", ["lr", "svm"])
@pytest.mark.parametrize("text", ["water unavailable", "electric fire"])
def test_binary_dimension_contributions_equal_selected_class_signed_base_margin(kind, text, tmp_path):
    path, _, encoder = make_embedding_artifact(tmp_path, kind)
    runtime = ClassifierRuntime(path, encoder=encoder)
    result = runtime.predict(text, "en", top_k=3, occlusion_limit=0)
    explanation = result["explanation"]
    vector = encoder.encode([result["text_normalized"]])[0]
    classes = list(runtime.base.classes_)
    raw_margin = float(runtime.base.decision_function(vector[None, :])[0])
    expected = raw_margin if result["prediction"] == classes[1] else -raw_margin
    assert explanation["feature_kind"] == "embedding_dimension"
    assert explanation["method"] == "linear_base_margin"
    assert explanation["top_terms"] == []
    assert {row["term"] for row in explanation["top_features"]} == {f"embedding_dimension_{i}" for i in range(3)}
    assert all(row["feature_kind"] == "embedding_dimension" for row in explanation["top_features"])
    assert explanation["base_margin"] == pytest.approx(expected, abs=1e-6)
    assert explanation["base_margin"] == pytest.approx(explanation["intercept"] + explanation["all_feature_contributions_sum"])
    assert explanation["all_feature_contributions_sum"] == pytest.approx(sum(row["contribution"] for row in explanation["top_features"]))
    assert "not calibrated probability" in explanation["semantics"]


@pytest.mark.parametrize("kind", ["lr", "svm"])
def test_multiclass_dimension_sum_matches_selected_class_decision_function(kind, tmp_path):
    path, _, encoder = make_embedding_artifact(tmp_path, kind, multiclass=True)
    runtime = ClassifierRuntime(path, encoder=encoder)
    result = runtime.predict("road surface damage", "en", top_k=3, occlusion_limit=0)
    explanation = result["explanation"]
    vector = encoder.encode([result["text_normalized"]])
    selected = list(runtime.base.classes_).index(result["prediction"])
    raw = runtime.base.decision_function(vector)
    assert raw.shape == (1, 3)
    assert explanation["base_margin"] == pytest.approx(float(raw[0, selected]), abs=1e-6)


def test_embedding_term_effects_are_bounded_token_deletions_not_dimension_names_or_shap(tmp_path):
    path, _, encoder = make_embedding_artifact(tmp_path)
    runtime = ClassifierRuntime(path, encoder=encoder)
    text = "water unavailable near hospital"
    result = runtime.predict(text, "en", top_k=5, occlusion_limit=2)
    explanation = result["explanation"]
    assert "approximate" in explanation["term_method"]
    assert "not SHAP" in explanation["term_method"]
    assert {row["term"] for row in explanation["top_terms"]} == {"water", "unavailable"}
    assert all("embedding_dimension" not in row["term"] for row in explanation["top_terms"])
    assert all("coefficient" not in row for row in explanation["top_terms"])
    removed = {"water": "unavailable near hospital", "unavailable": "water near hospital"}
    target = result["prediction"]
    for row in explanation["top_terms"]:
        after = runtime.predict_many([removed[row["term"]]], ["en"])[0]
        expected = result["probabilities"][target] - after["probabilities"][target]
        assert row["probability_drop_on_deletion"] == pytest.approx(expected)


def test_single_token_does_not_invent_an_empty_text_occlusion(tmp_path):
    path, _, encoder = make_embedding_artifact(tmp_path)
    runtime = ClassifierRuntime(path, encoder=encoder)
    result = runtime.predict("water", "en")
    assert result["explanation"]["top_terms"] == []
    assert "term_method" in result["explanation"]
    assert len(result["explanation"]["top_features"]) == 3


def test_copied_artifact_tree_preserves_model_resource_identity_and_predictions(tmp_path):
    path, root, encoder = make_embedding_artifact(tmp_path)
    first = ClassifierRuntime(path, encoder=encoder)
    moved = tmp_path / "moved-artifacts"
    shutil.copytree(root, moved)
    second = ClassifierRuntime(moved / "category" / path.name, encoder=ThreeDimensionalEncoder())
    assert first.fingerprint == second.fingerprint
    assert first.resource_fingerprint == second.resource_fingerprint
    assert first.predict_many(["water unavailable"], ["en"]) == second.predict_many(["water unavailable"], ["en"])


def test_encoder_snapshot_changes_resource_fingerprint_even_when_joblib_bytes_are_unchanged(tmp_path):
    path, root, encoder = make_embedding_artifact(tmp_path)
    first = ClassifierRuntime(path, encoder=encoder)
    (root / "minilm_encoder" / "synthetic_weights.json").write_text('{"fixture":"changed weights"}', encoding="utf-8")
    second = ClassifierRuntime(path, encoder=ThreeDimensionalEncoder())
    assert first.fingerprint == second.fingerprint
    assert first.resource_fingerprint != second.resource_fingerprint


def test_missing_encoder_snapshot_is_rejected_even_with_test_encoder_injection(tmp_path):
    path, root, encoder = make_embedding_artifact(tmp_path)
    (root / "minilm_encoder" / "synthetic_weights.json").unlink()
    with pytest.raises(ValueError, match="encoder resources are missing"):
        ClassifierRuntime(path, encoder=encoder)


def test_runtime_reuses_same_cached_encoder_and_refreshes_it_for_changed_resource_bytes(tmp_path, monkeypatch):
    path, root, _ = make_embedding_artifact(tmp_path)
    monkeypatch.setattr(runtime_module, "SentenceEncoder", ThreeDimensionalEncoder)
    monkeypatch.setattr(runtime_module, "_ENCODERS", {})
    first = ClassifierRuntime(path)
    second = ClassifierRuntime(path)
    assert first.encoder is second.encoder
    assert first.encoder.kwargs["allow_download"] is False
    assert first.predict_many(["pany nei"], ["banglish"])[0]["text_normalized"] == "pani nai"
    assert first.encoder.calls
    (root / "minilm_encoder" / "synthetic_weights.json").write_text("new snapshot bytes", encoding="utf-8")
    third = ClassifierRuntime(path)
    assert third.encoder is not first.encoder
    assert third.resource_fingerprint != first.resource_fingerprint


def test_policy_resource_binding_rejects_changed_encoder_while_model_file_hashes_stay_fixed(tmp_path):
    category_path, root, category_encoder = make_embedding_artifact(tmp_path, target="category")
    severity_path, _, severity_encoder = make_embedding_artifact(tmp_path, target="severity")
    category = ClassifierRuntime(category_path, encoder=category_encoder)
    severity = ClassifierRuntime(severity_path, encoder=severity_encoder)
    fingerprints = {"category": category.fingerprint, "severity": severity.fingerprint}
    resources = {"category": category.resource_fingerprint, "severity": severity.resource_fingerprint}
    policy = {
        "kind": "priority_review_policy", "schema_version": 1, "threshold": 0.6, "status": "selected",
        "confidence_definition": CONFIDENCE_DEFINITION, "classifier_fingerprints": fingerprints,
        "classifier_resource_fingerprints": resources,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    assert load_review_policy(policy_path, fingerprints, resources)["status"] == "selected"
    (root / "minilm_encoder" / "synthetic_weights.json").write_text("changed pretrained bytes", encoding="utf-8")
    changed_category = ClassifierRuntime(category_path, encoder=ThreeDimensionalEncoder())
    changed_severity = ClassifierRuntime(severity_path, encoder=ThreeDimensionalEncoder())
    assert changed_category.fingerprint == category.fingerprint
    assert changed_severity.fingerprint == severity.fingerprint
    updated_resources = {"category": changed_category.resource_fingerprint, "severity": changed_severity.resource_fingerprint}
    with pytest.raises(ValueError, match="resources do not match"):
        load_review_policy(policy_path, fingerprints, updated_resources)
