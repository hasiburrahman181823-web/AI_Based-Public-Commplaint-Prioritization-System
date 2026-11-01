"""Train-only features, prefit calibration, serialization and offline embedding IO."""

from __future__ import annotations

import sys
import types

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ml_training.modeling import (
    SentenceEncoder,
    build_embedding_estimator,
    build_text_estimator,
    calibrate_prefit,
)

TRAIN_TEXT = ["পানি নেই", "পানি সমস্যা", "রাস্তা ভাঙা", "রাস্তা বন্ধ", "বিদ্যুৎ নেই", "বিদ্যুৎ সমস্যা"]
TRAIN_Y = ["water", "water", "roads", "roads", "power", "power"]
VAL_TEXT = ["পানি নতুন শব্দ", "রাস্তা নতুন শব্দ", "বিদ্যুৎ নতুন শব্দ"]
VAL_Y = ["water", "roads", "power"]


def test_baseline_parameters_match_experiment_definition():
    char = build_text_estimator("char_svm", seed=123)
    assert char["tfidf"].analyzer == "char_wb"
    assert char["tfidf"].ngram_range == (1, 5)
    assert char["tfidf"].lowercase is False
    assert char["tfidf"].max_features is None
    assert isinstance(char["classifier"], LinearSVC)
    assert char["classifier"].random_state == 123
    word = build_text_estimator("word_lr", seed=123)
    assert word["tfidf"].analyzer == "word"
    assert word["tfidf"].ngram_range == (1, 1)
    assert word["tfidf"].max_features is None
    assert word["tfidf"].lowercase is False
    assert isinstance(word["classifier"], LogisticRegression)
    assert word["classifier"].max_iter == 2000
    assert word["classifier"].class_weight == "balanced"


def test_word_tokens_preserve_bengali_vowel_signs():
    estimator = build_text_estimator("word_lr").fit(TRAIN_TEXT, TRAIN_Y)
    assert "পানি" in estimator["tfidf"].vocabulary_
    assert "বিদ্যুৎ" in estimator["tfidf"].vocabulary_
    assert estimator["tfidf"].build_analyzer()("পানি নেই") == ["পানি", "নেই"]


@pytest.mark.parametrize("name", ["char_svm", "word_lr"])
def test_calibration_does_not_refit_train_vocabulary_or_base_coefficients(name):
    estimator = build_text_estimator(name).fit(TRAIN_TEXT, TRAIN_Y)
    vocabulary = dict(estimator["tfidf"].vocabulary_)
    idf = estimator["tfidf"].idf_.copy()
    coefficients = estimator["classifier"].coef_.copy()
    calibrated = calibrate_prefit(estimator, VAL_TEXT, VAL_Y)
    assert estimator["tfidf"].vocabulary_ == vocabulary
    np.testing.assert_array_equal(estimator["tfidf"].idf_, idf)
    np.testing.assert_array_equal(estimator["classifier"].coef_, coefficients)
    probabilities = calibrated.predict_proba(VAL_TEXT)
    assert probabilities.shape == (3, 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


@pytest.mark.parametrize("name", ["minilm_lr", "minilm_svm"])
def test_embedding_classifier_fits_vectors_and_calibrates(name):
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    estimator = build_embedding_estimator(name, seed=21).fit(vectors, ["low", "low", "high", "high"])
    coefficients = estimator.coef_.copy()
    calibrated = calibrate_prefit(estimator, vectors[[0, 2]], ["low", "high"])
    np.testing.assert_array_equal(estimator.coef_, coefficients)
    assert calibrated.predict_proba(vectors).shape == (4, 2)


@pytest.mark.parametrize("labels", [["water", "roads"], ["water", "roads", "unknown"]])
def test_calibration_requires_all_and_only_training_classes(labels):
    estimator = build_text_estimator("word_lr").fit(TRAIN_TEXT, TRAIN_Y)
    with pytest.raises(ValueError, match="exactly the training classes"):
        calibrate_prefit(estimator, VAL_TEXT[:len(labels)], labels)


def test_calibration_rejects_unfitted_and_invalid_inputs():
    with pytest.raises(ValueError, match="Fit the base estimator"):
        calibrate_prefit(build_text_estimator("word_lr"), VAL_TEXT, VAL_Y)
    fitted = build_text_estimator("word_lr").fit(TRAIN_TEXT, TRAIN_Y)
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        calibrate_prefit(fitted, [], [])
    with pytest.raises(ValueError, match="same number of rows"):
        calibrate_prefit(fitted, VAL_TEXT[:2], VAL_Y)


def test_calibrated_model_joblib_roundtrip(tmp_path):
    base = build_text_estimator("char_svm").fit(TRAIN_TEXT, TRAIN_Y)
    model = calibrate_prefit(base, VAL_TEXT, VAL_Y)
    destination = tmp_path / "model.joblib"
    joblib.dump({"estimator": model, "feature_type": "tfidf"}, destination)
    loaded = joblib.load(destination)["estimator"]
    np.testing.assert_array_equal(loaded.predict(VAL_TEXT), model.predict(VAL_TEXT))
    np.testing.assert_allclose(loaded.predict_proba(VAL_TEXT), model.predict_proba(VAL_TEXT))


@pytest.mark.parametrize("name", ["unknown", "minilm_lr"])
def test_text_factory_rejects_unknown_names(name):
    with pytest.raises(ValueError, match="Unknown text model"):
        build_text_estimator(name)


@pytest.mark.parametrize("name", ["unknown", "char_svm"])
def test_embedding_factory_rejects_unknown_names(name):
    with pytest.raises(ValueError, match="Unknown embedding model"):
        build_embedding_estimator(name)


def test_sentence_encoder_is_lazy_offline_and_normalizes_mock_embeddings(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        def __init__(self, model_name, **kwargs):
            calls.append(("load", model_name, kwargs))

        def encode(self, texts, **kwargs):
            calls.append(("encode", texts, kwargs))
            return np.tile(np.array([[3, 4]], dtype=np.float64), (len(texts), 1))

        def save(self, path):
            calls.append(("save", path))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    encoder = SentenceEncoder("local-model", batch_size=2, revision="pinned-revision")
    assert calls == []
    vectors = encoder.encode(["পানি নেই", "No water"])
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.6, 0.8]], rtol=1e-6)
    assert calls[0] == ("load", "local-model", {"device": "cpu", "local_files_only": True, "trust_remote_code": False, "revision": "pinned-revision"})
    assert calls[1][2] == {"batch_size": 2, "show_progress_bar": False, "convert_to_numpy": True, "normalize_embeddings": True}
    encoder.encode(["Again"])
    assert len([call for call in calls if call[0] == "load"]) == 1
    path = encoder.save(tmp_path / "encoder")
    assert path.is_dir()
    assert calls[-1] == ("save", str(path))


def test_explicit_download_permission_is_passed_to_encoder(monkeypatch):
    captured = {}

    class FakeModel:
        def __init__(self, model_name, **kwargs):
            captured.update(kwargs)

        def encode(self, texts, **kwargs):
            return [[1, 0]]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    SentenceEncoder(allow_download=True).encode(["test"])
    assert captured["local_files_only"] is False
    assert captured["trust_remote_code"] is False


def test_offline_model_error_is_actionable_without_retries(monkeypatch):
    calls = []

    def failing_model(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("No model files")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=failing_model))
    with pytest.raises(RuntimeError, match="Offline loading is enabled"):
        SentenceEncoder("not-cached").encode(["sample"])
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True


def test_optional_dependency_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="pip install sentence-transformers"):
        SentenceEncoder().encode(["sample"])


@pytest.mark.parametrize("texts", [[], "sample", [""], [None]])
def test_encoder_rejects_invalid_text_without_loading(texts):
    encoder = SentenceEncoder()
    with pytest.raises(ValueError, match="expects"):
        encoder.encode(texts)
    assert encoder._model is None


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": True}, {"model_name": ""}, {"device": ""}, {"allow_download": 1}, {"revision": ""}])
def test_encoder_validates_configuration(kwargs):
    with pytest.raises(ValueError):
        SentenceEncoder(**kwargs)
