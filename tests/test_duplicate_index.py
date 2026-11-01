"""Cache, retrieval and graph invariants with synthetic encoders only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import types

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml import duplicate_index as index_module
from ml.duplicate_index import DuplicateFinder, EmbeddingCache, connected_components
from ml_training.scripts.cluster_duplicates import cluster_predictions, main as cluster_main


class TestEncoder:
    __test__ = False

    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([
            self.vectors[text] if text in self.vectors else
            np.frombuffer(hashlib.sha256(text.encode()).digest()[:4], dtype=np.uint8).astype(float) + 1
            for text in texts
        ], dtype=np.float32)


class LocationScore:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, values):
        positive = np.asarray(values)[:, -1] * 0.8 + 0.1
        return np.column_stack([1 - positive, positive])


@pytest.fixture
def feature_stubs(monkeypatch):
    # Retrieval is tested independently of the separately tested extractors.
    def preprocess(row, variants_path=None):
        result = dict(row)
        result["text"] = row["text"].lower().strip()
        for name in ("category", "location", "timestamp"):
            result.setdefault(name, "")
        return result

    def matrix(left, right, cosines, names, gazetteer, category_source="provided"):
        output = []
        for a, b, cosine in zip(left, right, cosines):
            values = {"cosine": cosine,
                      "loc_match": float(bool(a["location"] and a["location"] == b["location"])),
                      "category_match": float(category_source == "provided" and a["category"] == b["category"])}
            output.append([values.get(name, 0) for name in names])
        return np.asarray(output)

    monkeypatch.setitem(sys.modules, "ml.duplicate_features", types.SimpleNamespace(
        preprocess_complaint=preprocess, feature_matrix=matrix,
    ))
    monkeypatch.setitem(sys.modules, "ml.location", types.SimpleNamespace(
        LocationGazetteer=lambda *args, **kwargs: object(),
    ))


def write_bundle(directory, name="B2", threshold=0.5, feature_names=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "resources").mkdir(exist_ok=True)
    resources = {}
    for filename in ("banglish_variants.json", "location_gazetteer.json"):
        relative = "resources/" + filename
        path = directory / relative
        path.write_text("{}", encoding="utf-8")
        resources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle = {
        "kind": "banglapsg_duplicate_model", "schema_version": 1,
        "model_name": name, "threshold": threshold, "artifact_root_relative": ".",
        "variants_file": "resources/banglish_variants.json",
        "gazetteer_file": "resources/location_gazetteer.json", "resource_sha256": resources,
        "feature_config": {"category_source": "provided"}, "feature_names": feature_names or ["cosine", "loc_match"],
        "encoder_fingerprint": "test-encoder-v1", "encoder_dir": "minilm_encoder",
        "estimator": LocationScore() if name in ("C0", "P1", "P2") else None,
        "vectorizer": TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5)).fit(["water unavailable", "road damaged"]) if name == "B1" else None,
        "positive_class": 1,
    }
    path = directory / f"{name}.joblib"
    joblib.dump(bundle, path)
    return path


def complaint(identifier, text="water unavailable", **extra):
    return {"id": identifier, "text": text, "lang": "en", **extra}


def test_cache_encodes_unique_misses_in_bounded_batches_and_persists(tmp_path):
    encoder = TestEncoder()
    cache = EmbeddingCache(tmp_path / "cache.sqlite3", "encoder+normalizer-v1")
    first = cache.get_or_encode(["one", "two", "one", "three"], encoder, batch_size=2)
    assert first.shape == (4, 4)
    assert np.allclose(np.linalg.norm(first, axis=1), 1)
    assert encoder.calls == [["one", "two"], ["three"]]
    other = TestEncoder()
    second = EmbeddingCache(cache.path, cache.namespace).get_or_encode(["three", "one"], other)
    assert other.calls == []
    assert np.array_equal(second, first[[3, 0]])
    cache.get_or_encode(["one", "four"], encoder)
    assert encoder.calls[-1] == ["four"]


def test_cache_namespace_prevents_reuse_across_encoder_or_dictionary_changes(tmp_path):
    path = tmp_path / "cache.sqlite3"
    left = EmbeddingCache(path, "encoder-A+dict-A").get_or_encode(["text"], TestEncoder({"text": [1, 0]}))
    other = TestEncoder({"text": [0, 1, 0]})
    right = EmbeddingCache(path, "encoder-B+dict-A").get_or_encode(["text"], other)
    assert left.shape == (1, 2) and right.shape == (1, 3)
    assert other.calls == [["text"]]


def test_cache_normalizes_large_finite_vectors_without_float32_norm_overflow(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3", "large")
    vectors = cache.get_or_encode(["large"], TestEncoder({"large": [1e30, 1e30]}))
    assert np.isfinite(vectors).all()
    assert np.linalg.norm(vectors[0]) == pytest.approx(1)
    assert np.array_equal(cache.get_or_encode(["large"], TestEncoder()), vectors)


@pytest.mark.parametrize("values", [[[0, 0]], [[float("nan"), 1]], [1, 2]])
def test_cache_rejects_invalid_new_vectors(tmp_path, values):
    with pytest.raises(ValueError, match="Encoder"):
        EmbeddingCache(tmp_path / "cache.sqlite3", "test").get_or_encode(["text"], types.SimpleNamespace(encode=lambda texts: values))


def test_cache_rejects_dimension_change_and_corrupted_blob(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3", "test")
    cache.get_or_encode(["one"], TestEncoder({"one": [1, 0]}))
    with pytest.raises(ValueError, match="dimension changed"):
        cache.get_or_encode(["two"], TestEncoder({"two": [1, 0, 0]}))
    with sqlite3.connect(cache.path) as connection:
        connection.execute("UPDATE embeddings SET vector=?", (b"bad",))
    with pytest.raises(ValueError, match="Corrupt"):
        cache.get_or_encode(["one"], TestEncoder())


@pytest.mark.parametrize("texts,batch", [("one", 64), ([""], 64), (["one"], 0), (["one"], True)])
def test_cache_validates_inputs(tmp_path, texts, batch):
    with pytest.raises(ValueError):
        EmbeddingCache(tmp_path / "cache.sqlite3", "test").get_or_encode(texts, TestEncoder(), batch)


def test_repeated_queries_reuse_index_embeddings_and_skip_self(tmp_path, feature_stubs):
    model = write_bundle(tmp_path / "model")
    encoder = TestEncoder({"query": [1, 0], "close": [1, 0], "far": [0, 1]})
    finder = DuplicateFinder(model, tmp_path / "cache", encoder=encoder)
    corpus = [complaint("q", "query"), complaint("a", "close"), complaint("b", "far")]
    result = finder.find_duplicates(complaint("q", "query"), corpus)
    assert [row["id"] for row in result] == ["a"]
    assert result[0]["embedding_cosine"] == pytest.approx(1)
    matrix = finder._matrix
    calls = list(encoder.calls)
    assert finder.find_duplicates(complaint("q", "query"), corpus) == result
    assert encoder.calls == calls and finder._matrix is matrix
    assert finder.last_query_stats["corpus_size"] == 3
    assert finder.last_query_stats["eligible_candidates"] == 2


def test_text_and_metadata_updates_invalidate_context_but_only_new_text_is_encoded(tmp_path, feature_stubs):
    model = write_bundle(tmp_path / "model", "P2", threshold=0.5)
    encoder = TestEncoder()
    finder = DuplicateFinder(model, tmp_path / "cache", encoder=encoder)
    query = complaint("new", "query", location="Mirpur")
    corpus = [complaint("a", "old", location="Mirpur")]
    assert finder.find_duplicates(query, corpus)[0]["id"] == "a"
    calls = len(encoder.calls)
    corpus[0]["location"] = "Uttara"
    assert finder.find_duplicates(query, corpus) == []
    assert len(encoder.calls) == calls
    corpus[0]["text"] = "changed"
    finder.find_duplicates(query, corpus)
    assert encoder.calls[-1] == ["changed"]


def test_learned_reranking_happens_after_candidate_truncation(tmp_path, feature_stubs):
    model = write_bundle(tmp_path / "model", "P1", threshold=0.5)
    encoder = TestEncoder({"query": [1, 0], "near": [1, 0], "lessnear": [0.8, 0.6]})
    finder = DuplicateFinder(model, tmp_path / "cache", encoder=encoder, top_k=1)
    query = complaint("new", "query", location="Mirpur")
    corpus = [complaint("a", "near", location="Uttara"), complaint("b", "lessnear", location="Mirpur")]
    assert finder.find_duplicates(query, corpus) == []
    assert finder.last_query_stats["exhaustive_candidates"] is False
    expanded = finder.find_duplicates(query, corpus, top_k=2)
    assert [item["id"] for item in expanded] == ["b"]
    assert expanded[0]["score"] == pytest.approx(0.9)
    assert expanded[0]["retrieval_rank"] == 2


def test_lexical_ties_are_deterministic_and_original_ids_do_not_influence_scores(tmp_path, feature_stubs):
    finder = DuplicateFinder(write_bundle(tmp_path / "model"), tmp_path / "cache", encoder=TestEncoder(), top_k=2)
    query = complaint("new", "same", original_id="shared-truth")
    corpus = [complaint(identifier, "same", original_id=identifier) for identifier in ("c", "a", "b")]
    result = finder.find_duplicates(query, corpus)
    assert [item["id"] for item in result] == ["a", "b"]
    for row in corpus:
        row["original_id"] = "shared-truth"
    assert finder.find_duplicates(query, corpus) == result


def test_b1_uses_saved_sparse_tfidf_without_embedding_dependencies(tmp_path, feature_stubs):
    finder = DuplicateFinder(write_bundle(tmp_path / "model", "B1", threshold=0.99), tmp_path / "cache")
    result = finder.find_duplicates(complaint("query"), [complaint("a"), complaint("b", "road damaged")])
    assert [row["id"] for row in result] == ["a"]
    assert result[0]["embedding_cosine"] is None
    assert finder.cache is None and finder.encoder is None


def test_resources_and_relative_layout_survive_move_but_tampering_fails(tmp_path, feature_stubs):
    original = tmp_path / "model"
    model = write_bundle(original)
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    finder = DuplicateFinder(moved / model.name, tmp_path / "cache", encoder=TestEncoder())
    assert finder.find_duplicates(complaint("q", "same"), [complaint("a", "same")])
    (moved / "resources" / "banglish_variants.json").write_text('{"changed":"yes"}')
    with pytest.raises(ValueError, match="fingerprint changed"):
        DuplicateFinder(moved / model.name, tmp_path / "other-cache", encoder=TestEncoder())


def real_feature_bundle(directory, name):
    """Fit a small real feature pipeline; encoder vectors are test data only."""
    from ml.duplicate_features import FEATURE_SETS, feature_matrix, preprocess_complaint
    from ml.location import DEFAULT_GAZETTEER_PATH, LocationGazetteer
    from ml.normalize import DEFAULT_VARIANTS_PATH

    directory.mkdir(parents=True, exist_ok=True)
    resources_dir = directory / "resources"
    resources_dir.mkdir()
    shutil.copyfile(DEFAULT_GAZETTEER_PATH, resources_dir / "location_gazetteer.json")
    shutil.copyfile(DEFAULT_VARIANTS_PATH, resources_dir / "banglish_variants.json")
    gazetteer = LocationGazetteer(resources_dir / "location_gazetteer.json")
    raw = [
        complaint("bangla", "মিরপুর ১০ এ পানি নেই ৩ দিন ধরে", lang="bn", category="water",
                  location="মিরপুর ১০", timestamp="2026-01-04T06:00:00Z"),
        complaint("english", "No water in Mirpur 10 for three days", category="water",
                  location="Mirpur 10", timestamp="2026-01-04T12:00:00+06:00"),
        complaint("banglish", "mirpur-10 pany nei 3 din dhore", lang="banglish", category="water",
                  location="mirpur-10", timestamp="2026-01-04T06:00:00Z"),
        complaint("different", "No electricity in Uttara sector 4 for six hours", category="electricity",
                  location="Uttara sector 4", timestamp="2026-01-12T06:00:00Z"),
        complaint("unknown", "Service unavailable", category="", location="", timestamp=""),
    ]
    prepared = [preprocess_complaint(record, resources_dir / "banglish_variants.json") for record in raw]
    vectors = np.asarray([[1, 0, 0], [0.98, 0.2, 0], [0.95, 0.25, 0.05], [0.1, 0, 1], [0.4, 0.4, 0.4]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pair_indices = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (3, 4)]
    cosines = np.asarray([vectors[a] @ vectors[b] for a, b in pair_indices])
    names = list(FEATURE_SETS[name])
    features = feature_matrix([prepared[a] for a, _ in pair_indices], [prepared[b] for _, b in pair_indices],
                              cosines, names, gazetteer, category_source="provided")
    estimator = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(random_state=42))])
    estimator.fit(features, [1, 1, 1, 0, 0, 0, 0, 0])
    resources = {"resources/" + filename: hashlib.sha256((resources_dir / filename).read_bytes()).hexdigest()
                 for filename in ("location_gazetteer.json", "banglish_variants.json")}
    bundle = {"kind": "banglapsg_duplicate_model", "schema_version": 1, "model_name": name,
              "threshold": 0.0, "artifact_root_relative": ".", "feature_names": names,
              "feature_config": {"category_source": "provided", "fuzzy_threshold": 90, "time_gap_unit": "hours"},
              "positive_class": 1, "estimator": estimator, "vectorizer": None,
              "encoder_dir": "minilm_encoder", "encoder_fingerprint": "mocked-normalized-3d-test-encoder",
              "variants_file": "resources/banglish_variants.json",
              "gazetteer_file": "resources/location_gazetteer.json", "resource_sha256": resources}
    model = directory / f"{name}.joblib"
    joblib.dump(bundle, model)
    encoder = TestEncoder({record["text"]: vector for record, vector in zip(prepared, vectors)})
    return {"model": model, "bundle": bundle, "raw": raw, "prepared": prepared, "vectors": vectors,
            "encoder": encoder, "gazetteer": gazetteer, "estimator": estimator}


@pytest.mark.parametrize("name", ["C0", "P1", "P2"])
def test_real_feature_pipeline_matches_training_order_and_saved_threshold(tmp_path, name):
    from ml.duplicate_features import feature_matrix

    fixture = real_feature_bundle(tmp_path / "model", name)
    finder = DuplicateFinder(fixture["model"], tmp_path / "cache", encoder=fixture["encoder"])
    query, corpus = fixture["raw"][0], fixture["raw"][1:]
    results = finder.find_duplicates(query, corpus)
    assert {record["id"] for record in results} == {record["id"] for record in corpus}
    result_by_id = {row["id"]: row for row in results}
    for index, expected in enumerate(fixture["prepared"][1:], 1):
        cosine = float(fixture["vectors"][0] @ fixture["vectors"][index])
        values = feature_matrix([fixture["prepared"][0]], [expected], [cosine], fixture["bundle"]["feature_names"],
                                fixture["gazetteer"], category_source="provided")
        probability = float(fixture["estimator"].predict_proba(values)[0, 1])
        actual = result_by_id[expected["id"]]
        assert actual["score"] == pytest.approx(probability, abs=1e-7)
        assert list(actual["features"]) == fixture["bundle"]["feature_names"]
        assert np.allclose(list(actual["features"].values()), values[0], atol=1e-7)
    same_incident = result_by_id["english"]
    if name in ("P1", "P2"):
        assert same_incident["features"]["loc_match"] == 1
        assert same_incident["features"]["loc_distance_flag"] == 0
    if name == "P2":
        assert same_incident["features"]["time_gap"] == 0
        assert same_incident["features"]["duration_gap"] == 0
        assert same_incident["features"]["interval_overlap"] == 1
        assert result_by_id["unknown"]["features"]["time_missing"] == 1
    calls, matrix = list(fixture["encoder"].calls), finder._matrix
    assert finder.find_duplicates(query, corpus) == results
    assert fixture["encoder"].calls == calls and finder._matrix is matrix
    # The numerical threshold remains the only decision boundary; changing it
    # does not refit features or a classifier at retrieval time.
    finder.threshold = np.nextafter(max(row["score"] for row in results), np.inf)
    assert finder.find_duplicates(query, corpus) == []


@pytest.mark.parametrize("name", ["C0", "P1", "P2"])
def test_real_pair_features_and_inference_scores_are_symmetric(tmp_path, name):
    fixture = real_feature_bundle(tmp_path / "model", name)
    finder = DuplicateFinder(fixture["model"], tmp_path / "cache", encoder=fixture["encoder"])
    for left, right in ((0, 1), (0, 3), (2, 4)):
        forward = finder.find_duplicates(fixture["raw"][left], [fixture["raw"][right]])[0]
        backward = finder.find_duplicates(fixture["raw"][right], [fixture["raw"][left]])[0]
        assert forward["score"] == pytest.approx(backward["score"], abs=1e-7)
        assert forward["features"] == backward["features"]


def test_real_resource_snapshot_changes_cache_namespace_without_reusing_stale_text_vectors(tmp_path):
    fixture = real_feature_bundle(tmp_path / "model", "P2")
    first = DuplicateFinder(fixture["model"], tmp_path / "cache", encoder=fixture["encoder"])
    first.find_duplicates(fixture["raw"][0], fixture["raw"][1:])
    dictionary = fixture["model"].parent / "resources" / "location_gazetteer.json"
    content = json.loads(dictionary.read_text(encoding="utf-8"))
    content["description"] = "Changed gazetteer resource snapshot for cache isolation test"
    dictionary.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    fixture["bundle"]["resource_sha256"]["resources/location_gazetteer.json"] = hashlib.sha256(dictionary.read_bytes()).hexdigest()
    joblib.dump(fixture["bundle"], fixture["model"])
    second_encoder = TestEncoder(fixture["encoder"].vectors)
    second = DuplicateFinder(fixture["model"], tmp_path / "cache", encoder=second_encoder)
    second.find_duplicates(fixture["raw"][0], fixture["raw"][1:])
    assert second.cache.namespace != first.cache.namespace
    assert sum(map(len, second_encoder.calls)) == len(fixture["raw"])


def test_raw_evidence_changes_refresh_real_time_features_without_reencoding(tmp_path):
    fixture = real_feature_bundle(tmp_path / "model", "P2")
    finder = DuplicateFinder(fixture["model"], tmp_path / "cache", encoder=fixture["encoder"])
    query = fixture["raw"][0]
    candidate = dict(fixture["raw"][1])
    initial = finder.find_duplicates(query, [candidate])[0]
    assert initial["features"]["duration_gap"] == 0
    calls = list(fixture["encoder"].calls)
    # Text embeddings can remain identical while approved source evidence gets
    # corrected; raw_text is therefore part of the index-context fingerprint.
    candidate["raw_text"] = "No water in Mirpur 10 for six days"
    corrected = finder.find_duplicates(query, [candidate])[0]
    assert corrected["features"]["duration_gap"] == 72
    assert fixture["encoder"].calls == calls


def test_query_and_corpus_validation_and_empty_cases(tmp_path, feature_stubs):
    finder = DuplicateFinder(write_bundle(tmp_path / "model", threshold=1.01), tmp_path / "cache", encoder=TestEncoder())
    with pytest.raises(ValueError, match="Build an index"):
        finder.find_duplicates(complaint("q"))
    assert finder.find_duplicates(complaint("q"), []) == []
    assert finder.find_duplicates(complaint("q", "same"), [complaint("a", "same")]) == []
    with pytest.raises(ValueError, match="Duplicate complaint id"):
        finder.build_index([complaint("a"), complaint("a", "other")])
    with pytest.raises(ValueError, match="lang"):
        finder.find_duplicates({"id": "q", "text": "hello"}, [])
    with pytest.raises(ValueError, match="top_k"):
        finder.find_duplicates(complaint("q"), [], top_k=0)


def test_fifty_thousand_rows_use_one_linear_index_and_cache_unique_texts(tmp_path, feature_stubs):
    encoder = TestEncoder()
    finder = DuplicateFinder(write_bundle(tmp_path / "model", threshold=-1), tmp_path / "cache", encoder=encoder, top_k=5)
    corpus = [complaint(f"c{index:05}", f"text {index % 4}") for index in range(50_000)]
    result = finder.find_duplicates(complaint("new", "text 0"), corpus)
    assert len(result) == 5
    assert finder._matrix.shape == (50_000, 4)
    assert finder._matrix.nbytes == 50_000 * 4 * 4
    assert sum(map(len, encoder.calls)) == 4
    calls = len(encoder.calls)
    finder.find_duplicates(complaint("another", "text 0"), None)
    assert len(encoder.calls) == calls


def test_convenience_function_uses_configured_reusable_finder(tmp_path, feature_stubs, monkeypatch):
    monkeypatch.setattr(index_module, "_default_finder", None)
    finder = index_module.configure_duplicate_finder(write_bundle(tmp_path / "model"), tmp_path / "cache", encoder=TestEncoder())
    result = index_module.find_duplicates(complaint("q", "same"), [complaint("a", "same")])
    assert result[0]["id"] == "a" and index_module._default_finder is finder


def test_convenience_missing_artifact_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "_default_finder", None)
    monkeypatch.setenv("BANGLAPSG_DUPLICATE_MODEL", str(tmp_path / "missing.joblib"))
    with pytest.raises(ValueError, match="configure_duplicate_finder"):
        index_module.find_duplicates(complaint("q"), [])


def test_connected_components_are_transitive_stable_and_keep_singletons():
    edges = [("a", "b", 0.9), ("b", "c", 0.9), ("a", "c", 0.1), ("d", "e", 0.2)]
    expected = [["a", "b", "c"], ["d"], ["e"], ["f"]]
    assert connected_components(edges, 0.8, ["f"]) == expected
    assert connected_components(reversed(edges), 0.8, ["f"]) == expected
    assert connected_components([], 1.01) == []


@pytest.mark.parametrize("edges,threshold", [([("a", "b", float("nan"))], 0.5), ([("", "b", 1)], 0.5), ([], float("inf"))])
def test_components_reject_invalid_values(edges, threshold):
    with pytest.raises(ValueError):
        connected_components(edges, threshold)


def write_csv(path, rows, columns=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_cluster_cli_partitions_splits_and_keeps_supplied_singletons(tmp_path):
    predictions = write_csv(tmp_path / "predictions.csv", [
        {"id_a": "a", "id_b": "b", "score": "0.9", "threshold": "0.8", "split": "test", "model": "P2"},
        {"id_a": "c", "id_b": "d", "score": "0.2", "threshold": "0.8", "split": "val", "model": "P2"},
    ])
    singletons = write_csv(tmp_path / "complaints.csv", [{"id": "z", "split": "test"}])
    rows, summary = cluster_predictions(predictions, all_complaints=[singletons])
    by_id = {row["id"]: row for row in rows}
    assert by_id["a"]["duplicate_group_id"] == by_id["b"]["duplicate_group_id"]
    assert by_id["c"]["duplicate_group_id"] != by_id["d"]["duplicate_group_id"]
    assert by_id["z"]["group_size"] == 1
    assert summary["splits"]["test"]["complaints"] == 3
    destination = tmp_path / "clusters.csv"
    assert cluster_main(["--predictions", str(predictions), "--all-complaints", str(singletons), "--output", str(destination)]) == 0
    assert destination.read_text().startswith("id,duplicate_group_id,split,group_size")
    assert cluster_main(["--predictions", str(predictions), "--output", str(destination)]) == 2


def test_cluster_rejects_split_leakage_and_mixed_model_thresholds(tmp_path):
    rows = [{"id_a": "a", "id_b": "b", "score": ".9", "threshold": ".8", "split": "train", "model": "P1"},
            {"id_a": "a", "id_b": "c", "score": ".9", "threshold": ".8", "split": "test", "model": "P1"}]
    path = write_csv(tmp_path / "predictions.csv", rows)
    with pytest.raises(ValueError, match="crosses"):
        cluster_predictions(path)
    rows[1]["id_a"] = "d"
    rows[1]["model"] = "P2"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="multiple models"):
        cluster_predictions(path)
    selected, _ = cluster_predictions(path, model="P2")
    assert {row["id"] for row in selected} == {"c", "d"}
    rows[1]["model"] = "P1"
    rows[1]["threshold"] = ".7"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="saved threshold"):
        cluster_predictions(path)


def test_cluster_rejects_original_group_leakage_and_input_overwrite(tmp_path):
    path = write_csv(tmp_path / "predictions.csv", [
        {"id_a": "a", "id_b": "b", "score": "1", "split": "train", "original_id_a": "incident", "original_id_b": "b"},
        {"id_a": "c", "id_b": "d", "score": "1", "split": "test", "original_id_a": "incident", "original_id_b": "d"},
    ])
    with pytest.raises(ValueError, match="original_id"):
        cluster_predictions(path, threshold=0.5)
    assert cluster_main(["--predictions", str(path), "--threshold", ".5", "--output", str(path), "--overwrite"]) == 2
