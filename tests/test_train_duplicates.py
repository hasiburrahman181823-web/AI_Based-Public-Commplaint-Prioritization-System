"""Synthetic Phase 4 runner integration; no pretrained checkpoint is downloaded."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from ml.duplicate_features import FEATURE_SETS, feature_matrix
from ml.duplicate_index import DuplicateFinder
from ml.location import LocationGazetteer
from ml_training import duplicate_metrics
from ml_training.duplicate_data import load_duplicate_data
from ml_training.duplicate_metrics import tune_threshold
from ml_training.scripts import train_duplicates as training


class FixtureSentenceEncoder:
    """Deterministic vectors exercise integration, not multilingual accuracy."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.calls = args, kwargs, []
        self.saved = []
        self.__class__.instances.append(self)

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = np.asarray([
            np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest()[:12], dtype=np.uint8).astype(float) + 1
            for text in texts
        ], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text('{"fixture_encoder":true,"dimensions":12}', encoding="utf-8")
        self.saved.append(path)
        return path


COMPLAINT_COLUMNS = ("id", "original_id", "text", "lang", "category", "severity", "source_type", "location", "timestamp")
PAIR_COLUMNS = ("text_a", "text_b", "loc_a", "loc_b", "time_a", "time_b", "label", "pair_type",
                "pair_id", "split", "id_a", "id_b", "original_id_a", "original_id_b", "gold_label")


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def make_duplicate_fixture(root):
    pair_dir, complaint_dir = root / "pairs", root / "complaints"
    complaints, pair_rows = {}, {}
    phrases = {
        "water": {"bn": "পানি নেই ৩ দিন ধরে", "en": "no water for three days", "banglish": "pany nei 3 din dhore"},
        "electricity": {"bn": "বিদ্যুৎ নেই ৩ দিন ধরে", "en": "no electricity for three days", "banglish": "biddut nei 3 din dhore"},
    }
    for split_index, split in enumerate(("train", "val", "test")):
        rows = []
        by_key = {}
        marker = {"train": "ttttt", "val": "vvvvv", "test": "zzzzz"}[split]
        for group in range(4):
            category = "water" if group < 2 else "electricity"
            place = "Mirpur 10" if group % 2 == 0 else "Uttara sector 4"
            for language in ("bn", "en", "banglish"):
                row = {
                    "id": f"{split}-{group}-{language}", "original_id": f"{split}-incident-{group}",
                    "text": f"{phrases[category][language]} {place} {marker}", "lang": language,
                    "category": category, "severity": "high", "source_type": "synthetic_test",
                    "location": place, "timestamp": f"2026-01-{1 + split_index * 6 + group:02}T06:00:00Z",
                }
                rows.append(row)
                by_key[(group, language)] = row
        specs = [
            ((0, "bn"), (0, "en"), 1, "positive_cross_language"),
            ((1, "bn"), (1, "banglish"), 1, "positive_cross_language"),
            ((2, "en"), (2, "banglish"), 1, "positive_cross_language"),
            ((3, "bn"), (3, "en"), 1, "positive_cross_language"),
            ((0, "bn"), (1, "en"), 0, "hard_negative"),
            ((2, "bn"), (3, "banglish"), 0, "hard_negative"),
            ((0, "en"), (2, "banglish"), 0, "easy_negative"),
            ((1, "banglish"), (3, "bn"), 0, "easy_negative"),
        ]
        pairs = []
        for index, (left, right, label, kind) in enumerate(specs):
            a, b = by_key[left], by_key[right]
            pairs.append({
                "text_a": a["text"], "text_b": b["text"], "loc_a": a["location"], "loc_b": b["location"],
                "time_a": a["timestamp"], "time_b": b["timestamp"], "label": str(label), "gold_label": str(label),
                "pair_type": kind, "pair_id": f"{split}-pair-{index}", "split": split,
                "id_a": a["id"], "id_b": b["id"], "original_id_a": a["original_id"], "original_id_b": b["original_id"],
            })
        write_csv(complaint_dir / f"{split}.csv", rows, COMPLAINT_COLUMNS)
        write_csv(pair_dir / split / "pairs.csv", pairs, PAIR_COLUMNS)
        complaints[split], pair_rows[split] = rows, pairs
    return {"root": root, "pair_dir": pair_dir, "complaint_paths": [complaint_dir / f"{split}.csv" for split in ("train", "val", "test")],
            "complaints": complaints, "pair_rows": pair_rows}


def args_for(fixture, models=("B1",), classifier="lr", suffix=""):
    argv = ["--pairs", str(fixture["pair_dir"]), "--models", *models, "--classifier", classifier,
            "--results-dir", str(fixture["root"] / ("results" + suffix)),
            "--artifacts-dir", str(fixture["root"] / ("artifacts" + suffix)),
            "--cache-dir", str(fixture["root"] / ("cache" + suffix)),
            "--embedding-model", "synthetic-local-encoder", "--embedding-batch-size", "7", "--seed", "24"]
    for path in fixture["complaint_paths"]:
        argv.extend(["--complaints", str(path)])
    return training.build_parser().parse_args(argv)


@pytest.fixture(scope="module")
def b1_run(tmp_path_factory):
    fixture = make_duplicate_fixture(tmp_path_factory.mktemp("duplicates-b1"))
    args = args_for(fixture)
    manifest = training.run_experiment(args)
    return {**fixture, "args": args, "manifest": manifest}


@pytest.fixture(scope="module")
def learned_run(tmp_path_factory):
    fixture = make_duplicate_fixture(tmp_path_factory.mktemp("duplicates-all"))
    args = args_for(fixture, models=("B1", "B2", "C0", "P1", "P2"))
    with pytest.MonkeyPatch.context() as monkeypatch:
        before = len(FixtureSentenceEncoder.instances)
        monkeypatch.setattr(training, "SentenceEncoder", FixtureSentenceEncoder)
        manifest = training.run_experiment(args)
        encoder = FixtureSentenceEncoder.instances[before:]
        yield {**fixture, "args": args, "manifest": manifest, "encoders": encoder}


@pytest.fixture
def fast_pr(monkeypatch):
    def plot(truth, scores, output, model_name):
        Path(output).write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-test-placeholder")
    monkeypatch.setattr(duplicate_metrics, "_plot_pr", plot)


def test_b1_end_to_end_exports_reports_tables_and_real_pr_png(b1_run):
    run = b1_run
    results = Path(run["args"].results_dir)
    saved = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    assert saved == run["manifest"]
    assert saved["models_requested"] == ["B1"]
    assert saved["counts"]["test"]["pairs"] == len(run["pair_rows"]["test"])
    record = saved["models"][0]
    assert record["name"] == "B1" and record["feature_names"] == ["cosine"]
    assert record["test_pair_fingerprint"] == saved["test_pair_fingerprint"]
    for split in ("validation", "test"):
        prediction_path = results / "B1" / split / "predictions.csv"
        predictions = read_csv(prediction_path)
        assert len(predictions) == len(run["pair_rows"]["val" if split == "validation" else "test"])
        assert all(row["model"] == "B1" for row in predictions)
        assert (results / "B1" / split / "precision_recall_curve.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        report = record[split]
        assert report["status"] == "ok" and report["n"] == len(predictions)
        assert report["pr_auc_definition"].startswith("average precision")
        assert {"hard_negative", "easy_negative"}.issubset(report["by_pair_type"])
        assert {"bn-en", "bn-banglish", "en-banglish"}.issubset(report["by_language_pair"])
        assert all(not Path(path).is_absolute() for path in report["files"].values())
    table = read_csv(results / "ablation_table.csv")
    assert len(table) == 1 and table[0]["model"] == "B1"
    markdown = (results / "ablation_table.md").read_text(encoding="utf-8")
    assert "B1" in markdown and "validation" in markdown and "average precision" in markdown


def test_b1_vocabulary_uses_training_complaints_only_and_threshold_uses_validation(b1_run):
    run = b1_run
    bundle = joblib.load(Path(run["args"].artifacts_dir) / "B1.joblib")
    vocabulary = bundle["vectorizer"].vocabulary_
    assert "ttttt" in vocabulary and "vvvvv" not in vocabulary and "zzzzz" not in vocabulary
    validation = read_csv(Path(run["args"].results_dir) / "B1" / "validation" / "predictions.csv")
    selected = tune_threshold([int(row["label"]) for row in validation], [float(row["score"]) for row in validation])
    assert bundle["threshold"] == pytest.approx(selected["threshold"])
    assert run["manifest"]["models"][0]["threshold_selection"]["threshold"] == bundle["threshold"]


def test_saved_b1_bundle_reloads_with_real_inference_preprocessing(b1_run):
    run = b1_run
    finder = DuplicateFinder(Path(run["args"].artifacts_dir) / "B1.joblib", run["root"] / "retrieval-cache")
    query, corpus = run["complaints"]["test"][0], run["complaints"]["test"][1:]
    matches = finder.find_duplicates(query, corpus, top_k=100)
    vectors = finder.bundle["vectorizer"].transform([row["text"] for row in finder._records])
    qtext = finder._preprocess(query, variants_path=finder.variants_path)["text"]
    vector = finder.bundle["vectorizer"].transform([qtext])
    scores = np.asarray(vectors @ vector.T).reshape(-1) if not hasattr(vectors @ vector.T, "toarray") else (vectors @ vector.T).toarray().reshape(-1)
    expected = {row["id"] for row, score in zip(finder._records, scores) if score >= finder.threshold}
    assert {row["id"] for row in matches} == expected
    assert finder.encoder is None and finder.cache is None


def test_all_ablation_models_share_identical_test_rows_and_fingerprint(learned_run):
    run = learned_run
    manifest = run["manifest"]
    assert {row["name"] for row in manifest["models"]} == {"B1", "B2", "C0", "P1", "P2"}
    assert {row["test_pair_fingerprint"] for row in manifest["models"]} == {manifest["test_pair_fingerprint"]}
    expected = [row["pair_id"] for row in run["pair_rows"]["test"]]
    for model in manifest["models"]:
        predictions = read_csv(Path(run["args"].results_dir) / model["name"] / "test" / "predictions.csv")
        assert [row["pair_id"] for row in predictions] == expected
        assert [int(row["label"]) for row in predictions] == [int(row["gold_label"]) for row in run["pair_rows"]["test"]]
        assert model["feature_names"] == FEATURE_SETS[model["name"]]
        assert not {"label", "gold_label", "pair_type", "original_id", "id_a", "id_b"} & set(model["feature_names"])
    table = read_csv(Path(run["args"].results_dir) / "ablation_table.csv")
    assert len(table) == len(manifest["models"])
    assert {row["test_pair_fingerprint"] for row in table} == {manifest["test_pair_fingerprint"]}


def test_minilm_branch_uses_one_saved_encoder_and_cache_for_unique_texts(learned_run):
    run = learned_run
    assert len(run["encoders"]) == 1
    encoder = run["encoders"][0]
    assert encoder.args[2] is False
    assert len(encoder.saved) == 1
    calls = [text for call in encoder.calls for text in call]
    assert len(calls) == len(set(calls))
    assert all(len(call) <= run["args"].embedding_batch_size for call in encoder.calls)
    assert (Path(run["args"].artifacts_dir) / "minilm_encoder" / "config.json").is_file()
    assert Path(run["args"].cache_dir, "embeddings.sqlite3").is_file()
    for name in ("B2", "C0", "P1", "P2"):
        bundle = joblib.load(Path(run["args"].artifacts_dir) / f"{name}.joblib")
        assert bundle["encoder_fingerprint"] == run["manifest"]["embedding"]["fingerprint"]
        assert "minilm_encoder/config.json" in bundle["resource_sha256"]


@pytest.mark.parametrize("name", ["C0", "P1", "P2"])
def test_saved_learned_scorer_and_retrieval_match_training_features_and_threshold(learned_run, name):
    run = learned_run
    model_path = Path(run["args"].artifacts_dir) / f"{name}.joblib"
    encoder = FixtureSentenceEncoder()
    # Training and inference use the exact same namespace/resource hashes, so
    # known text embeddings need no new encode calls on this persisted cache.
    finder = DuplicateFinder(model_path, run["args"].cache_dir, encoder=encoder)
    metadata = {row["id"]: row for row in run["complaints"]["test"]}
    predictions = read_csv(Path(run["args"].results_dir) / name / "test" / "predictions.csv")
    for row in predictions:
        query, candidate = metadata[row["id_a"]], metadata[row["id_b"]]
        result = finder.find_duplicates(query, [candidate])
        expected_score, threshold = float(row["score"]), float(row["threshold"])
        if expected_score >= threshold:
            assert len(result) == 1
            assert result[0]["id"] == candidate["id"]
            assert result[0]["score"] == pytest.approx(expected_score, abs=1e-6)
            assert list(result[0]["features"]) == FEATURE_SETS[name]
        else:
            assert result == []
    assert encoder.calls == []


def test_real_feature_audits_follow_the_nested_ablation_contract(learned_run):
    root = Path(learned_run["args"].results_dir)
    assert set(FEATURE_SETS["C0"]) < set(FEATURE_SETS["P1"]) < set(FEATURE_SETS["P2"])
    for split in ("train", "val", "test"):
        rows = read_csv(root / "feature_audits" / f"{split}.csv")
        assert len(rows) == len(learned_run["pair_rows"][split])
        assert list(rows[0]) == ["pair_id", "label", *FEATURE_SETS["P2"]]
        assert all(np.isfinite(float(row[name])) for row in rows for name in FEATURE_SETS["P2"])


def test_gradient_boosting_branch_uses_the_same_score_and_feature_contract(tmp_path, monkeypatch, fast_pr):
    fixture = make_duplicate_fixture(tmp_path)
    monkeypatch.setattr(training, "SentenceEncoder", FixtureSentenceEncoder)
    args = args_for(fixture, models=("P2",), classifier="gradient_boosting")
    manifest = training.run_experiment(args)
    record = manifest["models"][0]
    assert record["classifier"] == "gradient_boosting"
    bundle = joblib.load(Path(args.artifacts_dir) / "P2.joblib")
    assert bundle["estimator"].__class__.__name__ == "GradientBoostingClassifier"
    assert bundle["estimator"].n_features_in_ == len(FEATURE_SETS["P2"])
    finder = DuplicateFinder(Path(args.artifacts_dir) / "P2.joblib", args.cache_dir, encoder=FixtureSentenceEncoder())
    predictions = read_csv(Path(args.results_dir) / "P2" / "test" / "predictions.csv")
    metadata = {row["id"]: row for row in fixture["complaints"]["test"]}
    for row in predictions:
        matches = finder.find_duplicates(metadata[row["id_a"]], [metadata[row["id_b"]]])
        assert bool(matches) == (float(row["score"]) >= float(row["threshold"]))


def test_changing_test_labels_does_not_change_fitted_b1_or_validation_threshold(tmp_path, fast_pr):
    fixture = make_duplicate_fixture(tmp_path)
    first = training.run_experiment(args_for(fixture, suffix="_first"))
    test_path = fixture["pair_dir"] / "test" / "pairs.csv"
    changed = read_csv(test_path)
    for row in changed:
        row["gold_label"] = str(1 - int(row["gold_label"]))
    write_csv(test_path, changed, PAIR_COLUMNS)
    second = training.run_experiment(args_for(fixture, suffix="_second"))
    assert first["models"][0]["threshold_selection"] == second["models"][0]["threshold_selection"]
    assert first["test_pair_fingerprint"] != second["test_pair_fingerprint"]
    before = read_csv(Path(args_for(fixture, suffix="_first").results_dir) / "B1" / "test" / "predictions.csv")
    after = read_csv(Path(args_for(fixture, suffix="_second").results_dir) / "B1" / "test" / "predictions.csv")
    assert [row["score"] for row in before] == [row["score"] for row in after]


def test_generated_candidate_labels_require_explicit_weak_label_opt_in(tmp_path, fast_pr):
    fixture = make_duplicate_fixture(tmp_path)
    columns = [column for column in PAIR_COLUMNS if column != "gold_label"]
    for split in ("train", "val", "test"):
        path = fixture["pair_dir"] / split / "pairs.csv"
        rows = read_csv(path)
        for row in rows:
            row.pop("gold_label")
        write_csv(path, rows, columns)
    args = args_for(fixture)
    with pytest.raises(ValueError, match="gold_label"):
        training.run_experiment(args)
    assert not Path(args.results_dir, "metrics.json").exists()
    args.allow_weak_labels = True
    manifest = training.run_experiment(args)
    assert any("weak generated" in message for message in manifest["warnings"])


def test_category_none_sets_explicit_missing_features_without_using_gold_category(tmp_path, monkeypatch, fast_pr):
    fixture = make_duplicate_fixture(tmp_path)
    monkeypatch.setattr(training, "SentenceEncoder", FixtureSentenceEncoder)
    args = args_for(fixture, models=("C0",))
    args.category_source = "none"
    manifest = training.run_experiment(args)
    audit = read_csv(Path(args.results_dir) / "feature_audits" / "test.csv")
    assert all(float(row["category_match"]) == 0 and float(row["category_missing"]) == 1 for row in audit)
    assert manifest["design"]["category_source"] == "none"


@pytest.mark.parametrize("kind", ["output_in_pairs", "shared_outputs", "cache_in_results", "natural_file_as_output"])
def test_output_path_isolation_is_enforced_before_training(tmp_path, kind):
    fixture = make_duplicate_fixture(tmp_path)
    args = args_for(fixture)
    if kind == "output_in_pairs":
        args.results_dir = str(fixture["pair_dir"] / "results")
    elif kind == "shared_outputs":
        args.results_dir = args.artifacts_dir
    elif kind == "cache_in_results":
        args.cache_dir = str(Path(args.results_dir) / "cache")
    else:
        args.artifacts_dir = str(fixture["complaint_paths"][0])
    with pytest.raises(ValueError):
        training.run_experiment(args)


def test_existing_outputs_are_protected_and_missing_validation_class_fails(tmp_path, fast_pr):
    fixture = make_duplicate_fixture(tmp_path)
    args = args_for(fixture)
    training.run_experiment(args)
    with pytest.raises(ValueError, match="Output already exists"):
        training.run_experiment(args)
    path = fixture["pair_dir"] / "val" / "pairs.csv"
    rows = read_csv(path)
    for row in rows:
        row["gold_label"] = "1"
    write_csv(path, rows, PAIR_COLUMNS)
    args.overwrite = True
    with pytest.raises(ValueError, match="val pairs must contain both"):
        training.run_experiment(args)


def test_main_reports_bad_pair_input_without_traceback(tmp_path, capsys):
    fixture = make_duplicate_fixture(tmp_path)
    path = fixture["pair_dir"] / "val" / "pairs.csv"
    path.write_text('text_a,text_b\n"unterminated', encoding="utf-8")
    argv = ["--pairs", str(fixture["pair_dir"]), "--models", "B1", "--results-dir", str(tmp_path / "results"),
            "--artifacts-dir", str(tmp_path / "artifacts"), "--cache-dir", str(tmp_path / "cache")]
    for complaint_path in fixture["complaint_paths"]:
        argv.extend(["--complaints", str(complaint_path)])
    assert training.main(argv) == 2
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err and "Traceback" not in captured.err
