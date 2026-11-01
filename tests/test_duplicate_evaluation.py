"""Synthetic duplicate-input, threshold, stratified-metric and error-export checks."""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
import pytest

from ml_training.duplicate_data import PAIR_REQUIRED, language_pair, load_duplicate_data
from ml_training.duplicate_metrics import evaluate_duplicate_scores, tune_threshold
from ml_training.scripts.analyze_duplicate_errors import export_errors, main as error_main


COMPLAINT_COLUMNS = ("id", "original_id", "text", "lang", "category", "location", "timestamp", "split")
PAIR_COLUMNS = (*PAIR_REQUIRED, "split", "label", "gold_label")


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def dataset(tmp_path):
    complaints, pairs = [], {split: [] for split in ("train", "val", "test")}
    for split in pairs:
        for number in range(3):
            complaints.append({
                "id": f"{split}-r{number}", "original_id": f"{split}-g{0 if number < 2 else 1}",
                "text": f"Complaint {split} {number}", "lang": ("bn", "en", "banglish")[number],
                "category": "water" if number < 2 else "roads", "location": "Mirpur 10",
                "timestamp": "2026-01-01T12:00:00+06:00", "split": split,
            })
        own = complaints[-3:]
        for index, (a, b, kind, label) in enumerate(((own[0], own[1], "positive_cross_language", "1"),
                                                    (own[0], own[2], "easy_negative", "0"))):
            pairs[split].append({
                "pair_id": f"{split}-p{index}", "split": split,
                "id_a": a["id"], "id_b": b["id"], "original_id_a": a["original_id"], "original_id_b": b["original_id"],
                "text_a": a["text"], "text_b": b["text"], "loc_a": a["location"], "loc_b": b["location"],
                "time_a": "2026-01-01T06:00:00Z", "time_b": "2026-01-01T06:00:00Z",
                "label": label, "gold_label": label, "pair_type": kind,
            })
    complaint_path = write_csv(tmp_path / "complaints.csv", COMPLAINT_COLUMNS, complaints)
    pair_dir = tmp_path / "pairs"
    for split, rows in pairs.items():
        write_csv(pair_dir / split / "pairs.csv", PAIR_COLUMNS, rows)
    return {"pairs": pairs, "complaints": complaints, "pair_dir": pair_dir,
            "complaints_path": complaint_path, "root": tmp_path}


def load(dataset, **kwargs):
    return load_duplicate_data(dataset["pair_dir"], [dataset["complaints_path"]], **kwargs)


def rewrite_pairs(dataset, split, columns=PAIR_COLUMNS):
    write_csv(dataset["pair_dir"] / split / "pairs.csv", columns, dataset["pairs"][split])


def rewrite_complaints(dataset):
    write_csv(dataset["complaints_path"], COMPLAINT_COLUMNS, dataset["complaints"])


def test_loads_every_pair_enriches_independent_metadata_and_normalizes_time(dataset):
    data = load(dataset)
    for split in ("train", "val", "test"):
        frame = data[split]
        assert len(frame) == 2
        assert frame["label"].tolist() == [1, 0]
        assert frame["label_source"].tolist() == ["gold_label", "gold_label"]
        assert frame["language_pair"].tolist() == ["bn-en", "bn-banglish"]
        assert frame["category_b"].tolist() == ["water", "roads"]
        assert frame["time_a"].tolist() == ["2026-01-01T06:00:00Z"] * 2
        assert frame["candidate_label"].tolist() == ["1", "0"]


def test_combined_file_and_empty_split_headers_are_supported(dataset):
    combined = dataset["root"] / "all_pairs.csv"
    write_csv(combined, PAIR_COLUMNS, dataset["pairs"]["train"] + dataset["pairs"]["val"])
    data = load_duplicate_data(combined, [dataset["complaints_path"]])
    assert len(data["train"]) == len(data["val"]) == 2
    assert data["test"].empty
    assert {"lang_a", "category_b", "language_pair", "label_source"} <= set(data["test"].columns)


def test_candidate_labels_require_explicit_weak_opt_in(dataset):
    columns = tuple(name for name in PAIR_COLUMNS if name != "gold_label")
    for split in dataset["pairs"]:
        rewrite_pairs(dataset, split, columns)
    with pytest.raises(ValueError, match="adjudicated gold_label"):
        load(dataset)
    data = load(dataset, allow_weak_labels=True)
    assert all(frame["label_source"].eq("label").all() for frame in (data["train"], data["val"], data["test"]))
    assert any("not an independently annotated" in note for note in data["warnings"])


def test_human_label_supported_but_gold_takes_precedence(dataset):
    columns = tuple(name for name in PAIR_COLUMNS if name != "gold_label") + ("human_label",)
    for split, rows in dataset["pairs"].items():
        for row in rows:
            row["human_label"] = row["gold_label"]
        rewrite_pairs(dataset, split, columns)
    assert load(dataset)["test"]["label_source"].eq("human_label").all()
    dataset["pairs"]["test"][0]["gold_label"] = "0"
    rewrite_pairs(dataset, "test", (*PAIR_COLUMNS, "human_label"))
    data = load(dataset)
    assert data["test"]["label"].iloc[0] == 0
    assert any("takes precedence" in note for note in data["warnings"])


@pytest.mark.parametrize("value", ["", "skip", "2", "1.0"])
def test_no_unlabeled_or_nonbinary_rows_are_silently_dropped(dataset, value):
    dataset["pairs"]["test"][0]["gold_label"] = value
    rewrite_pairs(dataset, "test")
    with pytest.raises(ValueError, match="no rows will be dropped"):
        load(dataset)


@pytest.mark.parametrize("field,value,fragment", [
    ("id_a", "missing", "missing from supplied metadata"),
    ("original_id_a", "wrong-group", "original_id_a"),
    ("text_a", "Different incident", "text_a"),
    ("loc_a", "Uttara", "loc_a"),
    ("time_a", "2026-01-02", "time_a"),
    ("time_a", "not a time", "invalid ISO"),
])
def test_pair_metadata_must_match_complaint_table(dataset, field, value, fragment):
    dataset["pairs"]["test"][0][field] = value
    rewrite_pairs(dataset, "test")
    with pytest.raises(ValueError, match=fragment):
        load(dataset)


def test_complaint_groups_must_be_disjoint_even_when_not_used_in_pairs(dataset):
    dataset["complaints"].append(dict(dataset["complaints"][0], id="unused-val", split="val"))
    rewrite_complaints(dataset)
    with pytest.raises(ValueError, match="Split leakage"):
        load(dataset)


def test_conflicting_pair_split_and_duplicate_pair_ids_fail(dataset):
    dataset["pairs"]["test"][0]["split"] = "val"
    rewrite_pairs(dataset, "test")
    with pytest.raises(ValueError, match="split column disagrees"):
        load(dataset)
    dataset["pairs"]["test"][0]["split"] = "test"
    dataset["pairs"]["test"][0]["pair_id"] = dataset["pairs"]["train"][0]["pair_id"]
    rewrite_pairs(dataset, "test")
    with pytest.raises(ValueError, match="Duplicate pair_id"):
        load(dataset)


def test_duplicate_reversed_pair_is_rejected(dataset):
    row = dict(dataset["pairs"]["test"][0], pair_id="test-reverse")
    for left, right in (("id_a", "id_b"), ("original_id_a", "original_id_b"), ("text_a", "text_b"),
                        ("loc_a", "loc_b"), ("time_a", "time_b")):
        row[left], row[right] = row[right], row[left]
    dataset["pairs"]["test"].append(row)
    rewrite_pairs(dataset, "test")
    with pytest.raises(ValueError, match="Duplicate unordered"):
        load(dataset)


def test_supplied_language_columns_are_replaced_without_duplicate_headers(dataset):
    for split, rows in dataset["pairs"].items():
        for row in rows:
            row["lang_a"], row["category_b"] = "wrong", "not-a-feature-label"
        rewrite_pairs(dataset, split, (*PAIR_COLUMNS, "lang_a", "category_b"))
    frame = load(dataset)["test"]
    assert len(frame.columns) == len(set(frame.columns))
    assert frame["lang_a"].eq("bn").all()
    assert frame["category_b"].tolist() == ["water", "roads"]


@pytest.mark.parametrize("a,b,expected", [("en", "bn", "bn-en"), ("banglish", "bn", "bn-banglish"),
                                          ("banglish", "en", "en-banglish"), ("bn", "bn", "bn-bn")])
def test_language_pair_is_unordered_and_fixed(a, b, expected):
    assert language_pair(a, b) == expected


def test_invalid_schema_and_malformed_quote_fail_clearly(dataset):
    path = dataset["pair_dir"] / "test" / "pairs.csv"
    path.write_text('item_id,annotator,label\npair:example,alice,1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Adjudicate the long-form"):
        load(dataset)
    header = ",".join(PAIR_COLUMNS)
    path.write_text(header + '\n"unterminated\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot parse UTF-8 CSV"):
        load(dataset)


def test_threshold_tuning_matches_bruteforce_with_conservative_ties():
    truth = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.9, 0.9, 0.4, 0.1, 0.4, -0.1])
    candidates = [*np.unique(scores), np.nextafter(scores.max(), np.inf)]
    best = []
    for threshold in candidates:
        prediction = scores >= threshold
        tp, fp, fn = sum(prediction & (truth == 1)), sum(prediction & (truth == 0)), sum(~prediction & (truth == 1))
        f1 = 2 * tp / (2 * tp + fp + fn)
        best.append((f1, threshold))
    expected_f1, expected_threshold = max(best)
    chosen = tune_threshold(truth, scores)
    assert chosen["threshold"] == expected_threshold
    assert chosen["f1"] == pytest.approx(expected_f1)
    assert chosen["candidates_examined"] == len(candidates)
    assert chosen["decision_rule"] == "score >= threshold"


def test_threshold_ties_choose_none_for_no_positive_validation():
    chosen = tune_threshold([0, 0], [-0.7, 0.9])
    assert chosen["threshold"] > 0.9
    assert chosen["tie_break"] == "higher_threshold"


@pytest.mark.parametrize("truth,scores", [([], []), ([0, 2], [0.1, 0.2]), ([0, 1], [float("nan"), 1]),
                                          ([0, 1], [[0, 1], [1, 0]]), ([0], [0, 1])])
def test_invalid_threshold_inputs_fail(truth, scores):
    with pytest.raises(ValueError):
        tune_threshold(truth, scores)


def test_evaluation_exports_same_test_ids_and_required_breakdowns(dataset, tmp_path):
    frame = load(dataset)["test"]
    result = evaluate_duplicate_scores(frame, [0.8, 0.7], threshold=0.75, output_dir=tmp_path / "evaluation", model_name="synthetic")
    assert result["n"] == len(frame)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}
    assert result["pr_auc"] == result["average_precision"]
    assert "step-integral" in result["pr_auc_definition"]
    for required in ("bn-en", "bn-banglish", "en-banglish"):
        assert required in result["by_language_pair"]
    assert result["by_language_pair"]["en-banglish"]["status"] == "skipped_empty"
    assert result["by_language_pair"]["en-banglish"]["f1"] is None
    assert result["by_pair_type"]["hard_negative"]["status"] == "skipped_empty"
    assert result["by_pair_type"]["easy_negative"]["pr_auc"] is None
    assert result["by_pair_type"]["easy_negative"]["false_positive_rate"] == 0
    assert "Only one true label" in result["by_pair_type"]["easy_negative"]["pr_auc_undefined_reason"]
    predictions = pd.read_csv(result["files"]["predictions.csv"], dtype=str, keep_default_na=False)
    assert predictions["pair_id"].tolist() == frame["pair_id"].tolist()
    assert predictions["prediction"].tolist() == ["1", "0"]
    assert json.loads((tmp_path / "evaluation" / "metrics.json").read_text()) == result
    assert (tmp_path / "evaluation" / "precision_recall_curve.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_empty_evaluation_records_nulls_and_headers(tmp_path):
    frame = pd.DataFrame(columns=["pair_id", "label", "pair_type", "language_pair"])
    result = evaluate_duplicate_scores(frame, [], 0.5, tmp_path)
    assert result["status"] == "skipped_empty"
    assert result["precision"] is None and result["pr_auc"] is None
    assert not (tmp_path / "precision_recall_curve.png").exists()
    assert pd.read_csv(tmp_path / "predictions.csv").empty


def test_predicted_errors_are_ranked_by_threshold_margin_and_keep_provenance(tmp_path):
    frame = pd.DataFrame({"pair_id": ["fp-low", "fp-high", "fn-high", "fn-low"], "label": [0, 0, 1, 1],
                          "pair_type": ["hard_negative", "easy_negative", "positive_cross_language", "positive_paraphrase"],
                          "language_pair": ["bn-en", "bn-banglish", "en-banglish", "bn-bn"],
                          "text_a": ["a", "b", "c", "d"], "original_id_a": ["g1", "g2", "g3", "g4"]})
    report = evaluate_duplicate_scores(frame, [0.7, 0.9, 0.4, 0.1], 0.5, tmp_path / "prediction")
    result = export_errors(report["files"]["predictions.csv"], tmp_path / "errors", top_k=1)
    positives = pd.read_csv(tmp_path / "errors" / "false_positives.csv")
    negatives = pd.read_csv(tmp_path / "errors" / "false_negatives.csv")
    assert positives["pair_id"].tolist() == ["fp-high"]
    assert negatives["pair_id"].tolist() == ["fn-low"]
    assert "original_id_a" in positives and "language_pair" in negatives
    assert result["false_positives"] == result["false_negatives"] == 2
    assert result["exported_false_positives"] == result["exported_false_negatives"] == 1
    with pytest.raises(ValueError, match="already exist"):
        export_errors(report["files"]["predictions.csv"], tmp_path / "errors")


def test_mixed_threshold_error_ranking_and_invalid_cli_are_rejected(tmp_path, capsys):
    path = write_csv(tmp_path / "predictions.csv", ("pair_id", "label", "score", "threshold", "prediction"), [
        {"pair_id": "a", "label": "0", "score": "0.7", "threshold": "0.5", "prediction": "1"},
        {"pair_id": "b", "label": "1", "score": "0.3", "threshold": "0.8", "prediction": "0"},
    ])
    with pytest.raises(ValueError, match="one model/threshold"):
        export_errors(path, tmp_path / "errors")
    assert error_main(["--predictions", str(path), "--output-dir", str(tmp_path / "errors"), "--top-k", "0"]) == 2
    assert "positive integer" in capsys.readouterr().err
