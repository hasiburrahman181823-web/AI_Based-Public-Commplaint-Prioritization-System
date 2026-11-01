"""Research-input validation preserves incident isolation and all accepted rows."""

from __future__ import annotations

import csv
import json

import pytest

from ml_training.config import SCHEMA_COLUMNS
from ml_training.experiment_data import (
    load_experiment_data,
    preprocess_text,
    read_experiment_csv,
)


NATURAL_COLUMNS = ("text", "lang", "category", "location", "timestamp")


def record(identifier, group, text="পানি নেই", category="water", severity="low", lang="bn"):
    return {
        "id": identifier, "original_id": group, "text": text, "lang": lang,
        "category": category, "severity": severity, "source_type": "fixture",
        "location": "Dhaka", "timestamp": "2026-01-01T06:00:00Z",
    }


def write_csv(path, rows, columns=SCHEMA_COLUMNS):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def experiment(tmp_path):
    rows = {
        "train": [
            record("train-1", "train-water"),
            record("train-2", "train-roads", text="রাস্তা ভাঙা", category="roads", severity="high"),
        ],
        "val": [
            record("val-1", "val-water", text="পানি সমস্যা"),
            record("val-2", "val-roads", text="রাস্তা বন্ধ", category="roads", severity="high"),
        ],
        "test": [
            record("test-1", "test-water", text="water unavailable", lang="en"),
            record("test-2", "test-roads", text="rasta bhanga", category="roads", severity="high", lang="banglish"),
        ],
    }
    for split in rows:
        write_csv(tmp_path / f"{split}.csv", rows[split])
    return {"directory": tmp_path, "rows": rows}


def test_disjoint_standard_splits_return_exact_rows_and_training_taxonomy(experiment):
    loaded = load_experiment_data(experiment["directory"], severity_order=["low", "high"])
    assert loaded["category_labels"] == ["roads", "water"]
    assert loaded["severity_labels"] == ["low", "high"]
    assert loaded["natural"].empty
    assert loaded["warnings"] == []
    for split in ("train", "val", "test"):
        assert len(loaded[split]) == len(experiment["rows"][split])
        assert set(loaded[split]["id"]) == {row["id"] for row in experiment["rows"][split]}


@pytest.mark.parametrize("first,second", [("train", "val"), ("train", "test"), ("val", "test")])
def test_original_incident_cannot_cross_source_splits(experiment, first, second):
    experiment["rows"][second][0]["original_id"] = experiment["rows"][first][0]["original_id"]
    write_csv(experiment["directory"] / f"{second}.csv", experiment["rows"][second])
    with pytest.raises(ValueError, match="Split leakage: original_id"):
        load_experiment_data(experiment["directory"])


@pytest.mark.parametrize("first,second", [("train", "val"), ("train", "test"), ("val", "test")])
def test_row_identifiers_are_globally_unique_across_source_splits(experiment, first, second):
    experiment["rows"][second][0]["id"] = experiment["rows"][first][0]["id"]
    write_csv(experiment["directory"] / f"{second}.csv", experiment["rows"][second])
    with pytest.raises(ValueError, match="Row ID.*occurs in"):
        load_experiment_data(experiment["directory"])


@pytest.mark.parametrize("source_split", ["train", "val", "test"])
@pytest.mark.parametrize("identifier_field,fragment", [("id", "Row ID"), ("original_id", "Split leakage")])
def test_natural_data_cannot_reuse_source_ids_or_groups(experiment, source_split, identifier_field, fragment):
    source = experiment["rows"][source_split][0]
    natural = record("natural-unique-id", "natural-unique-group", text="An independent natural complaint", lang="en")
    natural[identifier_field] = source[identifier_field]
    path = write_csv(experiment["directory"] / "natural.csv", [natural])
    with pytest.raises(ValueError, match=fragment):
        load_experiment_data(experiment["directory"], path)


def test_five_column_natural_schema_retains_rows_without_inventing_severity(experiment):
    rows = [
        {"text": "Naturally written water complaint", "lang": "en", "category": "water", "location": "Road 1", "timestamp": ""},
        {"text": "rasta bhanga ekhane", "lang": "banglish", "category": "roads", "location": "Road 2", "timestamp": ""},
    ]
    path = write_csv(experiment["directory"] / "natural.csv", rows, NATURAL_COLUMNS)
    loaded = load_experiment_data(experiment["directory"], path, ["low", "high"])
    natural = loaded["natural"]
    assert len(natural) == 2
    assert natural["severity"].tolist() == ["", ""]
    assert natural["id"].str.startswith("natural_").all()
    assert natural["original_id"].str.startswith("natural_").all()
    assert natural["location"].tolist() == ["Road 1", "Road 2"]
    assert any("no severity labels" in warning for warning in loaded["warnings"])
    assert any("manual provenance audit" in warning for warning in loaded["warnings"])
    repeated = load_experiment_data(experiment["directory"], path, ["low", "high"])
    assert natural["id"].tolist() == repeated["natural"]["id"].tolist()


def test_optional_natural_severity_is_retained_when_present(experiment):
    natural = record("natural-id", "natural-group", text="A severe natural road report", category="roads", severity="high", lang="en")
    path = write_csv(experiment["directory"] / "natural.csv", [natural])
    loaded = load_experiment_data(experiment["directory"], path, ["low", "high"])
    assert loaded["natural"]["severity"].tolist() == ["high"]
    assert not any("no severity labels" in warning for warning in loaded["warnings"])


@pytest.mark.parametrize("order", [["low"], ["low", "low"], ["low", " "], ["", "high"]])
def test_invalid_severity_order_is_rejected(experiment, order):
    with pytest.raises(ValueError, match="severity_order needs at least two distinct nonempty"):
        load_experiment_data(experiment["directory"], severity_order=order)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_severity_labels_outside_explicit_order_are_rejected(experiment, split):
    experiment["rows"][split][0]["severity"] = "unexpected"
    write_csv(experiment["directory"] / f"{split}.csv", experiment["rows"][split])
    with pytest.raises(ValueError, match="severity labels not in --severity-order"):
        load_experiment_data(experiment["directory"], severity_order=["low", "high"])


def test_natural_unknown_severity_is_rejected_instead_of_recoded(experiment):
    row = record("natural-id", "natural-group", text="Natural water issue", severity="urgent", lang="en")
    path = write_csv(experiment["directory"] / "natural.csv", [row])
    with pytest.raises(ValueError, match="natural: severity labels not in --severity-order"):
        load_experiment_data(experiment["directory"], path, ["low", "high"])


@pytest.mark.parametrize("split", ["val", "test"])
def test_unknown_held_out_category_fails_instead_of_dropping_records(experiment, split):
    experiment["rows"][split][0]["category"] = "unseen-category"
    write_csv(experiment["directory"] / f"{split}.csv", experiment["rows"][split])
    with pytest.raises(ValueError, match="categories absent from training.*unseen-category"):
        load_experiment_data(experiment["directory"])


def test_unknown_natural_category_fails_instead_of_dropping_records(experiment):
    row = {"text": "Another natural topic", "lang": "en", "category": "unseen", "location": "", "timestamp": ""}
    path = write_csv(experiment["directory"] / "natural.csv", [row], NATURAL_COLUMNS)
    with pytest.raises(ValueError, match="natural: categories absent from training"):
        load_experiment_data(experiment["directory"], path)


@pytest.mark.parametrize("split", ["train", "val"])
def test_empty_training_or_calibration_split_fails(experiment, split):
    write_csv(experiment["directory"] / f"{split}.csv", [])
    with pytest.raises(ValueError, match=rf"{split} split is empty"):
        load_experiment_data(experiment["directory"])


def test_empty_test_split_is_allowed_with_explicit_warning(experiment):
    write_csv(experiment["directory"] / "test.csv", [])
    loaded = load_experiment_data(experiment["directory"])
    assert loaded["test"].empty
    assert any("Test split is empty" in warning for warning in loaded["warnings"])


def test_natural_normalized_text_overlap_warns_without_dropping_record(experiment):
    rows = [{"text": "  পানি নেই https://example.com ", "lang": "bn", "category": "water", "location": "", "timestamp": ""}]
    path = write_csv(experiment["directory"] / "natural.csv", rows, NATURAL_COLUMNS)
    loaded = load_experiment_data(experiment["directory"], path)
    assert len(loaded["natural"]) == 1
    assert any("1 natural rows exactly match normalized source text" in warning for warning in loaded["warnings"])


def test_duplicate_source_text_with_distinct_row_ids_is_retained(experiment):
    experiment["rows"]["train"].append(record("train-repeat", "train-water", text="পানি নেই"))
    write_csv(experiment["directory"] / "train.csv", experiment["rows"]["train"])
    loaded = load_experiment_data(experiment["directory"])
    assert len(loaded["train"]) == 3
    assert loaded["train"]["text"].eq("পানি নেই").sum() == 2
    assert loaded["train"]["id"].is_unique


def test_numeric_identifiers_labels_and_na_strings_are_not_inferred(tmp_path):
    row = record("0001", "0000", category="01", severity="002", text="NA", lang="en")
    frame = read_experiment_csv(write_csv(tmp_path / "leading.csv", [row]))
    assert frame.loc[0, "id"] == "0001"
    assert frame.loc[0, "original_id"] == "0000"
    assert frame.loc[0, "category"] == "01"
    assert frame.loc[0, "severity"] == "002"
    assert frame.loc[0, "raw_text"] == "NA"
    assert frame.loc[0, "text"] == "na"


def test_shared_preprocessing_applies_language_specific_normalization():
    assert preprocess_text("  পানি ১২ https://example.com ＴＥＳＴ ", "bn") == "পানি 12 test"
    assert preprocess_text("PANY NEI biddut paniiii", "banglish") == "pani nai bidyut pani"
    assert preprocess_text("PANY NEI FOOOD", "en") == "pany nei foood"
    with pytest.raises(ValueError, match="Unsupported language"):
        preprocess_text("complaint", "unknown")


def test_custom_banglish_dictionary_is_shared_with_direct_preprocessing(tmp_path):
    variants = tmp_path / "variants.json"
    variants.write_text(json.dumps({"custom": "canonical"}), encoding="utf-8")
    assert preprocess_text("CUSTOM", "banglish", variants) == "canonical"


def test_cleaning_retains_raw_text_for_auditing_and_rejects_empty_output(tmp_path):
    row = record("id-1", "group-1", text="PANY NEI ২", lang="banglish")
    path = write_csv(tmp_path / "clean.csv", [row])
    frame = read_experiment_csv(path)
    assert frame.loc[0, "raw_text"] == "PANY NEI ২"
    assert frame.loc[0, "text"] == "pani nai 2"
    row["text"] = "https://example.com"
    write_csv(path, [row])
    with pytest.raises(ValueError, match="text became empty after cleaning"):
        read_experiment_csv(path)


@pytest.mark.parametrize("field", ["id", "original_id", "text", "category", "severity"])
def test_source_required_values_must_be_nonempty(tmp_path, field):
    row = record("id-1", "group-1")
    row[field] = " "
    with pytest.raises(ValueError, match=f"'{field}' contains blank values"):
        read_experiment_csv(write_csv(tmp_path / "blank.csv", [row]))


def test_source_schema_requires_full_prepared_columns_but_natural_schema_does_not(tmp_path):
    natural = {"text": "Some water problem", "lang": "en", "category": "water", "location": "", "timestamp": ""}
    path = write_csv(tmp_path / "five.csv", [natural], NATURAL_COLUMNS)
    with pytest.raises(ValueError, match="missing columns"):
        read_experiment_csv(path)
    assert len(read_experiment_csv(path, natural=True)) == 1


def test_malformed_width_and_duplicate_source_ids_are_rejected(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text(
        ",".join(SCHEMA_COLUMNS) + "\n" + ",".join(["x"] * (len(SCHEMA_COLUMNS) + 1)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fields; expected"):
        read_experiment_csv(path)
    row = record("same-id", "same-group")
    write_csv(path, [row, row])
    with pytest.raises(ValueError, match="duplicate id values"):
        read_experiment_csv(path)
