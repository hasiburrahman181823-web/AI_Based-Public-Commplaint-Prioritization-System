"""CSV validation, schema behavior, export round trips and CLI contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from ml_training.config import SCHEMA_COLUMNS
from ml_training.scripts.prepare_data import (
    DataValidationError,
    load_and_prepare,
    validate_mapping,
    write_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "ml_training" / "scripts" / "prepare_data.py"


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(**overrides):
    result = {
        "id": "0001",
        "original_id": "0010",
        "text": " WATER  নেই  ১২৩ ",
        "lang": "bn",
        "category": "water",
        "severity": "02",
    }
    result.update(overrides)
    return result


def test_default_schema_preserves_string_labels_and_leading_zero_ids(tmp_path):
    source = write_csv(tmp_path / "source.csv", [row(category="NA", severity="02")])
    frame = load_and_prepare(source)
    assert list(frame.columns) == list(SCHEMA_COLUMNS)
    assert frame.iloc[0].to_dict() == {
        "id": "0001", "original_id": "0010", "text": "water নেই 123", "lang": "bn",
        "category": "NA", "severity": "02", "source_type": "banglapsg",
        "location": "", "timestamp": "",
    }
    assert all(isinstance(value, str) for value in frame.iloc[0])


def test_mapping_renames_required_and_optional_source_columns(tmp_path):
    source = write_csv(tmp_path / "source.csv", [{
        "message": "PANY NEI", "topic": "পানি", "level": "high",
        "group": "009", "language": "banglish",
    }])
    mapping = {
        "text": "message", "category": "topic", "severity": "level",
        "original_id": "group", "lang": "language",
    }
    frame = load_and_prepare(source, mapping)
    assert frame.loc[0, "text"] == "pani nai"
    assert frame.loc[0, "category"] == "পানি"
    assert frame.loc[0, "original_id"] == "009"
    assert frame.loc[0, "id"].startswith("row_")


def test_missing_column_error_identifies_mapping_and_available_headers(tmp_path):
    source = write_csv(tmp_path / "source.csv", [{"message": "water", "category": "water"}])
    with pytest.raises(DataValidationError) as error:
        load_and_prepare(source, {"text": "message", "severity": "level"})
    message = str(error.value)
    assert "Missing required CSV columns" in message
    assert "'severity' -> 'level'" in message
    assert "Available headers" in message
    assert "COLUMN_MAPPING" in message


@pytest.mark.parametrize(
    "mapping, message",
    [
        ([], "JSON object"),
        ({"unknown": "x"}, "Unknown column mapping keys"),
        ({"text": ""}, "nonempty CSV header"),
        ({"text": 1}, "nonempty CSV header"),
        ({"text": "category"}, "different CSV header"),
    ],
)
def test_invalid_mappings_have_actionable_errors(mapping, message):
    with pytest.raises(DataValidationError, match=message):
        validate_mapping(mapping)


@pytest.mark.parametrize("text", ["", "  \n\t", "https://example.com/path"])
def test_blank_or_cleaned_empty_text_is_rejected(tmp_path, text):
    source = write_csv(tmp_path / "source.csv", [row(text=text)])
    with pytest.raises(DataValidationError, match="Column 'text' contains empty values.*2"):
        load_and_prepare(source)


@pytest.mark.parametrize("label", ["category", "severity"])
def test_empty_label_is_not_silently_dropped(tmp_path, label):
    source = write_csv(tmp_path / "source.csv", [row(**{label: " "})])
    with pytest.raises(DataValidationError, match=f"Column '{label}' contains empty values"):
        load_and_prepare(source)


def test_fallback_ids_are_unique_stable_and_group_same_text_across_labels(tmp_path):
    records = [
        row(id="", original_id="", text=" PANI  NAI "),
        row(id="", original_id="", text="pani nai", category="other", severity="03"),
        row(id="", original_id="", text=" PANI  NAI "),
    ]
    source = write_csv(tmp_path / "source.csv", records)
    with pytest.warns(UserWarning, match="Generated original_id"):
        first = load_and_prepare(source)
    with pytest.warns(UserWarning, match="Generated original_id"):
        second = load_and_prepare(source)
    pd.testing.assert_frame_equal(first, second)
    assert first["id"].is_unique
    assert first["original_id"].nunique() == 1
    assert first["original_id"].iloc[0].startswith("auto_")


def test_language_heuristic_is_conservative_and_explicit_banglish_is_normalized(tmp_path):
    records = [
        row(id="bn", text="পানি নেই", lang=""),
        row(id="latin", text="PANY NEI", lang=""),
        row(id="explicit", text="PANY NEI", lang="BANGlish"),
    ]
    source = write_csv(tmp_path / "source.csv", records)
    with pytest.warns(UserWarning, match="Supply lang=banglish"):
        frame = load_and_prepare(source)
    assert frame["lang"].tolist() == ["bn", "en", "banglish"]
    assert frame["text"].tolist() == ["পানি নেই", "pany nei", "pani nai"]


def test_custom_banglish_dictionary_is_used(tmp_path):
    source = write_csv(tmp_path / "source.csv", [row(text="JOL", lang="banglish")])
    variants = tmp_path / "variants.json"
    variants.write_text('{"jol": "pani"}', encoding="utf-8")
    assert load_and_prepare(source, variants_path=variants).loc[0, "text"] == "pani"


def test_invalid_language_is_rejected(tmp_path):
    source = write_csv(tmp_path / "source.csv", [row(lang="bengali")])
    with pytest.raises(DataValidationError, match="Invalid lang.*2"):
        load_and_prepare(source)


def test_duplicate_ids_are_rejected_even_for_different_groups(tmp_path):
    source = write_csv(tmp_path / "source.csv", [row(), row(original_id="different")])
    with pytest.raises(DataValidationError, match="Duplicate id values.*2, 3"):
        load_and_prepare(source)


def test_timestamp_offsets_are_normalized_and_optional_metadata_preserved(tmp_path):
    source = write_csv(tmp_path / "source.csv", [
        row(timestamp="2026-01-01T12:00:00+06:00", location=" Dhaka ", source_type=" sms "),
        row(id="0002", timestamp="2026-01-01T06:00:00"),
    ])
    frame = load_and_prepare(source)
    assert frame["timestamp"].tolist() == ["2026-01-01T06:00:00Z"] * 2
    assert frame.loc[0, "location"] == "Dhaka"
    assert frame.loc[0, "source_type"] == "sms"


@pytest.mark.parametrize("stamp", ["yesterday", "2026-02-30T12:00:00Z", "NaT"])
def test_invalid_timestamps_are_rejected(tmp_path, stamp):
    source = write_csv(tmp_path / "source.csv", [row(timestamp=stamp)])
    with pytest.raises(DataValidationError, match="Invalid ISO-8601 timestamp.*2"):
        load_and_prepare(source)


def test_empty_csv_and_duplicate_headers_have_clear_errors(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("text,category,severity\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="no data rows"):
        load_and_prepare(source)
    source.write_text("text,text,category,severity\na,b,water,1\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Duplicate source CSV headers"):
        load_and_prepare(source)


@pytest.mark.parametrize(
    "records, fields",
    [
        ("unquoted, comma,water,2\nanother, unquoted,water,3\n", 4),
        ("water,2\nelectricity,3\n", 2),
    ],
)
def test_inconsistent_csv_width_is_rejected_without_shifting_labels(tmp_path, records, fields):
    source = tmp_path / "malformed.csv"
    source.write_text("text,category,severity\n" + records, encoding="utf-8")
    with pytest.raises(DataValidationError) as error:
        load_and_prepare(source)
    assert f"CSV record 2 has {fields} fields; expected 3" in str(error.value)
    assert "missing/extra fields" in str(error.value)


def test_unmatched_csv_quote_has_clear_read_error(tmp_path):
    source = tmp_path / "malformed.csv"
    source.write_text('text,category,severity\n"unclosed text,water,2\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="Cannot read UTF-8 CSV.*unexpected end of data"):
        load_and_prepare(source)


def test_correctly_quoted_commas_and_multiline_text_are_supported(tmp_path):
    source = write_csv(tmp_path / "source.csv", [row(text="first, second\nthird")])
    frame = load_and_prepare(source)
    assert frame.loc[0, "text"] == "first, second third"
    assert frame.loc[0, "category"] == "water"


def test_missing_source_error_explains_how_to_provide_input(tmp_path):
    with pytest.raises(DataValidationError, match="CSV not found.*Place your data"):
        load_and_prepare(tmp_path / "missing.csv")


def test_output_formats_round_trip_schema_and_guard_overwrite(tmp_path):
    source = write_csv(tmp_path / "source.csv", [row()])
    frame = load_and_prepare(source)
    splits = {"train": frame, "val": frame.iloc[:0].copy(), "test": frame.iloc[:0].copy()}
    destination = tmp_path / "prepared"
    manifest = {"rows": 1, "warnings": [], "note": "বাংলা"}
    paths = write_outputs(splits, destination, "both", manifest)
    assert len(paths) == 7
    assert json.loads((destination / "manifest.json").read_text(encoding="utf-8")) == manifest
    for name, expected in splits.items():
        csv_frame = pd.read_csv(destination / f"{name}.csv", dtype=str, keep_default_na=False)
        parquet_frame = pd.read_parquet(destination / f"{name}.parquet")
        pd.testing.assert_frame_equal(csv_frame, expected.reset_index(drop=True), check_dtype=False)
        pd.testing.assert_frame_equal(parquet_frame, expected.reset_index(drop=True), check_dtype=False)
    with pytest.raises(DataValidationError, match="already exists.*--overwrite"):
        write_outputs(splits, destination, "csv", manifest)
    updated = {"rows": 1, "note": "replacement"}
    paths = write_outputs(splits, destination, "csv", updated, overwrite=True)
    assert len(paths) == 4
    assert not list(destination.glob("*.parquet"))
    assert json.loads((destination / "manifest.json").read_text(encoding="utf-8")) == updated


def test_cli_direct_script_from_different_directory_and_error_exit(tmp_path):
    records = [row(id=f"{i:04}", original_id=f"group-{i}") for i in range(20)]
    source = write_csv(tmp_path / "source.csv", records)
    destination = tmp_path / "prepared"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(source), "--output-dir", str(destination), "--format", "csv"],
        cwd=tmp_path, text=True, capture_output=True, encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert "Loaded 20 rows / 20 original groups" in completed.stdout
    assert "Leakage check: PASS" in completed.stdout
    assert "Wrote 4 files" in completed.stdout
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 20
    assert sum(stats["rows"] for stats in manifest["splits"].values()) == 20
    bad = subprocess.run(
        [sys.executable, "-m", "ml_training.scripts.prepare_data", "--input", str(tmp_path / "missing.csv")],
        cwd=REPO_ROOT, text=True, capture_output=True, encoding="utf-8",
    )
    assert bad.returncode == 2
    assert "ERROR: CSV not found" in bad.stderr
    assert "Traceback" not in bad.stderr
