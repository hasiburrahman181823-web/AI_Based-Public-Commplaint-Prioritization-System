"""Offline transliteration, immutable translation queues and linked split safety."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ml.transliterate import transliterate_bangla
from ml_training.config import SCHEMA_COLUMNS
from ml_training.scripts.build_multilingual_set import (
    DERIVED_COLUMNS, QUEUE_COLUMNS, SPLITS, MultilingualValidationError,
    export_multilingual, import_translations, main,
)


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_splits(path: Path):
    return {name: read_csv(path / f"{name}.csv") for name in SPLITS}


@pytest.fixture
def source_splits(tmp_path):
    directory = tmp_path / "processed"
    for split in SPLITS:
        rows = []
        for index in range(3):
            source = {
                "id": f"{split}-bn-{index}", "original_id": f"{split}-incident-{index}",
                "text": "পানি নাই 2 দিন।", "lang": "bn", "category": "water",
                "severity": "02", "source_type": "banglapsg", "location": "Dhaka",
                "timestamp": "2026-10-29T10:30:00Z",
            }
            rows.append(source)
            # An existing English variant must not make this group count twice.
            rows.append(dict(source, id=f"{split}-en-{index}", text="No water for 2 days.", lang="en"))
        write_csv(directory / f"{split}.csv", rows, SCHEMA_COLUMNS)
    return directory


def fill_queue(directory: Path, *, count=None):
    queue = read_csv(directory / "translation_queue.csv")
    for index, row in enumerate(queue):
        if count is None or index < count:
            row["english_text"] = "No water for 2 days."
    path = directory / "filled.csv"
    write_csv(path, queue, QUEUE_COLUMNS)
    return path


@pytest.mark.parametrize("text, expected", [
    ("পানি নাই", "pani nai"), ("বিদ্যুৎ", "bidyut"),
    ("বাংলা", "bangla"), ("জল", "jol"),
    ("ঢাকা ১২৩!", "dhaka 123!"), ("নেই", "nei"),
    ("ড় ঢ় য়", "r rh y"), ("পা\u200dনি", "pani"),
    ("LATIN: 42?", "latin: 42?"), ("", ""),
])
def test_transliteration_rules(text, expected):
    assert transliterate_bangla(text) == expected


def test_transliteration_rejects_missing_values():
    with pytest.raises(TypeError, match="string"):
        transliterate_bangla(None)


def test_export_samples_original_groups_and_retains_splits(source_splits, tmp_path):
    output = tmp_path / "multilingual"
    manifest = export_multilingual(source_splits, output, 9, seed=42)
    queue = read_csv(output / "translation_queue.csv")
    assert len(queue) == 9
    assert len({row["original_id"] for row in queue}) == 9
    assert all(row["english_text"] == "" for row in queue)
    for split, rows in read_splits(output).items():
        assert len(rows) == 6
        assert list(rows[0]) == list(DERIVED_COLUMNS)
        assert all(row["original_id"].startswith(split) for row in rows)
        for source in [row for row in rows if row["lang"] == "bn"]:
            assert source["is_derived"] == "false"
            assert source["derivation"] == "source"
            derived = [row for row in rows if row["parent_id"] == source["id"]]
            assert len(derived) == 1
            assert derived[0]["is_derived"] == "true"
            assert derived[0]["lang"] == "banglish"
            assert derived[0]["text"] == "pani nai 2 din।"
            for field in ("original_id", "category", "severity", "source_type", "location", "timestamp"):
                assert derived[0][field] == source[field]
    assert manifest["translation_status"] == {"completed": 0, "pending": 9}


def test_sampling_reproducible_even_if_input_row_order_changes(source_splits, tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    export_multilingual(source_splits, one, 4, seed=7)
    for name in SPLITS:
        path = source_splits / f"{name}.csv"
        write_csv(path, list(reversed(read_csv(path))), SCHEMA_COLUMNS)
    export_multilingual(source_splits, two, 4, seed=7)
    assert read_csv(one / "translation_queue.csv") == read_csv(two / "translation_queue.csv")
    assert read_splits(one) == read_splits(two)


def test_selects_one_stable_bn_representative_per_group(source_splits, tmp_path):
    path = source_splits / "train.csv"
    rows = read_csv(path)
    rows.append(dict(rows[0], id="aaa-first-bn", text="বিদ্যুৎ নাই"))
    write_csv(path, rows, SCHEMA_COLUMNS)
    output = tmp_path / "out"
    export_multilingual(source_splits, output, 9)
    queue = read_csv(output / "translation_queue.csv")
    selected = next(row for row in queue if row["original_id"] == "train-incident-0")
    assert selected["source_id"] == "aaa-first-bn"


def test_input_leakage_rejected_before_writes(source_splits, tmp_path):
    path = source_splits / "test.csv"
    rows = read_csv(path)
    rows[0]["original_id"] = "train-incident-0"
    write_csv(path, rows, SCHEMA_COLUMNS)
    output = tmp_path / "out"
    with pytest.raises(MultilingualValidationError, match="Split leakage"):
        export_multilingual(source_splits, output, 3)
    assert not output.exists()


def test_sample_shortage_has_clear_error(source_splits, tmp_path):
    with pytest.raises(MultilingualValidationError, match="only 9"):
        export_multilingual(source_splits, tmp_path / "out", 10)


def test_export_requires_explicit_overwrite(source_splits, tmp_path):
    output = tmp_path / "out"
    export_multilingual(source_splits, output, 3)
    with pytest.raises(MultilingualValidationError, match="overwrite"):
        export_multilingual(source_splits, output, 3)
    export_multilingual(source_splits, output, 3, overwrite=True)


def test_export_never_overwrites_phase1_splits(source_splits):
    before = (source_splits / "train.csv").read_bytes()
    with pytest.raises(MultilingualValidationError, match="must differ"):
        export_multilingual(source_splits, source_splits, 3, overwrite=True)
    assert (source_splits / "train.csv").read_bytes() == before


def test_parquet_only_input_explains_csv_option(tmp_path):
    source = tmp_path / "processed"
    source.mkdir()
    (source / "train.parquet").touch()
    with pytest.raises(MultilingualValidationError, match="--format csv"):
        export_multilingual(source, tmp_path / "out", 1)


def test_full_import_adds_english_without_leakage(source_splits, tmp_path):
    output = tmp_path / "out"
    export_multilingual(source_splits, output, 9)
    before = read_splits(output)
    manifest = import_translations(output, fill_queue(output))
    seen_groups = set()
    for split, rows in read_splits(output).items():
        assert len(rows) == 9
        originals = {row["id"]: row for row in rows if row["lang"] == "bn"}
        assert [row for row in rows if row["lang"] != "en"] == before[split]
        groups = {row["original_id"] for row in rows}
        assert not groups & seen_groups
        seen_groups.update(groups)
        for english in [row for row in rows if row["lang"] == "en"]:
            parent = originals[english["parent_id"]]
            assert english["text"] == "no water for 2 days."
            assert english["is_derived"] == "true"
            assert english["derivation"] == "manual_translation"
            for field in ("original_id", "category", "severity", "source_type", "location", "timestamp"):
                assert english[field] == parent[field]
    assert manifest["translation_status"] == {"completed": 9, "pending": 0}


def test_import_partial_requires_explicit_option(source_splits, tmp_path):
    output = tmp_path / "out"
    export_multilingual(source_splits, output, 3)
    filled = fill_queue(output, count=1)
    before = read_splits(output)
    with pytest.raises(MultilingualValidationError, match="2 English translations"):
        import_translations(output, filled)
    assert read_splits(output) == before
    manifest = import_translations(output, filled, allow_partial=True)
    assert manifest["translation_status"] == {"completed": 1, "pending": 2}
    assert sum(len(rows) for rows in read_splits(output).values()) == 7


@pytest.mark.parametrize("field", ["original_id", "source_id", "split", "source_text", "category", "severity", "location", "timestamp"])
def test_import_rejects_changed_metadata(source_splits, tmp_path, field):
    output = tmp_path / "out"
    export_multilingual(source_splits, output, 3)
    filled = fill_queue(output)
    queue = read_csv(filled)
    queue[0][field] = "changed"
    write_csv(filled, queue, QUEUE_COLUMNS)
    with pytest.raises(MultilingualValidationError, match=f"immutable field '{field}'"):
        import_translations(output, filled)


@pytest.mark.parametrize("mutation, message", [("duplicate", "Duplicate translation_id"), ("unknown", "Unknown translation_id")])
def test_import_rejects_duplicate_and_unknown_queue_ids(source_splits, tmp_path, mutation, message):
    output = tmp_path / "out"
    export_multilingual(source_splits, output, 3)
    filled = fill_queue(output)
    queue = read_csv(filled)
    if mutation == "duplicate":
        queue.append(dict(queue[0]))
    else:
        queue[0]["translation_id"] = "unknown"
    write_csv(filled, queue, QUEUE_COLUMNS)
    with pytest.raises(MultilingualValidationError, match=message):
        import_translations(output, filled)


def test_import_rejects_untranslated_bangla(source_splits, tmp_path):
    output = tmp_path / "out"
    export_multilingual(source_splits, output, 3)
    filled = fill_queue(output)
    queue = read_csv(filled)
    queue[0]["english_text"] = queue[0]["source_text"]
    write_csv(filled, queue, QUEUE_COLUMNS)
    with pytest.raises(MultilingualValidationError, match="still contains Bangla"):
        import_translations(output, filled)


def test_import_rejects_missing_source_and_empty_english(source_splits, tmp_path):
    output = tmp_path / "out"
    export_multilingual(source_splits, output, 3)
    filled = fill_queue(output, count=0)
    with pytest.raises(MultilingualValidationError, match="missing/blank"):
        import_translations(output, filled)
    queue = read_csv(filled)
    queue[0]["source_text"] = ""
    write_csv(filled, queue, QUEUE_COLUMNS)
    with pytest.raises(MultilingualValidationError, match="source_text"):
        import_translations(output, filled)


def test_import_requires_explicit_overwrite_and_has_stable_ids(source_splits, tmp_path):
    output = tmp_path / "out"
    export_multilingual(source_splits, output, 3)
    filled = fill_queue(output)
    import_translations(output, filled)
    before = read_splits(output)
    with pytest.raises(MultilingualValidationError, match="already exist"):
        import_translations(output, filled)
    import_translations(output, filled, overwrite=True)
    assert read_splits(output) == before


def test_import_checks_saved_base_rows_against_snapshot(source_splits, tmp_path):
    output = tmp_path / "out"
    export_multilingual(source_splits, output, 9)
    filled = fill_queue(output)
    path = output / "train.csv"
    rows = read_csv(path)
    rows[0]["text"] = "Edited after export"
    write_csv(path, rows, DERIVED_COLUMNS)
    with pytest.raises(MultilingualValidationError, match="changed or disappeared"):
        import_translations(output, filled)


def test_cli_errors_are_actionable_without_traceback(tmp_path, capsys):
    result = main(["export", "--input-dir", str(tmp_path / "missing"),
                   "--output-dir", str(tmp_path / "out"), "--sample-size", "3"])
    assert result == 2
    assert "ERROR: CSV not found:" in capsys.readouterr().err
