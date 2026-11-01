"""Miniature archive fixtures only; no supplied research records are bundled."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from openpyxl import Workbook

from ml_training.scripts.import_research_datasets import (
    AUXILIARY_COLUMNS, SOURCE_COLUMN_MAPPING, import_datasets, inspect_banglapsg,
    main, sample_bangladual,
)


def make_workbook_archive(path, headers=None, rows=None, *, sheets=None):
    workbook = Workbook()
    workbook.active.title = "Sheet1"
    workbook.active.append(headers or ["No.", "Grievance Text", "Category", "Severity"])
    for row in (rows if rows is not None else [[1, "পানি নেই", "Water", "High"], [2, "রাস্তা ভাঙা", "Road", "Low"]]):
        workbook.active.append(row)
    if sheets:
        for name in sheets:
            workbook.create_sheet(name)
    stream = io.BytesIO()
    workbook.save(stream)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("source/grievances.xlsx", stream.getvalue())
    return path


def csv_content(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def make_auxiliary_archive(path):
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("dataset/Sample Data/bangla_samples.csv", csv_content(["Banglish", "Bangla", "Source"], [
            {"Banglish": "pani nai", "Bangla": "পানি নেই", "Source": "miniature source"},
            {"Banglish": "rasta kharap", "Bangla": "রাস্তা খারাপ", "Source": "miniature source"},
        ]))
        archive.writestr("dataset/Sample Data/english_samples.csv", csv_content(["Banglish", "English", "Source"], [
            {"Banglish": "pani nai", "English": "No water", "Source": "miniature source"},
            {"Banglish": "rasta kharap", "English": "Bad road", "Source": "miniature source"},
        ]))
        archive.writestr("dataset/Sample Data/sample_merged_data.csv", csv_content(["Banglish", "Bangla", "English", "Source_bangla", "Source_english"], [
            {"Banglish": "test", "Bangla": "পরীক্ষা", "English": "test", "Source_bangla": "a", "Source_english": "b"},
            {"Banglish": "empty targets", "Bangla": "", "English": "", "Source_bangla": "", "Source_english": ""},
        ]))
        # These must only be inventoried. Neither is valid executable/parquet data.
        archive.writestr("dataset/Processing Scripts/do_not_run.py", "raise RuntimeError('archive code must never execute')")
        archive.writestr("dataset/Merged Dual-Task Dataset/huge.parquet", b"not parquet: must never open")
        archive.writestr("../../outside_payload.py", "raise RuntimeError('must not extract')")
    return path


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_actual_header_mapping_exports_source_values_and_deterministic_row_ids(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip")
    first, columns, report = inspect_banglapsg(source)
    second, _, _ = inspect_banglapsg(source)
    assert first == second
    assert report["rows"] == 2 and report["canonical_ready"] is True
    assert report["column_mapping"] == SOURCE_COLUMN_MAPPING
    assert report["category_counts"] == {"Road": 1, "Water": 1}
    assert report["severity_counts"] == {"High": 1, "Low": 1}
    assert first[0]["text"] == "পানি নেই"
    assert first[0]["source_no"] == "1"
    assert first[0]["original_id"] == ""  # row order is not incident linkage
    assert first[0]["lang"] == first[0]["location"] == first[0]["timestamp"] == ""
    assert first[0]["id"].startswith("banglapsg_")
    assert first[0]["id"] != first[1]["id"]
    assert first[0]["source_2_Grievance Text"] == first[0]["text"]
    assert first[0]["source_type"] == "banglapsg"
    assert len(report["archive"]["sha256"]) == 64
    assert set(columns) == set(first[0])


def test_missing_severity_creates_staging_and_required_label_report(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip", ["No.", "Grievance Text", "Category"], [[1, "পানি নেই", "Water"]])
    output = tmp_path / "raw"
    report = import_datasets(source, output_dir=output, auxiliary_output=tmp_path / "aux")
    assert report["status"] == "requires_labels"
    assert not (output / "banglapsg.csv").exists()
    columns, rows = read_csv(output / "banglapsg_source_staging.csv")
    assert "severity" not in columns
    assert rows[0]["category"] == "Water"
    assert report["banglapsg"]["missing_required_headers"] == ["severity"]
    assert report["banglapsg"]["issues"]


def test_blank_required_value_is_preserved_for_review_not_filled_or_dropped(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip", rows=[[1, "পানি নেই", "Water", None]])
    rows, _, report = inspect_banglapsg(source)
    assert len(rows) == 1 and rows[0]["severity"] == ""
    assert report["canonical_ready"] is False
    assert report["missing_required_values"] == {"severity": 1}


def test_optional_ids_and_incident_metadata_are_preserved_as_strings(tmp_path):
    headers = ["No.", "Grievance Text", "Category", "Severity", "id", "original_id", "location", "timestamp", "lang"]
    source = make_workbook_archive(tmp_path / "input.zip", headers, [[1, "পানি নেই", "Water", "01", "0007", "incident-01", "Mirpur 10", "2026-01-01T12:00:00+06:00", "bn"]])
    rows, _, report = inspect_banglapsg(source)
    assert report["canonical_ready"] is True
    assert rows[0]["id"] == "0007" and rows[0]["original_id"] == "incident-01"
    assert rows[0]["severity"] == "01"
    assert rows[0]["location"] == "Mirpur 10"


def test_blank_formatted_rows_skipped_and_unnamed_nonempty_values_retained(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip", ["No.", "Grievance Text", "Category", "Severity", None], [
        [1, "পানি নেই", "Water", "High", "extra observation"],
        [None, None, None, None, None],
        [2, "রাস্তা ভাঙা", "Road", "Low", None],
    ])
    rows, columns, report = inspect_banglapsg(source)
    assert len(rows) == 2 and report["formatted_blank_rows_skipped"] == 1
    assert "source_5_unnamed" in columns
    assert rows[0]["source_5_unnamed"] == "extra observation"


def test_formula_cells_are_not_evaluated_and_formula_text_is_inert_provenance(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip", rows=[["=1+1", "পানি নেই", "Water", "High"]])
    rows, _, report = inspect_banglapsg(source)
    assert rows[0]["source_no"] == ""  # openpyxl fixture has no cached formula result
    assert json.loads(rows[0]["source_formulas_json"]) == {"1": "=1+1"}
    assert rows[0]["source_formulas_json"].startswith("{")
    assert report["formula_cells_by_source_column"] == {"No.": 1}
    assert report["formula_cells_missing_cached_value"] == {"No.": 1}
    assert rows[0]["id"].startswith("banglapsg_")


def test_formula_severity_without_cached_value_requires_annotation(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip", rows=[[1, "পানি নেই", "Water", '=IF(1,"High","Low")']])
    rows, _, report = inspect_banglapsg(source)
    assert report["canonical_ready"] is False
    assert rows[0]["severity"] == ""


def test_duplicate_supplied_ids_do_not_become_ready_data(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip", ["Grievance Text", "Category", "Severity", "id"], [
        ["পানি নেই", "Water", "High", "same-id"], ["রাস্তা ভাঙা", "Road", "Low", "same-id"],
    ])
    rows, _, report = inspect_banglapsg(source)
    assert len(rows) == 2 and report["canonical_ready"] is False
    assert "Duplicate supplied row IDs" in " ".join(report["issues"])


def test_multiple_sheets_require_explicit_selection_and_mapping_overrides_work(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip", ["complaint", "label", "priority"], [["পানি নেই", "Water", "High"]], sheets=["Notes"])
    with pytest.raises(ValueError, match="multiple sheets"):
        inspect_banglapsg(source)
    rows, _, report = inspect_banglapsg(source, sheet="Sheet1", column_mapping={"text": "complaint", "category": "label", "severity": "priority"})
    assert report["canonical_ready"] is True and rows[0]["text"] == "পানি নেই"


def test_unknown_or_invalid_mapping_is_rejected(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip")
    with pytest.raises(ValueError, match="Unknown"):
        inspect_banglapsg(source, column_mapping={"fake": "Column"})
    with pytest.raises(ValueError, match="distinct"):
        inspect_banglapsg(source, column_mapping={"severity": "Category"})


def test_auxiliary_samples_are_bounded_separate_and_have_provenance(tmp_path):
    source = make_auxiliary_archive(tmp_path / "aux.zip")
    rows, report = sample_bangladual(source, 3)
    assert len(rows) == report["exported_language_pairs"] == 3
    assert set(rows[0]) == set(AUXILIARY_COLUMNS)
    assert {row["source_type"] for row in rows} == {"auxiliary_translation", "auxiliary_transliteration"}
    assert all(row["is_incident_gold"] == "false" for row in rows)
    assert all(row["source_member"] and row["upstream_source"] and row["source_csv_row"] for row in rows)
    assert all("category" not in row and "severity" not in row and "original_id" not in row for row in rows)
    assert report["parquet_members_opened"] == 0 and report["packaged_code_executed"] is False
    assert any(item["member"].endswith(".parquet") for item in report["inventory"])
    assert not (tmp_path.parent / "outside_payload.py").exists()
    assert sample_bangladual(source, 3)[0] == rows


def test_auxiliary_missing_targets_skipped_and_finite_sample_exhaustion_is_reported(tmp_path):
    rows, report = sample_bangladual(make_auxiliary_archive(tmp_path / "aux.zip"), 100)
    assert len(rows) == 6
    assert report["requested_max_language_pairs"] == 100
    assert all(row["text_target"] for row in rows)


@pytest.mark.parametrize("sample_size", [0, -1, True, 1.5])
def test_auxiliary_invalid_bounds_rejected(tmp_path, sample_size):
    source = make_auxiliary_archive(tmp_path / "aux.zip")
    with pytest.raises(ValueError, match="positive integer"):
        sample_bangladual(source, sample_size)


def test_output_guard_full_import_and_json_fingerprints(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip")
    auxiliary = make_auxiliary_archive(tmp_path / "aux.zip")
    output, aux_output = tmp_path / "raw", tmp_path / "aux"
    report = import_datasets(source, auxiliary, output, aux_output, 4)
    assert json.loads((output / "dataset_import_report.json").read_text(encoding="utf-8")) == report
    columns, rows = read_csv(output / "banglapsg.csv")
    assert len(rows) == 2 and "severity" in columns
    assert len(read_csv(aux_output / "bangladual_language_sample.csv")[1]) == 4
    assert len(report["banglapsg"]["output_sha256"]) == 64
    with pytest.raises(ValueError, match="already exists"):
        import_datasets(source, auxiliary, output, aux_output, 4)
    assert import_datasets(source, auxiliary, output, aux_output, 4, overwrite=True)["status"] == "ready"


def test_incomplete_new_source_cannot_leave_stale_ready_csv(tmp_path):
    output, auxiliary = tmp_path / "raw", tmp_path / "aux"
    complete = make_workbook_archive(tmp_path / "complete.zip")
    incomplete = make_workbook_archive(tmp_path / "incomplete.zip", ["Grievance Text", "Category"], [["পানি নেই", "Water"]])
    import_datasets(complete, output_dir=output, auxiliary_output=auxiliary)
    with pytest.raises(ValueError, match="previous banglapsg.csv"):
        import_datasets(incomplete, output_dir=output, auxiliary_output=auxiliary, overwrite=True)


def test_output_tree_isolation_and_archive_inputs_are_preserved(tmp_path):
    source = make_workbook_archive(tmp_path / "input.zip")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="non-nested"):
        import_datasets(source, output_dir=tmp_path / "same", auxiliary_output=tmp_path / "same")
    with pytest.raises(ValueError, match="outside"):
        import_datasets(source, output_dir=tmp_path, auxiliary_output=tmp_path.parent / "aux-sibling")
    assert source.read_bytes() == before


def test_cli_success_missing_labels_and_file_errors_have_distinct_exit_codes(tmp_path, capsys):
    source = make_workbook_archive(tmp_path / "input.zip")
    common = ["--output-dir", str(tmp_path / "raw"), "--auxiliary-output", str(tmp_path / "aux")]
    assert main(["--banglapsg-zip", str(source), *common]) == 0
    assert "Canonical data ready: True" in capsys.readouterr().out
    incomplete = make_workbook_archive(tmp_path / "incomplete.zip", ["Grievance Text", "Category"], [["পানি নেই", "Water"]])
    assert main(["--banglapsg-zip", str(incomplete), "--output-dir", str(tmp_path / "staging"), "--auxiliary-output", str(tmp_path / "aux-staging")]) == 3
    assert "REQUIRED LABEL REVIEW" in capsys.readouterr().err
    assert main(["--banglapsg-zip", str(tmp_path / "missing.zip"), "--output-dir", str(tmp_path / "missing-output"), "--auxiliary-output", str(tmp_path / "aux-missing")]) == 2
    assert "ERROR:" in capsys.readouterr().err
