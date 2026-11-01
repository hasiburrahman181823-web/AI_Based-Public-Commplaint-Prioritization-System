"""Import supplied research archives without executing packaged code.

BanglaPSG XLSX values are read from the ZIP in memory. English/Banglish language
samples from BanglaDual stay in a separate auxiliary CSV and are never treated
as incident labels. No files are extracted to paths supplied by ZIP members.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from contextlib import ExitStack
import csv
from datetime import date, datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_training.config import SCHEMA_COLUMNS

SOURCE_COLUMN_MAPPING = {
    "text": "Grievance Text", "category": "Category", "severity": "Severity",
    "source_no": "No.", "id": "id", "original_id": "original_id",
    "lang": "lang", "source_type": "source_type", "location": "location", "timestamp": "timestamp",
}
AUXILIARY_COLUMNS = (
    "auxiliary_pair_id", "text_source", "lang_source", "text_target", "lang_target",
    "source_type", "is_incident_gold", "source_member", "source_csv_row", "upstream_source",
)
MAX_XLSX_BYTES = 64 * 1024 * 1024
MAX_INNER_XLSX_BYTES = 256 * 1024 * 1024
MAX_SAMPLE_CSV_BYTES = 16 * 1024 * 1024


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def archive_inventory(archive: ZipFile) -> list[dict]:
    names = [entry.filename for entry in archive.infolist()]
    if len(set(names)) != len(names):
        raise ValueError("Archive contains duplicate member names; member selection would be ambiguous")
    return [
        {"member": entry.filename, "uncompressed_bytes": entry.file_size,
         "compressed_bytes": entry.compress_size, "crc32": f"{entry.CRC:08x}"}
        for entry in archive.infolist() if not entry.is_dir()
    ]


def _value_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Workbook contains a non-finite number")
        if value.is_integer():
            return str(int(value))
    return str(value)


def _validate_mapping(overrides: dict | None) -> dict[str, str]:
    mapping = dict(SOURCE_COLUMN_MAPPING)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("Column mapping must be a JSON object: canonical name -> source header")
        if set(overrides) - set(mapping):
            raise ValueError(f"Unknown column mapping keys: {sorted(set(overrides)-set(mapping))}")
        mapping.update(overrides)
    if any(not isinstance(value, str) or not value.strip() for value in mapping.values()):
        raise ValueError("Mapped source headers must be nonempty strings")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Each canonical column must map to a distinct source header")
    return mapping


def inspect_banglapsg(
    path: str | Path, *, workbook_member: str | None = None, sheet: str | None = None,
    column_mapping: dict | None = None,
) -> tuple[list[dict[str, str]], list[str], dict]:
    """Read cached source values and preserve formula text without calculating it.

    Unknown incident lineage is left blank. Generated IDs identify workbook
    rows, not incidents; prepare_data subsequently groups exact normalized text
    when original_id is absent. Missing severity is never inferred from category.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("XLSX import requires openpyxl. Install requirements.txt.") from exc
    path = Path(path)
    mapping = _validate_mapping(column_mapping)
    fingerprint = file_fingerprint(path)
    with ZipFile(path) as archive:
        inventory = archive_inventory(archive)
        candidates = [entry for entry in archive.infolist() if entry.filename.lower().endswith(".xlsx") and not entry.is_dir()]
        if workbook_member is None:
            if len(candidates) != 1:
                raise ValueError("BanglaPSG archive needs exactly one XLSX or an explicit --workbook-member")
            selected = candidates[0]
        else:
            matches = [entry for entry in candidates if entry.filename == workbook_member]
            if len(matches) != 1:
                raise ValueError("--workbook-member must name one XLSX member in the archive")
            selected = matches[0]
        if selected.file_size > MAX_XLSX_BYTES:
            raise ValueError("Workbook member exceeds the bounded XLSX import size")
        content = archive.read(selected)
    # The outer ZIP's XLSX is itself a ZIP. Bound decompressed XML size before
    # openpyxl parses it; nothing from either archive is executed or extracted.
    with ZipFile(io.BytesIO(content)) as inner:
        inner_entries = inner.infolist()
        if sum(entry.file_size for entry in inner_entries) > MAX_INNER_XLSX_BYTES:
            raise ValueError("Workbook XML exceeds the bounded import size")
    formula_book = load_workbook(io.BytesIO(content), read_only=True, data_only=False, keep_links=False)
    cached_book = load_workbook(io.BytesIO(content), read_only=True, data_only=True, keep_links=False)
    try:
        sheet_names = formula_book.sheetnames
        if sheet is None:
            if len(sheet_names) != 1:
                raise ValueError("Workbook has multiple sheets; choose --sheet explicitly")
            sheet = sheet_names[0]
        if sheet not in sheet_names:
            raise ValueError(f"Sheet {sheet!r} not found; available sheets: {sheet_names}")
        raw_rows = formula_book[sheet].iter_rows()
        value_rows = cached_book[sheet].iter_rows()
        raw_header, value_header = next(raw_rows, ()), next(value_rows, ())
        if any(cell.data_type == "f" for cell in raw_header):
            raise ValueError("Formula-based workbook headers are not supported")
        headers = [_value_text(cell.value).strip() for cell in value_header]
        named = [header for header in headers if header]
        if not named or len(set(named)) != len(named):
            raise ValueError("Workbook needs nonempty, unique named source headers")
        parsed = []
        nonempty_columns = {index for index, header in enumerate(headers) if header}
        formulas_by_column = Counter()
        cached_formula_missing = Counter()
        blank_rows = 0
        for excel_row, (raw_cells, cached_cells) in enumerate(zip(raw_rows, value_rows), start=2):
            raw = [cell.value for cell in raw_cells]
            values = [_value_text(cell.value) for cell in cached_cells]
            formulas = {
                str(index + 1): _value_text(cell.value)
                for index, cell in enumerate(raw_cells) if cell.data_type == "f"
            }
            if not formulas and all(not value.strip() for value in values):
                blank_rows += 1
                continue
            width = max(len(headers), len(values), len(raw))
            values.extend([""] * (width - len(values)))
            headers.extend([""] * (width - len(headers)))
            for index, value in enumerate(values):
                if value.strip() or str(index + 1) in formulas:
                    nonempty_columns.add(index)
            for column in formulas:
                key = headers[int(column) - 1] or f"unnamed_column_{column}"
                formulas_by_column[key] += 1
                if not values[int(column) - 1].strip():
                    cached_formula_missing[key] += 1
            parsed.append((excel_row, values, formulas))
        if not parsed:
            raise ValueError("Selected workbook sheet has no nonblank data records")
        source_columns = {
            f"source_{index+1}_{headers[index] or 'unnamed'}": {"column_number": index + 1, "header": headers[index] or None}
            for index in sorted(nonempty_columns)
        }
        source_indices = {header: index for index, header in enumerate(headers) if header}
        missing_headers = [field for field in ("text", "category", "severity") if mapping[field] not in source_indices]
        canonical = []
        missing_values = Counter()
        category_counts, severity_counts, source_no_counts = Counter(), Counter(), Counter()
        used_ids = set()
        duplicate_ids = []
        workbook_hash = hashlib.sha256(content).hexdigest()
        for excel_row, values, formulas in parsed:
            def supplied(field: str) -> str:
                index = source_indices.get(mapping[field])
                return values[index] if index is not None and index < len(values) else ""
            record = {field: supplied(field).strip() for field in SCHEMA_COLUMNS if field in mapping}
            record["id"] = record.get("id") or f"banglapsg_{workbook_hash[:12]}_{sheet_names.index(sheet)+1}_{excel_row:06}"
            record["original_id"] = record.get("original_id", "")
            record["source_type"] = record.get("source_type") or "banglapsg"
            # Missing source columns stay absent in staging, rather than adding
            # dummy labels that might be mistaken for reviewed observations.
            for field in missing_headers:
                record.pop(field, None)
            for field in ("text", "category", "severity"):
                if field not in missing_headers and not record.get(field, "").strip():
                    missing_values[field] += 1
            category_counts.update([record["category"]] if record.get("category") else [])
            severity_counts.update([record["severity"]] if record.get("severity") else [])
            source_no = supplied("source_no").strip()
            if source_no:
                source_no_counts[source_no] += 1
            if record["id"] in used_ids:
                duplicate_ids.append(record["id"])
            used_ids.add(record["id"])
            record.update(source_member=selected.filename, source_sheet=sheet, source_excel_row=str(excel_row),
                          source_no=source_no, source_formulas_json=json.dumps(formulas, ensure_ascii=False, sort_keys=True) if formulas else "")
            for export_name, details in source_columns.items():
                index = details["column_number"] - 1
                record[export_name] = values[index] if index < len(values) else ""
            canonical.append(record)
        issues = []
        if missing_headers:
            issues.append(f"Required source columns missing: {missing_headers}; fill reviewed labels/text before preparation.")
        if missing_values:
            issues.append(f"Required source values missing: {dict(missing_values)}; no values were imputed.")
        if duplicate_ids:
            issues.append(f"Duplicate supplied row IDs: {len(duplicate_ids)}; resolve them before preparation.")
        ready = not issues
        present_schema = [field for field in SCHEMA_COLUMNS if field not in missing_headers]
        columns = [*present_schema, "source_member", "source_sheet", "source_excel_row", "source_no", "source_formulas_json", *source_columns]
        report = {
            "archive": fingerprint, "inventory": inventory, "workbook_member": selected.filename,
            "workbook_sha256": workbook_hash, "sheet_names": sheet_names, "selected_sheet": sheet,
            "source_headers": [header or None for header in headers], "column_mapping": mapping,
            "source_export_columns": source_columns, "rows": len(canonical), "formatted_blank_rows_skipped": blank_rows,
            "canonical_ready": ready, "missing_required_headers": missing_headers,
            "missing_required_values": dict(missing_values), "issues": issues,
            "category_counts": dict(sorted(category_counts.items())), "severity_counts": dict(sorted(severity_counts.items())),
            "source_no_duplicate_excess": sum(count-1 for count in source_no_counts.values() if count > 1),
            "formula_cells_by_source_column": dict(formulas_by_column), "formula_cells_missing_cached_value": dict(cached_formula_missing),
            "formula_handling": "Cached source values read without recalculation; original formula strings preserved as inert JSON provenance.",
            "id_policy": "Use supplied id when present; otherwise deterministic workbook-hash/sheet/physical-row ID. Row IDs do not establish incident identity.",
            "original_id_policy": "Preserve explicit original_id only. Absent lineage stays blank; prepare_data groups exact normalized text later.",
            "metadata_gaps": [field for field in ("original_id", "lang", "location", "timestamp") if not any(row.get(field, "") for row in canonical)],
            "version_binding": "Local archive bytes are fingerprinted; no catalogue version/file-hash match has been established.",
            "licensing": "Local package contains no verified licence declaration; preserve attribution and verify applicable release terms before redistribution.",
        }
        return canonical, columns, report
    finally:
        formula_book.close()
        cached_book.close()


def sample_bangladual(path: str | Path, sample_size: int = 2000) -> tuple[list[dict[str, str]], dict]:
    """Read bounded packaged CSV examples only, round-robin across sample files.

    Large Parquet members and packaged Python files are inventoried but never
    opened. The bound is exported language pairs, not incident observations.
    """
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("auxiliary_sample_size must be a positive integer")
    path = Path(path)
    fingerprint = file_fingerprint(path)
    with ZipFile(path) as archive, ExitStack() as stack:
        inventory = archive_inventory(archive)
        members = [entry for entry in archive.infolist()
                   if entry.filename.lower().endswith(".csv") and "/sample data/" in "/" + entry.filename.lower().replace("\\", "/")]
        if not members:
            raise ValueError("BanglaDual archive contains no packaged Sample Data CSV members")
        readers = []
        sampled_member_schemas = {}
        for entry in sorted(members, key=lambda item: item.filename):
            if entry.file_size > MAX_SAMPLE_CSV_BYTES:
                raise ValueError(f"Sample CSV exceeds bounded member size: {entry.filename}")
            stream = stack.enter_context(io.TextIOWrapper(archive.open(entry), encoding="utf-8-sig", newline=""))
            reader = csv.DictReader(stream, strict=True)
            headers = reader.fieldnames or []
            if len(set(headers)) != len(headers) or "Banglish" not in headers or not ({"Bangla", "English"} & set(headers)):
                raise ValueError(f"Unexpected auxiliary sample headers in {entry.filename}: {headers}")
            sampled_member_schemas[entry.filename] = headers

            def pairs(reader=reader, member=entry.filename):
                for source_row, row in enumerate(reader, start=2):
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"Malformed auxiliary CSV record {source_row} in {member}")
                    source = row["Banglish"].strip()
                    if not source:
                        continue
                    for target_column, language, kind in (("Bangla", "bn", "auxiliary_transliteration"), ("English", "en", "auxiliary_translation")):
                        target = row.get(target_column, "").strip()
                        if not target:
                            continue
                        upstream = row.get("Source_bangla" if language == "bn" else "Source_english", row.get("Source", "")).strip()
                        key = json.dumps([member, source_row, language], ensure_ascii=False)
                        yield {
                            "auxiliary_pair_id": "bangladual_" + hashlib.sha256(key.encode("utf-8")).hexdigest(),
                            "text_source": source, "lang_source": "banglish", "text_target": target,
                            "lang_target": language, "source_type": kind, "is_incident_gold": "false",
                            "source_member": member, "source_csv_row": str(source_row), "upstream_source": upstream,
                        }
            readers.append(iter(pairs()))
        queue, result = deque(readers), []
        while queue and len(result) < sample_size:
            iterator = queue.popleft()
            try:
                result.append(next(iterator))
            except StopIteration:
                continue
            queue.append(iterator)
        duplicate_content = Counter((row["text_source"], row["text_target"], row["lang_target"]) for row in result)
        report = {
            "archive": fingerprint, "inventory": inventory, "requested_max_language_pairs": sample_size,
            "exported_language_pairs": len(result), "sample_csv_schemas": sampled_member_schemas,
            "member_counts": dict(sorted(Counter(row["source_member"] for row in result).items())),
            "source_type_counts": dict(sorted(Counter(row["source_type"] for row in result).items())),
            "repeated_content_excess": sum(count-1 for count in duplicate_content.values() if count > 1),
            "sampling": "Bounded deterministic round-robin over packaged sample CSVs; not a random or representative sample of full Parquet data.",
            "parquet_members_opened": 0, "packaged_code_executed": False, "incident_gold": False,
            "usage": "Auxiliary translation/transliteration examples only. No incident category, severity, event linkage, location or event time is established.",
            "licensing": "Candidate official catalogue record pxw2bb4wyt/1 states CC BY 4.0; local byte-level release binding remains unverified. Preserve upstream per-row attribution.",
        }
        return result, report


def _write_csv(path: Path, columns, records) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def import_datasets(
    banglapsg_zip: str | Path, bangladual_zip: str | Path | None = None,
    output_dir: str | Path = "ml_training/data/raw",
    auxiliary_output: str | Path = "ml_training/data/auxiliary",
    auxiliary_sample_size: int = 2000, overwrite: bool = False,
    workbook_member: str | None = None, sheet: str | None = None,
    column_mapping: dict | None = None,
) -> dict:
    output, auxiliary = Path(output_dir).resolve(), Path(auxiliary_output).resolve()
    if output == auxiliary or output in auxiliary.parents or auxiliary in output.parents:
        raise ValueError("Raw and auxiliary output directories must be separate non-nested directories")
    for input_path in (banglapsg_zip, bangladual_zip):
        if input_path is not None:
            source = Path(input_path).resolve()
            if source in (output, auxiliary) or output in source.parents or auxiliary in source.parents:
                raise ValueError("Archive inputs must be outside output directories")
    rows, columns, grievance_report = inspect_banglapsg(banglapsg_zip, workbook_member=workbook_member, sheet=sheet, column_mapping=column_mapping)
    auxiliary_rows, auxiliary_report = sample_bangladual(bangladual_zip, auxiliary_sample_size) if bangladual_zip else (None, None)
    filename = "banglapsg.csv" if grievance_report["canonical_ready"] else "banglapsg_source_staging.csv"
    if filename != "banglapsg.csv" and (output / "banglapsg.csv").exists():
        raise ValueError("Source labels are incomplete and a previous banglapsg.csv exists; use a fresh output directory to avoid stale canonical data")
    output_files = [output / filename, output / "dataset_import_report.json"]
    if auxiliary_rows is not None:
        output_files.extend((auxiliary / "bangladual_language_sample.csv", auxiliary / "bangladual_sample_manifest.json"))
    if not overwrite and any(path.exists() for path in output_files):
        raise ValueError("Import output already exists; use a new output directory or --overwrite")
    report = {"schema_version": 1, "status": "ready" if grievance_report["canonical_ready"] else "requires_labels",
              "banglapsg": grievance_report, "bangladual_auxiliary": auxiliary_report,
              "data_redistributed_in_code_package": False}
    output.mkdir(parents=True, exist_ok=True)
    if auxiliary_rows is not None:
        auxiliary.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dataset-import-", dir=output) as temporary:
        stage = Path(temporary)
        _write_csv(stage / filename, columns, rows)
        grievance_report["output_file"] = str(output / filename)
        grievance_report["output_sha256"] = file_fingerprint(stage / filename)["sha256"]
        if auxiliary_rows is not None:
            _write_csv(stage / "bangladual_language_sample.csv", AUXILIARY_COLUMNS, auxiliary_rows)
            auxiliary_report["output_file"] = str(auxiliary / "bangladual_language_sample.csv")
            auxiliary_report["output_sha256"] = file_fingerprint(stage / "bangladual_language_sample.csv")["sha256"]
            (stage / "bangladual_sample_manifest.json").write_text(json.dumps(auxiliary_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "dataset_import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(stage / filename, output / filename)
        os.replace(stage / "dataset_import_report.json", output / "dataset_import_report.json")
        if auxiliary_rows is not None:
            os.replace(stage / "bangladual_language_sample.csv", auxiliary / "bangladual_language_sample.csv")
            os.replace(stage / "bangladual_sample_manifest.json", auxiliary / "bangladual_sample_manifest.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banglapsg-zip", required=True)
    parser.add_argument("--bangladual-zip")
    parser.add_argument("--output-dir", default="ml_training/data/raw")
    parser.add_argument("--auxiliary-output", default="ml_training/data/auxiliary")
    parser.add_argument("--auxiliary-sample-size", type=int, default=2000)
    parser.add_argument("--workbook-member")
    parser.add_argument("--sheet")
    parser.add_argument("--column-map", help="Optional JSON canonical-name -> workbook-header overrides")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mapping = json.loads(resolve_path(args.column_map).read_text(encoding="utf-8-sig")) if args.column_map else None
        report = import_datasets(resolve_path(args.banglapsg_zip), resolve_path(args.bangladual_zip) if args.bangladual_zip else None,
                                 resolve_path(args.output_dir), resolve_path(args.auxiliary_output), args.auxiliary_sample_size,
                                 args.overwrite, args.workbook_member, args.sheet, mapping)
        print(f"BanglaPSG source rows: {report['banglapsg']['rows']}")
        print(f"Source mapping: {json.dumps(report['banglapsg']['column_mapping'], ensure_ascii=True)}")
        print(f"Canonical data ready: {report['banglapsg']['canonical_ready']}")
        print(f"Saved source data: {report['banglapsg']['output_file']}")
        if report["bangladual_auxiliary"] is not None:
            print(f"Auxiliary language pairs: {report['bangladual_auxiliary']['exported_language_pairs']} (not incident gold)")
            print(f"Saved auxiliary data: {report['bangladual_auxiliary']['output_file']}")
        for issue in report["banglapsg"]["issues"]:
            print(f"REQUIRED LABEL REVIEW: {issue}", file=sys.stderr)
        return 0 if report["status"] == "ready" else 3
    except (ValueError, OSError, RuntimeError, BadZipFile, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
