"""Collect saved experiment evidence into a Markdown report without inventing results.

Run ``python make_thesis_report.py`` from any directory. Defaults resolve
against this repository, not the shell's current working directory. The
collector does not execute notebooks, import model artifacts, recompute scores,
select a winning model, or turn synthetic test output into research evidence.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import html
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
FIGURES = {".png", ".jpg", ".jpeg", ".webp"}
TABLES = {".csv", ".tsv"}
PRIVATE_COLUMNS = {"text", "text_a", "text_b", "contact", "phone", "email", "reason", "password", "access_token", "jwt_secret", "admin_password_hash"}


def _path(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value.absolute() if value.is_absolute() else (ROOT / value).absolute()


def _linklike(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _cell(value: object, limit: int = 500) -> str:
    """Escape data as text, never executable Markdown/HTML or a table boundary."""
    if value is None:
        value = "undefined (null)"
    elif isinstance(value, bool):
        value = "true" if value else "false"
    value = str(value)
    if len(value) > limit:
        value = value[:limit] + " … [cell truncated; inspect source]"
    value = html.escape(value, quote=False).replace("\\", "\\\\")
    for character in ("|", "`", "[", "]", "*", "_", "#"):
        value = value.replace(character, "\\" + character)
    return value.replace("\r\n", " ↵ ").replace("\r", " ↵ ").replace("\n", " ↵ ")


def _href(source: Path, output: Path) -> str:
    return quote(Path(os.path.relpath(source, output.parent)).as_posix(), safe="/.-_")


def _sha(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Result changed during collection: {path.name}; stop writers and retry")
    return digest.hexdigest(), after.st_size


def _inventory(root: Path) -> list[dict]:
    records = []
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(subdirectories):
            path = parent / name
            if _linklike(path):
                subdirectories.remove(name)
                records.append({"path": path, "kind": "excluded link/junction directory", "sha256": None, "bytes": None})
        for name in filenames:
            path = parent / name
            if name == ".gitkeep":
                continue
            if _linklike(path) or not path.resolve().is_relative_to(root):
                records.append({"path": path, "kind": "excluded link/outside source root", "sha256": None, "bytes": None})
            elif not stat.S_ISREG(path.stat().st_mode):
                records.append({"path": path, "kind": "excluded nonregular file", "sha256": None, "bytes": None})
            else:
                digest, size = _sha(path)
                records.append({"path": path, "kind": "saved artifact", "sha256": digest, "bytes": size})
    return sorted(records, key=lambda record: record["path"].relative_to(root).as_posix())


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _scalars(value, prefix: str = "$", depth: int = 0):
    if depth > 30:
        yield prefix, "[nested value not expanded; inspect source]"
    elif isinstance(value, dict) and value:
        for key, child in value.items():
            child_path = prefix + "[" + json.dumps(key, ensure_ascii=False) + "]"
            if key.casefold() in PRIVATE_COLUMNS:
                yield child_path, "[row-level/private field omitted; inspect source privately]"
            else:
                yield from _scalars(child, child_path, depth + 1)
    elif isinstance(value, list) and value:
        for index, child in enumerate(value):
            yield from _scalars(child, f"{prefix}[{index}]", depth + 1)
    else:
        yield prefix, "{}" if value == {} else "[]" if value == [] else value


def _provenance(value) -> str:
    if not isinstance(value, dict):
        return "No top-level provenance declaration; research status is unverified."
    parts = []
    for key in ("experiment_kind", "kind", "dataset_role", "purpose", "synthetic", "is_synthetic", "label_source", "weak_labels", "is_incident_gold", "ranker_status"):
        if key in value and not isinstance(value[key], (dict, list)):
            parts.append(f"{key}={value[key]}")
    synthetic = value.get("synthetic") is True or value.get("is_synthetic") is True
    synthetic |= any(token in str(value.get("experiment_kind", "")).casefold() for token in ("synthetic", "fixture", "smoke"))
    weak = value.get("weak_labels") is True or value.get("is_incident_gold") is False or "weak" in str(value.get("label_source", "")).casefold()
    warning = "DECLARED SYNTHETIC/FIXTURE: do not present as observed research performance. " if synthetic else ""
    if weak:
        warning += "DECLARED WEAK/NON-GOLD LABELS: do not claim independent incident-gold evaluation. "
    return warning + ("Reported metadata: " + "; ".join(parts) if parts else "No top-level provenance declaration; research status is unverified.")


def _json_preview(path: Path, max_rows: int) -> list[str]:
    with path.open(encoding="utf-8-sig") as stream:
        payload = json.load(stream, parse_float=Decimal, parse_constant=_invalid_constant, object_pairs_hook=_unique_object)
    lines = [_cell(_provenance(payload), 2000), "", "| Saved field path | Saved value |", "| --- | --- |"]
    total = 0
    for field, value in _scalars(payload):
        total += 1
        if total <= max_rows:
            lines.append(f"| {_cell(field)} | {_cell(value)} |")
    if total > max_rows:
        lines += ["", f"Preview limited to {max_rows} of {total} scalar/terminal fields; the linked JSON retains the complete evidence."]
    return lines


def _table_preview(path: Path, max_rows: int, max_columns: int) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",", strict=True)
        header = next(reader, [])
        if not header:
            return ["Empty table: no columns or measurements supplied."]
        if len(set(header)) != len(header) or any(not value.strip() for value in header):
            raise ValueError("Table has empty or duplicate column names")
        private = any(value.strip().casefold() in PRIVATE_COLUMNS for value in header)
        sample, count = [], 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Malformed table at record {reader.line_num}")
            count += 1
            if not private and len(sample) < max_rows:
                sample.append(row[:max_columns])
    if private:
        return [f"Row-level/private-data table: {count} saved records. Values are not copied into this report; review the linked source privately before sharing."]
    selected = header[:max_columns]
    lines = [f"Saved table: {count} data records; {len(header)} columns. Empty cells remain empty, not zero.", "",
             "| " + " | ".join(_cell(value) for value in selected) + " |", "| " + " | ".join("---" for _ in selected) + " |"]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in sample)
    if count > max_rows or len(header) > max_columns:
        lines += ["", f"Preview limited to {max_rows} records and {max_columns} columns; the linked table remains complete."]
    return lines


def make_report(results_dir: str | Path = "results", output: str | Path = "docs/thesis_results.md", *,
                overwrite: bool = False, max_table_rows: int = 40, max_columns: int = 20,
                max_json_rows: int = 1000, max_preview_bytes: int = 5 * 1024 * 1024) -> dict:
    """Hash every regular artifact, render bounded evidence, then atomically publish."""
    for name, value in (("max_table_rows", max_table_rows), ("max_columns", max_columns),
                        ("max_json_rows", max_json_rows), ("max_preview_bytes", max_preview_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    source, destination = _path(results_dir), _path(output)
    if _linklike(source) or not source.is_dir():
        raise ValueError("Results directory must exist and must not be a link/junction")
    source = source.resolve()
    if _linklike(destination) or any(_linklike(parent) for parent in destination.parents):
        raise ValueError("Report output and its parents must not be links/junctions")
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("Report output must be outside the results source tree")
    if destination.suffix.lower() != ".md":
        raise ValueError("Report output must use the .md extension")
    if destination.exists() and not overwrite:
        raise ValueError("Report exists; pass --overwrite to replace the generated report")
    records = _inventory(source)
    saved = [record for record in records if record["sha256"] is not None]
    lines = ["# Thesis results — saved evidence", "",
             "Generated by `make_thesis_report.py` from the selected results folder. This is an evidence collector, not an experiment or an automatic thesis conclusion.", "",
             "## Evidence status", ""]
    if not saved:
        lines += ["No evaluated research results are saved in this results folder. No performance numbers, natural-test counts, expert-label counts, model winner or significance claim have been generated.", "",
                  "Complete the reviewed dataset/annotation workflow, run the real experiments, then regenerate this document. Unit tests and synthetic UI/encoder fixtures are implementation checks; they are not thesis results."]
    else:
        lines += ["Saved artifacts were discovered. Their research validity has not been independently verified by this collector. Explicit synthetic/weak-label declarations are flagged; absence of a flag is not proof of independent gold labels.", "",
                  "Values below are copied from saved files. No languages are pooled, runs ranked, metrics recomputed, confidence intervals inferred or missing values changed to zero. Repeated metrics in a manifest and a cohort report remain separate source observations."]
    lines += ["", "Before using these results, verify each run's incident split isolation, cohort support, label provenance, checkpoint/resources, validation-only tuning and final-holdout status. Report the actual natural-test size and expert/annotator counts from collection records; unknown counts stay unknown.", "",
              "Raw predictions, error examples and contacts can contain private complaint content. Tables with recognized row-level/private columns are linked but not copied. Review all other report content and artifacts before publishing.", "",
              "## Artifact inventory", "",
              "All files except `.gitkeep` are listed below. SHA-256 hashes identify saved bytes, not scientific correctness. Excluded links and unsupported/oversized files are recorded explicitly.", "",
              "| Artifact | Bytes | SHA-256 | Collection status |", "| --- | ---: | --- | --- |"]
    warnings = []
    sections = []
    for index, record in enumerate(records, 1):
        path = record["path"]
        name = path.relative_to(source).as_posix()
        status = record["kind"]
        link = f"[Source file]({_href(path, destination)})" if record["sha256"] else "No link followed"
        content = []
        if record["sha256"]:
            suffix = path.suffix.lower()
            if suffix in FIGURES:
                status = "figure included"
                content = [f"![Saved figure {index}]({_href(path, destination)})", "", "Caption/source name: " + _cell(name)]
            elif suffix in TABLES | {".json"}:
                if record["bytes"] > max_preview_bytes:
                    status = "linked only: file exceeds preview byte limit"
                    content = [f"The file exceeds the {max_preview_bytes}-byte preview limit. No values were inferred; inspect the complete linked source."]
                else:
                    try:
                        content = _json_preview(path, max_json_rows) if suffix == ".json" else _table_preview(path, max_table_rows, max_columns)
                        status = "saved JSON fields included" if suffix == ".json" else "table inspected; preview/private-data policy applied"
                    except (OSError, UnicodeError, ValueError, csv.Error, RecursionError) as exc:
                        status = "linked only: could not safely parse"
                        warning = f"{name}: {type(exc).__name__} while parsing; inspect source (no replacement measurements generated)."
                        warnings.append(warning)
                        content = [_cell(warning, 2000)]
            else:
                status = "linked only: unsupported preview format"
                content = ["This artifact is retained in the inventory. It was not executed, deserialized or embedded. SVG/HTML/Markdown files are linked rather than inserted as active content."]
            # Detect concurrent writes between fingerprinting and previewing.
            if _sha(path)[0] != record["sha256"]:
                raise ValueError(f"Result changed during collection: {path.name}; stop writers and retry")
        title = f"### Artifact {index}: {_cell(name)}"
        sections += [title, "", link, "", *content, ""]
        lines.append(f"| {_cell(name)} | {record['bytes'] if record['bytes'] is not None else 'not read'} | {record['sha256'] or 'not read'} | {_cell(status)} |")
    if not records:
        lines.append("| No saved artifacts | — | — | Awaiting experiments |")
    lines += ["", "## Saved metrics, tables and figures", "", *sections]
    if not sections:
        lines += ["No result sections can be populated until result artifacts exist.", ""]
    lines += ["## Interpretation and reporting", "",
              "Use the prespecified comparisons in [research questions](research_questions.md), the [chapter outline](thesis_outline.md), and [validity/ethics guidance](thesis_validity_ethics.md). Do not equate a passing test suite or a working deployment with measured classification, duplicate or ranking quality.", "",
              "A proposed model that does not outperform its baseline is still a reportable result. State the measured effect, uncertainty and cohort support; explain observed trade-offs and limitations without selecting only favorable runs or inventing an improvement.", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
                                         prefix=".thesis-results-", suffix=".tmp", delete=False) as stream:
            temporary_name = stream.name
            stream.write("\n".join(lines).rstrip() + "\n")
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return {"output": str(destination), "saved_artifacts": len(saved), "excluded_artifacts": len(records) - len(saved),
            "warnings": warnings, "research_measurements_computed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="docs/thesis_results.md")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-table-rows", type=int, default=40)
    parser.add_argument("--max-columns", type=int, default=20)
    parser.add_argument("--max-json-rows", type=int, default=1000)
    parser.add_argument("--max-preview-bytes", type=int, default=5 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        result = make_report(args.results_dir, args.output, overwrite=args.overwrite,
                             max_table_rows=args.max_table_rows, max_columns=args.max_columns,
                             max_json_rows=args.max_json_rows, max_preview_bytes=args.max_preview_bytes)
    except (OSError, ValueError) as exc:
        print("Cannot build thesis report: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
