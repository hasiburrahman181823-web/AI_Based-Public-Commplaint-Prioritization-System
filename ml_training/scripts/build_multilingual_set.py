"""Build linked Bengali/Banglish/English evaluation records without re-splitting.

Export samples Bengali incident groups and writes a manual translation queue.
Import validates a filled queue and adds English rows. No network is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.normalize import clean_text, normalize_banglish
from ml.transliterate import TRANSLITERATOR_VERSION, transliterate_bangla
from ml_training.config import SCHEMA_COLUMNS

SPLITS = ("train", "val", "test")
DERIVED_COLUMNS = (*SCHEMA_COLUMNS, "is_derived", "derivation", "parent_id")
QUEUE_COLUMNS = (
    "translation_id", "original_id", "source_id", "split", "source_text",
    "category", "severity", "location", "timestamp", "english_text",
)
MANIFEST_VERSION = 1


class MultilingualValidationError(ValueError):
    """Invalid source splits, translation queue, or saved multilingual dataset."""


def _path(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _id(kind: str, source_id: str) -> str:
    return f"{kind}_" + hashlib.sha256(source_id.encode("utf-8")).hexdigest()


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise MultilingualValidationError(f"CSV not found: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            if len(header) != len(set(header)):
                raise MultilingualValidationError(f"Duplicate headers in {path}.")
            if set(header) != set(columns):
                raise MultilingualValidationError(
                    f"Unexpected columns in {path}. Required: {list(columns)}; got: {header}."
                )
            result = []
            for number, fields in enumerate(reader, 2):
                if len(fields) != len(header):
                    raise MultilingualValidationError(
                        f"CSV record {number} in {path} has {len(fields)} fields; expected {len(header)}."
                    )
                result.append(dict(zip(header, fields)))
            return result
    except (csv.Error, UnicodeError) as exc:
        raise MultilingualValidationError(f"Cannot read UTF-8 CSV {path}: {exc}") from exc


def _validate_splits(rows: list[dict[str, str]], *, derived: bool = False) -> None:
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    for row in rows:
        for field in ("id", "original_id", "text", "category", "severity"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise MultilingualValidationError(f"Missing/blank {field} in a dataset row.")
        if row.get("split") not in SPLITS:
            raise MultilingualValidationError(f"Invalid split for row {row['id']!r}.")
        if row.get("lang") not in ("bn", "en", "banglish"):
            raise MultilingualValidationError(f"Invalid lang for row {row['id']!r}.")
        if row["id"] in seen_ids:
            raise MultilingualValidationError(f"Duplicate row id: {row['id']!r}.")
        seen_ids.add(row["id"])
        old_split = group_splits.setdefault(row["original_id"], row["split"])
        if old_split != row["split"]:
            raise MultilingualValidationError(
                f"Split leakage: original_id {row['original_id']!r} appears in {old_split} and {row['split']}."
            )
        if derived and row.get("is_derived") not in ("true", "false"):
            raise MultilingualValidationError(f"Invalid is_derived for row {row['id']!r}.")


def _queue_record(source: dict[str, str]) -> dict[str, str]:
    return {
        "translation_id": _id("en", source["id"]),
        "original_id": source["original_id"], "source_id": source["id"],
        "split": source["split"], "source_text": source["text"],
        "category": source["category"], "severity": source["severity"],
        "location": source["location"], "timestamp": source["timestamp"],
        "english_text": "",
    }


def _source_record(source: dict[str, str]) -> dict[str, str]:
    return dict(source, is_derived="false", derivation="source", parent_id="")


def _banglish_record(source: dict[str, str]) -> dict[str, str]:
    row = dict(source)
    row.update(
        id=_id("banglish", source["id"]), lang="banglish",
        text=normalize_banglish(transliterate_bangla(source["text"])),
        is_derived="true", derivation="rule_transliteration", parent_id=source["id"],
    )
    return row


def _stats(rows: list[dict[str, str]]) -> dict:
    return {
        name: {
            "rows": len(part := [row for row in rows if row["split"] == name]),
            "original_groups": len({row["original_id"] for row in part}),
            "language_counts": {lang: sum(row["lang"] == lang for row in part)
                                for lang in ("bn", "banglish", "en")},
        }
        for name in SPLITS
    }


def _write_dataset(
    output_dir: Path, rows: list[dict[str, str]], manifest: dict,
    queue: list[dict[str, str]] | None = None,
) -> None:
    """Write every file to a stage directory before replacing destinations."""
    _validate_splits(rows, derived=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest["splits"] = _stats(rows)
    with tempfile.TemporaryDirectory(prefix=".multilingual-", dir=output_dir) as temporary:
        stage = Path(temporary)
        filenames = []
        for name in SPLITS:
            filename = f"{name}.csv"
            with (stage / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DERIVED_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(row for row in rows if row["split"] == name)
            filenames.append(filename)
        if queue is not None:
            with (stage / "translation_queue.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=QUEUE_COLUMNS)
                writer.writeheader()
                writer.writerows(queue)
            filenames.append("translation_queue.csv")
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        filenames.append("manifest.json")
        for filename in filenames:
            os.replace(stage / filename, output_dir / filename)


def export_multilingual(
    input_dir: str | Path, output_dir: str | Path, sample_size: int,
    seed: int = 42, overwrite: bool = False,
) -> dict:
    """Sample one Bengali source per incident group, retaining its existing split."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise MultilingualValidationError("Input and output directories must differ; preserve the phase-1 splits.")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise MultilingualValidationError("sample_size must be a positive integer.")
    output_names = [*(f"{name}.csv" for name in SPLITS), "manifest.json", "translation_queue.csv"]
    if not overwrite and any((output_dir / name).exists() for name in output_names):
        raise MultilingualValidationError(
            f"Multilingual output already exists in {output_dir}; use a new directory or --overwrite."
        )
    rows = []
    input_hashes = {}
    for split in SPLITS:
        path = input_dir / f"{split}.csv"
        if not path.is_file() and (input_dir / f"{split}.parquet").is_file():
            raise MultilingualValidationError(
                f"CSV split missing: {path}. Re-run prepare_data with --format csv or --format both."
            )
        rows.extend(dict(row, split=split) for row in _read_csv(path, SCHEMA_COLUMNS))
        input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _validate_splits(rows)
    # Stable representative and ordering make sampling independent of CSV row order.
    by_group: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda row: row["id"]):
        if row["lang"] == "bn":
            by_group.setdefault(row["original_id"], row)
    if sample_size > len(by_group):
        raise MultilingualValidationError(
            f"Requested {sample_size} Bengali incident groups, but only {len(by_group)} are available."
        )
    chosen_groups = sorted(random.Random(seed).sample(sorted(by_group), sample_size))
    sources = [by_group[group] for group in chosen_groups]
    base_rows = [version for source in sources for version in (_source_record(source), _banglish_record(source))]
    manifest = {
        "schema_version": MANIFEST_VERSION, "kind": "banglapsg_multilingual",
        "seed": seed, "sample_size": sample_size,
        "input_dir": str(input_dir.resolve()), "input_sha256": input_hashes,
        "transliterator": TRANSLITERATOR_VERSION,
        "selection": "one bn representative per sampled original_id; existing split retained",
        "selected_sources": sources,
        # Preserve the exact exported Banglish baseline even if rules later change.
        "base_rows": base_rows,
        "translation_status": {"completed": 0, "pending": sample_size},
    }
    _write_dataset(output_dir, base_rows, manifest, [_queue_record(source) for source in sources])
    return manifest


def _load_snapshot(work_dir: Path) -> tuple[dict, list[dict[str, str]], list[dict[str, str]]]:
    path = work_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MultilingualValidationError(f"Cannot read export manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("kind") != "banglapsg_multilingual" or manifest.get("schema_version") != MANIFEST_VERSION:
        raise MultilingualValidationError("The work directory must contain a supported multilingual export manifest.")
    sources, base_rows = manifest.get("selected_sources"), manifest.get("base_rows")
    for records, columns, name in (
        (sources, (*SCHEMA_COLUMNS, "split"), "selected_sources"),
        (base_rows, (*DERIVED_COLUMNS, "split"), "base_rows"),
    ):
        if not isinstance(records, list) or not records:
            raise MultilingualValidationError(f"Manifest {name} must be a nonempty row snapshot.")
        if any(not isinstance(row, dict) or set(row) != set(columns)
               or any(not isinstance(value, str) for value in row.values()) for row in records):
            raise MultilingualValidationError(f"Malformed manifest {name} snapshot.")
        _validate_splits(records, derived=name == "base_rows")
    if any(row["lang"] != "bn" for row in sources):
        raise MultilingualValidationError("Manifest selected_sources must contain Bengali originals.")
    if len({row["original_id"] for row in sources}) != len(sources):
        raise MultilingualValidationError("Manifest must select exactly one source per original_id.")
    if type(manifest.get("sample_size")) is not int or manifest["sample_size"] != len(sources):
        raise MultilingualValidationError("Manifest sample_size does not match its selected source snapshot.")
    expected_base_ids = set()
    base_by_id = {row["id"]: row for row in base_rows}
    for source in sources:
        source_id = source["id"]
        expected_base_ids.update((source_id, _id("banglish", source_id)))
        if base_by_id.get(source_id) != _source_record(source):
            raise MultilingualValidationError("Manifest source row differs from its saved base row.")
        variant = base_by_id.get(_id("banglish", source_id))
        expected = dict(source, id=_id("banglish", source_id), lang="banglish", is_derived="true",
                        derivation="rule_transliteration", parent_id=source_id)
        if variant is None or any(variant[key] != value for key, value in expected.items() if key != "text"):
            raise MultilingualValidationError("Manifest Banglish metadata differs from its source.")
    if set(base_by_id) != expected_base_ids:
        raise MultilingualValidationError("Manifest base rows contain missing or unexpected versions.")
    return manifest, sources, base_rows


def import_translations(
    work_dir: str | Path, translations: str | Path, allow_partial: bool = False,
    overwrite: bool = False,
) -> dict:
    """Validate immutable queue metadata and add English versions to saved splits."""
    work_dir, translations = Path(work_dir), Path(translations)
    manifest, sources, base_rows = _load_snapshot(work_dir)
    existing_rows = [dict(row, split=split) for split in SPLITS
                     for row in _read_csv(work_dir / f"{split}.csv", DERIVED_COLUMNS)]
    _validate_splits(existing_rows, derived=True)
    base_by_id = {row["id"]: row for row in base_rows}
    existing_by_id = {row["id"]: row for row in existing_rows}
    if any(existing_by_id.get(key) != row for key, row in base_by_id.items()):
        raise MultilingualValidationError("Saved source/Banglish rows changed or disappeared; restore the export before importing.")
    english_ids = {_id("en", source["id"]) for source in sources}
    if set(existing_by_id) - set(base_by_id) - english_ids:
        raise MultilingualValidationError("Saved dataset contains unexpected rows; restore the export before importing.")
    if set(existing_by_id) & english_ids and not overwrite:
        raise MultilingualValidationError("English versions already exist; pass --overwrite to replace them from this translation file.")

    expected = {_id("en", source["id"]): _queue_record(source) for source in sources}
    filled = _read_csv(translations, QUEUE_COLUMNS)
    seen: set[str] = set()
    english: list[dict[str, str]] = []
    source_by_id = {source["id"]: source for source in sources}
    for record in filled:
        translation_id = record["translation_id"]
        if translation_id in seen:
            raise MultilingualValidationError(f"Duplicate translation_id: {translation_id!r}.")
        seen.add(translation_id)
        if translation_id not in expected:
            raise MultilingualValidationError(f"Unknown translation_id: {translation_id!r}.")
        expected_row = expected[translation_id]
        for field in QUEUE_COLUMNS[:-1]:
            if record[field] != expected_row[field]:
                raise MultilingualValidationError(
                    f"Translation {translation_id!r}: immutable field {field!r} was changed. Only english_text may be edited."
                )
        text = clean_text(record["english_text"])
        if not text:
            continue
        if any("\u0980" <= char <= "\u09ff" for char in text):
            raise MultilingualValidationError(
                f"Translation {translation_id!r} still contains Bangla script; supply an English translation."
            )
        row = dict(source_by_id[record["source_id"]])
        row.update(id=translation_id, text=text, lang="en", is_derived="true",
                   derivation="manual_translation", parent_id=record["source_id"])
        english.append(row)
    pending = len(expected) - len(english)
    if pending and not allow_partial:
        raise MultilingualValidationError(
            f"{pending} English translations are missing/blank. Fill every english_text or explicitly pass --allow-partial."
        )
    english.sort(key=lambda row: (row["original_id"], row["id"]))
    all_rows = [*base_rows, *english]
    manifest["translation_status"] = {"completed": len(english), "pending": pending}
    manifest["translation_file_sha256"] = hashlib.sha256(translations.read_bytes()).hexdigest()
    _write_dataset(work_dir, all_rows, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Sample Bengali incidents and write a translation queue")
    export.add_argument("--input-dir", default="ml_training/data/processed")
    export.add_argument("--output-dir", default="ml_training/data/multilingual")
    export.add_argument("--sample-size", type=int, required=True)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--overwrite", action="store_true")
    importer = commands.add_parser("import", help="Validate a filled queue and add English versions")
    importer.add_argument("--work-dir", default="ml_training/data/multilingual")
    importer.add_argument("--translations", required=True)
    importer.add_argument("--allow-partial", action="store_true")
    importer.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "export":
            output_dir = _path(args.output_dir)
            manifest = export_multilingual(_path(args.input_dir), output_dir, args.sample_size, args.seed, args.overwrite)
        else:
            output_dir = _path(args.work_dir)
            manifest = import_translations(output_dir, _path(args.translations), args.allow_partial, args.overwrite)
        print(f"Selected incident groups: {manifest['sample_size']}")
        for split, stats in manifest["splits"].items():
            print(f"{split}: rows={stats['rows']} groups={stats['original_groups']} languages={stats['language_counts']}")
        status = manifest["translation_status"]
        print(f"English translations: {status['completed']} complete / {status['pending']} pending")
        print("Leakage check: PASS (original split assignments retained)")
        print(f"Wrote multilingual files to {output_dir}")
        return 0
    except (MultilingualValidationError, OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
