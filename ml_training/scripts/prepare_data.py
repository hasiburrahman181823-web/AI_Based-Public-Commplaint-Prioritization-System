"""Validate, normalize and split a local BanglaPSG CSV.

Run from the repository root:
    python -m ml_training.scripts.prepare_data
or:
    python ml_training/scripts/prepare_data.py --format csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings

# Support direct script execution from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from dotenv import load_dotenv

from ml.normalize import clean_text, normalize_banglish
from ml_training.config import COLUMN_MAPPING, REQUIRED_COLUMNS, SCHEMA_COLUMNS, SPLIT_RATIOS
from ml_training.splitting import split_grouped


class DataValidationError(ValueError):
    """An actionable error in the source CSV or mapping configuration."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def validate_mapping(overrides: dict[str, str] | None = None) -> dict[str, str]:
    mapping = dict(COLUMN_MAPPING)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise DataValidationError("Column mapping must be a JSON object: output_name -> CSV_header.")
        mapping.update(overrides)
    unknown = set(mapping) - set(SCHEMA_COLUMNS)
    if unknown:
        raise DataValidationError(f"Unknown column mapping keys: {sorted(unknown)}.")
    for key in SCHEMA_COLUMNS:
        if key not in mapping or not isinstance(mapping[key], str) or not mapping[key].strip():
            raise DataValidationError(f"Column mapping for '{key}' must be a nonempty CSV header string.")
    if len(set(mapping.values())) != len(mapping):
        raise DataValidationError("Each standardized column must map to a different CSV header.")
    return mapping


def _bad_rows(mask: pd.Series) -> str:
    # Header is record 1; CSV records can span more than one physical line.
    return ", ".join(str(int(i) + 2) for i in mask[mask].index[:10])


def _require_nonempty(frame: pd.DataFrame, name: str) -> None:
    bad = frame[name].eq("")
    if bad.any():
        raise DataValidationError(
            f"Column '{name}' contains empty values at CSV record(s) {_bad_rows(bad)}. "
            "Fill these values before preparing the data."
        )


def load_and_prepare(
    input_path: str | Path,
    column_mapping: dict[str, str] | None = None,
    variants_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return exactly the documented schema, with all values as strings.

    Source labels are stripped but not recoded. Rows are never silently dropped.
    Blank optional identifiers are generated; duplicate supplied IDs are errors.
    """
    path = Path(input_path)
    mapping = validate_mapping(column_mapping)
    if not path.is_file():
        raise DataValidationError(
            f"CSV not found: {path}. Place your data there or pass --input PATH."
        )
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            duplicates = sorted({name for name in header if header.count(name) > 1})
            if duplicates:
                raise DataValidationError(f"Duplicate source CSV headers: {duplicates}. Rename them first.")
            missing = [f"'{key}' -> '{mapping[key]}'" for key in REQUIRED_COLUMNS if mapping[key] not in header]
            if missing:
                raise DataValidationError(
                    f"Missing required CSV columns: {', '.join(missing)}. "
                    f"Available headers: {header}. Edit COLUMN_MAPPING in ml_training/config.py "
                    "or pass --column-map mapping.json."
                )
            records = []
            for record_number, row in enumerate(reader, start=2):
                # Strict width checks prevent pandas from silently treating an
                # over-wide first field as an inferred index and shifting labels.
                if len(row) != len(header):
                    raise DataValidationError(
                        f"CSV record {record_number} has {len(row)} fields; expected {len(header)}. "
                        "Quote text containing commas and check for missing/extra fields."
                    )
                records.append(row)
        raw = pd.DataFrame(records, columns=header, dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError, csv.Error) as exc:
        raise DataValidationError(f"Cannot read UTF-8 CSV '{path}': {exc}") from exc
    if raw.empty:
        raise DataValidationError("The source CSV has no data rows. Add data beneath the header.")

    frame = pd.DataFrame(index=raw.index)
    for name in SCHEMA_COLUMNS:
        source = mapping[name]
        frame[name] = raw[source].str.strip() if source in raw.columns else ""
    for name in REQUIRED_COLUMNS:
        _require_nonempty(frame, name)

    frame["text"] = frame["text"].map(clean_text)
    _require_nonempty(frame, "text")
    cleaned_source = frame["text"].copy()

    frame["lang"] = frame["lang"].str.lower()
    bad_lang = ~frame["lang"].isin(["", "bn", "en", "banglish"])
    if bad_lang.any():
        raise DataValidationError(
            f"Invalid lang at CSV record(s) {_bad_rows(bad_lang)}. Use bn, en or banglish."
        )
    inferred = frame["lang"].eq("")
    frame.loc[inferred, "lang"] = frame.loc[inferred, "text"].map(
        lambda text: "bn" if any("\u0980" <= char <= "\u09ff" for char in text) else "en"
    )
    if inferred.any():
        warnings.warn(
            f"Inferred lang for {int(inferred.sum())} row(s): Bangla script -> bn, otherwise en. "
            "Supply lang=banglish to enable Banglish normalization for romanized Bangla.",
            UserWarning,
            stacklevel=2,
        )
    is_banglish = frame["lang"].eq("banglish")
    frame.loc[is_banglish, "text"] = frame.loc[is_banglish, "text"].map(
        lambda text: normalize_banglish(text, variants_path=variants_path)
    )
    _require_nonempty(frame, "text")

    # Keep exact cleaned repeats together regardless of label or severity.
    missing_groups = frame["original_id"].eq("")
    frame.loc[missing_groups, "original_id"] = cleaned_source[missing_groups].map(
        lambda text: "auto_" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    if missing_groups.any():
        warnings.warn(
            f"Generated original_id for {int(missing_groups.sum())} row(s) from cleaned source text. "
            "Provide shared original_id values for translations, paraphrases and augmentations.",
            UserWarning,
            stacklevel=2,
        )

    # Include a duplicate occurrence number to give repeated records unique row IDs.
    occurrences: dict[str, int] = {}
    for idx in frame.index[frame["id"].eq("")]:
        payload = json.dumps(
            [frame.at[idx, name] for name in SCHEMA_COLUMNS if name != "id"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        occurrences[digest] = occurrences.get(digest, 0) + 1
        frame.at[idx, "id"] = f"row_{digest}_{occurrences[digest]}"
    duplicate_ids = frame["id"].duplicated(keep=False)
    if duplicate_ids.any():
        raise DataValidationError(
            f"Duplicate id values at CSV record(s) {_bad_rows(duplicate_ids)}. "
            "Each row needs a unique id; related rows should share original_id instead."
        )
    frame.loc[frame["source_type"].eq(""), "source_type"] = "banglapsg"
    for idx, value in frame.loc[frame["timestamp"].ne(""), "timestamp"].items():
        try:
            # Naive timestamps are interpreted as UTC; offset timestamps convert to UTC.
            stamp = pd.to_datetime(value, format="ISO8601", utc=True, errors="raise")
            if pd.isna(stamp):
                raise ValueError("timestamp is not a real date/time")
            frame.at[idx, "timestamp"] = stamp.isoformat().replace("+00:00", "Z")
        except (ValueError, TypeError, OverflowError) as exc:
            raise DataValidationError(
                f"Invalid ISO-8601 timestamp at CSV record {idx + 2}: {value!r}. "
                "Use e.g. 2026-01-01T12:00:00+06:00 or leave it empty."
            ) from exc
    return frame.loc[:, list(SCHEMA_COLUMNS)].reset_index(drop=True)


def write_outputs(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    output_format: str,
    manifest: dict,
    overwrite: bool = False,
) -> list[Path]:
    """Stage all output files before replacing destinations; avoid partial write errors."""
    extensions = ["csv", "parquet"] if output_format == "both" else [output_format]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse stale files from a previous run even if the requested format changed.
    existing = [output_dir / f"{split}.{ext}" for split in splits for ext in ("csv", "parquet")]
    existing.append(output_dir / "manifest.json")
    if not overwrite and any(path.exists() for path in existing):
        raise DataValidationError(
            f"Prepared output already exists in {output_dir}. Use a new --output-dir "
            "or pass --overwrite to replace existing split files."
        )
    names = []
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_dir) as temporary:
        stage = Path(temporary)
        for split, data in splits.items():
            for ext in extensions:
                filename = f"{split}.{ext}"
                destination = stage / filename
                if ext == "csv":
                    data.to_csv(destination, index=False, encoding="utf-8")
                else:
                    data.to_parquet(destination, index=False, engine="pyarrow")
                names.append(filename)
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        names.append("manifest.json")
        for name in names:
            os.replace(stage / name, output_dir / name)
    if overwrite:
        for path in existing:
            if path.name not in names and path.is_file():
                path.unlink()  # Only known obsolete split formats inside output_dir.
    return [output_dir / name for name in names]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.getenv("BANGLAPSG_RAW_PATH", "ml_training/data/raw/banglapsg.csv"))
    parser.add_argument("--output-dir", default=os.getenv("BANGLAPSG_OUTPUT_DIR", "ml_training/data/processed"))
    parser.add_argument("--format", choices=("both", "csv", "parquet"), default="both")
    parser.add_argument("--seed", type=int, default=os.getenv("BANGLAPSG_RANDOM_SEED", "42"))
    parser.add_argument("--column-map", type=Path, help="JSON canonical column name -> source CSV header")
    parser.add_argument("--variants", type=Path, help="Replacement JSON Banglish alias dictionary")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing prepared splits")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env", override=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        input_path = resolve_path(args.input)
        output_dir = resolve_path(args.output_dir)
        # Never overwrite a raw input with one of the known output filenames.
        if input_path.resolve() in {
            (output_dir / f"{split}.{ext}").resolve()
            for split in ("train", "val", "test") for ext in ("csv", "parquet")
        }:
            raise DataValidationError("The input CSV must be separate from the output split files.")
        overrides = None
        if args.column_map:
            overrides = json.loads(resolve_path(args.column_map).read_text(encoding="utf-8-sig"))
        variants = resolve_path(args.variants) if args.variants else None
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            frame = load_and_prepare(input_path, overrides, variants)
            splits = split_grouped(frame, ratios=SPLIT_RATIOS, seed=args.seed)
        messages = list(dict.fromkeys(str(item.message) for item in captured))
        # Defense in depth: refuse to write if the splitting invariant is violated.
        seen_groups: set[str] = set()
        for name, split in splits.items():
            groups = set(split["original_id"])
            if seen_groups & groups:
                raise DataValidationError(f"Internal split leakage detected in {name}; no output written.")
            seen_groups.update(groups)
        if sum(len(split) for split in splits.values()) != len(frame):
            raise DataValidationError("Internal split row-count mismatch; no output written.")
        manifest = {
            "schema_version": 1,
            "input_file": str(input_path.resolve()),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "seed": args.seed,
            "target_ratios": dict(zip(("train", "val", "test"), SPLIT_RATIOS)),
            "column_mapping": validate_mapping(overrides),
            "variants_file": str(variants or REPO_ROOT / "ml" / "banglish_variants.json"),
            "variants_sha256": hashlib.sha256(
                (variants or REPO_ROOT / "ml" / "banglish_variants.json").read_bytes()
            ).hexdigest(),
            "rows": len(frame),
            "original_groups": int(frame["original_id"].nunique()),
            "warnings": messages,
            "splits": {
                name: {
                    "rows": len(split),
                    "row_fraction": len(split) / len(frame),
                    "groups": int(split["original_id"].nunique()),
                    "category_counts": {str(k): int(v) for k, v in split["category"].value_counts().sort_index().items()},
                    "files": [f"{name}.{ext}" for ext in (["csv", "parquet"] if args.format == "both" else [args.format])],
                }
                for name, split in splits.items()
            },
        }
        files = write_outputs(splits, output_dir, args.format, manifest, args.overwrite)
        for message in messages:
            print(f"WARNING: {message}", file=sys.stderr)
        print(f"Loaded {len(frame)} rows / {frame['original_id'].nunique()} original groups")
        print("split    rows  groups  row_fraction  category_counts")
        for name, stats in manifest["splits"].items():
            print(f"{name:<7} {stats['rows']:>5} {stats['groups']:>7}  {stats['row_fraction']:.4f}        "
                  f"{json.dumps(stats['category_counts'], ensure_ascii=False, sort_keys=True)}")
        print("Leakage check: PASS (original_id sets are disjoint)")
        print(f"Wrote {len(files)} files to {output_dir}")
        return 0
    except (DataValidationError, ValueError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
