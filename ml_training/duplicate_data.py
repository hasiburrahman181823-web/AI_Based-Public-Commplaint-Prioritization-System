"""Strict, incident-isolated duplicate-pair inputs with independent metadata.

The generated ``label`` column is a candidate suggestion. By default evaluation
requires a separately adjudicated ``gold_label`` or ``human_label`` column.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from ml.normalize import clean_text

SPLITS = ("train", "val", "test")
LANGUAGE_ORDER = ("bn", "en", "banglish")
PAIR_REQUIRED = (
    "text_a", "text_b", "loc_a", "loc_b", "time_a", "time_b", "pair_type",
    "pair_id", "id_a", "id_b", "original_id_a", "original_id_b",
)
COMPLAINT_REQUIRED = ("id", "original_id", "text", "lang", "category", "location", "timestamp")


def _read_csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"CSV not found: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, [])
            if len(set(header)) != len(header):
                raise ValueError(f"Duplicate CSV headers in {path}")
            missing = set(required) - set(header)
            if missing:
                suffix = " Adjudicate the long-form annotation export into one gold label per pair first." if "item_id" in header else ""
                raise ValueError(f"{path}: missing columns {sorted(missing)}.{suffix}")
            rows = []
            for number, row in enumerate(reader, 2):
                if len(row) != len(header):
                    raise ValueError(f"{path}: CSV record {number} has {len(row)} fields; expected {len(header)}")
                rows.append([value.strip() for value in row])
        return pd.DataFrame(rows, columns=header, dtype=str)
    except (csv.Error, UnicodeError) as exc:
        raise ValueError(f"Cannot parse UTF-8 CSV {path}: {exc}") from exc


def _timestamp(value: str, description: str) -> str:
    if not value:
        return ""
    try:
        stamp = pd.to_datetime(value, format="ISO8601", utc=True, errors="raise")
        if pd.isna(stamp):
            raise ValueError("not a real date/time")
        return stamp.isoformat().replace("+00:00", "Z")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{description}: invalid ISO-8601 timestamp {value!r}") from exc


def language_pair(lang_a: str, lang_b: str) -> str:
    """Canonical unordered language pair, independent of target/pair type."""
    if lang_a not in LANGUAGE_ORDER or lang_b not in LANGUAGE_ORDER:
        raise ValueError("Languages must be bn, en or banglish")
    left, right = sorted((lang_a, lang_b), key=LANGUAGE_ORDER.index)
    return f"{left}-{right}"


def load_duplicate_data(
    pair_dir: str | Path,
    complaint_paths: list[str | Path],
    allow_weak_labels: bool = False,
) -> dict:
    """Return enriched train/val/test frames and explicit label-provenance notes.

    ``pair_dir`` is a directory with train/pairs.csv, val/pairs.csv and
    test/pairs.csv, or a combined pairs.csv with a split column. Complaint
    metadata supplies language and category; neither is inferred from pair_type
    or the duplicate label. All accepted pairs are retained, with no random
    row-level re-splitting and no automatic vote aggregation.
    """
    if not complaint_paths:
        raise ValueError("Supply complaint CSVs so languages/categories come from independent complaint metadata")
    complaints: dict[str, dict] = {}
    group_owner: dict[str, str] = {}
    row_owner: dict[str, str] = {}

    def own_group(group: str, split: str, description: str) -> None:
        if not split:
            return
        if split not in SPLITS:
            raise ValueError(f"{description}: split must be train, val or test")
        previous = group_owner.setdefault(group, split)
        if previous != split:
            raise ValueError(f"Split leakage: original_id {group!r} crosses {previous} and {split} ({description})")

    for value in complaint_paths:
        path = Path(value)
        frame = _read_csv(path, COMPLAINT_REQUIRED)
        default_split = path.stem if path.stem in SPLITS else ""
        for row in frame.to_dict("records"):
            for field in ("id", "original_id", "text", "lang", "category"):
                if not row[field]:
                    raise ValueError(f"{path}: complaint field {field!r} must be nonempty")
            if row["lang"] not in LANGUAGE_ORDER:
                raise ValueError(f"{path}: lang must be bn, en or banglish")
            identifier = row["id"]
            if identifier in complaints:
                raise ValueError(f"Complaint id {identifier!r} is repeated across supplied metadata files")
            row["timestamp"] = _timestamp(row["timestamp"], f"{path}: {identifier}")
            row["split"] = row.get("split", "") or default_split
            own_group(row["original_id"], row["split"], str(path))
            complaints[identifier] = row

    source = Path(pair_dir)
    if source.is_file():
        combined = _read_csv(source, (*PAIR_REQUIRED, "split"))
        if not combined["split"].isin(SPLITS).all():
            raise ValueError(f"{source}: split must be train, val or test")
        frames = {split: combined.loc[combined["split"].eq(split)].copy().reset_index(drop=True) for split in SPLITS}
    else:
        frames = {}
        for split in SPLITS:
            path = source / split / "pairs.csv"
            frame = _read_csv(path, PAIR_REQUIRED)
            if "split" in frame and not frame["split"].eq(split).all():
                raise ValueError(f"{path}: split column disagrees with the split directory")
            frame["split"] = split
            frames[split] = frame

    notes = []
    used_pair_ids: set[str] = set()
    used_orientations: set[tuple[str, str]] = set()
    label_columns: dict[str, str] = {}
    for split, frame in frames.items():
        label_column = next((column for column in ("gold_label", "human_label") if column in frame), None)
        if label_column is None:
            if not allow_weak_labels or "label" not in frame:
                raise ValueError(
                    f"{split}: add an adjudicated gold_label (preferred) or human_label column. "
                    "Phase 2 label values are generated candidate suggestions, not independent truth. "
                    "Use --allow-weak-labels only for an explicitly labeled pipeline/synthetic experiment."
                )
            label_column = "label"
            notes.append(f"{split}: weak generated label suggestions are being used; this is not an independently annotated research evaluation.")
        label_columns[split] = label_column
        if not frame[label_column].isin(("0", "1")).all():
            raise ValueError(f"{split}: {label_column} must contain an explicit 0 or 1 on every pair. Adjudicate skips/missing labels first; no rows will be dropped.")
        if "gold_label" in frame and "human_label" in frame:
            comparable = frame["human_label"].isin(("0", "1"))
            if (frame.loc[comparable, "gold_label"] != frame.loc[comparable, "human_label"]).any():
                notes.append(f"{split}: gold_label takes precedence over differing human_label values; retain independent annotations for the agreement audit.")
        if "label" in frame:
            frame["candidate_label"] = frame["label"]
        frame["label"] = frame[label_column].astype(int)
        frame["label_source"] = label_column
        for field in ("pair_id", "id_a", "id_b", "original_id_a", "original_id_b", "text_a", "text_b", "pair_type"):
            if frame[field].eq("").any():
                raise ValueError(f"{split}: '{field}' contains blank values")

        enriched = []
        for row in frame.to_dict("records"):
            pair_id = row["pair_id"]
            if pair_id in used_pair_ids:
                raise ValueError(f"Duplicate pair_id {pair_id!r}")
            used_pair_ids.add(pair_id)
            if row["id_a"] == row["id_b"]:
                raise ValueError(f"Pair {pair_id!r} compares a complaint with itself")
            orientation = tuple(sorted((row["id_a"], row["id_b"])))
            if orientation in used_orientations:
                raise ValueError(f"Duplicate unordered complaint pair {orientation}; remove repeated/reversed pairs upstream")
            used_orientations.add(orientation)
            for side in ("a", "b"):
                identifier = row[f"id_{side}"]
                if identifier not in complaints:
                    raise ValueError(f"Pair {pair_id!r}: complaint id {identifier!r} missing from supplied metadata. Include manual-paraphrase CSVs as well.")
                complaint = complaints[identifier]
                if row[f"original_id_{side}"] != complaint["original_id"]:
                    raise ValueError(f"Pair {pair_id!r}: original_id_{side} disagrees with complaint metadata")
                if clean_text(row[f"text_{side}"]) != clean_text(complaint["text"]):
                    raise ValueError(f"Pair {pair_id!r}: text_{side} disagrees with complaint metadata")
                if clean_text(row[f"loc_{side}"]) != clean_text(complaint["location"]):
                    raise ValueError(f"Pair {pair_id!r}: loc_{side} disagrees with complaint metadata")
                row[f"time_{side}"] = _timestamp(row[f"time_{side}"], f"Pair {pair_id!r}")
                if row[f"time_{side}"] != complaint["timestamp"]:
                    raise ValueError(f"Pair {pair_id!r}: time_{side} disagrees with complaint metadata")
                known_split = complaint["split"]
                if known_split and known_split != split:
                    raise ValueError(f"Pair {pair_id!r}: complaint {identifier!r} belongs to {known_split}, not {split}")
                previous = row_owner.setdefault(identifier, split)
                if previous != split:
                    raise ValueError(f"Split leakage: complaint id {identifier!r} crosses {previous} and {split}")
                own_group(complaint["original_id"], split, f"pair {pair_id}")
                row[f"lang_{side}"] = complaint["lang"]
                row[f"category_{side}"] = complaint["category"]
            row["language_pair"] = language_pair(row["lang_a"], row["lang_b"])
            enriched.append(row)
        # Keep a complete, stable schema even when an input split has no pairs.
        output_columns = list(dict.fromkeys([*frame.columns, "lang_a", "category_a", "lang_b", "category_b", "language_pair"]))
        frames[split] = pd.DataFrame(enriched, columns=output_columns)
        if not frames[split].empty:
            frames[split]["label"] = frames[split]["label"].astype(int)
    notes.append("Pair type is used only for evaluation breakdowns; it is never a model feature. Category/language features come from complaint metadata, not pair labels.")
    return {**frames, "warnings": notes}
