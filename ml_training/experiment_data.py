"""Validated, group-isolated experiment inputs shared by all model families."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

from ml.normalize import clean_text, normalize_banglish
from ml_training.config import SCHEMA_COLUMNS

LANGUAGES = ("bn", "en", "banglish")
SPLITS = ("train", "val", "test")


def preprocess_text(text: str, lang: str, variants_path: str | Path | None = None) -> str:
    if lang not in LANGUAGES:
        raise ValueError(f"Unsupported language {lang!r}; use bn, en or banglish")
    return normalize_banglish(text, variants_path) if lang == "banglish" else clean_text(text)


def read_experiment_csv(path: Path, natural: bool = False) -> pd.DataFrame:
    """Strict widths and string values avoid inferred indices and numeric IDs."""
    if not path.is_file():
        raise ValueError(f"CSV not found: {path}. Prepare CSV splits first (--format both or csv).")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        if len(set(header)) != len(header):
            raise ValueError(f"Duplicate CSV headers: {path}")
        required = ("text", "lang", "category", "location", "timestamp") if natural else SCHEMA_COLUMNS
        missing = set(required) - set(header)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        records = []
        for number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path}: CSV record {number} has {len(row)} fields; expected {len(header)}")
            records.append(row)
    frame = pd.DataFrame(records, columns=header, dtype=str)
    for field in SCHEMA_COLUMNS:
        if field not in frame:
            frame[field] = ""
        frame[field] = frame[field].str.strip()
    for field in ("text", "category") + (() if natural else ("id", "original_id", "severity")):
        if frame[field].eq("").any():
            raise ValueError(f"{path}: '{field}' contains blank values")
    if not frame["lang"].isin(LANGUAGES).all():
        raise ValueError(f"{path}: lang must be bn, en or banglish; annotate language explicitly")
    if natural:
        for index in frame.index:
            digest = hashlib.sha256(frame.at[index, "text"].encode("utf-8")).hexdigest()
            if not frame.at[index, "id"]:
                frame.at[index, "id"] = f"natural_{digest}_{index}"
            if not frame.at[index, "original_id"]:
                frame.at[index, "original_id"] = f"natural_{digest}"
    frame["raw_text"] = frame["text"]
    frame["text"] = [preprocess_text(text, lang) for text, lang in zip(frame["text"], frame["lang"])]
    if frame["text"].eq("").any():
        raise ValueError(f"{path}: text became empty after cleaning")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate id values")
    return frame.reset_index(drop=True)


def load_experiment_data(
    input_dir: str | Path,
    natural_test: str | Path | None = None,
    severity_order: list[str] | None = None,
) -> dict:
    """Load prepared splits and optional independently collected natural data.

    Category taxonomy comes from training only. Unknown held-out categories are
    rejected rather than silently removed. Severity order is user-defined from
    least to most severe; natural rows may lack severity and remain in category
    evaluation. Priority (1–5) is never implicitly substituted for severity.
    """
    directory = Path(input_dir)
    frames = {split: read_experiment_csv(directory / f"{split}.csv") for split in SPLITS}
    for split in ("train", "val"):
        if frames[split].empty:
            raise ValueError(f"{split} split is empty; independent training and calibration data are required")
    empty = frames["test"].iloc[0:0].copy()
    frames["natural"] = read_experiment_csv(Path(natural_test), natural=True) if natural_test else empty
    warnings = []
    group_owner: dict[str, str] = {}
    row_owner: dict[str, str] = {}
    for split, frame in frames.items():
        for identifier in frame["id"]:
            if identifier in row_owner:
                raise ValueError(f"Row ID {identifier!r} occurs in {row_owner[identifier]} and {split}")
            row_owner[identifier] = split
        for group in frame["original_id"].unique():
            previous = group_owner.setdefault(group, split)
            if previous != split:
                raise ValueError(f"Split leakage: original_id {group!r} occurs in {previous} and {split}")
    categories = sorted(frames["train"]["category"].unique().tolist())
    for split, frame in frames.items():
        unknown = set(frame["category"]) - set(categories)
        if unknown:
            raise ValueError(
                f"{split}: categories absent from training: {sorted(unknown)}. "
                "Correct the taxonomy/group split upstream; held-out rows will not be dropped."
            )
    order = list(severity_order or [])
    if order:
        if len(order) < 2 or len(set(order)) != len(order) or any(not label.strip() for label in order):
            raise ValueError("severity_order needs at least two distinct nonempty labels, least to most severe")
        for split, frame in frames.items():
            unknown = set(frame.loc[frame["severity"].ne(""), "severity"]) - set(order)
            if unknown:
                raise ValueError(f"{split}: severity labels not in --severity-order: {sorted(unknown)}")
    if not frames["natural"].empty:
        source_texts = set(pd.concat([frames[split]["text"] for split in SPLITS]))
        overlaps = int(frames["natural"]["text"].isin(source_texts).sum())
        if overlaps:
            warnings.append(f"{overlaps} natural rows exactly match normalized source text; audit incident independence before reporting natural results.")
        if not frames["natural"]["severity"].ne("").any():
            warnings.append("Natural data has no severity labels; natural severity evaluation will be skipped. Source priority judgments are not mapped automatically.")
        warnings.append("Natural incident independence requires a manual provenance audit; text hashes cannot identify paraphrases or translations.")
    if frames["test"].empty:
        warnings.append("Test split is empty; test cohort metrics will be skipped.")
    return {**frames, "category_labels": categories, "severity_labels": order, "warnings": warnings}
