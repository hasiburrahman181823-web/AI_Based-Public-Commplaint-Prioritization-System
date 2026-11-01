"""Build bounded, reproducible candidate pairs inside existing data splits.

Labels are suggestions for human annotation, including derived-text positives.
No model or network service is called. Run from the repository root:
    python -m ml_training.scripts.generate_pairs --max-pairs-per-type 1000
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from ml_training.config import SCHEMA_COLUMNS

SPLITS = ("train", "val", "test")
PAIR_TYPES = ("positive_paraphrase", "positive_cross_language", "hard_negative", "easy_negative")
PAIR_COLUMNS = (
    "text_a", "text_b", "loc_a", "loc_b", "time_a", "time_b", "label", "pair_type",
    "pair_id", "split", "id_a", "id_b", "original_id_a", "original_id_b",
    "is_derived_a", "is_derived_b", "derivation_a", "derivation_b", "parent_id_a", "parent_id_b",
)
EXTRA_COLUMNS = ("is_derived", "derivation", "parent_id", "text_style")


class PairValidationError(ValueError):
    """Invalid split data or manual paraphrase input."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _read_csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise PairValidationError(f"CSV not found: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            if len(set(header)) != len(header):
                raise PairValidationError(f"Duplicate CSV headers in {path}")
            missing = sorted(set(required) - set(header))
            if missing:
                raise PairValidationError(f"Missing required columns in {path}: {', '.join(missing)}")
            rows = []
            for number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise PairValidationError(
                        f"{path}: CSV record {number} has {len(row)} fields; expected {len(header)}"
                    )
                rows.append(row)
        return pd.DataFrame(rows, columns=header, dtype=str)
    except (UnicodeError, csv.Error) as exc:
        raise PairValidationError(f"Cannot read UTF-8 CSV {path}: {exc}") from exc


def _validate_frame(frame: pd.DataFrame, description: str) -> pd.DataFrame:
    missing = sorted(set(SCHEMA_COLUMNS) - set(frame.columns))
    if missing:
        raise PairValidationError(f"{description}: missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    for column in (*SCHEMA_COLUMNS, *EXTRA_COLUMNS):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    for column in ("id", "original_id", "text", "category"):
        bad = frame[column].eq("")
        if bad.any():
            raise PairValidationError(f"{description}: '{column}' contains {int(bad.sum())} empty values")
    frame["lang"] = frame["lang"].str.lower()
    if not frame["lang"].isin(("bn", "en", "banglish")).all():
        raise PairValidationError(f"{description}: lang must be bn, en, or banglish")
    flags = frame["is_derived"].str.lower()
    if not flags.isin(("", "false", "true", "0", "1")).all():
        raise PairValidationError(f"{description}: is_derived must be true/false or 1/0")
    frame["is_derived"] = flags.map(lambda value: "true" if value in ("true", "1") else "false")
    for index, value in frame.loc[frame["timestamp"].ne(""), "timestamp"].items():
        try:
            stamp = pd.to_datetime(value, format="ISO8601", utc=True, errors="raise")
            if pd.isna(stamp):
                raise ValueError("not a real timestamp")
            frame.at[index, "timestamp"] = stamp.isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, OverflowError) as exc:
            raise PairValidationError(f"{description}: invalid ISO-8601 timestamp {value!r}") from exc
    return frame


def _validate_splits(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if set(splits) != set(SPLITS):
        raise PairValidationError("Supply exactly train, val, and test dataframes")
    validated = {}
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    for split in SPLITS:
        frame = _validate_frame(splits[split], split).reset_index(drop=True)
        ids = set(frame["id"])
        if frame["id"].duplicated().any() or ids & seen_ids:
            raise PairValidationError(f"IDs must be globally unique; duplicate id in {split}")
        groups = set(frame["original_id"])
        if groups & seen_groups:
            raise PairValidationError(f"Split leakage: original_id appears in multiple splits ({split})")
        seen_ids.update(ids)
        seen_groups.update(groups)
        validated[split] = frame
    return validated


def load_splits(input_dir: str | Path, paraphrases: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load split CSVs and optionally attach explicitly identified manual paraphrases.

    A paraphrase must identify an existing original_id and its existing split.
    It is never allowed to introduce a group or move a group between splits.
    """
    directory = Path(input_dir)
    splits = _validate_splits({name: _read_csv(directory / f"{name}.csv", SCHEMA_COLUMNS) for name in SPLITS})
    if paraphrases is None:
        return splits
    manual = _read_csv(Path(paraphrases), (*SCHEMA_COLUMNS, "split"))
    manual = _validate_frame(manual, "manual paraphrases")
    manual["split"] = manual["split"].str.strip().str.lower()
    group_split = {group: split for split, frame in splits.items() for group in frame["original_id"]}
    for row in manual.to_dict("records"):
        expected = group_split.get(row["original_id"])
        if expected is None:
            raise PairValidationError(f"Manual paraphrase {row['id']!r} references unknown original_id {row['original_id']!r}")
        if row["split"] != expected:
            raise PairValidationError(
                f"Manual paraphrase {row['id']!r} requests split {row['split']!r}; its original_id belongs to {expected}"
            )
    manual["is_derived"] = "true"
    manual["derivation"] = "manual_paraphrase"
    for split in SPLITS:
        additions = manual.loc[manual["split"].eq(split)].drop(columns="split")
        splits[split] = pd.concat([splits[split], additions], ignore_index=True)
    return _validate_splits(splits)


def _normalized_location(value: str) -> str:
    return " ".join(value.casefold().split())


def text_style(row: dict) -> tuple[str, ...]:
    """Use a supplied style, or language + whitespace-token length bucket.

    Explicit and inferred styles never match one another. Inferred buckets are
    short <=12 tokens, medium <=30 tokens, and long >30 tokens. This transparent
    heuristic is not a learned writing-style classifier.
    """
    explicit = " ".join(row.get("text_style", "").casefold().split())
    if explicit:
        return ("explicit", explicit)
    length = len(row["text"].split())
    return ("inferred", row["lang"], "short" if length <= 12 else "medium" if length <= 30 else "long")


def hard_negative_eligible(a: dict, b: dict) -> bool:
    """Require same category/style plus known different locations OR UTC dates.

    Missing metadata never counts as a difference. Naive ISO timestamps are
    interpreted as UTC when loading; date comparisons use the normalized UTC
    calendar date, so different times on the same date do not qualify.
    """
    if a["original_id"] == b["original_id"] or a["category"] != b["category"] or text_style(a) != text_style(b):
        return False
    loc_a, loc_b = _normalized_location(a["location"]), _normalized_location(b["location"])
    date_a, date_b = a["timestamp"][:10], b["timestamp"][:10]
    return bool((loc_a and loc_b and loc_a != loc_b) or (date_a and date_b and date_a != date_b))


def _pair(a: dict, b: dict, split: str, pair_type: str) -> dict:
    if b["id"] < a["id"]:
        a, b = b, a
    digest = hashlib.sha256(json.dumps([split, a["id"], b["id"]], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    result = {
        "text_a": a["text"], "text_b": b["text"], "loc_a": a["location"], "loc_b": b["location"],
        "time_a": a["timestamp"], "time_b": b["timestamp"],
        "label": 1 if pair_type.startswith("positive_") else 0,
        "pair_type": pair_type, "pair_id": "pair_" + digest, "split": split,
        "id_a": a["id"], "id_b": b["id"],
        "original_id_a": a["original_id"], "original_id_b": b["original_id"],
    }
    for name in ("is_derived", "derivation", "parent_id"):
        result[f"{name}_a"], result[f"{name}_b"] = a[name], b[name]
    return result


def generate_pairs(
    splits: dict[str, pd.DataFrame], max_pairs_per_type: int = 1000, seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Sample at most ``max_pairs_per_type`` of EACH type in EACH input split.

    Sampling uses at most max(200, cap*30) attempts per type per split, without
    materializing Cartesian products. Repeated candidates consume attempts, so
    a sparse or exhausted candidate pool may underfill the cap. Warnings and
    the summary state emitted counts and actual attempts, not an estimated total
    candidate population. Paraphrase type takes priority over cross-language.
    """
    if isinstance(max_pairs_per_type, bool) or not isinstance(max_pairs_per_type, int) or max_pairs_per_type < 1:
        raise PairValidationError("max_pairs_per_type must be a positive integer")
    splits = _validate_splits(splits)
    rng = np.random.default_rng(seed)
    all_pairs = []
    summary = {"seed": seed, "max_pairs_per_type": max_pairs_per_type, "cap_scope": "per_type_per_split", "label_status": "candidate_suggestions_require_human_annotation", "splits": {}}
    limit = max(200, max_pairs_per_type * 30)

    for split in SPLITS:
        # Stable ordering makes sampling reproducible even when CSV rows reorder.
        rows = splits[split].sort_values("id", kind="stable").to_dict("records")
        groups: dict[str, list[int]] = {}
        hard_buckets: dict[tuple, list[int]] = {}
        category_buckets: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(row["original_id"], []).append(index)
            hard_buckets.setdefault((row["category"], text_style(row)), []).append(index)
            category_buckets.setdefault(row["category"], []).append(index)
        manual_groups = []
        cross_groups = []
        positive_pool_sizes = {"positive_paraphrase": 0, "positive_cross_language": 0}
        for members in groups.values():
            manual = [i for i in members if rows[i]["derivation"].casefold() == "manual_paraphrase"]
            if manual and len(members) >= 2:
                manual_groups.append((members, manual))
                positive_pool_sizes["positive_paraphrase"] += len(manual) * (len(members) - len(manual)) + len(manual) * (len(manual) - 1) // 2
            languages: dict[str, list[int]] = {}
            for i in members:
                if rows[i]["derivation"].casefold() != "manual_paraphrase":
                    languages.setdefault(rows[i]["lang"], []).append(i)
            if len(languages) >= 2:
                cross_groups.append(list(languages.values()))
                counts = [len(values) for values in languages.values()]
                positive_pool_sizes["positive_cross_language"] += (sum(counts) ** 2 - sum(count ** 2 for count in counts)) // 2
        def has_metadata_difference(members: list[int]) -> bool:
            # A qualifying pair exists iff a known metadata field has at least
            # two values across at least two groups. Missing values are ignored.
            for field in ("location", "timestamp"):
                values, original_groups = set(), set()
                for index in members:
                    value = _normalized_location(rows[index][field]) if field == "location" else rows[index][field][:10]
                    if value:
                        values.add(value)
                        original_groups.add(rows[index]["original_id"])
                if len(values) >= 2 and len(original_groups) >= 2:
                    return True
            return False

        hard_groups = [members for members in hard_buckets.values() if has_metadata_difference(members)]
        easy_groups = list(category_buckets.values())
        seen: set[tuple[str, str]] = set()
        output = []
        stats = {"rows": len(rows), "original_groups": len(groups), "sampled_candidate_counts": {}, "attempts": {}, "warnings": []}

        def choose(values):
            return values[int(rng.integers(len(values)))]

        for pair_type in PAIR_TYPES:
            emitted, attempts = 0, 0
            # Positive-pool sizes are available from group/language counts in
            # linear time. Stop once a small pool is exhausted instead of
            # repeatedly drawing already emitted pairs until the attempt cap.
            desired = min(max_pairs_per_type, positive_pool_sizes.get(pair_type, max_pairs_per_type))
            available = bool(manual_groups) if pair_type == "positive_paraphrase" else bool(cross_groups) if pair_type == "positive_cross_language" else bool(hard_groups) if pair_type == "hard_negative" else len(easy_groups) >= 2 and len(groups) >= 2
            while available and emitted < desired and attempts < limit:
                attempts += 1
                if pair_type == "positive_paraphrase":
                    members, manual = choose(manual_groups)
                    first, second = choose(manual), choose(members)
                    if first == second:
                        continue
                elif pair_type == "positive_cross_language":
                    languages = choose(cross_groups)
                    lang_a, lang_b = rng.choice(len(languages), size=2, replace=False)
                    first, second = choose(languages[int(lang_a)]), choose(languages[int(lang_b)])
                elif pair_type == "hard_negative":
                    members = choose(hard_groups)
                    member_a, member_b = rng.choice(len(members), size=2, replace=False)
                    first, second = members[int(member_a)], members[int(member_b)]
                    if not hard_negative_eligible(rows[int(first)], rows[int(second)]):
                        continue
                else:
                    cat_a, cat_b = rng.choice(len(easy_groups), size=2, replace=False)
                    first, second = choose(easy_groups[int(cat_a)]), choose(easy_groups[int(cat_b)])
                    if rows[first]["original_id"] == rows[second]["original_id"]:
                        continue
                a, b = rows[int(first)], rows[int(second)]
                key = tuple(sorted((a["id"], b["id"])))
                if key in seen:
                    continue
                seen.add(key)
                output.append(_pair(a, b, split, pair_type))
                emitted += 1
            stats["sampled_candidate_counts"][pair_type] = emitted
            stats["attempts"][pair_type] = attempts
            if emitted < max_pairs_per_type:
                message = (
                    f"{split}/{pair_type}: emitted {emitted}/{max_pairs_per_type} requested pairs "
                    f"after {attempts}/{limit} bounded attempts; eligible candidates unavailable/exhausted "
                    "or sampling budget reached."
                )
                stats["warnings"].append(message)
                warnings.warn(message, UserWarning, stacklevel=2)
        all_pairs.extend(sorted(output, key=lambda row: row["pair_id"]))
        summary["splits"][split] = stats
    result = pd.DataFrame(all_pairs, columns=PAIR_COLUMNS)
    summary["total_pairs"] = len(result)
    return result, summary


def write_pairs(pairs: pd.DataFrame, summary: dict, output_dir: Path, overwrite: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    relative = [Path("pairs.csv"), *(Path(split) / "pairs.csv" for split in SPLITS), Path("summary.json")]
    if not overwrite and any((output_dir / name).exists() for name in relative):
        raise PairValidationError(f"Pair output already exists in {output_dir}; pass --overwrite or use another --output-dir")
    with tempfile.TemporaryDirectory(prefix=".pairs-", dir=output_dir) as temporary:
        stage = Path(temporary)
        pairs.to_csv(stage / "pairs.csv", index=False, encoding="utf-8")
        for split in SPLITS:
            (stage / split).mkdir()
            pairs.loc[pairs["split"].eq(split)].to_csv(stage / split / "pairs.csv", index=False, encoding="utf-8")
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name in relative:
            (output_dir / name).parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, output_dir / name)
    return [output_dir / name for name in relative]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="ml_training/data/multilingual")
    parser.add_argument("--output-dir", default="ml_training/data/pairs")
    parser.add_argument("--paraphrases", help="Manual paraphrase CSV with schema columns plus split")
    parser.add_argument("--max-pairs-per-type", type=int, default=1000, help="Cap for EACH pair type in EACH split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        input_dir, output_dir = resolve_path(args.input_dir), resolve_path(args.output_dir)
        input_paths = {(input_dir / f"{split}.csv").resolve() for split in SPLITS}
        if args.paraphrases:
            input_paths.add(resolve_path(args.paraphrases).resolve())
        output_paths = {(output_dir / "pairs.csv").resolve(), (output_dir / "summary.json").resolve()}
        output_paths.update((output_dir / split / "pairs.csv").resolve() for split in SPLITS)
        if input_paths & output_paths:
            raise PairValidationError("Pair output paths must be separate from input CSV files")
        splits = load_splits(input_dir, resolve_path(args.paraphrases) if args.paraphrases else None)
        pairs, summary = generate_pairs(splits, args.max_pairs_per_type, args.seed)
        files = write_pairs(pairs, summary, output_dir, args.overwrite)
        print("Labels are candidate suggestions; review them in the annotation app.")
        print("split  positive_paraphrase  positive_cross_language  hard_negative  easy_negative")
        for split, stats in summary["splits"].items():
            counts = stats["sampled_candidate_counts"]
            print(f"{split:<5}  " + "  ".join(str(counts[name]) for name in PAIR_TYPES))
        print(f"Leakage check: PASS; wrote {len(pairs)} pairs in {len(files)} files to {output_dir}")
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
