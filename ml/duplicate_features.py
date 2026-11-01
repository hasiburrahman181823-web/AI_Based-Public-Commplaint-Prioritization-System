"""Symmetric, label-free duplicate features shared by research and inference.

No feature uses pair_type, a duplicate label, original_id, or record identity.
The C0 control keeps category/language covariates identical across P1/P2 so
P2-minus-P1 isolates time features and P1-minus-C0 isolates location features.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ml.location import LocationGazetteer, location_pair_features
from ml.normalize import clean_text, normalize_banglish
from ml.time_features import extract_time_features, time_pair_features

LANGUAGES = ("bn", "en", "banglish")
LANGUAGE_PAIRS = ("bn-bn", "bn-en", "bn-banglish", "en-en", "en-banglish", "banglish-banglish")
BASE_FEATURES = ["cosine", "category_match", "category_missing", *["lang_" + pair.replace("-", "_") for pair in LANGUAGE_PAIRS]]
LOCATION_FEATURES = ["loc_match", "loc_distance_flag", "loc_missing", "loc_conflict"]
TIME_FEATURES = ["time_gap", "time_missing", "duration_gap", "duration_missing", "interval_overlap", "interval_missing"]
FEATURE_SETS = {
    "B1": ["cosine"], "B2": ["cosine"], "C0": BASE_FEATURES,
    "P1": [*BASE_FEATURES, *LOCATION_FEATURES],
    "P2": [*BASE_FEATURES, *LOCATION_FEATURES, *TIME_FEATURES],
}


def preprocess_complaint(record: dict, variants_path: str | Path | None = None) -> dict:
    """Copy and validate a complaint; preserve raw wording for evidence rules.

    ``lang`` is explicit: romanized Bangla is not guessed to be English. Missing
    category/location/timestamp remain unknown. No wall-clock time is inferred.
    """
    if not isinstance(record, dict):
        raise ValueError("A complaint must be a dict with text and lang")
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        raise ValueError("A complaint needs nonempty text")
    language = record.get("lang")
    if language not in LANGUAGES:
        raise ValueError("A complaint needs explicit lang=bn, en or banglish")
    result = dict(record)
    for name in ("id", "original_id", "category", "location", "timestamp"):
        value = record.get(name, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"Complaint {name} must be a string when supplied")
        result[name] = value.strip()
    raw = record.get("raw_text", record["text"])
    if not isinstance(raw, str):
        raise ValueError("raw_text must be a string")
    result["raw_text"] = clean_text(raw)
    result["text"] = normalize_banglish(record["text"], variants_path) if language == "banglish" else clean_text(record["text"])
    if not result["text"]:
        raise ValueError("Complaint text is empty after cleaning")
    # Parse timestamps even for feature subsets that do not consume time; the
    # same input record must not have different validity across ablations.
    extract_time_features(result["raw_text"], result["timestamp"])
    return result


def pair_feature_dict(
    a: dict, b: dict, cosine: float, gazetteer: LocationGazetteer,
    category_source: str = "provided",
) -> dict[str, float]:
    if category_source not in ("provided", "none"):
        raise ValueError("category_source must be provided or none")
    if not math.isfinite(float(cosine)) or not -1.00001 <= float(cosine) <= 1.00001:
        raise ValueError("cosine must be finite and between -1 and 1")
    if a.get("lang") not in LANGUAGES or b.get("lang") not in LANGUAGES:
        raise ValueError("Both complaints need lang=bn, en or banglish")
    left, right = sorted((a["lang"], b["lang"]), key=LANGUAGES.index)
    language = f"{left}-{right}"
    ca, cb = a.get("category", "").strip().casefold(), b.get("category", "").strip().casefold()
    known_categories = category_source == "provided" and bool(ca and cb)
    features = {
        "cosine": float(np.clip(cosine, -1, 1)),
        "category_match": float(known_categories and ca == cb),
        "category_missing": float(not known_categories),
        **{"lang_" + pair.replace("-", "_"): float(language == pair) for pair in LANGUAGE_PAIRS},
    }
    features.update(location_pair_features(
        gazetteer.extract(a.get("raw_text", a["text"]), a.get("location", "")),
        gazetteer.extract(b.get("raw_text", b["text"]), b.get("location", "")),
    ))
    features.update(time_pair_features(
        extract_time_features(a.get("raw_text", a["text"]), a.get("timestamp", "")),
        extract_time_features(b.get("raw_text", b["text"]), b.get("timestamp", "")),
    ))
    return {key: float(value) for key, value in features.items()}


def feature_matrix(
    records_a: list[dict], records_b: list[dict], cosines,
    feature_names: list[str], gazetteer: LocationGazetteer,
    category_source: str = "provided",
) -> np.ndarray:
    """Build ordered symmetric features, with explicit missingness indicators."""
    scores = np.asarray(cosines, dtype=float)
    if scores.ndim != 1 or len(records_a) != len(records_b) or len(records_a) != len(scores):
        raise ValueError("Complaint sides and cosine scores must have the same length")
    allowed = set(FEATURE_SETS["P2"])
    if not feature_names or len(set(feature_names)) != len(feature_names) or set(feature_names) - allowed:
        raise ValueError("feature_names must be unique supported duplicate feature names")
    rows = [pair_feature_dict(a, b, score, gazetteer, category_source) for a, b, score in zip(records_a, records_b, scores)]
    matrix = np.asarray([[row[name] for name in feature_names] for row in rows], dtype=np.float64)
    matrix = matrix.reshape(len(rows), len(feature_names))
    if not np.isfinite(matrix).all():
        raise ValueError("Duplicate features contain non-finite values")
    return matrix
