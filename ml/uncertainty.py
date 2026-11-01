"""Empirical validation selection for a joint category/severity review policy.

Confidence is the smaller of the two calibrated maximum class probabilities.
It is a conservative ranking statistic, not a calibrated probability that both
targets are correct. Validation error constraints are empirical, not guarantees
on a future population or independently estimated risk bounds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CONFIDENCE_DEFINITION = "min(category_max_probability,severity_max_probability)"


def _arrays(confidences, errors):
    try:
        confidence = np.asarray(list(confidences), dtype=float)
        raw_error = np.asarray(list(errors))
        error = raw_error.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Confidence and errors must be aligned numeric sequences") from exc
    if confidence.ndim != 1 or error.shape != confidence.shape or not len(confidence):
        raise ValueError("Validation needs nonempty aligned one-dimensional confidence/error sequences")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Confidence must be finite and between 0 and 1")
    if not np.isfinite(error).all() or not np.isin(error, (0, 1)).all():
        raise ValueError("Errors must contain only 0 or 1")
    return confidence, error.astype(np.int8)


def coverage_error_curve(confidences, errors) -> list[dict]:
    """Return every distinct confidence operating point in O(n log n).

    Accepted predictions satisfy confidence >= threshold. Equal-confidence
    observations move together; the function never chooses a row-level tie.
    Reject-all and threshold zero are included. A reject-all threshold can be
    just above one so confidence-one observations can also be sent to review.
    """
    confidence, error = _arrays(confidences, errors)
    order = np.argsort(-confidence, kind="stable")
    values, ordered_errors = confidence[order], error[order]
    boundaries = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values) - 1]
    cumulative = np.cumsum(ordered_errors, dtype=np.int64)
    rows = [{"threshold": float(np.nextafter(float(values[0]), math.inf)), "coverage": 0.0,
             "error_rate": None, "n_accepted": 0, "n_errors": 0}]
    for index in boundaries:
        count, mistakes = int(index + 1), int(cumulative[index])
        rows.append({"threshold": float(values[index]), "coverage": count / len(values),
                     "error_rate": mistakes / count, "n_accepted": count, "n_errors": mistakes})
    if rows[-1]["threshold"] != 0:
        rows.append({**rows[-1], "threshold": 0.0})
    return rows


def select_review_threshold(
    confidences, errors, max_error_rate: float = 0.1, min_accepted: int = 1,
) -> dict:
    """Maximize accepted validation coverage under an observed-error constraint."""
    if isinstance(max_error_rate, bool) or not isinstance(max_error_rate, (int, float)) or not math.isfinite(max_error_rate) or not 0 <= max_error_rate <= 1:
        raise ValueError("max_error_rate must be finite and between 0 and 1")
    if isinstance(min_accepted, bool) or not isinstance(min_accepted, int) or min_accepted < 1:
        raise ValueError("min_accepted must be a positive integer")
    curve = coverage_error_curve(confidences, errors)
    feasible = [row for row in curve if row["n_accepted"] >= min_accepted and row["error_rate"] <= max_error_rate]
    selected = max(feasible, key=lambda row: (row["coverage"], row["threshold"])) if feasible else {
        **curve[0], "threshold": float(np.nextafter(1.0, math.inf)),
    }
    return {
        **selected, "status": "selected" if feasible else "reject_all",
        "max_error_rate": float(max_error_rate), "min_accepted": min_accepted,
        "n_validation": max(row["n_accepted"] for row in curve),
        "selection_policy": "maximum validation coverage satisfying empirical accepted-error constraint; higher threshold breaks equal-coverage ties",
        "decision_rule": "accept iff confidence >= threshold; otherwise needs_human_review",
        "confidence_definition": CONFIDENCE_DEFINITION,
        "risk_interpretation": "empirical validation error, not a guarantee or calibrated joint correctness probability",
    }


def load_review_policy(path: str | Path, classifier_fingerprints: dict[str, str] | None = None,
                       classifier_resource_fingerprints: dict[str, str] | None = None) -> dict:
    """Validate a saved policy and optionally bind it to the exact two models.

    Call with both classifier SHA-256 hashes at production startup. A policy
    tuned for other artifacts must not silently control the active classifiers.
    """
    try:
        policy = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load review policy {path}: {exc}") from exc
    if not isinstance(policy, dict) or policy.get("kind") != "priority_review_policy" or policy.get("schema_version") != 1:
        raise ValueError("Not a supported priority review policy")
    threshold = policy.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Review threshold must be a nonnegative finite number")
    if policy.get("status") not in ("selected", "reject_all"):
        raise ValueError("Review policy status must be selected or reject_all")
    if policy["status"] == "selected" and threshold > 1:
        raise ValueError("A selected non-reject-all threshold cannot exceed 1")
    if policy["status"] == "reject_all" and threshold <= 1:
        raise ValueError("A reject_all policy needs a threshold above 1 so future confidence values are also rejected")
    if policy.get("confidence_definition") != CONFIDENCE_DEFINITION:
        raise ValueError("Review policy uses an incompatible confidence definition")
    fingerprints = policy.get("classifier_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {"category", "severity"} or any(
        not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in fingerprints.values()
    ):
        raise ValueError("Policy must record both classifier SHA-256 fingerprints")
    if classifier_fingerprints is not None and fingerprints != classifier_fingerprints:
        raise ValueError("Review policy classifier fingerprints do not match the active classifiers")
    if classifier_resource_fingerprints is not None and policy.get("classifier_resource_fingerprints") != classifier_resource_fingerprints:
        raise ValueError("Review policy preprocessing/encoder resources do not match; export validation predictions and retune the policy")
    return policy
