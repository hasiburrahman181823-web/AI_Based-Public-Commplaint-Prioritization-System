"""Descriptive priority-score, review, and observed-error disparities.

The report is an audit of the supplied evaluation rows, not a causal fairness
test. Area/language score differences may reflect differences in complaint
severity, missing metadata, composition, or annotation. Conditional summaries
help inspect composition but do not establish a causal effect of language/area.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ml_training.scripts.train_classifiers import _publish, resolve_path

UNKNOWN = "__unknown__"


def _strings(frame: pd.DataFrame, field: str) -> pd.Series:
    return frame[field].fillna("").astype(str).str.strip() if field in frame else pd.Series("", index=frame.index, dtype=str)


def _numeric(series: pd.Series, name: str, allow_missing: bool = True) -> pd.Series:
    raw = series.fillna("").astype(str).str.strip()
    missing = raw.eq("")
    values = pd.to_numeric(raw.mask(missing), errors="coerce").astype(float)
    if ((~missing) & (~np.isfinite(values))).any() or (not allow_missing and missing.any()):
        raise ValueError(f"{name} must contain finite numeric values" + (" or empty values" if allow_missing else " on every row"))
    return values


def _binary(series: pd.Series, name: str) -> pd.Series:
    raw = series.fillna("").astype(str).str.strip().str.lower()
    if not raw.isin(("", "0", "1", "true", "false")).all():
        raise ValueError(f"{name} must use 0/1, true/false, or empty values")
    return raw.map({"0": 0.0, "1": 1.0, "true": 1.0, "false": 0.0, "": float("nan")}).astype(float)


def _stats(frame: pd.DataFrame, minimum: int) -> dict:
    values = frame["_score"]
    error = frame["_error"].dropna()
    review = frame["_review"].dropna()
    priority_error = frame["_priority_error"].dropna()
    under = frame["_under_priority"].dropna()
    return {
        "n": len(frame), "mean_score": float(values.mean()), "median_score": float(values.median()),
        "std_score": float(values.std(ddof=0)), "q25_score": float(values.quantile(0.25)),
        "q75_score": float(values.quantile(0.75)), "min_score": float(values.min()), "max_score": float(values.max()),
        "eligible_distribution_comparison": len(frame) >= minimum,
        "n_labeled_error": len(error), "error_rate": float(error.mean()) if len(error) else None,
        "eligible_error_comparison": len(error) >= minimum,
        "n_review_observed": len(review), "review_rate": float(review.mean()) if len(review) else None,
        "eligible_review_comparison": len(review) >= minimum,
        "n_priority_labeled": len(priority_error), "priority_mae": float(priority_error.mean()) if len(priority_error) else None,
        "under_prioritization_rate": float(under.mean()) if len(under) else None,
        "eligible_priority_error_comparison": len(priority_error) >= minimum,
    }


def _disparity(rows: list[dict], minimum: int) -> dict:
    eligible = [row for row in rows if row["eligible_distribution_comparison"] and row["group"] != UNKNOWN]
    result = {"status": "ok" if len(eligible) >= 2 else "insufficient_groups", "min_group_size": minimum,
              "eligible_groups": len(eligible), "mean_score_gap": None, "mean_score_ratio": None,
              "highest_mean_group": None, "lowest_mean_group": None, "error_rate_gap": None,
              "review_rate_gap": None, "priority_mae_gap": None, "under_prioritization_gap": None,
              "pairwise": []}
    if len(eligible) >= 2:
        highest = max(eligible, key=lambda row: (row["mean_score"], row["group"]))
        lowest = min(eligible, key=lambda row: (row["mean_score"], row["group"]))
        result.update(mean_score_gap=highest["mean_score"] - lowest["mean_score"],
                      mean_score_ratio=highest["mean_score"] / lowest["mean_score"] if lowest["mean_score"] > 0 else None,
                      highest_mean_group=highest["group"], lowest_mean_group=lowest["group"])
    for metric, flag, key in (
        ("error_rate", "eligible_error_comparison", "error_rate_gap"),
        ("review_rate", "eligible_review_comparison", "review_rate_gap"),
        ("priority_mae", "eligible_priority_error_comparison", "priority_mae_gap"),
        ("under_prioritization_rate", "eligible_priority_error_comparison", "under_prioritization_gap"),
    ):
        supported = [row[metric] for row in eligible if row[flag] and row[metric] is not None]
        if len(supported) >= 2:
            result[key] = max(supported) - min(supported)
    # Bound large-area outputs. Extremal summary gaps above stay linear in the
    # number of groups; creating every pair for thousands of areas is avoided.
    if len(eligible) <= 50:
        for left, right in itertools.combinations(sorted(eligible, key=lambda row: row["group"]), 2):
            result["pairwise"].append({"group_a": left["group"], "group_b": right["group"],
                "mean_score_difference_a_minus_b": left["mean_score"] - right["mean_score"],
                "error_rate_difference_a_minus_b": left["error_rate"] - right["error_rate"]
                    if left["eligible_error_comparison"] and right["eligible_error_comparison"] else None})
    else:
        result["pairwise_note"] = "More than 50 supported groups; pairwise rows omitted to avoid quadratic output. Extremal gaps are retained."
    return result


def audit_priority_fairness(frame: pd.DataFrame, min_group_size: int = 20) -> dict:
    """Audit score distributions plus errors only where compatible truth exists.

    Optional ``priority_true``/``priority_pred`` are explicitly aligned integer
    judgments on the same 1–5 scale. They are never inferred from an arbitrary
    rule/LTR score. Classification error uses the available true/predicted
    category/severity pairs, or an explicit binary classification_error column.
    """
    if isinstance(min_group_size, bool) or not isinstance(min_group_size, int) or min_group_size < 1:
        raise ValueError("min_group_size must be a positive integer")
    if not {"priority_score", "lang"} <= set(frame.columns):
        raise ValueError("Fairness input requires priority_score and lang")
    if frame.empty:
        raise ValueError("Fairness input is empty")
    data = frame.copy()
    data["_score"] = _numeric(data["priority_score"], "priority_score", allow_missing=False)
    data["_language"] = _strings(data, "lang").str.casefold().replace("", UNKNOWN)
    area, sources = pd.Series("", index=data.index, dtype=str), []
    for column in ("canonical_area_id", "area_id", "location"):
        if column in data:
            available = _strings(data, column).str.casefold().str.replace(r"\s+", " ", regex=True)
            area = area.where(area.ne(""), available)
            sources.append(column)
    data["_area"] = area.replace("", UNKNOWN)
    data["_review"] = _binary(data["review_required"], "review_required") if "review_required" in data else np.nan
    notes = []
    target_counts = {}
    if "classification_error" in data:
        data["_error"] = _binary(data["classification_error"], "classification_error")
        error_definition = "explicit classification_error supplied by caller"
    else:
        has_truth = pd.Series(False, index=data.index)
        mistakes = pd.Series(False, index=data.index)
        for target in ("category", "severity"):
            truth, predicted = _strings(data, target + "_true"), _strings(data, target + "_pred")
            known = truth.ne("") & predicted.ne("")
            target_counts[target] = int(known.sum())
            has_truth |= known
            mistakes |= known & truth.ne(predicted)
        data["_error"] = mistakes.astype(float).where(has_truth)
        error_definition = "incorrect on at least one available category/severity true-predicted pair; rows with no available truth excluded"
        if target_counts.get("category") != target_counts.get("severity"):
            notes.append("Classification error targets have unequal truth coverage; inspect target counts before comparing rates.")
    data["_priority_error"], data["_under_priority"] = np.nan, np.nan
    if ("priority_true" in data) != ("priority_pred" in data):
        raise ValueError("Supply both aligned priority_true and priority_pred, or neither")
    if "priority_true" in data:
        truth = _numeric(data["priority_true"], "priority_true")
        predicted = _numeric(data["priority_pred"], "priority_pred")
        for values in (truth, predicted):
            present = values.dropna()
            if ((present < 1) | (present > 5) | (present % 1 != 0)).any():
                raise ValueError("Priority error needs aligned integer priority judgments on the same 1–5 scale; never compare raw rule/LTR scores to expert judgments")
        known = truth.notna() & predicted.notna()
        data["_priority_error"] = (predicted - truth).abs().where(known)
        data["_under_priority"] = predicted.lt(truth).astype(float).where(known)
    category_column = next((key for key in ("category_true", "category_pred", "category") if key in data and _strings(data, key).ne("").any()), None)
    severity_column = next((key for key in ("severity_true", "severity_pred", "severity") if key in data and _strings(data, key).ne("").any()), None)
    if any(key and key.endswith("_pred") for key in (category_column, severity_column)):
        notes.append("Some conditional strata use predicted category/severity. Conditioning on model outputs can obscure upstream disparities; these are descriptive controls, not verified need or causal adjustment.")
    groups, disparities, conditional, conditional_disparities = [], {}, [], []
    for dimension, field in (("language", "_language"), ("area", "_area")):
        rows = [{"dimension": dimension, "group": str(group), **_stats(subset, min_group_size)}
                for group, subset in data.groupby(field, sort=True, dropna=False)]
        groups.extend(rows)
        disparities[dimension] = _disparity(rows, min_group_size)
    data["_category_context"] = _strings(data, category_column).replace("", UNKNOWN) if category_column else UNKNOWN
    data["_severity_context"] = _strings(data, severity_column).replace("", UNKNOWN) if severity_column else UNKNOWN
    if category_column or severity_column:
        for context, subset in data.groupby(["_category_context", "_severity_context"], sort=True, dropna=False):
            for dimension, field in (("language", "_language"), ("area", "_area")):
                rows = [{"dimension": dimension, "group": str(group), "category_context": str(context[0]),
                         "severity_context": str(context[1]), **_stats(part, min_group_size)}
                        for group, part in subset.groupby(field, sort=True, dropna=False)]
                conditional.extend(rows)
                conditional_disparities.append({"dimension": dimension, "category_context": str(context[0]),
                                               "severity_context": str(context[1]), **_disparity(rows, min_group_size)})
    if not data["_error"].notna().any():
        notes.append("No compatible classification truth was supplied; error-rate fields are null, not zero.")
    if not data["_priority_error"].notna().any():
        notes.append("No aligned 1–5 priority truth/prediction pair was supplied; priority errors are not inferred from priority_score.")
    return {
        "schema_version": 1, "n": len(data), "min_group_size": min_group_size,
        "overall": _stats(data, min_group_size), "groups": groups, "disparities": disparities,
        "conditional_groups": conditional, "conditional_disparities": conditional_disparities,
        "conditioning_columns": {"category": category_column, "severity": severity_column},
        "area_fields_in_fallback_order": sources, "unknown_group": UNKNOWN,
        "classification_error_definition": error_definition, "classification_target_truth_counts": target_counts,
        "priority_error_scale": "explicit aligned integer judgments 1–5 only",
        "interpretation": "descriptive sample disparities, not causal bias findings or a fairness guarantee",
        "warnings": [*notes, "Small groups retain descriptive statistics but are excluded from disparity comparisons; error-rate comparisons also require sufficient labeled support.",
                     "Case/spacing-normalized raw location fallbacks are not a gazetteer; prefer canonical_area_id or area_id for consistent multilingual areas.",
                     "Area/language differences can reflect need, label composition, missingness, annotation or model behavior. Conditional summaries do not remove all confounding."],
    }


def read_input(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if len(set(header)) != len(header) or not {"priority_score", "lang"} <= set(header):
            raise ValueError("Fairness CSV needs unique headers including priority_score,lang")
        records = []
        for number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Invalid fairness CSV field count at record {number}")
            records.append(row)
    return pd.DataFrame(records, columns=header, dtype=str)


def export_fairness_report(input_path: str | Path, output_dir: str | Path,
                           min_group_size: int = 20, overwrite: bool = False) -> dict:
    source, output = Path(input_path).resolve(), Path(output_dir).resolve()
    names = ("fairness.json", "groups.csv", "conditional_groups.csv", "fairness.md")
    if source in {output / name for name in names}:
        raise ValueError("Fairness outputs must not overwrite the input")
    if not overwrite and any((output / name).exists() for name in names):
        raise ValueError("Fairness output exists; choose a fresh directory or --overwrite")
    result = audit_priority_fairness(read_input(source), min_group_size)
    result["input_file"] = str(source)
    result["input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    lines = ["# Descriptive priority fairness audit", "", result["interpretation"], "",
             "| Dimension | Eligible groups | Mean score gap | Observed error-rate gap | Status |",
             "| --- | ---: | ---: | ---: | --- |"]
    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"
    for dimension, report in result["disparities"].items():
        lines.append(f"| {dimension} | {report['eligible_groups']} | {number(report['mean_score_gap'])} | {number(report['error_rate_gap'])} | {report['status']} |")
    lines.extend(["", "Differences are descriptive and may reflect legitimate complaint composition or missing labels. Inspect support, conditional groups and source provenance before interpretation.", ""])
    with tempfile.TemporaryDirectory(prefix=".fairness-", dir=output) as temporary:
        stage = Path(temporary)
        (stage / "fairness.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pd.DataFrame(result["groups"]).to_csv(stage / "groups.csv", index=False, encoding="utf-8")
        pd.DataFrame(result["conditional_groups"], columns=list(result["conditional_groups"][0]) if result["conditional_groups"] else ["dimension", "group", "category_context", "severity_context"]).to_csv(stage / "conditional_groups.csv", index=False, encoding="utf-8")
        (stage / "fairness.md").write_text("\n".join(lines), encoding="utf-8")
        _publish(stage, output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/fairness")
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = export_fairness_report(resolve_path(args.input), resolve_path(args.output_dir), args.min_group_size, args.overwrite)
        print(json.dumps({"n": report["n"], "disparities": report["disparities"], "interpretation": report["interpretation"]}, ensure_ascii=True, allow_nan=False))
        print(f"Saved fairness audit: {resolve_path(args.output_dir) / 'fairness.json'}")
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
