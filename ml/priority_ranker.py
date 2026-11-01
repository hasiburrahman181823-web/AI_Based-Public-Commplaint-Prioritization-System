"""Optional pairwise logistic ranking over transparent priority-rule features.

The model learns a linear utility, not a probability or a 1–5 priority label.
Experts must supply query/queue context and final relevance judgments. Missing
or insufficient training judgments produce an explicit rule-based fallback.
"""

from __future__ import annotations

from bisect import bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score

JUDGMENT_COLUMNS = (
    "id", "original_id", "query_id", "split", "text", "lang", "category",
    "severity", "location", "timestamp", "duplicate_group_size",
    "expert_priority", "expert_rank",
)


def load_expert_judgments(path: str | Path) -> pd.DataFrame:
    """Read adjudicated expert judgments, retaining query and incident isolation.

    ``expert_priority`` is an integer 1–5, higher meaning more urgent. Optional
    ``expert_rank`` is positive, with 1 meaning most urgent. Supply either every
    rank or no rank within a query. Two experts' separate labels must first be
    adjudicated into this one-item-per-row format; this function never votes.
    """
    path = Path(path)
    required = set(JUDGMENT_COLUMNS) - {"expert_rank"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if len(set(header)) != len(header) or not required <= set(header):
            raise ValueError(f"{path}: need unique headers including {sorted(required)}")
        records = []
        for number, values in enumerate(reader, 2):
            if len(values) != len(header):
                raise ValueError(f"{path}: wrong CSV width at record {number}")
            records.append([value.strip() for value in values])
    frame = pd.DataFrame(records, columns=header, dtype=str)
    if "expert_rank" not in frame:
        frame["expert_rank"] = ""
    for field in ("id", "original_id", "query_id", "text", "category", "severity"):
        if frame[field].eq("").any():
            raise ValueError(f"Expert judgment '{field}' values must be nonempty")
    if frame["id"].duplicated().any():
        raise ValueError("Expert judgments need one final row per globally unique complaint id")
    if not frame["split"].isin(("train", "val", "test")).all():
        raise ValueError("Expert split must be train, val or test")
    if not frame["lang"].isin(("bn", "en", "banglish")).all():
        raise ValueError("Expert lang must be bn, en or banglish")
    if not frame["expert_priority"].isin(tuple(str(i) for i in range(1, 6))).all():
        raise ValueError("expert_priority must be an independently reviewed integer from 1 to 5")
    if not frame["duplicate_group_size"].str.fullmatch(r"[1-9]\d*").all():
        raise ValueError("duplicate_group_size must be a positive integer report count, not people affected")
    frame["expert_priority"] = frame["expert_priority"].astype(int)
    frame["duplicate_group_size"] = frame["duplicate_group_size"].map(int)
    for identity in ("original_id", "query_id"):
        if not frame.empty and frame.groupby(identity)["split"].nunique().gt(1).any():
            raise ValueError(f"Split leakage: {identity} occurs in multiple partitions")
    ranks = []
    for value in frame["expert_rank"]:
        if value == "":
            ranks.append(None)
            continue
        try:
            rank = float(value)
        except ValueError as exc:
            raise ValueError("expert_rank must be empty or a positive finite rank, with 1 most urgent") from exc
        if not math.isfinite(rank) or rank <= 0:
            raise ValueError("expert_rank must be empty or a positive finite rank, with 1 most urgent")
        ranks.append(rank)
    frame["expert_rank"] = pd.Series(ranks, index=frame.index, dtype="float64")
    warnings = []
    for query, group in frame.groupby("query_id", sort=True):
        count = int(group["expert_rank"].notna().sum())
        if count not in (0, len(group)):
            raise ValueError(f"Query {query!r} mixes present and missing expert_rank values")
        if count and group["expert_priority"].nunique() > 1:
            relation = spearmanr(-group["expert_priority"].to_numpy(), group["expert_rank"].to_numpy()).statistic
            if not np.isfinite(relation) or relation < 0:
                warnings.append(f"Query {query!r}: relevance and explicit rank disagree or lack rank variation; NDCG and Spearman use their stated separate judgments.")
    frame.attrs["warnings"] = warnings
    return frame


def make_pairwise_training_data(
    frame: pd.DataFrame, feature_values, max_pairs_per_query: int = 1000, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample unequal-relevance within-query comparisons without a product table.

    The cap counts unordered comparisons before mirrored training orientations.
    Grade bins define at most ten Cartesian blocks for a 1–5 scale. Sampled
    offsets unrank directly into those blocks, so a large query does not create
    all O(n²) candidate pairs. Tied relevance is excluded from training.
    """
    values = np.asarray(feature_values, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(frame) or not np.isfinite(values).all():
        raise ValueError("feature_values must be a finite matrix aligned with judgment rows")
    if isinstance(max_pairs_per_query, bool) or not isinstance(max_pairs_per_query, int) or max_pairs_per_query < 1:
        raise ValueError("max_pairs_per_query must be a positive integer")
    for name in ("id", "query_id", "expert_priority"):
        if name not in frame:
            raise ValueError(f"Missing pairwise judgment field {name}")
    if not frame["expert_priority"].isin((1, 2, 3, 4, 5)).all():
        raise ValueError("Pairwise relevance must be in the expert 1–5 scale")
    rng = random.Random(seed)
    differences, targets = [], []
    report = {"unordered_comparisons": 0, "mirrored_training_rows": 0, "queries": {},
              "sampling": "uniform eligible Cartesian-block offsets without product materialization"}
    for query, positions in frame.groupby("query_id", sort=True).indices.items():
        positions = sorted(map(int, positions), key=lambda position: str(frame.iloc[position]["id"]))
        bins: dict[int, list[int]] = {}
        for position in positions:
            bins.setdefault(int(frame.iloc[position]["expert_priority"]), []).append(position)
        grades = sorted(bins)
        blocks, endpoints, total = [], [], 0
        for lower_index, lower in enumerate(grades):
            for higher in grades[lower_index + 1:]:
                size = len(bins[lower]) * len(bins[higher])
                blocks.append((bins[lower], bins[higher], total))
                total += size
                endpoints.append(total)
        selected = sorted(rng.sample(range(total), min(max_pairs_per_query, total)))
        for offset in selected:
            low, high, start = blocks[bisect_right(endpoints, offset)]
            low_offset, high_offset = divmod(offset - start, len(high))
            difference = values[high[high_offset]] - values[low[low_offset]]
            differences.extend((difference, -difference))
            targets.extend((1, 0))
        report["queries"][str(query)] = {"eligible_unordered_comparisons": total, "sampled_comparisons": len(selected)}
        report["unordered_comparisons"] += len(selected)
    matrix = np.asarray(differences, dtype=float).reshape(len(differences), values.shape[1])
    report["mirrored_training_rows"] = len(targets)
    return matrix, np.asarray(targets, dtype=np.int8), report


def fit_priority_ranker(
    train_frame: pd.DataFrame, feature_values, feature_names: Sequence[str],
    rules_config_sha256: str, *, seed: int = 42, min_train_queries: int = 3,
    min_comparisons: int = 20, min_train_items: int = 10,
    max_pairs_per_query: int = 1000,
) -> dict:
    """Fit linear utility on training expert comparisons or save an explicit fallback."""
    names = list(feature_names)
    if not names or len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature_names must contain distinct nonempty strings")
    values = np.asarray(feature_values, dtype=float)
    if values.shape != (len(train_frame), len(names)) or not np.isfinite(values).all():
        raise ValueError("Training features must be finite and match row/feature counts")
    if "split" not in train_frame or not train_frame["split"].eq("train").all():
        raise ValueError("fit_priority_ranker accepts training split judgments only")
    for name, value in (("min_train_queries", min_train_queries), ("min_comparisons", min_comparisons), ("min_train_items", min_train_items)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(rules_config_sha256, str) or len(rules_config_sha256) != 64 or any(c not in "0123456789abcdef" for c in rules_config_sha256):
        raise ValueError("rules_config_sha256 must be the SHA-256 of the frozen rules file")
    differences, targets, sampling = make_pairwise_training_data(train_frame, values, max_pairs_per_query, seed)
    n_queries = int(train_frame["query_id"].nunique())
    informative_queries = sum(row["sampled_comparisons"] > 0 for row in sampling["queries"].values())
    reasons = []
    if len(train_frame) < min_train_items:
        reasons.append(f"fewer than {min_train_items} expert-labeled training items")
    if informative_queries < min_train_queries:
        reasons.append(f"fewer than {min_train_queries} independent informative training queries")
    if sampling["unordered_comparisons"] < min_comparisons:
        reasons.append(f"fewer than {min_comparisons} unequal-relevance training comparisons")
    if not len(differences) or not np.any(np.abs(differences) > 1e-12):
        reasons.append("no informative feature differences between expert-priority levels")
    artifact = {
        "kind": "banglapsg_priority_ranker", "schema_version": 1,
        "status": "fallback_rules" if reasons else "trained",
        "reason": "; ".join(reasons) if reasons else None, "estimator": None,
        "feature_names": names, "rules_config_sha256": rules_config_sha256,
        "seed": seed, "n_training_items": len(train_frame), "n_training_queries": n_queries,
        "n_informative_training_queries": informative_queries,
        "sampling": sampling, "training_target": "within-query differences of expert_priority; ties excluded",
        "minima": {"training_items": min_train_items, "training_queries": min_train_queries, "comparisons": min_comparisons},
        "utility_semantics": "linear relative ranking utility; not a calibrated probability or 1–5 priority rating",
    }
    if not reasons:
        model = LogisticRegression(fit_intercept=False, C=1.0, max_iter=3000, random_state=seed)
        model.fit(differences, targets)
        artifact["estimator"] = model
    return artifact


def _preprocessing_identity(value) -> dict[str, str]:
    required = {"normalization_dictionary_sha256", "normalizer_code_sha256"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("Priority preprocessing fingerprints must identify the dictionary and normalizer code")
    if any(not isinstance(key, str) or not isinstance(digest, str) or len(digest) != 64
           or any(char not in "0123456789abcdef" for char in digest) for key, digest in value.items()):
        raise ValueError("Priority preprocessing fingerprints must contain valid SHA-256 values")
    return value


def ranker_score(artifact: dict, rule_result: dict) -> dict:
    """Return rule fallback or additive linear utility in the saved feature order."""
    if not isinstance(artifact, dict) or artifact.get("kind") != "banglapsg_priority_ranker" or artifact.get("schema_version") != 1:
        raise ValueError("Not a supported priority ranker artifact")
    if artifact.get("rules_config_sha256") != rule_result.get("config_sha256"):
        raise ValueError("Priority rules configuration changed since ranker training; use the matching snapshot")
    if "rules_preprocessing_fingerprints" in artifact:
        expected = _preprocessing_identity(artifact["rules_preprocessing_fingerprints"])
        observed = _preprocessing_identity(rule_result.get("preprocessing_fingerprints"))
        if expected != observed:
            raise ValueError("Priority preprocessing changed since ranker training; restore the dictionary/code snapshots")
    if artifact.get("status") == "fallback_rules":
        score = float(rule_result["priority_score"])
        if not math.isfinite(score):
            raise ValueError("Rule fallback score must be finite")
        return {"ranking_score": score, "method": "rules", "fallback_reason": artifact.get("reason"),
                "contributions": rule_result.get("rule_breakdown", []),
                "score_semantics": "configured additive rule points"}
    if artifact.get("status") != "trained" or artifact.get("estimator") is None:
        raise ValueError("Priority ranker has no trained estimator or valid rule fallback")
    names = artifact["feature_names"]
    try:
        values = np.asarray([rule_result["feature_values"][name] for name in names], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Rule features do not match the saved ranker feature order") from exc
    estimator = artifact["estimator"]
    coefficients = np.asarray(estimator.coef_, dtype=float)
    if values.shape != (len(names),) or coefficients.shape != (1, len(names)) or not np.isfinite(values).all() or not np.isfinite(coefficients).all():
        raise ValueError("Priority ranking features/coefficients are not finite and aligned")
    bias = float(np.asarray(estimator.intercept_).reshape(-1)[0]) if np.asarray(estimator.intercept_).size else 0.0
    contributions = [{"name": name, "feature_value": float(value), "coefficient": float(coefficient),
                      "contribution": float(value * coefficient)} for name, value, coefficient in zip(names, values, coefficients[0])]
    score = float(np.dot(values, coefficients[0]) + bias)
    return {"ranking_score": score, "method": "pairwise_logistic", "bias": bias,
            "contributions": sorted(contributions, key=lambda row: (-abs(row["contribution"]), row["name"])),
            "score_semantics": artifact["utility_semantics"]}


def load_priority_ranker(path: str | Path, rules_config_sha256: str, preprocessing_fingerprints: dict | None = None) -> dict:
    """Load only a trusted self-created joblib artifact and verify rules identity."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("kind") != "banglapsg_priority_ranker" or artifact.get("schema_version") != 1:
        raise ValueError("Not a supported priority ranker artifact")
    if artifact.get("rules_config_sha256") != rules_config_sha256:
        raise ValueError("Priority ranker rules fingerprint does not match the configured rules")
    if "rules_preprocessing_fingerprints" in artifact:
        expected = _preprocessing_identity(artifact["rules_preprocessing_fingerprints"])
        if preprocessing_fingerprints is not None and expected != _preprocessing_identity(preprocessing_fingerprints):
            raise ValueError("Priority ranker preprocessing fingerprints do not match the current dictionary/code")
    if artifact.get("status") not in ("trained", "fallback_rules"):
        raise ValueError("Unsupported priority ranker training status")
    return artifact


def ranking_metrics(expert_priority, scores, expert_ranks=None, k: int = 5) -> dict:
    """Tie-aware exponential-gain NDCG and rank-direction-safe Spearman.

    A constant relevance query has no discriminatory NDCG evidence, represented
    by None rather than a manufactured perfect score. A constant predicted rank
    or gold rank similarly has undefined Spearman correlation.
    """
    relevance, predicted = np.asarray(expert_priority, dtype=float), np.asarray(scores, dtype=float)
    if relevance.ndim != 1 or predicted.shape != relevance.shape or not np.isfinite(relevance).all() or not np.isfinite(predicted).all():
        raise ValueError("Expert relevance and predicted scores must be aligned finite vectors")
    if not np.isin(relevance, (1, 2, 3, 4, 5)).all():
        raise ValueError("expert_priority relevance must use the 1–5 scale")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    n = len(relevance)
    result = {"n_items": n, "k_requested": k, "k_effective": min(k, n), "ndcg_at_5": None,
              "spearman": None, "ndcg_undefined_reason": None, "spearman_undefined_reason": None,
              "ndcg_gain": "2**expert_priority - 1; equal predicted scores use tie-averaged DCG"}
    if n < 2:
        result["ndcg_undefined_reason"] = result["spearman_undefined_reason"] = "At least two judged items are required in a query"
        return result
    if np.unique(relevance).size < 2:
        result["ndcg_undefined_reason"] = "All expert relevance grades are equal; query is non-discriminatory"
    else:
        gains = np.exp2(relevance) - 1
        result["ndcg_at_5"] = float(ndcg_score(gains[None, :], predicted[None, :], k=min(k, n), ignore_ties=False))
    if expert_ranks is None:
        ranks = rankdata(-relevance, method="average")
    else:
        ranks = np.asarray(expert_ranks, dtype=float)
        if ranks.shape != relevance.shape or not np.isfinite(ranks).all() or (ranks <= 0).any():
            raise ValueError("Expert ranks must be aligned positive finite ranks, 1 most urgent")
    predicted_ranks = rankdata(-predicted, method="average")
    if np.unique(ranks).size < 2 or np.unique(predicted_ranks).size < 2:
        result["spearman_undefined_reason"] = "Constant expert or predicted rank makes Spearman undefined"
    else:
        value = float(spearmanr(ranks, predicted_ranks).statistic)
        if math.isfinite(value):
            result["spearman"] = value
        else:
            result["spearman_undefined_reason"] = "Spearman correlation is non-finite for this query"
    return result


def evaluate_rankings(frame: pd.DataFrame, score_sets: dict[str, Sequence[float]], k: int = 5) -> dict:
    """Compare every scorer on exactly the same ordered query-item cohort."""
    if len(frame) and frame["split"].nunique() > 1:
        raise ValueError("Evaluate one held-out partition at a time")
    fingerprints = [
        {"id": row["id"], "original_id": row["original_id"], "query_id": row["query_id"],
         "expert_priority": int(row["expert_priority"]),
         "expert_rank": None if pd.isna(row["expert_rank"]) else float(row["expert_rank"])}
        for row in frame.to_dict("records")
    ]
    cohort_id = hashlib.sha256(json.dumps(fingerprints, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    reports = {}
    for name, scores in score_sets.items():
        values = np.asarray(scores, dtype=float)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError(f"{name}: scores must be finite and aligned with all held-out items")
        queries = []
        for query, positions in frame.groupby("query_id", sort=True).indices.items():
            group = frame.iloc[positions]
            explicit = group["expert_rank"].to_numpy() if group["expert_rank"].notna().all() else None
            metrics = ranking_metrics(group["expert_priority"].to_numpy(), values[positions], explicit, k)
            queries.append({"query_id": str(query), **metrics})
        ndcg = [row["ndcg_at_5"] for row in queries if row["ndcg_at_5"] is not None]
        correlations = [row["spearman"] for row in queries if row["spearman"] is not None]
        reports[name] = {
            "status": "ok" if len(frame) else "skipped_no_expert_items", "cohort_fingerprint": cohort_id,
            "n_items": len(frame), "n_queries": len(queries), "ndcg_eligible_queries": len(ndcg),
            "spearman_eligible_queries": len(correlations), "ndcg_at_5": float(np.mean(ndcg)) if ndcg else None,
            "spearman": float(np.mean(correlations)) if correlations else None, "queries": queries,
            "aggregation": "unweighted macro mean over queries with defined metric; all query supports retained",
        }
    return reports
