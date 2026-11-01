"""One cached, thread-safe ML service per API worker.

Inference proposes duplicate candidates; the backend adds a conservative area
guard before automatic grouping. A semantic score cannot overrule known area
disagreement, ambiguity, or missing location evidence.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
import math
import threading
from typing import Mapping, Sequence
import uuid

from ml.analysis import PriorityAnalyzer, source_report_key
from ml.location import LocationGazetteer
from ml.priority_ranker import ranker_score
from ml.time_features import extract_time_features, time_pair_features


def _records(values: Sequence[dict]) -> list[dict]:
    if isinstance(values, (str, bytes, dict)):
        raise ValueError("Existing complaints must be a sequence of records")
    result, seen = [], set()
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"].strip():
            raise ValueError("Existing complaints need nonempty string IDs")
        row = deepcopy(value)
        row["id"] = row["id"].strip()
        if row["id"] in seen:
            raise ValueError("Existing complaint IDs must be unique")
        seen.add(row["id"])
        if isinstance(row.get("timestamp"), (datetime, date)):
            row["timestamp"] = row["timestamp"].isoformat()
        result.append(row)
    return result


def _text_field(row: dict, name: str) -> str:
    value = row.get(name)
    if value is None:
        return ""
    if isinstance(value, (datetime, date)) and name == "timestamp":
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError(f"Complaint {name} must be a string when supplied")
    return value


def safe_duplicate_candidates(
    record: dict, result: dict, existing_records: Sequence[dict] | Mapping[str, dict],
    gazetteer: LocationGazetteer | None = None,
) -> tuple[list[dict], list[str]]:
    """Retain scored suggestions but mark only exact known-area matches eligible.

    Parent/child areas are compatible context, not precise equal locations.
    Timestamp differences alone do not prove separate incidents; only supported
    disjoint ongoing intervals add a temporal veto. The returned flags are
    necessary guard conditions, not verified same-incident labels.
    """
    gazetteer = gazetteer or LocationGazetteer()
    existing = dict(existing_records) if isinstance(existing_records, Mapping) else {row["id"]: row for row in _records(existing_records)}
    query = gazetteer.extract(_text_field(record, "text"), _text_field(record, "location"))
    query_time = extract_time_features(_text_field(record, "text"), _text_field(record, "timestamp"))
    raw_candidates = result.get("duplicate_candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("Analyzer duplicate_candidates must be a list")
    selected, warnings = {}, []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str) or not candidate["id"].strip():
            raise ValueError("Duplicate candidates need nonempty string IDs")
        candidate = deepcopy(candidate)
        identifier = candidate["id"] = candidate["id"].strip()
        if identifier == record.get("id"):
            continue
        score = candidate.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Duplicate candidate score must be finite")
        qualified = candidate.get("is_duplicate", True)
        if not isinstance(qualified, bool):
            raise ValueError("Candidate is_duplicate must be boolean")
        if "threshold" in candidate:
            threshold = candidate["threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
                raise ValueError("Duplicate candidate threshold must be finite")
            qualified = qualified and score >= threshold
        status, time_status = "unknown", "not_comparable"
        candidate_area = None
        eligible = False
        reasons = []
        other = existing.get(identifier)
        if not qualified:
            status = "below_threshold"
            reasons.append("below_duplicate_threshold")
        elif other is None:
            status = "unknown_record"
            reasons.append("candidate_not_in_current_store")
        else:
            context = gazetteer.extract(_text_field(other, "text"), _text_field(other, "location"))
            left, right = query.get("canonical_id"), context.get("canonical_id")
            candidate_area = right
            if query.get("conflict") or context.get("conflict"):
                status = "conflict"
                reasons.append("location_evidence_conflict")
            elif query.get("ambiguous") or context.get("ambiguous"):
                status = "ambiguous"
                reasons.append("location_evidence_ambiguous")
            elif not left or not right:
                status = "unknown"
                reasons.append("canonical_area_unavailable")
            elif left == right:
                status, eligible = "matched_exact", True
            elif left in context.get("ancestor_ids", []) or right in query.get("ancestor_ids", []):
                status = "parent_child"
                reasons.append("location_granularity_differs")
            else:
                status = "different"
                reasons.append("different_canonical_areas")
            other_time = extract_time_features(_text_field(other, "text"), _text_field(other, "timestamp"))
            temporal = time_pair_features(query_time, other_time)
            if not temporal["interval_missing"]:
                time_status = "overlap" if temporal["interval_overlap"] else "interval_disjoint"
                if not temporal["interval_overlap"]:
                    eligible = False
                    reasons.append("supported_ongoing_intervals_disjoint")
        candidate.update(
            auto_group_eligible=eligible,
            area_guard_status=status,
            time_guard_status=time_status,
            query_area_id=query.get("canonical_id"),
            candidate_area_id=candidate_area,
            auto_group_block_reasons=reasons,
        )
        old = selected.get(identifier)
        if old is None or candidate["score"] > old["score"]:
            selected[identifier] = candidate
        if reasons and qualified:
            warnings.extend(reasons)
    return sorted(selected.values(), key=lambda item: (-item["score"], item["id"])), list(dict.fromkeys(warnings))


class MLService:
    """Own one analyzer and embedding index; serialize access with ``lock``.

    FastAPI synchronous handlers run this CPU work in the thread pool. The root
    submission coordinator may hold ``lock`` across refresh, analysis, DB commit
    and final index refresh. Across workers, the database must supply a separate
    write/transaction lock; this in-process lock does not provide that guarantee.
    """

    def __init__(self, settings, analyzer=None):
        self.settings = settings
        self.analyzer = analyzer
        self.lock = threading.RLock()
        self._initialized = False
        self._signature = None
        self._existing: dict[str, dict] = {}
        self._guard_gazetteer = None
        self._refresh_failed = False

    def _make_analyzer(self, records: list[dict]):
        return PriorityAnalyzer(
            category_model=getattr(self.settings, "category_model", "artifacts/category/char_svm.joblib"),
            severity_model=getattr(self.settings, "severity_model", "artifacts/severity/char_svm.joblib"),
            review_policy=getattr(self.settings, "review_policy", "artifacts/review_policy.json"),
            rules_config=getattr(self.settings, "rules_config", "config/priority_rules.yaml"),
            ranker=getattr(self.settings, "priority_ranker", "artifacts/priority_ranker.joblib"),
            duplicate_model=getattr(self.settings, "duplicate_model", "artifacts/duplicates/P2.joblib"),
            existing_complaints=records,
            cache_dir=getattr(self.settings, "embedding_cache_dir", ".cache/backend"),
            top_k=getattr(self.settings, "duplicate_top_k", 100),
        )

    def initialize(self, existing_records: Sequence[dict]):
        with self.lock:
            records = _records(existing_records)
            if self.analyzer is None:
                self.analyzer = self._make_analyzer(records)
            if self._guard_gazetteer is None:
                finder = getattr(self.analyzer, "finder", None)
                self._guard_gazetteer = getattr(finder, "gazetteer", None)
                # Cosine-only duplicate bundles still snapshot a gazetteer,
                # although their scorer itself does not consume its features.
                if self._guard_gazetteer is None and isinstance(getattr(finder, "bundle", None), dict) and finder.bundle.get("gazetteer_file"):
                    config = finder.bundle.get("feature_config", {})
                    self._guard_gazetteer = LocationGazetteer(
                        finder._resource(finder.bundle["gazetteer_file"]),
                        fuzzy_threshold=config.get("fuzzy_threshold", 90),
                    )
                if self._guard_gazetteer is None:
                    self._guard_gazetteer = getattr(self.analyzer, "gazetteer", None) or LocationGazetteer()
            self._initialized = True
            self.refresh(records)
            if getattr(self.settings, "require_ml_ready", False) and not self.readiness()["ready"]:
                raise RuntimeError("Required calibrated classifiers, duplicate model and review policy are not ready")
            return self

    def refresh(self, records: Sequence[dict]):
        with self.lock:
            if not self._initialized:
                return self.initialize(records)
            copied = _records(records)
            # Include all supplied metadata, not only text: corrections to area,
            # timestamps or report lineage must refresh the shared context.
            signature = hashlib.sha256(json.dumps(sorted(copied, key=lambda row: row["id"]),
                                                  ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            if signature == self._signature and not self._refresh_failed:
                return self
            existing = {row["id"]: row for row in copied}
            finder = getattr(self.analyzer, "finder", None)
            if finder is not None:
                try:
                    finder.build_index(copied)
                except Exception:
                    self._refresh_failed = True
                    raise
            # Publish only after successful index construction.
            self.analyzer.existing = existing
            self._existing, self._signature = existing, signature
            self._refresh_failed = False
            return self

    def readiness(self) -> dict:
        with self.lock:
            if not self._initialized:
                return {"ready": False, "status": "degraded", "classifiers_loaded": False,
                        "duplicate_model_loaded": False, "review_policy_loaded": False,
                        "messages": ["ML service has not been initialized"]}
            if self._refresh_failed:
                return {"ready": False, "status": "degraded",
                        "classifiers_loaded": set(getattr(self.analyzer, "runtimes", {})) >= {"category", "severity"},
                        "duplicate_model_loaded": getattr(self.analyzer, "finder", None) is not None,
                        "review_policy_loaded": getattr(self.analyzer, "policy", None) is not None,
                        "messages": ["Duplicate index refresh failed; refresh successfully before serving analysis"]}
            supplied = getattr(self.analyzer, "readiness", None)
            if callable(supplied):
                report = supplied()
                if not isinstance(report, dict) or not isinstance(report.get("ready"), bool):
                    raise ValueError("Injected analyzer readiness must return a ready boolean")
                ready = report["ready"]
                return {"ready": ready, "status": "ready" if ready else "degraded",
                        "classifiers_loaded": bool(report.get("classifiers_loaded", ready)),
                        "duplicate_model_loaded": bool(report.get("duplicate_model_loaded", ready)),
                        "review_policy_loaded": bool(report.get("review_policy_loaded", ready)),
                        "messages": list(report.get("messages", []))}
            # An explicit ready flag is for a trusted dependency-injected test or
            # alternate implementation. The real analyzer exposes actual state.
            override = getattr(self.analyzer, "ready", None)
            if isinstance(override, bool):
                return {"ready": override, "status": "ready" if override else "degraded",
                        "classifiers_loaded": override, "duplicate_model_loaded": override,
                        "review_policy_loaded": override, "messages": [] if override else ["Injected analyzer reports unavailable models"]}
            classifiers = set(getattr(self.analyzer, "runtimes", {})) >= {"category", "severity"}
            duplicate = getattr(self.analyzer, "finder", None) is not None
            policy = getattr(self.analyzer, "policy", None) is not None
            messages = []
            if not classifiers:
                messages.append("Calibrated category/severity artifacts unavailable")
            if not duplicate:
                messages.append("Duplicate model unavailable")
            if not policy:
                messages.append("Compatible review policy unavailable")
            ready = classifiers and duplicate and policy
            return {"ready": ready, "status": "ready" if ready else "degraded",
                    "classifiers_loaded": classifiers, "duplicate_model_loaded": duplicate,
                    "review_policy_loaded": policy, "messages": messages}

    def _correct_report_volume(self, query: dict, result: dict) -> None:
        rules = getattr(self.analyzer, "rules", None)
        if rules is None:
            return
        lookup = {**self._existing, query["id"]: query}
        keys = {source_report_key(query, lookup)}
        for candidate in result["duplicate_candidates"]:
            if candidate["auto_group_eligible"]:
                keys.add(source_report_key(self._existing[candidate["id"]], lookup))
        unresolved = None in keys
        keys.discard(None)
        count = max(1, len(keys))
        rule_result = rules.score({**query, "category": result.get("category"), "severity": result.get("severity"),
                                   "duplicate_group_size": count})
        ranker = getattr(self.analyzer, "ranker", None)
        ranking = ranker_score(ranker, rule_result) if ranker is not None else {
            "ranking_score": rule_result["priority_score"], "method": "rules",
            "contributions": rule_result["rule_breakdown"], "reason": "No expert-trained ranking artifact configured",
        }
        result.update(priority_score=rule_result["priority_score"], priority_label=rule_result["priority_label"],
                      ranking_score=ranking["ranking_score"], ranking_method=ranking["method"],
                      duplicate_group_size=count, duplicate_lineage_incomplete=unresolved,
                      duplicate_group_size_semantics="estimated source-report count among self and area/time-eligible retrieved matches; not people affected or an exhaustive cluster size")
        explanation = result.setdefault("explanation", {})
        explanation.update(rule_breakdown=rule_result["rule_breakdown"], ranking=ranking,
                           matched_keywords=rule_result["matched_keywords"], negated_keywords=rule_result["negated_keywords"],
                           missing_fields=rule_result["missing_fields"], notes=rule_result["warnings"])

    def analyze(self, record: dict, existing_records: Sequence[dict] | None = None, persist: bool = False) -> dict:
        if persist:
            raise ValueError("MLService does not persist complaints; commit through the API transaction coordinator")
        with self.lock:
            if not self._initialized:
                self.initialize(existing_records or [])
            elif existing_records is not None:
                self.refresh(existing_records)
            if self._refresh_failed:
                raise RuntimeError("Duplicate index refresh is required before analysis")
            if not isinstance(record, dict):
                raise ValueError("Complaint input must be a record")
            query = deepcopy(record)
            if isinstance(query.get("timestamp"), (datetime, date)):
                query["timestamp"] = query["timestamp"].isoformat()
            result = deepcopy(self.analyzer.analyze_record(query))
            if not isinstance(result, dict):
                raise ValueError("Analyzer must return structured JSON data")
            query["id"] = query.get("id") or result.get("id") or "query_" + uuid.uuid4().hex
            query["lang"] = query.get("lang") or result.get("lang")
            gazetteer = self._guard_gazetteer
            context = gazetteer.extract(_text_field(query, "text"), _text_field(query, "location"))
            result["area_id"], result["location_context"] = context.get("canonical_id"), context
            candidates, warnings = safe_duplicate_candidates(query, result, self._existing, gazetteer)
            result["duplicate_candidates"] = candidates
            result["auto_group_candidate_count"] = sum(candidate["auto_group_eligible"] for candidate in candidates)
            result["duplicate_guard"] = {"policy": "exact_known_canonical_area_and_no_supported_disjoint_interval",
                                         "blocked_candidate_count": sum(not candidate["auto_group_eligible"] for candidate in candidates),
                                         "reasons": warnings}
            self._correct_report_volume(query, result)
            if warnings:
                reasons = list(result.get("review_reasons", []))
                reasons.append("duplicate_auto_group_guard_requires_review")
                result.update(review_required=True, review_status="needs_human_review",
                              review_reasons=list(dict.fromkeys(reasons)))
            # Reject NaN/objects before the result can reach a database JSON field.
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            return result
