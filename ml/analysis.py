"""One configurable, cached analysis service for complaint triage research.

The three-argument convenience API infers language heuristically. Use
PriorityAnalyzer.analyze_record with explicit lang for collected/evaluation
records. Missing models or a missing review policy never create confident
automatic decisions.
"""

from __future__ import annotations

import uuid
from pathlib import Path
import re

from ml.classifier_runtime import ClassifierRuntime
from ml.duplicate_index import DuplicateFinder
from ml.location import LocationGazetteer
from ml.normalize import clean_text
from ml.priority_ranker import load_priority_ranker, ranker_score
from ml.priority_rules import PriorityRules
from ml.uncertainty import load_review_policy

ROOT = Path(__file__).resolve().parents[1]
_BANGLISH_WORDS = {"pani", "pany", "nai", "nei", "bidyut", "biddut", "rasta", "ache", "achhe", "dhore", "onek", "ekhane", "amader", "agun", "baccha", "haspatal", "bonna"}


def _path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def infer_language(text: str) -> dict:
    cleaned = clean_text(text)
    if any("\u0980" <= char <= "\u09ff" and char.isalpha() for char in cleaned):
        return {"lang": "bn", "method": "Bangla-script heuristic", "ambiguous": False}
    hits = sorted(set(re.findall(r"[a-z]+", cleaned)) & _BANGLISH_WORDS)
    return {"lang": "banglish" if len(hits) >= 2 else "en", "method": "small romanized-Bangla lexicon heuristic",
            "ambiguous": len(hits) == 1, "matched_words": hits}


def source_report_key(record: dict, lookup: dict[str, dict]) -> str | None:
    """Resolve a source report, not an incident or a translated copy.

    Independent originals sharing original_id remain distinct. A derived row
    with no resolvable parent is excluded rather than increasing volume.
    """
    current, seen = record, set()
    while True:
        identifier = current["id"]
        if identifier in seen:
            raise ValueError("Cyclic source-report parent lineage")
        seen.add(identifier)
        explicit = current.get("source_report_id")
        if explicit:
            if not isinstance(explicit, str):
                raise ValueError("source_report_id must be a string")
            if not explicit.strip():
                raise ValueError("source_report_id must not be blank whitespace")
            return "source:" + explicit.strip()
        parent = current.get("parent_id")
        if parent:
            if not isinstance(parent, str):
                raise ValueError("parent_id must be a string")
            parent = parent.strip()
            if not parent:
                raise ValueError("parent_id must not be blank whitespace")
            if parent == identifier or parent in seen:
                raise ValueError("Cyclic source-report parent lineage")
            if parent in lookup:
                current = lookup[parent]
                continue
            return "source:" + parent
        derived = str(current.get("is_derived", "false")).strip().lower() in ("true", "1")
        if derived:
            return None
        return "source:" + identifier


class PriorityAnalyzer:
    def __init__(
        self,
        category_model: str | Path | None = "artifacts/category/char_svm.joblib",
        severity_model: str | Path | None = "artifacts/severity/char_svm.joblib",
        review_policy: str | Path | None = "artifacts/review_policy.json",
        rules_config: str | Path | None = None,
        ranker: str | Path | None = "artifacts/priority_ranker.joblib",
        duplicate_model: str | Path | None = "artifacts/duplicates/P2.joblib",
        existing_complaints: list[dict] | None = None,
        cache_dir: str | Path = ".cache/analysis_duplicates",
        top_k: int = 100,
        review_on_life_risk: bool = True,
    ):
        if not isinstance(review_on_life_risk, bool):
            raise ValueError("review_on_life_risk must be boolean")
        self.rules = PriorityRules(_path(rules_config))
        self.review_on_life_risk = review_on_life_risk
        self.runtimes, self.model_fingerprints, self.model_resource_fingerprints = {}, {}, {}
        self.missing_models = []
        for target, value in (("category", category_model), ("severity", severity_model)):
            path = _path(value)
            if path is not None and path.is_file():
                runtime = ClassifierRuntime(path, expected_target=target)
                self.runtimes[target] = runtime
                self.model_fingerprints[target] = runtime.fingerprint
                self.model_resource_fingerprints[target] = runtime.resource_fingerprint
            else:
                self.missing_models.append(target)
        policy_path = _path(review_policy)
        self.policy = None
        if policy_path is not None and policy_path.is_file() and len(self.runtimes) == 2:
            self.policy = load_review_policy(policy_path, self.model_fingerprints, self.model_resource_fingerprints)
        ranker_path = _path(ranker)
        self.ranker = load_priority_ranker(ranker_path, self.rules.config_sha256, self.rules.preprocessing_fingerprints) if ranker_path is not None and ranker_path.is_file() else None
        self.gazetteer = LocationGazetteer()
        self.finder = None
        self.existing: dict[str, dict] = {}
        duplicate_path = _path(duplicate_model)
        if existing_complaints is not None:
            for record in existing_complaints:
                if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"].strip():
                    raise ValueError("Existing complaint records need nonempty string IDs")
                identifier = record["id"].strip()
                if identifier in self.existing:
                    raise ValueError("Existing complaint IDs must be unique")
                self.existing[identifier] = dict(record, id=identifier)
            if duplicate_path is not None and duplicate_path.is_file():
                self.finder = DuplicateFinder(duplicate_path, _path(cache_dir), top_k=top_k)
                self.finder.build_index(list(self.existing.values()))

    def analyze_record(self, record: dict) -> dict:
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError("Complaint record needs text")
        text = record["text"]
        if not clean_text(text):
            raise ValueError("Complaint text is empty after cleaning")
        supplied_language = record.get("lang")
        language = ({"lang": supplied_language, "method": "provided", "ambiguous": False}
                    if supplied_language else infer_language(text))
        lang = language["lang"]
        if lang not in ("bn", "en", "banglish"):
            raise ValueError("lang must be bn, en or banglish")
        location = "" if record.get("location") is None else record.get("location", "")
        timestamp = "" if record.get("timestamp") is None else record.get("timestamp", "")
        if not isinstance(location, str) or not isinstance(timestamp, str):
            raise ValueError("Location and timestamp must be strings when provided")
        identifier = record.get("id") or "new_" + uuid.uuid4().hex
        if not isinstance(identifier, str):
            raise ValueError("id must be a string")
        identifier = identifier.strip()
        if not identifier:
            raise ValueError("id must be nonempty when supplied")
        predictions = {target: runtime.predict(text, lang) for target, runtime in self.runtimes.items()}
        category = predictions.get("category", {}).get("prediction")
        severity = predictions.get("severity", {}).get("prediction")
        confidence = min(prediction["confidence"] for prediction in predictions.values()) if len(predictions) == 2 else None
        context = self.gazetteer.extract(text, location)
        new = {"id": identifier, "original_id": record.get("original_id") or "",
               "text": text, "lang": lang, "category": category or "", "location": location, "timestamp": timestamp,
               **{key: record[key] for key in ("source_report_id", "parent_id", "is_derived", "derivation") if key in record}}
        candidates = self.finder.find_duplicates(new, None) if self.finder is not None else []
        # Incident identity is deliberately not a source-report identity.
        lineage_lookup = {**self.existing, identifier: new}
        keys = {source_report_key(new, lineage_lookup)}
        for candidate in candidates:
            keys.add(source_report_key(self.existing[candidate["id"]], lineage_lookup))
        unresolved_lineage = None in keys
        keys.discard(None)
        duplicate_count = max(1, len(keys))
        rules = self.rules.score({**new, "category": category, "severity": severity,
                                  "duplicate_group_size": duplicate_count})
        ranking = ranker_score(self.ranker, rules) if self.ranker is not None else {
            "ranking_score": rules["priority_score"], "method": "rules",
            "contributions": rules["rule_breakdown"], "reason": "No expert-trained ranking artifact configured",
        }
        reasons = []
        if self.missing_models:
            reasons.append("classification_models_unavailable:" + ",".join(self.missing_models))
        if self.policy is None:
            reasons.append("validated_review_policy_unavailable")
        elif self.policy["status"] == "reject_all":
            reasons.append("validation_policy_reject_all")
        elif confidence is None or confidence < self.policy["threshold"]:
            reasons.append("low_classification_confidence")
        if language["ambiguous"]:
            reasons.append("language_inference_ambiguous")
        if context.get("conflict") or context.get("ambiguous"):
            reasons.append("location_evidence_conflicting_or_ambiguous")
        if unresolved_lineage:
            reasons.append("duplicate_report_lineage_incomplete")
        if set(rules["missing_fields"]) & {"category", "severity"}:
            reasons.append("priority_label_mapping_unknown")
        if self.review_on_life_risk and rules["matched_keywords"]["life_risk"]:
            reasons.append("life_risk_keyword_requires_review")
        matched = {item["concept"] for item in rules["matched_keywords"]["life_risk"]}
        negated = {item["concept"] for item in rules["negated_keywords"]["life_risk"]}
        if matched & negated:
            reasons.append("life_risk_keyword_context_conflict")
        duplicate_stats = dict(self.finder.last_query_stats) if self.finder is not None else None
        return {
            "id": identifier, "category": category, "severity": severity,
            "priority_score": rules["priority_score"], "priority_label": rules["priority_label"],
            "ranking_score": float(ranking["ranking_score"]), "ranking_method": ranking["method"],
            "confidence": confidence,
            "confidence_definition": "minimum of calibrated category/severity top probabilities; not a calibrated joint, priority or duplicate probability",
            "classification_probabilities": {target: prediction["probabilities"] for target, prediction in predictions.items()},
            "lang": lang, "language_inference": language,
            "area_id": context.get("canonical_id"), "location_context": context,
            "explanation": {
                "category": predictions.get("category", {}).get("explanation"),
                "severity": predictions.get("severity", {}).get("explanation"),
                "rule_breakdown": rules["rule_breakdown"], "ranking": ranking,
                "matched_keywords": rules["matched_keywords"], "negated_keywords": rules["negated_keywords"],
                "missing_fields": rules["missing_fields"], "notes": rules["warnings"],
            },
            "review_required": bool(reasons), "review_status": "needs_human_review" if reasons else "scored",
            "review_reasons": reasons,
            "review_threshold": self.policy["threshold"] if self.policy is not None else None,
            "duplicate_candidates": candidates,
            "duplicate_detection_status": "available" if self.finder is not None else "not_configured",
            "duplicate_group_size": duplicate_count,
            "duplicate_group_size_semantics": "estimated distinct source reports among self and retrieved matches; derived versions collapsed; not unique people or exhaustive cluster size",
            "duplicate_lineage_incomplete": unresolved_lineage,
            "duplicate_retrieval": duplicate_stats,
            "provenance": {"classifier_fingerprints": self.model_fingerprints, "rules_config_sha256": self.rules.config_sha256,
                           "classifier_resource_fingerprints": self.model_resource_fingerprints,
                           "rules_preprocessing_fingerprints": self.rules.preprocessing_fingerprints},
        }

    def analyze_complaint(self, text, location=None, timestamp=None) -> dict:
        return self.analyze_record({"text": text, "location": location, "timestamp": timestamp})


_DEFAULT_ANALYZER: PriorityAnalyzer | None = None


def configure_analyzer(**kwargs) -> PriorityAnalyzer:
    """Configure artifacts, expert ranker and optional complaint store once."""
    global _DEFAULT_ANALYZER
    _DEFAULT_ANALYZER = PriorityAnalyzer(**kwargs)
    return _DEFAULT_ANALYZER


def analyze_complaint(text, location=None, timestamp=None) -> dict:
    global _DEFAULT_ANALYZER
    if _DEFAULT_ANALYZER is None:
        _DEFAULT_ANALYZER = PriorityAnalyzer()
    return _DEFAULT_ANALYZER.analyze_complaint(text, location, timestamp)
