"""Transparent configurable research priority rules with auditable evidence.

Keyword and duration rules are deterministic heuristics, not validated risk
estimates. Report-group size is a report-volume proxy and never a population
estimate. Source severity and output priority remain distinct constructs.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

import yaml

from ml.normalize import DEFAULT_VARIANTS_PATH, clean_text, load_variants, normalize_banglish
from ml.time_features import extract_time_features

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "priority_rules.yaml"
FEATURE_NAMES = ("severity", "category", "duration", "duplicate_reports", "vulnerable", "life_risk")
_TOKEN_BOUNDARY = r"[\w\u0980-\u09ff]"


def _number(value, description: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{description} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{description} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{description} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _mapping(raw, description: str) -> dict:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{description} must be a nonempty mapping")
    return raw


def _label_points(raw, description: str) -> dict[str, float]:
    result = {}
    for key, value in _mapping(raw, description).items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{description} keys must be nonempty strings")
        normalized = key.strip().casefold()
        if normalized in result:
            raise ValueError(f"{description} contains a duplicate case-insensitive label {key!r}")
        result[normalized] = _number(value, f"{description}.{key}")
    return result


def _keyword_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-‐‑‒–—_]", " ", clean_text(text))).strip()


def _phrase_pattern(phrase: str) -> re.Pattern:
    value = _keyword_text(phrase)
    if not value:
        raise ValueError("Keyword aliases cannot normalize to empty text")
    body = r"\s+".join(re.escape(word) for word in value.split())
    return re.compile(rf"(?<!{_TOKEN_BOUNDARY}){body}(?!{_TOKEN_BOUNDARY})")


class PriorityRules:
    """Load safe YAML once and score complaint evidence without hidden learning."""

    feature_names = FEATURE_NAMES

    def __init__(self, config_path: str | Path | None = None, variants_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        self.variants_path = Path(variants_path or DEFAULT_VARIANTS_PATH).resolve()
        load_variants(self.variants_path)
        self.preprocessing_fingerprints = {
            "normalization_dictionary_sha256": hashlib.sha256(self.variants_path.read_bytes()).hexdigest(),
            "normalizer_code_sha256": hashlib.sha256(Path(__file__).with_name("normalize.py").read_bytes()).hexdigest(),
            "priority_rules_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "time_features_code_sha256": hashlib.sha256(Path(__file__).with_name("time_features.py").read_bytes()).hexdigest(),
        }
        try:
            data = self.config_path.read_bytes()
            raw = yaml.safe_load(data.decode("utf-8-sig"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ValueError(f"Cannot load priority rule YAML {self.config_path}: {exc}") from exc
        self.config_sha256 = hashlib.sha256(data).hexdigest()
        if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw.get("schema_version") != 1:
            raise ValueError("Priority config must contain schema_version: 1")
        expected = {"schema_version", "weights", "severity_points", "category_points", "duration_bands", "duplicate_bands", "keyword_rules", "negation", "priority_bands", "max_score"}
        if set(raw) - expected:
            raise ValueError(f"Unknown priority config keys: {sorted(set(raw)-expected)}")
        weights = _mapping(raw.get("weights"), "weights")
        if set(weights) != set(self.feature_names):
            raise ValueError(f"weights must define exactly {list(self.feature_names)}")
        self.weights = {key: _number(weights[key], f"weights.{key}") for key in self.feature_names}
        self.severity_points = _label_points(raw.get("severity_points"), "severity_points")
        self.category_points = _label_points(raw.get("category_points"), "category_points")
        self.duration_bands = self._bands(raw.get("duration_bands"), "min_hours", "duration_bands")
        self.duplicate_bands = self._bands(raw.get("duplicate_bands"), "min_reports", "duplicate_bands")
        for band in self.duplicate_bands:
            if not band["minimum"].is_integer() or band["minimum"] < 1:
                raise ValueError("duplicate_bands.min_reports must be a positive integer")
        keyword_rules = _mapping(raw.get("keyword_rules"), "keyword_rules")
        if set(keyword_rules) != {"vulnerable", "life_risk"}:
            raise ValueError("keyword_rules must define vulnerable and life_risk")
        self.keyword_rules = {}
        for name, spec in keyword_rules.items():
            spec = _mapping(spec, f"keyword_rules.{name}")
            if set(spec) != {"points", "max_concepts", "concepts"}:
                raise ValueError(f"keyword_rules.{name} needs points, max_concepts, concepts")
            points = _number(spec["points"], f"keyword_rules.{name}.points")
            cap = spec["max_concepts"]
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
                raise ValueError(f"keyword_rules.{name}.max_concepts must be a positive integer")
            concepts = _mapping(spec["concepts"], f"keyword_rules.{name}.concepts")
            parsed = {}
            alias_owners = {}
            for concept, aliases in concepts.items():
                if not isinstance(concept, str) or not concept.strip() or not isinstance(aliases, list) or not aliases:
                    raise ValueError(f"{name} concepts need nonempty names and alias lists")
                if any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
                    raise ValueError(f"{name}.{concept} aliases must be nonempty strings")
                parsed[concept] = []
                for alias in aliases:
                    normalized = _keyword_text(alias)
                    previous = alias_owners.setdefault(normalized, concept)
                    if previous != concept:
                        raise ValueError(f"Ambiguous {name} alias {alias!r} in multiple concepts")
                    parsed[concept].append((normalized, _phrase_pattern(normalized)))
            self.keyword_rules[name] = {"points": points, "max_concepts": cap, "concepts": parsed}
        negation = raw.get("negation", {"before": [], "after": []})
        if not isinstance(negation, dict) or set(negation) != {"before", "after"}:
            raise ValueError("negation must contain before and after phrase lists")
        self.negation = {}
        for side in ("before", "after"):
            values = negation[side]
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"negation.{side} must be a list of nonempty phrases")
            self.negation[side] = [_keyword_text(value) for value in values]
        self.priority_bands = self._bands(raw.get("priority_bands"), "min_score", "priority_bands", points=False)
        if self.priority_bands[0]["minimum"] != 0 or self.priority_bands[0]["strict"]:
            raise ValueError("priority_bands must start at min_score: 0")
        labels = [band["label"] for band in self.priority_bands]
        if len(set(labels)) != len(labels):
            raise ValueError("priority_bands labels must be unique")
        self.max_score = None if raw.get("max_score") is None else _number(raw["max_score"], "max_score", positive=True)

    @staticmethod
    def _bands(values, minimum_key: str, description: str, points: bool = True) -> list[dict]:
        if not isinstance(values, list) or not values:
            raise ValueError(f"{description} must be a nonempty list")
        result = []
        for spec in values:
            if not isinstance(spec, dict) or minimum_key not in spec or "label" not in spec:
                raise ValueError(f"{description} entries need {minimum_key} and label")
            allowed = {minimum_key, "label", "strict"} | ({"points"} if points else set())
            if set(spec) - allowed or (points and "points" not in spec):
                raise ValueError(f"Unexpected/missing fields in {description}")
            if not isinstance(spec["label"], str) or not spec["label"].strip():
                raise ValueError(f"{description} labels must be nonempty strings")
            if not isinstance(spec.get("strict", False), bool):
                raise ValueError(f"{description}.strict must be boolean")
            result.append({"minimum": _number(spec[minimum_key], f"{description}.{minimum_key}"),
                           "points": _number(spec["points"], f"{description}.points") if points else None,
                           "label": spec["label"], "strict": spec.get("strict", False)})
        result.sort(key=lambda row: row["minimum"])
        if len({row["minimum"] for row in result}) != len(result):
            raise ValueError(f"{description} thresholds must be unique")
        return result

    def label_for_score(self, score: float) -> str:
        value = _number(score, "priority score")
        # Explicit loop avoids treating strict/non-strict thresholds alike.
        label = self.priority_bands[0]["label"]
        for band in self.priority_bands:
            if value > band["minimum"] if band["strict"] else value >= band["minimum"]:
                label = band["label"]
        return label

    def _negated(self, text: str, start: int, end: int) -> bool:
        # Stop at punctuation/clause boundaries. Adjacent explicit negation is
        # supported; scope, sarcasm and counterfactuals need human interpretation.
        before = re.split(r"[.,;:!?।\n]", text[:start])[-1].strip()
        after = re.split(r"[.,;:!?।\n]", text[end:])[0].strip()
        return any(_phrase_pattern(value).search(before) and before.endswith(value) for value in self.negation["before"]) or any(
            _phrase_pattern(value).match(after) is not None for value in self.negation["after"]
        )

    def _keywords(self, name: str, text: str) -> tuple[list[dict], list[dict]]:
        matched, negated = [], []
        for concept, aliases in self.keyword_rules[name]["concepts"].items():
            active = []
            negative = []
            for alias, pattern in aliases:
                for match in pattern.finditer(text):
                    evidence = {"concept": concept, "alias": alias, "matched_text": match.group(),
                                "start": match.start(), "end": match.end()}
                    (negative if self._negated(text, match.start(), match.end()) else active).append(evidence)
            active = [row for row in active if not any(neg["start"] <= row["start"] and row["end"] <= neg["end"] for neg in negative)]
            if active:
                # One evidence item per concept: longer phrases win ties, then
                # earlier position; repeating aliases never stacks rule points.
                matched.append(sorted(active, key=lambda item: (-len(item["matched_text"]), item["start"]))[0])
            negated.extend(negative)
        dedup = {(row["concept"], row["start"], row["end"]): row for row in negated}
        return matched, list(dedup.values())

    def score(self, complaint: dict) -> dict:
        if not isinstance(complaint, dict) or not isinstance(complaint.get("text"), str) or not complaint["text"].strip():
            raise ValueError("Complaint needs a nonempty text string")
        language = complaint.get("lang")
        if language is not None and language not in ("bn", "en", "banglish"):
            raise ValueError("lang must be bn, en or banglish when provided")
        text = normalize_banglish(complaint["text"], self.variants_path) if language == "banglish" else clean_text(complaint["text"])
        if not text:
            raise ValueError("Complaint text is empty after cleaning")
        keyword_text = _keyword_text(text)
        count = complaint.get("duplicate_group_size", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("duplicate_group_size must be a positive integer report count")
        features = {name: 0.0 for name in self.feature_names}
        evidence = {name: [] for name in self.feature_names}
        warnings, missing = [], []
        for name, points in (("severity", self.severity_points), ("category", self.category_points)):
            value = complaint.get(name)
            if not isinstance(value, str) or not value.strip() or value.strip().casefold() not in points:
                missing.append(name)
                warnings.append(f"Unknown or missing {name}; its rule contribution is zero and needs label review.")
                evidence[name] = [{"value": value if isinstance(value, str) else None, "known": False}]
            else:
                features[name] = points[value.strip().casefold()]
                evidence[name] = [{"value": value.strip(), "known": True}]
        timestamp = complaint.get("timestamp") or ""
        temporal = extract_time_features(complaint["text"], timestamp)
        hours = temporal["duration_hours"]
        if hours is None:
            missing.append("duration")
            warnings.append("Duration unavailable or ambiguous; no duration boost was imputed.")
        else:
            selected = None
            for band in self.duration_bands:
                if hours > band["minimum"] if band["strict"] else hours >= band["minimum"]:
                    selected = band
            if selected:
                features["duration"] = selected["points"]
            evidence["duration"] = [{"duration_hours": hours, "duration_days": temporal["duration_days"],
                                     "band": selected["label"] if selected else None,
                                     "matched_expressions": temporal["matched_expressions"]}]
        warnings.extend(temporal["errors"])
        selected = None
        for band in self.duplicate_bands:
            if count > band["minimum"] if band["strict"] else count >= band["minimum"]:
                selected = band
        features["duplicate_reports"] = selected["points"] if selected else 0.0
        evidence["duplicate_reports"] = [{"report_count": count, "band": selected["label"] if selected else None,
                                          "interpretation": "report volume, not unique people affected"}]
        matched, negated = {}, {}
        for name in ("vulnerable", "life_risk"):
            matched[name], negated[name] = self._keywords(name, keyword_text)
            spec = self.keyword_rules[name]
            features[name] = min(len(matched[name]), spec["max_concepts"]) * spec["points"]
            evidence[name] = matched[name]
        breakdown = [
            {"name": name, "points": features[name], "weight": self.weights[name],
             "contribution": features[name] * self.weights[name], "evidence": evidence[name]}
            for name in self.feature_names
        ]
        score = sum(row["contribution"] for row in breakdown)
        if not math.isfinite(score):
            raise ValueError("Priority score overflowed; reduce configured weights/points")
        unclipped = score
        if self.max_score is not None and score > self.max_score:
            score = self.max_score
            breakdown.append({"name": "score_cap", "points": score-unclipped, "weight": 1.0,
                              "contribution": score-unclipped, "evidence": [{"max_score": self.max_score}]})
        return {
            "priority_score": float(score), "priority_label": self.label_for_score(score),
            "unclipped_score": float(unclipped), "rule_breakdown": breakdown,
            "feature_values": features, "matched_keywords": matched, "negated_keywords": negated,
            "missing_fields": missing, "warnings": warnings, "config_sha256": self.config_sha256,
            "preprocessing_fingerprints": dict(self.preprocessing_fingerprints),
            "duration_hours": hours, "duration_days": temporal["duration_days"],
            "duplicate_group_size": count, "group_size_interpretation": "number of reports, not number of people affected",
            "text_normalized": text,
        }
