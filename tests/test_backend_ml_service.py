"""Backend ML lifecycle, thread safety and area guards with synthetic records."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import threading
import time
from types import SimpleNamespace

import pytest

from backend import ml_service
from backend.ml_service import MLService, safe_duplicate_candidates
from ml.analysis import PriorityAnalyzer
from ml.location import LocationGazetteer


def settings(**updates):
    values = dict(category_model=None, severity_model=None, review_policy=None,
                  rules_config=None, priority_ranker=None, duplicate_model=None,
                  require_ml_ready=False, embedding_cache_dir=".cache/backend-test", duplicate_top_k=100)
    values.update(updates)
    return SimpleNamespace(**values)


def complaint(identifier, text="pani nai for three days", location="Mirpur 10", **updates):
    result = {"id": identifier, "text": text, "lang": "banglish", "category": "Water",
              "location": location, "timestamp": ""}
    result.update(updates)
    return result


class FakeFinder:
    def __init__(self):
        self.gazetteer = LocationGazetteer()
        self.builds = []

    def build_index(self, records):
        self.builds.append(deepcopy(records))


class FakeAnalyzer:
    ready = True

    def __init__(self):
        self.finder = FakeFinder()
        self.existing = {}
        self.active, self.max_active = 0, 0
        self.counter_lock = threading.Lock()

    def analyze_record(self, record):
        with self.counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.003)
        with self.counter_lock:
            self.active -= 1
        return {"id": record.get("id", "new"), "lang": record.get("lang", "en"),
                "category": "Water", "severity": "High", "priority_score": 5.0,
                "priority_label": "moderate", "confidence": 0.95,
                "review_required": False, "review_reasons": [],
                "duplicate_candidates": [{"id": identifier, "score": 0.99, "threshold": 0.8,
                                          "is_duplicate": True} for identifier in self.existing]}


def test_initialize_loads_once_and_refresh_reuses_same_metadata_signature(monkeypatch):
    built = []

    def factory(**kwargs):
        built.append(kwargs)
        return FakeAnalyzer()

    monkeypatch.setattr(ml_service, "PriorityAnalyzer", factory)
    service = MLService(settings(require_ml_ready=True))
    records = [complaint("a")]
    assert service.readiness()["ready"] is False
    service.initialize(records)
    service.initialize(records)
    service.refresh(records)
    service.analyze(complaint("new"))
    assert len(built) == 1
    assert len(service.analyzer.finder.builds) == 1
    assert service.readiness()["ready"] is True


def test_refresh_copies_metadata_and_rebuilds_for_evidence_updates_only():
    analyzer = FakeAnalyzer()
    service = MLService(settings(), analyzer).initialize([complaint("a")])
    changed = [complaint("a", location="Mohammadpur")]
    service.refresh(changed)
    changed[0]["location"] = "Uttara Sector 4"
    assert service.analyzer.existing["a"]["location"] == "Mohammadpur"
    assert len(analyzer.finder.builds) == 2
    assert service.analyze(complaint("new"))["duplicate_candidates"][0]["auto_group_eligible"] is False


def test_parallel_requests_are_serialized_for_shared_analyzer():
    analyzer = FakeAnalyzer()
    service = MLService(settings(), analyzer).initialize([complaint("a")])
    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(lambda index: service.analyze(complaint(f"new-{index}")), range(20)))
    assert analyzer.max_active == 1
    assert all(response["duplicate_candidates"][0]["auto_group_eligible"] for response in responses)
    assert len(analyzer.finder.builds) == 1


@pytest.mark.parametrize("text,language", [
    ("মিরপুর ১০ এ তিন দিন ধরে পানি নেই", "bn"),
    ("mirpur-10 e 3 din dhore pani nai", "banglish"),
    ("No water in Mirpur 10 for three days", "en"),
])
def test_cross_language_mirpur_area_guard_permits_known_exact_canonical_match(text, language):
    query = complaint("new", text=text, lang=language, location=None)
    existing = [complaint("a", text="মিরপুর ১০ এ পানি নেই", lang="bn", location="মিরপুর ১০")]
    candidates, warnings = safe_duplicate_candidates(query, {"duplicate_candidates": [{"id": "a", "score": 0.99}]}, existing)
    assert warnings == []
    assert candidates[0]["auto_group_eligible"] is True
    assert candidates[0]["area_guard_status"] == "matched_exact"
    assert candidates[0]["query_area_id"] == candidates[0]["candidate_area_id"] == "dhaka_mirpur_10"


def test_same_wording_different_explicit_area_never_auto_groups_even_at_max_score():
    service = MLService(settings(), FakeAnalyzer()).initialize([complaint("a", text="No water for three days", location="Mirpur 10")])
    result = service.analyze(complaint("new", text="No water for three days", location="Uttara sector 4"))
    candidate = result["duplicate_candidates"][0]
    assert candidate["score"] == 0.99
    assert candidate["auto_group_eligible"] is False
    assert candidate["area_guard_status"] == "different"
    assert result["review_required"] is True
    assert result["auto_group_candidate_count"] == 0


@pytest.mark.parametrize("left,right,status", [
    ("", "", "unknown"),
    ("Mirpur", "Mirpur 10", "parent_child"),
    ("Mirpur 10 and Mohammadpur", "Mirpur 10", "ambiguous"),
])
def test_unknown_parent_or_ambiguous_locations_require_manual_grouping(left, right, status):
    candidates, warnings = safe_duplicate_candidates(
        complaint("new", text="No water", location=left), {"duplicate_candidates": [{"id": "a", "score": 1.0}]},
        [complaint("a", text="No water", location=right)],
    )
    assert candidates[0]["auto_group_eligible"] is False
    assert candidates[0]["area_guard_status"] == status
    assert warnings


def test_explicit_and_textual_location_conflict_blocks_automatic_merge():
    query = complaint("new", text="Mirpur 10 water problem", location="Uttara Sector 4")
    candidates, warnings = safe_duplicate_candidates(query, {"duplicate_candidates": [{"id": "a", "score": 1.0}]}, [complaint("a")])
    assert candidates[0]["area_guard_status"] == "conflict"
    assert "location_evidence_conflict" in warnings


def test_disjoint_supported_ongoing_intervals_block_but_report_dates_alone_do_not():
    old = complaint("a", text="No water for three days", timestamp="2026-01-01T00:00:00Z")
    query = complaint("new", text="No water for three days", timestamp="2026-02-01T00:00:00Z")
    candidates, _ = safe_duplicate_candidates(query, {"duplicate_candidates": [{"id": "a", "score": 1.0}]}, [old])
    assert candidates[0]["auto_group_eligible"] is False
    assert candidates[0]["time_guard_status"] == "interval_disjoint"
    query["text"] = old["text"] = "No water"
    candidates, _ = safe_duplicate_candidates(query, {"duplicate_candidates": [{"id": "a", "score": 1.0}]}, [old])
    assert candidates[0]["auto_group_eligible"] is True
    assert candidates[0]["time_guard_status"] == "not_comparable"


def test_unknown_candidates_self_matches_and_duplicate_orientations_are_safe():
    query = complaint("self")
    raw = {"duplicate_candidates": [{"id": "self", "score": 1.0}, {"id": "a", "score": 0.9},
                                    {"id": "a", "score": 0.95}, {"id": "missing", "score": 1.0}]}
    candidates, _ = safe_duplicate_candidates(query, raw, [complaint("a")])
    assert [item["id"] for item in candidates] == ["missing", "a"]
    assert candidates[0]["auto_group_eligible"] is False
    assert candidates[1]["score"] == 0.95


@pytest.mark.parametrize("score", [None, True, float("nan"), float("inf"), "0.99"])
def test_invalid_model_scores_are_rejected(score):
    with pytest.raises(ValueError, match="score"):
        safe_duplicate_candidates(complaint("new"), {"duplicate_candidates": [{"id": "a", "score": score}]}, [complaint("a")])


def test_below_threshold_candidate_cannot_be_grouped_and_does_not_force_review():
    candidates, warnings = safe_duplicate_candidates(complaint("new"), {"duplicate_candidates": [{"id": "a", "score": 0.2, "threshold": 0.8}]}, [complaint("a")])
    assert candidates[0]["auto_group_eligible"] is False
    assert candidates[0]["area_guard_status"] == "below_threshold"
    assert warnings == []


def test_real_analyzer_without_models_is_explicitly_degraded_and_reviewed():
    service = MLService(settings()).initialize([])
    readiness = service.readiness()
    assert readiness["ready"] is False
    assert readiness["classifiers_loaded"] is False
    result = service.analyze(complaint("new", text="No water for three days", lang="en"))
    assert result["category"] is None and result["confidence"] is None
    assert result["review_required"] is True
    assert result["duplicate_candidates"] == []
    assert result["duplicate_group_size"] == 1
    json.dumps(result, allow_nan=False)


def test_required_models_fail_startup_instead_of_inventing_readiness():
    with pytest.raises(RuntimeError, match="not ready"):
        MLService(settings(require_ml_ready=True)).initialize([])


def test_area_blocked_candidates_do_not_inflate_real_rule_report_volume():
    analyzer = PriorityAnalyzer(category_model=None, severity_model=None, review_policy=None,
                                ranker=None, duplicate_model=None, existing_complaints=[])
    records = [complaint("a", location="Mohammadpur"), complaint("b", location="Uttara Sector 4")]
    service = MLService(settings(), analyzer).initialize(records)
    original = analyzer.analyze_record

    def propose(record):
        result = original(record)
        result["duplicate_candidates"] = [{"id": "a", "score": 1.0}, {"id": "b", "score": 1.0}]
        result["duplicate_group_size"] = 3
        return result

    analyzer.analyze_record = propose
    result = service.analyze(complaint("new", text="No water for three days", lang="en"))
    assert result["duplicate_group_size"] == 1
    volume = next(row for row in result["explanation"]["rule_breakdown"] if row["name"] == "duplicate_reports")
    assert volume["contribution"] == 0


def test_service_does_not_persist_and_accepts_datetime_metadata():
    service = MLService(settings(), FakeAnalyzer()).initialize([complaint("a", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))])
    assert service.analyzer.existing["a"]["timestamp"] == "2026-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="does not persist"):
        service.analyze(complaint("new"), persist=True)


def test_refresh_failure_does_not_publish_new_corpus():
    analyzer = FakeAnalyzer()
    service = MLService(settings(), analyzer).initialize([complaint("a")])

    def fail(records):
        raise RuntimeError("bad index")

    analyzer.finder.build_index = fail
    with pytest.raises(RuntimeError, match="bad index"):
        service.refresh([complaint("b")])
    assert set(analyzer.existing) == {"a"}
    assert set(service._existing) == {"a"}
    assert service.readiness()["ready"] is False
    with pytest.raises(RuntimeError, match="refresh is required"):
        service.analyze(complaint("new"))


def test_cosine_bundle_uses_its_frozen_gazetteer_for_guard_and_stored_area(tmp_path):
    saved = tmp_path / "gazetteer.json"
    saved.write_text(json.dumps({"schema_version": 1, "areas": [
        {"id": "saved_area", "name": "Saved Alias", "parent_id": None, "aliases": ["saved alias"]}
    ]}), encoding="utf-8")

    class CosineFinder(FakeFinder):
        def __init__(self):
            self.gazetteer = None
            self.builds = []
            self.bundle = {"gazetteer_file": "resources/gazetteer.json", "feature_config": {"fuzzy_threshold": 90}}

        def _resource(self, relative):
            assert relative == self.bundle["gazetteer_file"]
            return saved

    analyzer = FakeAnalyzer()
    analyzer.finder = CosineFinder()
    service = MLService(settings(), analyzer).initialize([complaint("a", text="No water", location="Saved Alias")])
    result = service.analyze(complaint("new", text="No water", location="Saved Alias"))
    assert result["area_id"] == "saved_area"
    assert result["duplicate_candidates"][0]["auto_group_eligible"] is True
