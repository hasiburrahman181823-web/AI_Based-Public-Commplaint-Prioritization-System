"""Deterministic location/duration features; examples are synthetic test inputs."""

from __future__ import annotations

import json

import pytest

from ml.location import LocationGazetteer, location_pair_features, normalize_location_text
from ml.time_features import extract_time_features, time_pair_features


@pytest.fixture
def gazetteer():
    return LocationGazetteer()


@pytest.mark.parametrize("text", [
    "মিরপুর ১০ এ পানি নেই", "Mirpur 10 has no water", "mirpur-10 pani nai",
    "mirpur10 pani nai", "MIRPUR—১০, পানি নেই", "মিরপুর১০ এ পানি নেই",
])
def test_cross_script_mirpur_ten_has_one_canonical_id(gazetteer, text):
    result = gazetteer.extract(text)
    assert result["canonical_id"] == "dhaka_mirpur_10"
    assert result["confidence"] == 1
    assert result["parent_id"] == "dhaka_mirpur"
    assert result["candidate_ids"] == ["dhaka_mirpur_10"]


@pytest.mark.parametrize("text", ["উত্তরা সেক্টর ৪ বিদ্যুৎ নেই", "Uttara sector 4 power failure", "uttara-sector-4 biddut nai"])
def test_uttara_sector_is_specific_not_parent(gazetteer, text):
    assert gazetteer.extract(text)["canonical_id"] == "dhaka_uttara_sector_4"


def test_aliases_are_whole_tokens_not_substring_matches(gazetteer):
    assert gazetteer.extract("notmirpur power failure")["canonical_id"] is None
    assert gazetteer.extract("bananiland no water")["canonical_id"] is None
    assert gazetteer.extract("see https://mirpur10.example.com report")["canonical_id"] is None


def test_fuzzy_typo_can_match_a_known_area(gazetteer):
    result = gazetteer.extract("mohammadpr e pani nai")
    assert result["canonical_id"] == "dhaka_mohammadpur"
    assert result["method"] == "fuzzy"
    assert 0.9 <= result["confidence"] < 1


def test_fuzzy_numeric_guard_never_changes_neighborhood_number(gazetteer):
    result = gazetteer.extract("mirpr 11 e pani nai")
    assert result["canonical_id"] != "dhaka_mirpur_10"
    assert "dhaka_mirpur_10" not in result["candidate_ids"]
    assert gazetteer.extract("mirpur 99 e pani nai")["canonical_id"] == "dhaka_mirpur"


def test_fuzzy_refines_exact_parent_when_specific_alias_has_typo(gazetteer):
    result = gazetteer.extract("uttara secton 4 e pani nai")
    assert result["canonical_id"] == "dhaka_uttara_sector_4"
    assert result["method"] == "fuzzy"


def test_multiple_unrelated_locations_are_ambiguous(gazetteer):
    result = gazetteer.extract("Mirpur 10 and Mohammadpur both have no water")
    assert result["canonical_id"] is None
    assert result["ambiguous"] is True
    assert result["candidate_ids"] == ["dhaka_mirpur_10", "dhaka_mohammadpur"]


def test_explicit_metadata_precedence_retains_conflict(gazetteer):
    result = gazetteer.extract("Mirpur 10 has no water", explicit_location="Mohammadpur")
    assert result["canonical_id"] == "dhaka_mohammadpur"
    assert result["method"] == "conflict"
    assert result["conflict"] is True
    assert result["source"] == "explicit"
    assert set(result["candidate_ids"]) == {"dhaka_mohammadpur", "dhaka_mirpur_10"}


def test_unresolved_metadata_can_fall_back_to_text(gazetteer):
    result = gazetteer.extract("Mirpur 10 outage", explicit_location="unmapped neighborhood")
    assert result["canonical_id"] == "dhaka_mirpur_10"
    assert result["explicit_unresolved"] is True


def test_parent_child_known_compatible_but_not_exact_match(gazetteer):
    parent, child = gazetteer.extract("Mirpur"), gazetteer.extract("Mirpur 10")
    features = location_pair_features(parent, child)
    assert features == {"loc_match": 0, "loc_distance_flag": 0, "loc_missing": 0,
                        "loc_conflict": 0, "loc_compatible": 1}


def test_location_missing_is_not_a_match_or_known_distance(gazetteer):
    unknown = gazetteer.extract("pani nai")
    features = location_pair_features(unknown, unknown)
    assert features["loc_missing"] == 1
    assert features["loc_match"] == features["loc_distance_flag"] == 0
    known = location_pair_features(gazetteer.extract("Mirpur 10"), gazetteer.extract("Mirpur 11"))
    assert known["loc_distance_flag"] == 1
    assert known["loc_match"] == 0


def test_json_extensibility_and_bad_alias_or_hierarchy(tmp_path):
    path = tmp_path / "gazetteer.json"
    payload = {"schema_version": 1, "areas": [
        {"id": "parent", "parent_id": None, "aliases": ["new area"]},
        {"id": "child", "parent_id": "parent", "aliases": ["new area 3", "নতুন এলাকা ৩"]},
    ]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert LocationGazetteer(path).extract("নতুন এলাকা ৩")["canonical_id"] == "child"
    payload["areas"][1]["aliases"] = ["new-area"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Ambiguous normalized alias"):
        LocationGazetteer(path)
    payload["areas"][1]["aliases"] = ["child area"]
    payload["areas"][0]["parent_id"] = "child"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Cyclic"):
        LocationGazetteer(path)


def test_fuzzy_equal_candidates_remain_unknown_and_explicit_exact_is_not_false_conflict(tmp_path):
    path = tmp_path / "gazetteer.json"
    path.write_text(json.dumps({"schema_version": 1, "areas": [
        {"id": "a", "parent_id": None, "aliases": ["abcityaa"]},
        {"id": "b", "parent_id": None, "aliases": ["abcityab"]},
    ]}), encoding="utf-8")
    gazetteer = LocationGazetteer(path, fuzzy_threshold=85)
    result = gazetteer.extract("abcityac")
    assert result["canonical_id"] is None
    assert result["candidate_ids"] == ["a", "b"]
    assert result["ambiguous"] is True
    exact = gazetteer.extract("abcityaa", explicit_location="abcityaa")
    assert exact["canonical_id"] == "a"
    assert exact["conflict"] is False


@pytest.mark.parametrize("text,hours", [
    ("৩ দিন ধরে পানি নেই", 72), ("3 din dhore pani nai", 72),
    ("No water for three days", 72), ("তিন দিন ধরে পানি নেই", 72),
    ("tin din dhore pani nai", 72), ("বিদ্যুৎ নেই ২ ঘণ্টা ধরে", 2),
    ("No power for two hours", 2), ("dui ghonta dhore biddut nai", 2),
    ("for 0.5 days", 12), ("for 30 minutes", 0.5), ("for one week", 168),
    ("for zero hours", None),
])
def test_multilingual_duration_hours_and_days(text, hours):
    result = extract_time_features(text)
    assert result["duration_hours"] == hours
    assert result["duration_days"] == (hours / 24 if hours is not None else None)


def test_ongoing_duration_uses_supplied_utc_anchor_not_current_clock():
    result = extract_time_features("৩ দিন ধরে পানি নেই", "2026-10-29T12:00:00+06:00")
    assert result["timestamp_utc"] == "2026-10-29T06:00:00Z"
    assert result["interval_start_utc"] == "2026-10-26T06:00:00Z"
    assert result["interval_end_utc"] == "2026-10-29T06:00:00Z"
    assert result["ongoing"] is True


def test_naive_timestamp_is_documented_as_utc_and_missing_is_unknown():
    value = extract_time_features("for 2 hours", "2026-10-29T12:00:00")
    assert value["timestamp_utc"] == "2026-10-29T12:00:00Z"
    missing = extract_time_features("pani nai")
    assert missing["duration_hours"] is None
    assert missing["timestamp_utc"] is None
    assert missing["interval_start_utc"] is None


def test_bare_duration_is_not_automatically_an_ongoing_interval():
    value = extract_time_features("3 days water problem", "2026-10-29")
    assert value["duration_hours"] == 72
    assert value["ongoing"] is False
    assert value["interval_start_utc"] is None


@pytest.mark.parametrize("text", ["three days ago", "after 3 days", "৩ দিন আগে", "৩-৪ দিন ধরে", "for 2 days and 5 hours", "-3 days"])
def test_relative_points_ranges_and_distinct_durations_not_imputed(text):
    assert extract_time_features(text)["duration_hours"] is None


def test_equivalent_repeated_durations_are_consistent_not_ambiguous():
    value = extract_time_features("for 3 days, 72 hours")
    assert value["duration_hours"] == 72
    assert value["errors"] == []


def test_invalid_timestamp_fails_clearly():
    with pytest.raises(ValueError, match="ISO-8601"):
        extract_time_features("for 3 days", "not-a-timestamp")


def test_enormous_duration_stays_unknown_instead_of_infinite_model_feature():
    result = extract_time_features("for " + "9" * 500 + " days")
    assert result["duration_hours"] is None
    assert result["errors"]


def test_time_pair_gap_and_inferred_overlap_have_explicit_units():
    a = extract_time_features("for three days", "2026-10-29T12:00:00Z")
    b = extract_time_features("3 din dhore", "2026-10-30T12:00:00Z")
    features = time_pair_features(a, b)
    assert features["time_gap"] == 24
    assert features["duration_gap"] == 0
    assert features["interval_overlap"] == 1
    assert features["interval_gap"] == 0
    assert features["time_missing"] == features["duration_missing"] == features["interval_missing"] == 0


def test_time_missing_vs_real_zero_is_distinguishable():
    unknown = extract_time_features("no water")
    features = time_pair_features(unknown, unknown)
    assert features["time_gap"] == features["duration_gap"] == 0
    assert features["time_missing"] == features["duration_missing"] == features["interval_missing"] == 1
    zero = extract_time_features("for 0 hours", "2026-10-29T12:00:00Z")
    known = time_pair_features(zero, zero)
    assert known["duration_gap"] == known["time_gap"] == known["duration_missing"] == known["time_missing"] == 0
