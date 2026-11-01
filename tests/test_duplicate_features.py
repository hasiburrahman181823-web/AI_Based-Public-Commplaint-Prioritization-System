"""Shared duplicate feature invariants and inference/training input handling."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations_with_replacement

import numpy as np
import pytest

from ml.duplicate_features import (
    BASE_FEATURES, FEATURE_SETS, LANGUAGE_PAIRS, LANGUAGES,
    feature_matrix, pair_feature_dict, preprocess_complaint,
)
from ml.location import LocationGazetteer


@pytest.fixture(scope="module")
def gazetteer():
    return LocationGazetteer()


def complaint(text="PANY NEI ৩ din dhore", lang="banglish", **metadata):
    return preprocess_complaint({"text": text, "lang": lang, **metadata})


def test_preprocessing_copies_input_and_retains_raw_spelling_for_evidence():
    source = {"text": "  PANY NEI ৩ DIN DHORE https://example.com  ", "lang": "banglish",
              "id": " 0001 ", "original_id": " group-1 ", "category": " water ",
              "location": None, "timestamp": "2026-10-29T12:00:00+06:00"}
    before = deepcopy(source)
    result = preprocess_complaint(source)
    assert source == before
    assert result is not source
    assert result["text"] == "pani nai 3 din dhore"
    assert result["raw_text"] == "pany nei 3 din dhore"
    assert result["id"] == "0001" and result["original_id"] == "group-1"
    assert result["category"] == "water" and result["location"] == ""
    assert result["timestamp"] == source["timestamp"]


def test_explicit_raw_evidence_is_kept_separate_from_normalized_model_text(gazetteer):
    a = complaint("pani nai", raw_text="Mirpur-10 এ ৩ দিন ধরে পানি নেই", timestamp="2026-10-29T12:00:00+06:00")
    b = complaint("No water for three days at Mirpur 10", "en", timestamp="2026-10-29T06:00:00Z")
    values = pair_feature_dict(a, b, 0.4, gazetteer)
    assert a["text"] == "pani nai"
    assert values["loc_match"] == 1
    assert values["duration_gap"] == 0 and values["duration_missing"] == 0
    assert values["time_gap"] == 0 and values["time_missing"] == 0
    assert values["interval_overlap"] == 1 and values["interval_missing"] == 0


def test_identifiers_pair_types_and_labels_cannot_change_features(gazetteer):
    a = complaint(location="Mirpur 10", category="water", timestamp="2026-10-29T06:00:00Z",
                  id="source-a", original_id="same-incident", label=1, pair_type="positive_cross_language")
    b = complaint("No water for three days", "en", location="Mirpur-10", category="water",
                  timestamp="2026-10-29T06:00:00Z", id="source-b", original_id="same-incident")
    expected = pair_feature_dict(a, b, 0.7, gazetteer)
    changed_a, changed_b = dict(a), dict(b)
    changed_a.update(id="different-a", original_id="unrelated-a", label=0, pair_type="hard_negative", severity="high", split="test")
    changed_b.update(id="different-b", original_id="unrelated-b", label=0, pair_type="easy_negative", severity="low", split="train")
    assert pair_feature_dict(changed_a, changed_b, 0.7, gazetteer) == expected
    forbidden = {"id", "id_a", "id_b", "original_id", "original_id_a", "original_id_b", "pair_id", "pair_type", "label", "severity", "split"}
    assert set(expected).isdisjoint(forbidden)
    assert all(set(names).isdisjoint(forbidden) for names in FEATURE_SETS.values())


def test_pair_features_are_symmetric_despite_locations_dates_and_language_order(gazetteer):
    a = complaint("Mirpur 10 e 3 din dhore pani nai", "banglish", category="water", timestamp="2026-10-29T06:00:00Z")
    b = complaint("No water in Mohammadpur for two hours", "en", category="water", timestamp="2026-10-30T18:00:00+06:00")
    forward = pair_feature_dict(a, b, 0.3, gazetteer)
    reverse = pair_feature_dict(b, a, 0.3, gazetteer)
    assert forward == reverse
    assert forward["time_gap"] == 30
    assert forward["duration_gap"] == 70
    assert forward["loc_distance_flag"] == 1
    assert forward["lang_en_banglish"] == 1
    matrix = feature_matrix([a, b], [b, a], [0.3, 0.3], FEATURE_SETS["P2"], gazetteer)
    np.testing.assert_array_equal(matrix[0], matrix[1])


@pytest.mark.parametrize("left,right", list(combinations_with_replacement(LANGUAGES, 2)))
def test_language_pair_is_one_symmetric_one_hot_block(gazetteer, left, right):
    a, b = complaint("test complaint", left), complaint("another complaint", right)
    language = f"{left}-{right}"
    assert language in LANGUAGE_PAIRS
    for sides in ((a, b), (b, a)):
        values = pair_feature_dict(*sides, 0.2, gazetteer)
        one_hot = {pair: values["lang_" + pair.replace("-", "_")] for pair in LANGUAGE_PAIRS}
        assert sum(one_hot.values()) == 1
        assert one_hot[language] == 1


def test_category_source_none_masks_only_category_covariates(gazetteer):
    a = complaint(category="WATER", location="Mirpur 10")
    b = complaint("No water", "en", category="water", location="মিরপুর ১০")
    provided = pair_feature_dict(a, b, 0.5, gazetteer)
    masked = pair_feature_dict(a, b, 0.5, gazetteer, category_source="none")
    assert provided["category_match"] == 1 and provided["category_missing"] == 0
    assert masked["category_match"] == 0 and masked["category_missing"] == 1
    for name in provided:
        if not name.startswith("category_"):
            assert provided[name] == masked[name]


def test_missing_category_is_not_assumed_equal(gazetteer):
    values = pair_feature_dict(complaint(category=""), complaint("No water", "en"), 0.5, gazetteer)
    assert values["category_match"] == 0 and values["category_missing"] == 1


def test_canonical_location_equivalence_and_unknowns_have_distinct_features(gazetteer):
    known = pair_feature_dict(complaint(location="মিরপুর ১০"), complaint("No water", "en", location="Mirpur-10"), 0.5, gazetteer)
    unknown = pair_feature_dict(complaint(), complaint("No water", "en"), 0.5, gazetteer)
    assert known["loc_match"] == 1 and known["loc_missing"] == 0 and known["loc_distance_flag"] == 0
    assert unknown["loc_match"] == 0 and unknown["loc_missing"] == 1 and unknown["loc_distance_flag"] == 0


def test_explicit_text_location_conflict_is_not_hidden(gazetteer):
    a = complaint("Mirpur 10 e pani nai", location="Mohammadpur")
    b = complaint("No water in Mohammadpur", "en")
    values = pair_feature_dict(a, b, 0.5, gazetteer)
    assert values["loc_match"] == 1
    assert values["loc_conflict"] == 1


def test_missing_duration_and_anchor_are_distinct_from_real_zero_gap(gazetteer):
    missing = pair_feature_dict(complaint("pani nai"), complaint("No water", "en"), 0.5, gazetteer)
    a = complaint("for 0 hours", "en", timestamp="2026-10-29T12:00:00+06:00")
    b = complaint("0 ghonta dhore", timestamp="2026-10-29T06:00:00Z")
    known = pair_feature_dict(a, b, 0.5, gazetteer)
    assert missing["time_gap"] == known["time_gap"] == 0
    assert missing["duration_gap"] == known["duration_gap"] == 0
    assert missing["time_missing"] == missing["duration_missing"] == 1
    assert known["time_missing"] == known["duration_missing"] == 0


@pytest.mark.parametrize("name", list(FEATURE_SETS))
def test_matrix_shape_empty_shape_and_order_are_stable(gazetteer, name):
    names = FEATURE_SETS[name]
    a, b = complaint(location="Mirpur 10"), complaint("No water", "en", location="Mirpur 10")
    matrix = feature_matrix([a], [b], [0.8], names, gazetteer)
    assert matrix.shape == (1, len(names)) and matrix.dtype == np.float64
    values = pair_feature_dict(a, b, 0.8, gazetteer)
    np.testing.assert_array_equal(matrix[0], [values[field] for field in names])
    assert feature_matrix([], [], [], names, gazetteer).shape == (0, len(names))
    assert np.isfinite(matrix).all()


def test_feature_sets_are_nested_controls_for_location_and_time():
    assert FEATURE_SETS["C0"] == BASE_FEATURES
    assert FEATURE_SETS["P1"][:len(BASE_FEATURES)] == BASE_FEATURES
    assert FEATURE_SETS["P2"][:len(FEATURE_SETS["P1"])] == FEATURE_SETS["P1"]
    assert FEATURE_SETS["B1"] == FEATURE_SETS["B2"] == ["cosine"]


@pytest.mark.parametrize("record", [None, [], {}, {"text": "", "lang": "en"}, {"text": "test"}, {"text": "test", "lang": "english"}, {"text": 1, "lang": "en"}, {"text": "https://example.com", "lang": "en"}, {"text": "test", "lang": "en", "raw_text": None}])
def test_invalid_complaints_fail_explicitly(record):
    with pytest.raises(ValueError):
        preprocess_complaint(record)


@pytest.mark.parametrize("field", ["id", "original_id", "category", "location", "timestamp"])
def test_numeric_metadata_is_not_silently_coerced(field):
    with pytest.raises(ValueError, match=field):
        complaint(**{field: 123})


def test_timestamp_validation_is_identical_even_for_cosine_only_features(gazetteer):
    with pytest.raises(ValueError, match="ISO-8601"):
        complaint(timestamp="tomorrow")
    a, b = complaint(), complaint("No water", "en")
    a["timestamp"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="ISO-8601"):
        feature_matrix([a], [b], [0.5], ["cosine"], gazetteer)


@pytest.mark.parametrize("cosine", [float("nan"), float("inf"), -1.1, 1.1, "not-a-score"])
def test_invalid_cosines_fail(gazetteer, cosine):
    with pytest.raises(ValueError, match="cosine|convert"):
        pair_feature_dict(complaint(), complaint(), cosine, gazetteer)


@pytest.mark.parametrize("names", [[], ["cosine", "cosine"], ["pair_type"], ["unknown"]])
def test_invalid_feature_names_fail(gazetteer, names):
    with pytest.raises(ValueError, match="feature_names"):
        feature_matrix([], [], [], names, gazetteer)


@pytest.mark.parametrize("left,right,scores", [([{}], [], [0.5]), ([], [], [0.5]), ([], [], [[0.5]])])
def test_misaligned_or_nondimensional_pair_arrays_fail(gazetteer, left, right, scores):
    with pytest.raises(ValueError, match="same length"):
        feature_matrix(left, right, scores, ["cosine"], gazetteer)


def test_unknown_category_source_and_pair_language_fail(gazetteer):
    with pytest.raises(ValueError, match="category_source"):
        pair_feature_dict(complaint(), complaint(), 0.5, gazetteer, "predicted_without_mapping")
    with pytest.raises(ValueError, match="lang"):
        pair_feature_dict({"text": "test", "lang": "unknown"}, complaint(), 0.5, gazetteer)
