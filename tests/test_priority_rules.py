"""Transparent priority heuristics, multilingual evidence and config safety."""

from copy import deepcopy
import json

import pytest
import yaml

from ml.priority_rules import DEFAULT_CONFIG_PATH, FEATURE_NAMES, PriorityRules


def complaint(**updates):
    return {"text": "Service problem", "lang": "en", "category": "Water", "severity": "Low", **updates}


def custom_rules(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return PriorityRules(path)


def test_default_breakdown_adds_to_score_and_has_stable_ltr_features():
    rules = PriorityRules()
    result = rules.score(complaint(text="Fire outside hospital for three days", category="Healthcare", severity="Medium", duplicate_group_size=3))
    assert rules.feature_names == FEATURE_NAMES
    assert list(result["feature_values"]) == list(FEATURE_NAMES)
    assert result["feature_values"] == {"severity": 3, "category": 3, "duration": 2, "duplicate_reports": 1, "vulnerable": 1, "life_risk": 3}
    assert result["priority_score"] == sum(row["contribution"] for row in result["rule_breakdown"])
    assert result["priority_label"] == "urgent"
    assert len(result["config_sha256"]) == 64
    assert result["duration_hours"] == 72
    assert result["duration_days"] == 3
    assert result["group_size_interpretation"] == "number of reports, not number of people affected"
    json.dumps(result, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize("text,lang", [
    ("হাসপাতালে আগুন, ৩ দিন ধরে সমস্যা", "bn"),
    ("Fire outside hospital for three days", "en"),
    ("haspatale agun 3 din dhore somossa", "banglish"),
])
def test_equivalent_multilingual_evidence_gives_equivalent_rule_components(text, lang):
    result = PriorityRules().score(complaint(text=text, lang=lang))
    assert result["feature_values"]["duration"] == 2
    assert result["feature_values"]["life_risk"] == 3
    assert result["feature_values"]["vulnerable"] == 1
    assert result["matched_keywords"]["life_risk"][0]["concept"] == "fire"


@pytest.mark.parametrize("text,concept", [
    ("A child reports electric shock", "electric_shock"),
    ("শিশুরা বিদ্যুৎস্পৃষ্ট হচ্ছে", "electric_shock"),
    ("baccha current shock kheyeche", "electric_shock"),
    ("gas-leak outside school", "gas_leak"),
    ("স্কুলে গ্যাস লিক হচ্ছে", "gas_leak"),
    ("school e gas lik hocche", "gas_leak"),
    ("elderly people affected by flooding", "flood"),
    ("বৃদ্ধ বন্যায় আটকা", "flood"),
    ("boyoshko manush bonna te atka", "flood"),
])
def test_life_risk_and_vulnerable_aliases(text, concept):
    result = PriorityRules().score(complaint(text=text))
    assert result["feature_values"]["life_risk"] == 3
    assert result["feature_values"]["vulnerable"] == 1
    assert any(row["concept"] == concept for row in result["matched_keywords"]["life_risk"])


def test_keywords_do_not_match_english_or_bangla_subwords_or_url_content():
    result = PriorityRules().score(complaint(text="hospitality childish firework aguner https://example.com/fire/hospital"))
    assert result["feature_values"]["life_risk"] == 0
    assert result["feature_values"]["vulnerable"] == 0
    bn = PriorityRules().score(complaint(text="অগ্নিকাণ্ডজনিত হাসপাতালসমূহ", lang="bn"))
    assert bn["feature_values"]["life_risk"] == 0
    assert bn["feature_values"]["vulnerable"] == 0


@pytest.mark.parametrize("text,lang", [("No fire outside the building", "en"), ("আগুন নেই এখানে", "bn"), ("agun nai ekhane", "banglish"), ("No active fire outside", "en")])
def test_adjacent_negation_does_not_add_life_risk_boost(text, lang):
    result = PriorityRules().score(complaint(text=text, lang=lang))
    assert result["feature_values"]["life_risk"] == 0
    assert result["negated_keywords"]["life_risk"]


def test_separate_positive_evidence_is_not_cancelled_by_a_negated_mention():
    result = PriorityRules().score(complaint(text="No fire. Electric shock at hospital."))
    assert result["feature_values"]["life_risk"] == 3
    assert [row["concept"] for row in result["matched_keywords"]["life_risk"]] == ["electric_shock"]


def test_concept_and_category_caps_prevent_alias_stacking():
    result = PriorityRules().score(complaint(text="Fire আগুন agun fire; flood; gas leak. Hospital school children elderly."))
    assert len(result["matched_keywords"]["life_risk"]) == 3
    assert result["feature_values"]["life_risk"] == 3
    assert result["feature_values"]["vulnerable"] == 1
    assert len(result["matched_keywords"]["vulnerable"]) == 4


@pytest.mark.parametrize("hours,expected", [(0, 0), (24, 0), (25, 1), (48, 1), (49, 2), (120, 2)])
def test_duration_bands_are_strict_and_only_highest_band_applies(hours, expected):
    result = PriorityRules().score(complaint(text=f"Service unavailable for {hours} hours"))
    assert result["duration_hours"] == hours
    assert result["feature_values"]["duration"] == expected


@pytest.mark.parametrize("count,expected", [(1, 0), (2, 0), (3, 1), (9, 1), (10, 2), (100, 2)])
def test_duplicate_report_bands_are_not_a_people_affected_estimate(count, expected):
    result = PriorityRules().score(complaint(duplicate_group_size=count))
    assert result["feature_values"]["duplicate_reports"] == expected
    evidence = next(row for row in result["rule_breakdown"] if row["name"] == "duplicate_reports")["evidence"][0]
    assert evidence["report_count"] == count
    assert "not unique people" in evidence["interpretation"]


@pytest.mark.parametrize("count", [0, -1, 1.5, "3", True, None])
def test_invalid_duplicate_counts_are_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        PriorityRules().score(complaint(duplicate_group_size=count))


def test_unknown_labels_and_missing_duration_are_explicit_not_imputed():
    result = PriorityRules().score(complaint(severity="emergency", category="other"))
    assert result["feature_values"]["severity"] == result["feature_values"]["category"] == 0
    assert set(result["missing_fields"]) == {"severity", "category", "duration"}
    assert result["duration_hours"] is None
    assert len(result["warnings"]) == 3
    result = PriorityRules().score(complaint(text="Outage for 2-3 days"))
    assert result["duration_hours"] is None
    assert any("ambiguous" in note for note in result["warnings"])


def test_source_labels_are_case_insensitive_and_healthcare_has_higher_category_points_than_road():
    rules = PriorityRules()
    high = rules.score(complaint(severity=" hIgH ", category="hEaLtHcArE"))
    road = rules.score(complaint(severity="High", category="Road & Transport"))
    assert high["feature_values"]["severity"] == 5
    assert high["feature_values"]["category"] > road["feature_values"]["category"]


def test_custom_weights_caps_keywords_and_bands_are_read_from_yaml(tmp_path):
    def mutate(raw):
        raw["weights"]["severity"] = 2
        raw["keyword_rules"]["life_risk"]["max_concepts"] = 2
        raw["keyword_rules"]["life_risk"]["concepts"]["collapse"] = ["building collapse"]
        raw["max_score"] = 10
        raw["priority_bands"] = [{"min_score": 0, "label": "routine"}, {"min_score": 10, "label": "expert_review"}]
    rules = custom_rules(tmp_path, mutate)
    result = rules.score(complaint(text="Fire and building collapse", severity="High"))
    assert result["feature_values"]["life_risk"] == 6
    assert result["unclipped_score"] > 10
    assert result["priority_score"] == 10
    assert result["priority_label"] == "expert_review"
    assert result["rule_breakdown"][-1]["name"] == "score_cap"
    assert sum(row["contribution"] for row in result["rule_breakdown"]) == 10
    assert rules.config_sha256 == result["config_sha256"]


def test_score_does_not_mutate_input_and_is_deterministic():
    original = complaint(text="Gas leak at school for three days", timestamp="2026-10-30T12:00:00+06:00")
    saved = deepcopy(original)
    rules = PriorityRules()
    assert rules.score(original) == rules.score(original)
    assert original == saved


@pytest.mark.parametrize("text", ["", "   ", None, 3, "https://example.com"])
def test_invalid_or_cleaned_empty_text_rejected(text):
    with pytest.raises(ValueError, match="text"):
        PriorityRules().score(complaint(text=text))


def test_invalid_timestamp_and_language_rejected():
    with pytest.raises(ValueError, match="timestamp"):
        PriorityRules().score(complaint(timestamp="yesterday"))
    with pytest.raises(ValueError, match="lang"):
        PriorityRules().score(complaint(lang="guess"))


@pytest.mark.parametrize("mutation,fragment", [
    (lambda raw: raw.update(schema_version=True), "schema_version"),
    (lambda raw: raw.update(unknown=1), "Unknown priority"),
    (lambda raw: raw["weights"].update(life_risk=-1), "nonnegative"),
    (lambda raw: raw["weights"].update(life_risk=float("nan")), "finite"),
    (lambda raw: raw["weights"].pop("severity"), "exactly"),
    (lambda raw: raw["severity_points"].update(LOW=3), "case-insensitive"),
    (lambda raw: raw["duplicate_bands"][0].update(min_reports=1.5), "positive integer"),
    (lambda raw: raw["duration_bands"][0].update(strict="yes"), "boolean"),
    (lambda raw: raw["duration_bands"][0].update(min_hours=48), "unique"),
    (lambda raw: raw["keyword_rules"]["life_risk"].update(max_concepts=True), "positive integer"),
    (lambda raw: raw["keyword_rules"]["life_risk"]["concepts"].update(other=["fire"]), "Ambiguous"),
    (lambda raw: raw["priority_bands"][0].update(min_score=1), "start"),
    (lambda raw: raw["priority_bands"][1].update(label="low"), "unique"),
    (lambda raw: raw.update(max_score=0), "positive"),
])
def test_invalid_config_is_rejected(tmp_path, mutation, fragment):
    with pytest.raises(ValueError, match=fragment):
        custom_rules(tmp_path, mutation)


def test_unsafe_yaml_cannot_execute_python(tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:os.system ['echo unsafe']", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load"):
        PriorityRules(path)


def test_missing_config_error_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="Cannot load priority rule YAML"):
        PriorityRules(tmp_path / "missing.yaml")
