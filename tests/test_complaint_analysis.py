"""Synthetic integration of cached classifier explanations and complaint analysis.

The fixtures fit small calibrated classifiers to artificial examples. Their
predictions verify interfaces and invariants, not research performance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from ml import analysis
from ml.classifier_runtime import ClassifierRuntime, file_sha256
from ml.normalize import DEFAULT_VARIANTS_PATH
from ml.priority_ranker import fit_priority_ranker
from ml.priority_rules import DEFAULT_CONFIG_PATH, PriorityRules
from ml.uncertainty import CONFIDENCE_DEFINITION
from ml_training.modeling import build_text_estimator, calibrate_prefit


def make_classifier(root: Path, target: str, name: str = "char_svm", category_names=None) -> Path:
    category_names = category_names or ("Water", "Healthcare")
    terms = {
        category_names[0]: "water tap pipe dry pani nai",
        category_names[1]: "medical care clinic hospital health",
    }
    if len(category_names) > 2:
        terms[category_names[2]] = "road transport traffic pothole bridge"
    train_texts, train_labels, val_texts, val_labels = [], [], [], []
    for category, phrase in terms.items():
        label = category if target == "category" else "Low" if category == category_names[0] else "High"
        for index in range(5):
            train_texts.append(f"{phrase} training {index}")
            train_labels.append(label)
        for index in range(2):
            val_texts.append(f"{phrase} validation {index}")
            val_labels.append(label)
    base = build_text_estimator(name, seed=23).fit(train_texts, train_labels)
    calibrated = calibrate_prefit(base, val_texts, val_labels)
    labels = sorted(set(train_labels)) if target == "category" else ["Low", "High"]
    resource = root / "resources" / "banglish_variants.json"
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_bytes(DEFAULT_VARIANTS_PATH.read_bytes())
    bundle = {
        "kind": "banglapsg_classifier", "schema_version": 1, "target": target,
        "model_name": name, "feature_type": "tfidf", "estimator": calibrated,
        "labels": labels, "calibration": "held-out validation sigmoid fixture",
        "artifact_root_relative": "..", "encoder_dir": None,
        "variants_file": "resources/banglish_variants.json",
        "normalization_dictionary_sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
    }
    path = root / target / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


@pytest.fixture(scope="module")
def classifiers(tmp_path_factory):
    root = tmp_path_factory.mktemp("analysis-classifiers")
    return {"root": root, "category": make_classifier(root, "category"),
            "severity": make_classifier(root, "severity", "word_lr"),
            "multiclass": make_classifier(root / "multi", "category", "word_lr", ("Water", "Healthcare", "Road & Transport"))}


def make_policy(path: Path, classifiers, threshold: float = 0, status: str = "selected") -> Path:
    payload = {
        "kind": "priority_review_policy", "schema_version": 1,
        "threshold": threshold, "status": status, "confidence_definition": CONFIDENCE_DEFINITION,
        "classifier_fingerprints": {target: file_sha256(classifiers[target]) for target in ("category", "severity")},
        "classifier_resource_fingerprints": {target: ClassifierRuntime(classifiers[target], expected_target=target).resource_fingerprint
                                             for target in ("category", "severity")},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def analyzer_options(**updates):
    options = {"category_model": None, "severity_model": None, "review_policy": None,
               "ranker": None, "duplicate_model": None}
    options.update(updates)
    return options


@pytest.mark.parametrize("target,text", [("category", "water tap pipe problem"), ("severity", "medical clinic problem")])
def test_binary_explanations_sum_to_selected_signed_base_margin(classifiers, target, text):
    runtime = ClassifierRuntime(classifiers[target], expected_target=target)
    result = runtime.predict(text, "en", top_k=10000)
    detail = result["explanation"]
    base = runtime.base
    raw = float(np.asarray(base.decision_function([result["text_normalized"]])).reshape(-1)[0])
    base_classes = list(base.named_steps["classifier"].classes_)
    expected = raw if result["prediction"] == base_classes[1] else -raw
    assert detail["base_margin"] == pytest.approx(expected)
    assert detail["intercept"] + detail["all_feature_contributions_sum"] == pytest.approx(expected)
    assert sum(row["contribution"] for row in detail["top_features"]) + detail["intercept"] == pytest.approx(expected)
    assert all(row["contribution"] == pytest.approx(row["feature_value"] * row["coefficient"]) for row in detail["top_features"])
    assert "not calibrated probability" in detail["semantics"]
    assert detail["target_class"] == result["prediction"]
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["confidence"] == pytest.approx(max(result["probabilities"].values()))


def test_multiclass_word_explanation_matches_chosen_class_margin(classifiers):
    runtime = ClassifierRuntime(classifiers["multiclass"])
    result = runtime.predict("road transport bridge problem", "en", top_k=10000)
    classes = list(runtime.base.named_steps["classifier"].classes_)
    margin = runtime.base.decision_function([result["text_normalized"]])[0, classes.index(result["prediction"])]
    explanation = result["explanation"]
    assert explanation["base_margin"] == pytest.approx(margin)
    assert explanation["feature_kind"] == "word_token"
    assert explanation["top_terms"]


def test_runtime_loads_model_once_and_batches_without_reloading(classifiers, monkeypatch):
    calls = []
    original = joblib.load

    def count_load(path, *args, **kwargs):
        calls.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(joblib, "load", count_load)
    runtime = ClassifierRuntime(classifiers["category"])
    first = runtime.predict("pany nei", "banglish")
    repeated = runtime.predict_many(["pany nei", "hospital care"], ["banglish", "en"])
    assert calls == [classifiers["category"]]
    assert repeated[0]["prediction"] == first["prediction"]
    assert repeated[0]["probabilities"] == first["probabilities"]
    assert repeated[0]["text_normalized"] == "pani nai"
    assert runtime.predict_many([], []) == []


def test_runtime_resources_are_portable_and_changed_dictionary_is_rejected(classifiers, tmp_path):
    relocated = tmp_path / "copied"
    shutil.copytree(classifiers["root"], relocated)
    original = ClassifierRuntime(classifiers["category"]).predict("pany nei", "banglish")
    model = relocated / "category" / "char_svm.joblib"
    assert ClassifierRuntime(model).predict("pany nei", "banglish") == original
    (relocated / "resources" / "banglish_variants.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint changed"):
        ClassifierRuntime(model)


def test_invalid_classifier_bundle_or_wrong_target_is_not_silently_used(classifiers, tmp_path):
    wrong = tmp_path / "wrong.joblib"
    joblib.dump({"kind": "different_model"}, wrong)
    with pytest.raises(ValueError, match="Phase 3"):
        ClassifierRuntime(wrong)
    with pytest.raises(ValueError, match="target"):
        ClassifierRuntime(classifiers["category"], expected_target="severity")


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_explanation_limits_are_rejected(classifiers, top_k):
    with pytest.raises(ValueError, match="top_k"):
        ClassifierRuntime(classifiers["category"]).predict("water tap", "en", top_k=top_k)


def test_runtime_rejects_empty_cleaned_text_or_misaligned_languages(classifiers):
    runtime = ClassifierRuntime(classifiers["category"])
    with pytest.raises(ValueError, match="align"):
        runtime.predict_many(["water"], [])
    with pytest.raises(ValueError, match="empty"):
        runtime.predict("https://example.com", "en")


def test_no_models_returns_no_fabricated_classes_or_confidence_and_rules_still_work():
    service = analysis.PriorityAnalyzer(**analyzer_options())
    result = service.analyze_complaint("হাসপাতালে আগুন, ৩ দিন ধরে সমস্যা", "মিরপুর ১০", "2026-10-30T10:00:00+06:00")
    assert result["category"] is None and result["severity"] is None
    assert result["confidence"] is None
    assert result["classification_probabilities"] == {}
    assert result["review_required"] and result["review_status"] == "needs_human_review"
    assert "classification_models_unavailable:category,severity" in result["review_reasons"]
    assert result["area_id"] == "dhaka_mirpur_10"
    breakdown = {row["name"]: row for row in result["explanation"]["rule_breakdown"]}
    assert breakdown["duration"]["contribution"] == 2
    assert breakdown["life_risk"]["contribution"] == 3
    assert result["ranking_score"] == result["priority_score"]
    assert result["ranking_method"] == "rules"
    assert "not unique people" in result["duplicate_group_size_semantics"]
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_missing_policy_preserves_calibrated_predictions_but_requires_review(classifiers):
    service = analysis.PriorityAnalyzer(**analyzer_options(category_model=classifiers["category"], severity_model=classifiers["severity"]))
    result = service.analyze_record({"text": "water tap pipe dry", "lang": "en"})
    assert result["category"] in ("Water", "Healthcare")
    assert result["severity"] in ("Low", "High")
    assert 0 <= result["confidence"] <= 1
    assert result["review_required"]
    assert "validated_review_policy_unavailable" in result["review_reasons"]
    assert result["explanation"]["category"]["method"] == "linear_base_margin"


def test_bound_policy_accepts_supported_confidence_and_high_threshold_routes_review(classifiers, tmp_path):
    policy = make_policy(tmp_path / "accept.json", classifiers, threshold=0)
    service = analysis.PriorityAnalyzer(**analyzer_options(category_model=classifiers["category"], severity_model=classifiers["severity"], review_policy=policy))
    result = service.analyze_record({"text": "water tap dry", "lang": "en"})
    assert not result["review_required"] and result["review_status"] == "scored"
    maximums = [max(result["classification_probabilities"][target].values()) for target in ("category", "severity")]
    assert result["confidence"] == pytest.approx(min(maximums))
    assert "not a calibrated joint" in result["confidence_definition"]
    strict = make_policy(tmp_path / "strict.json", classifiers, threshold=1)
    strict_service = analysis.PriorityAnalyzer(**analyzer_options(category_model=classifiers["category"], severity_model=classifiers["severity"], review_policy=strict))
    strict_result = strict_service.analyze_record({"text": "water tap dry", "lang": "en"})
    assert strict_result["confidence"] < 1
    assert "low_classification_confidence" in strict_result["review_reasons"]


def test_reject_all_policy_and_life_risk_force_review_even_with_accepting_threshold(classifiers, tmp_path):
    reject = make_policy(tmp_path / "reject.json", classifiers, threshold=float(np.nextafter(1.0, np.inf)), status="reject_all")
    options = analyzer_options(category_model=classifiers["category"], severity_model=classifiers["severity"], review_policy=reject)
    assert "validation_policy_reject_all" in analysis.PriorityAnalyzer(**options).analyze_record({"text": "water tap", "lang": "en"})["review_reasons"]
    accept = make_policy(tmp_path / "accept.json", classifiers)
    options["review_policy"] = accept
    assert "life_risk_keyword_requires_review" in analysis.PriorityAnalyzer(**options).analyze_record({"text": "active fire near hospital", "lang": "en"})["review_reasons"]
    options["review_on_life_risk"] = False
    assert not analysis.PriorityAnalyzer(**options).analyze_record({"text": "active fire near hospital", "lang": "en"})["review_required"]


def test_policy_model_identity_mismatch_is_a_configuration_error(classifiers, tmp_path):
    policy = make_policy(tmp_path / "mismatch.json", classifiers)
    data = json.loads(policy.read_text())
    data["classifier_fingerprints"]["category"] = "0" * 64
    policy.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fingerprints"):
        analysis.PriorityAnalyzer(**analyzer_options(category_model=classifiers["category"], severity_model=classifiers["severity"], review_policy=policy))


def test_source_ground_truth_is_not_used_as_an_inference_label_override(classifiers, tmp_path):
    policy = make_policy(tmp_path / "accept.json", classifiers)
    service = analysis.PriorityAnalyzer(**analyzer_options(category_model=classifiers["category"], severity_model=classifiers["severity"], review_policy=policy))
    record = {"text": "water tap pipe dry", "lang": "en"}
    plain = service.analyze_record(record)
    altered = service.analyze_record({**record, "category": "Completely different label", "severity": "Nonexistent", "category_true": "Other", "severity_true": "Other"})
    for field in ("category", "severity", "priority_score", "ranking_score", "confidence"):
        assert plain[field] == altered[field]


def test_unmapped_classifier_labels_require_review_despite_high_model_confidence(tmp_path):
    root = tmp_path / "unknown-labels"
    paths = {"category": make_classifier(root, "category", category_names=("UnknownA", "UnknownB")),
             "severity": make_classifier(root, "severity", "word_lr", category_names=("UnknownA", "UnknownB"))}
    policy = make_policy(tmp_path / "accept.json", paths)
    service = analysis.PriorityAnalyzer(**analyzer_options(category_model=paths["category"], severity_model=paths["severity"], review_policy=policy))
    result = service.analyze_record({"text": "water tap", "lang": "en"})
    assert "category" in result["explanation"]["missing_fields"]
    assert "priority_label_mapping_unknown" in result["review_reasons"]


def test_language_inference_flags_only_ambiguous_heuristic_and_explicit_language_is_preserved(classifiers, tmp_path):
    assert analysis.infer_language("পানি নেই")["lang"] == "bn"
    assert analysis.infer_language("pani nai ekhane")["lang"] == "banglish"
    assert analysis.infer_language("water tap problem")["lang"] == "en"
    assert analysis.infer_language("pani problem")["ambiguous"]
    policy = make_policy(tmp_path / "accept.json", classifiers)
    service = analysis.PriorityAnalyzer(**analyzer_options(category_model=classifiers["category"], severity_model=classifiers["severity"], review_policy=policy))
    assert "language_inference_ambiguous" in service.analyze_complaint("pani problem")["review_reasons"]
    explicit = service.analyze_record({"text": "pani problem", "lang": "en"})
    assert explicit["language_inference"]["method"] == "provided"
    assert "language_inference_ambiguous" not in explicit["review_reasons"]


def test_configured_convenience_function_returns_the_required_json_fields(monkeypatch):
    service = analysis.PriorityAnalyzer(**analyzer_options())
    monkeypatch.setattr(analysis, "_DEFAULT_ANALYZER", service)
    result = analysis.analyze_complaint("3 din dhore pani nai", "Mirpur 10", "2026-10-30T10:00:00+06:00")
    required = {"category", "severity", "priority_score", "priority_label", "confidence", "explanation", "review_required", "duplicate_candidates"}
    assert required <= set(result)
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_duplicate_report_volume_counts_sources_and_collapses_derivations(tmp_path, monkeypatch):
    marker = tmp_path / "duplicate.joblib"
    marker.write_bytes(b"fake finder is injected; never loaded")
    corpus = [
        {"id": "source-a", "original_id": "incident-1", "text": "water problem", "lang": "en", "is_derived": "false"},
        {"id": "a-english", "original_id": "incident-1", "parent_id": "source-a", "text": "same problem", "lang": "en", "is_derived": "true"},
        {"id": "a-banglish", "original_id": "incident-1", "parent_id": "a-english", "text": "pani nai", "lang": "banglish", "is_derived": "true"},
        {"id": "source-b", "original_id": "incident-1", "text": "another report of water", "lang": "en", "is_derived": "false"},
        {"id": "c-derived", "original_id": "incident-1", "source_report_id": "source-c", "text": "third report", "lang": "en", "is_derived": "true"},
    ]

    class Finder:
        def __init__(self, *args, **kwargs):
            self.last_query_stats = {"exhaustive_candidates": False, "corpus_size": len(corpus)}
        def build_index(self, records):
            assert records == corpus
        def find_duplicates(self, query, records):
            assert records is None
            return [{"id": row["id"], "score": 0.8, "threshold": 0.5} for row in corpus if row["id"] != query["id"]]

    monkeypatch.setattr(analysis, "DuplicateFinder", Finder)
    service = analysis.PriorityAnalyzer(**analyzer_options(duplicate_model=marker, existing_complaints=corpus))
    result = service.analyze_record({"id": "new-independent-report", "original_id": "incident-1", "text": "water issue", "lang": "en"})
    # New source + source-a (including its two derived descendants) + independent
    # source-b + explicit source-c lineage. Shared incident ID is not a person
    # count and must not collapse genuinely independent source reports.
    assert result["duplicate_group_size"] == 4
    assert result["duplicate_detection_status"] == "available"
    duplicate_rule = next(row for row in result["explanation"]["rule_breakdown"] if row["name"] == "duplicate_reports")
    assert duplicate_rule["evidence"][0]["report_count"] == 4
    assert "not unique people" in result["duplicate_group_size_semantics"]


def test_source_report_identity_distinguishes_originals_from_incident_and_resolves_parent_chains():
    original_a = {"id": "a", "original_id": "incident-1", "is_derived": False}
    original_b = {"id": "b", "original_id": "incident-1", "is_derived": False}
    translated = {"id": "t", "original_id": "incident-1", "is_derived": True, "parent_id": "a"}
    paraphrased = {"id": "p", "original_id": "incident-1", "is_derived": True, "parent_id": "t"}
    lookup = {record["id"]: record for record in (original_a, original_b, translated, paraphrased)}
    assert analysis.source_report_key(original_a, lookup) != analysis.source_report_key(original_b, lookup)
    assert analysis.source_report_key(paraphrased, lookup) == analysis.source_report_key(original_a, lookup)
    assert analysis.source_report_key({"id": "d", "is_derived": "true", "original_id": "incident-1"}, lookup) is None
    assert analysis.source_report_key({"id": "e", "source_report_id": "explicit-source"}, lookup) == "source:explicit-source"
    lookup["cycle-a"] = {"id": "cycle-a", "parent_id": "cycle-b"}
    lookup["cycle-b"] = {"id": "cycle-b", "parent_id": "cycle-a"}
    with pytest.raises(ValueError, match="Cyclic"):
        analysis.source_report_key(lookup["cycle-a"], lookup)


def test_unidentified_identical_worded_queries_get_distinct_source_report_ids():
    service = analysis.PriorityAnalyzer(**analyzer_options())
    first = service.analyze_complaint("same water issue")
    second = service.analyze_complaint("same water issue")
    assert first["id"] != second["id"]


def test_review_policy_binds_preprocessing_resources_as_well_as_joblib_bytes(classifiers, tmp_path, monkeypatch):
    from ml import classifier_runtime

    policy = make_policy(tmp_path / "accept.json", classifiers)
    sha256 = classifier_runtime.file_sha256

    def different_preprocessing(path):
        return "f" * 64 if Path(path).name == "experiment_data.py" else sha256(path)

    monkeypatch.setattr(classifier_runtime, "file_sha256", different_preprocessing)
    with pytest.raises(ValueError, match="resources do not match"):
        analysis.PriorityAnalyzer(**analyzer_options(category_model=classifiers["category"], severity_model=classifiers["severity"], review_policy=policy))


def test_absent_or_insufficient_expert_ranker_keeps_rule_priority(tmp_path):
    service = analysis.PriorityAnalyzer(**analyzer_options())
    plain = service.analyze_record({"text": "school issue for three days", "lang": "en"})
    rules = PriorityRules()
    artifact = {"kind": "banglapsg_priority_ranker", "schema_version": 1, "status": "fallback_rules",
                "rules_config_sha256": rules.config_sha256, "feature_names": list(rules.feature_names),
                "rules_preprocessing_fingerprints": rules.preprocessing_fingerprints,
                "reason": "insufficient independently judged queries", "estimator": None}
    path = tmp_path / "fallback.joblib"
    joblib.dump(artifact, path)
    result = analysis.PriorityAnalyzer(**analyzer_options(ranker=path)).analyze_record({"text": "school issue for three days", "lang": "en"})
    assert result["priority_score"] == plain["priority_score"]
    assert result["ranking_score"] == plain["ranking_score"] == result["priority_score"]
    assert result["ranking_method"] == "rules"


def test_trained_ranker_additive_utility_is_separate_from_rule_points(tmp_path):
    rules = PriorityRules()
    rows = [{"id": "r1", "original_id": "i1", "query_id": "q1", "split": "train", "expert_priority": 1},
            {"id": "r2", "original_id": "i2", "query_id": "q1", "split": "train", "expert_priority": 5}]
    features = np.zeros((2, len(rules.feature_names)))
    features[1, :] = [5, 3, 2, 1, 1, 3]
    artifact = fit_priority_ranker(pd.DataFrame(rows), features, rules.feature_names, rules.config_sha256,
                                   min_train_queries=1, min_comparisons=1, min_train_items=2)
    assert artifact["status"] == "trained"
    artifact["rules_preprocessing_fingerprints"] = rules.preprocessing_fingerprints
    path = tmp_path / "ranker.joblib"
    joblib.dump(artifact, path)
    result = analysis.PriorityAnalyzer(**analyzer_options(ranker=path)).analyze_record({"text": "school fire for three days", "lang": "en"})
    ranked = result["explanation"]["ranking"]
    assert result["ranking_method"] == "pairwise_logistic"
    assert result["ranking_score"] == pytest.approx(ranked["bias"] + sum(row["contribution"] for row in ranked["contributions"]))
    assert result["priority_score"] == sum(row["contribution"] for row in result["explanation"]["rule_breakdown"])
    assert "not a calibrated probability" in ranked["score_semantics"]


def test_ranker_config_fingerprint_mismatch_is_rejected(tmp_path):
    path = tmp_path / "wrong-ranker.joblib"
    joblib.dump({"kind": "banglapsg_priority_ranker", "schema_version": 1, "status": "fallback_rules",
                 "rules_config_sha256": "0" * 64, "feature_names": []}, path)
    with pytest.raises(ValueError, match="fingerprint"):
        analysis.PriorityAnalyzer(**analyzer_options(ranker=path))


@pytest.mark.parametrize("record", [{}, {"text": None}, {"text": ""}, {"text": "https://example.com"},
                                   {"text": "problem", "lang": "unsupported"}, {"text": "problem", "location": 7},
                                   {"text": "problem", "timestamp": "not an ISO time"}])
def test_invalid_analysis_inputs_raise_clear_errors(record):
    with pytest.raises((ValueError, TypeError)):
        analysis.PriorityAnalyzer(**analyzer_options()).analyze_record(record)
