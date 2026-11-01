"""Expert ranking mechanics with synthetic judgments; no research score claims."""

from __future__ import annotations

import csv
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.priority_ranker import (
    JUDGMENT_COLUMNS, evaluate_rankings, fit_priority_ranker, load_expert_judgments,
    load_priority_ranker, make_pairwise_training_data, ranker_score, ranking_metrics,
)
from ml.priority_rules import DEFAULT_CONFIG_PATH, PriorityRules
from ml_training.scripts import train_priority_ranker as training


def judgment_rows():
    examples = [
        ("Minor road surface damage", "Road & Transport", "Low", 1),
        ("water unavailable for two days", "Water", "Medium", 1),
        ("road blocked for three days", "Road & Transport", "Medium", 3),
        ("Hospital without water for three days", "Healthcare", "High", 3),
        ("Active fire near children", "Electricity", "High", 10),
    ]
    rows = []
    for split, queries in (("train", 3), ("val", 1), ("test", 2)):
        for query_index in range(queries):
            query = f"{split}-q{query_index}"
            for index, (text, category, severity, size) in enumerate(examples):
                identifier = f"{query}-item{index}"
                rows.append(dict(id=identifier, original_id=identifier, query_id=query,
                                 split=split, text=text, lang="en", category=category,
                                 severity=severity, location="", timestamp="2026-10-30T08:00:00Z",
                                 duplicate_group_size=str(size), expert_priority=str(index + 1),
                                 expert_rank=str(5 - index)))
    return rows


def write_rows(path, rows, columns=JUDGMENT_COLUMNS):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def feature_rows(frame, rules):
    results = [rules.score(row) for row in frame.to_dict("records")]
    return results, np.asarray([[row["feature_values"][name] for name in rules.feature_names] for row in results], dtype=float).reshape(len(frame), len(rules.feature_names))


@pytest.fixture
def dataset(tmp_path):
    path = write_rows(tmp_path / "experts.csv", judgment_rows())
    frame = load_expert_judgments(path)
    rules = PriorityRules()
    return path, frame, rules


def test_expert_template_is_header_only():
    path = DEFAULT_CONFIG_PATH.parents[1] / "templates" / "expert_priority_judgments.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows == [list(JUDGMENT_COLUMNS)]


def test_load_preserves_explicit_scale_and_rank_direction(dataset):
    _, frame, _ = dataset
    assert set(frame["expert_priority"]) == {1, 2, 3, 4, 5}
    assert frame.iloc[0]["expert_rank"] == 5
    assert frame.iloc[4]["expert_rank"] == 1
    assert frame["duplicate_group_size"].map(type).eq(int).all()
    assert set(frame["split"]) == {"train", "val", "test"}


@pytest.mark.parametrize("identity", ["id", "original_id", "query_id"])
def test_rejects_id_incident_or_query_leakage(tmp_path, identity):
    rows = judgment_rows()
    first_test = next(index for index, row in enumerate(rows) if row["split"] == "test")
    rows[first_test][identity] = rows[0][identity]
    path = write_rows(tmp_path / "leak.csv", rows)
    with pytest.raises(ValueError, match="unique|leakage"):
        load_expert_judgments(path)


@pytest.mark.parametrize("field,value", [("expert_priority", ""), ("expert_priority", "6"),
                                         ("expert_priority", "2.5"), ("duplicate_group_size", "0"),
                                         ("duplicate_group_size", "True"), ("lang", "guess"),
                                         ("split", "training"), ("expert_rank", "0"),
                                         ("expert_rank", "nan")])
def test_invalid_judgments_do_not_become_source_or_default_labels(tmp_path, field, value):
    rows = judgment_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        load_expert_judgments(write_rows(tmp_path / "bad.csv", rows))


def test_optional_ranks_can_be_absent_but_not_partially_present(tmp_path):
    rows = judgment_rows()
    for row in rows:
        row["expert_rank"] = ""
    frame = load_expert_judgments(write_rows(tmp_path / "unranked.csv", rows))
    assert frame["expert_rank"].isna().all()
    rows[0]["expert_rank"] = "1"
    with pytest.raises(ValueError, match="mixes"):
        load_expert_judgments(write_rows(tmp_path / "partial.csv", rows))


def test_pairwise_samples_are_mirrored_and_seeded_without_tied_relevance(dataset):
    _, frame, rules = dataset
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    _, values = feature_rows(train, rules)
    differences, target, report = make_pairwise_training_data(train, values, 4, 7)
    assert report["unordered_comparisons"] == 12
    assert differences.shape == (24, len(rules.feature_names))
    assert np.array_equal(target[::2], np.ones(12))
    assert np.array_equal(target[1::2], np.zeros(12))
    assert np.allclose(differences[::2], -differences[1::2])
    again = make_pairwise_training_data(train, values, 4, 7)
    assert np.array_equal(differences, again[0])
    assert report == again[2]


def test_large_query_pair_output_is_bounded_by_cap():
    count = 5000
    frame = pd.DataFrame({"id": [f"row-{i:06}" for i in range(count)], "query_id": ["queue"] * count,
                          "expert_priority": [1 if i < count // 2 else 5 for i in range(count)]})
    values = np.column_stack((np.arange(count), np.ones(count)))
    differences, labels, report = make_pairwise_training_data(frame, values, max_pairs_per_query=9)
    assert report["queries"]["queue"]["eligible_unordered_comparisons"] == 2500 * 2500
    assert report["unordered_comparisons"] == 9
    assert differences.shape == (18, 2) and labels.shape == (18,)


def test_pairwise_row_reordering_keeps_same_sampled_item_differences(dataset):
    _, frame, rules = dataset
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    _, values = feature_rows(train, rules)
    first = make_pairwise_training_data(train, values, max_pairs_per_query=6, seed=3)
    reordered = train.sample(frac=1, random_state=8).reset_index(drop=True)
    _, reordered_values = feature_rows(reordered, rules)
    second = make_pairwise_training_data(reordered, reordered_values, max_pairs_per_query=6, seed=3)
    assert np.allclose(first[0], second[0])
    assert first[2] == second[2]


def test_trained_ranker_has_exact_additive_utility_and_frozen_rules_identity(dataset, tmp_path):
    _, frame, rules = dataset
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    results, values = feature_rows(train, rules)
    fingerprint = hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest()
    artifact = fit_priority_ranker(train, values, rules.feature_names, fingerprint)
    assert artifact["status"] == "trained"
    output = ranker_score(artifact, results[-1])
    assert output["method"] == "pairwise_logistic"
    assert output["ranking_score"] == pytest.approx(sum(row["contribution"] for row in output["contributions"]) + output["bias"])
    assert output["ranking_score"] == pytest.approx(float(artifact["estimator"].decision_function(values[-1:])[0]))
    path = tmp_path / "ranker.joblib"
    joblib.dump(artifact, path)
    loaded = load_priority_ranker(path, fingerprint)
    assert ranker_score(loaded, results[-1])["ranking_score"] == pytest.approx(output["ranking_score"])
    with pytest.raises(ValueError, match="fingerprint"):
        load_priority_ranker(path, "0" * 64)
    changed = dict(results[-1], config_sha256="0" * 64)
    with pytest.raises(ValueError, match="changed"):
        ranker_score(artifact, changed)


@pytest.mark.parametrize("field", ["normalization_dictionary_sha256", "normalizer_code_sha256"])
def test_saved_preprocessing_identity_rejects_changed_dictionary_or_normalizer(dataset, tmp_path, field):
    _, frame, rules = dataset
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    results, values = feature_rows(train, rules)
    artifact = fit_priority_ranker(train, values, rules.feature_names, results[0]["config_sha256"])
    artifact["rules_preprocessing_fingerprints"] = dict(rules.preprocessing_fingerprints)
    path = tmp_path / "ranker.joblib"
    joblib.dump(artifact, path)
    loaded = load_priority_ranker(path, rules.config_sha256, rules.preprocessing_fingerprints)
    assert ranker_score(loaded, results[0])["method"] == "pairwise_logistic"
    changed = dict(rules.preprocessing_fingerprints)
    changed[field] = "0" * 64
    with pytest.raises(ValueError, match="preprocessing"):
        load_priority_ranker(path, rules.config_sha256, changed)
    changed_result = dict(results[0], preprocessing_fingerprints=changed)
    with pytest.raises(ValueError, match="preprocessing"):
        ranker_score(loaded, changed_result)
    missing_result = {key: value for key, value in results[0].items() if key != "preprocessing_fingerprints"}
    with pytest.raises(ValueError, match="preprocessing"):
        ranker_score(loaded, missing_result)


def test_insufficient_judgments_fall_back_without_estimating_weights(dataset):
    _, frame, rules = dataset
    train = frame.loc[frame["split"].eq("train")].iloc[:3].reset_index(drop=True)
    results, values = feature_rows(train, rules)
    artifact = fit_priority_ranker(train, values, rules.feature_names, results[0]["config_sha256"])
    assert artifact["status"] == "fallback_rules" and artifact["estimator"] is None
    assert "fewer than" in artifact["reason"]
    output = ranker_score(artifact, results[0])
    assert output["method"] == "rules"
    assert output["ranking_score"] == results[0]["priority_score"]


def test_fit_refuses_validation_or_test_items(dataset):
    _, frame, rules = dataset
    results, values = feature_rows(frame, rules)
    with pytest.raises(ValueError, match="training split"):
        fit_priority_ranker(frame, values, rules.feature_names, results[0]["config_sha256"])


def test_rank_metrics_have_correct_high_priority_and_rank_one_direction():
    perfect = ranking_metrics([1, 3, 5], [0, 2, 4], [3, 2, 1])
    reverse = ranking_metrics([1, 3, 5], [4, 2, 0], [3, 2, 1])
    assert perfect["ndcg_at_5"] == pytest.approx(1)
    assert perfect["spearman"] == pytest.approx(1)
    assert reverse["ndcg_at_5"] < perfect["ndcg_at_5"]
    assert reverse["spearman"] == pytest.approx(-1)
    assert perfect["k_effective"] == 3


def test_ties_use_expected_dcg_and_do_not_claim_constant_ranking_correlation():
    report = ranking_metrics([1, 5], [0, 0])
    gains = [1, 31]
    expected = np.mean(gains) * (1 + 1 / np.log2(3))
    ideal = 31 + 1 / np.log2(3)
    assert report["ndcg_at_5"] == pytest.approx(expected / ideal)
    assert report["spearman"] is None
    assert report["spearman_undefined_reason"]


@pytest.mark.parametrize("grades,scores", [([], []), ([3], [1]), ([3, 3], [1, 2])])
def test_non_discriminatory_queries_have_explicit_null_metrics(grades, scores):
    report = ranking_metrics(grades, scores)
    assert report["ndcg_at_5"] is None
    assert report["spearman"] is None


def test_comparison_keeps_identical_query_items_and_metric_support(dataset):
    _, frame, _ = dataset
    test = frame.loc[frame["split"].eq("test")].reset_index(drop=True)
    reports = evaluate_rankings(test, {"perfect": test["expert_priority"], "reverse": -test["expert_priority"]})
    assert reports["perfect"]["cohort_fingerprint"] == reports["reverse"]["cohort_fingerprint"]
    assert reports["perfect"]["n_items"] == reports["reverse"]["n_items"] == 10
    assert reports["perfect"]["n_queries"] == 2
    assert reports["perfect"]["ndcg_eligible_queries"] == 2
    assert reports["perfect"]["spearman"] == pytest.approx(1)


def test_cli_trains_saves_and_compares_real_rules_on_same_test_cohort(dataset, tmp_path):
    path, _, _ = dataset
    args = training.build_parser().parse_args(["--judgments", str(path), "--output-dir", str(tmp_path / "results"),
                                             "--artifact", str(tmp_path / "artifacts" / "priority.joblib")])
    report = training.run_experiment(args)
    assert report["ranker_status"] == "trained"
    assert set(report["comparisons"]["test"]) == {"severity_only", "rule_based", "ltr"}
    assert len({row["cohort_fingerprint"] for row in report["comparisons"]["test"].values()}) == 1
    assert (tmp_path / "results" / "ranking_comparison.csv").is_file()
    assert (tmp_path / "results" / "ranking_comparison.md").is_file()
    assert json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8")) == report
    assert (tmp_path / "artifacts" / "priority.joblib.rules.yaml").is_file()
    variants_snapshot = tmp_path / "artifacts" / "priority.joblib.variants.json"
    assert variants_snapshot.is_file()
    assert hashlib.sha256(variants_snapshot.read_bytes()).hexdigest() == report["rules_preprocessing_fingerprints"]["normalization_dictionary_sha256"]
    artifact = joblib.load(tmp_path / "artifacts" / "priority.joblib")
    assert artifact["variants_snapshot"] == variants_snapshot.name
    assert artifact["rules_preprocessing_fingerprints"] == report["rules_preprocessing_fingerprints"]
    with pytest.raises(ValueError, match="exists"):
        training.run_experiment(args)


def test_empty_expert_template_saves_honest_rule_fallback_and_null_comparison(tmp_path):
    path = write_rows(tmp_path / "empty.csv", [])
    args = training.build_parser().parse_args(["--judgments", str(path), "--output-dir", str(tmp_path / "results"),
                                             "--artifact", str(tmp_path / "artifacts" / "fallback.joblib")])
    report = training.run_experiment(args)
    assert report["ranker_status"] == "fallback_rules"
    assert report["n_training_items"] == 0
    for split in ("val", "test"):
        for model in report["comparisons"][split].values():
            assert model["status"] == "skipped_no_expert_items"
            assert model["ndcg_at_5"] is None and model["spearman"] is None


def test_invalid_cli_input_returns_clear_error_without_traceback(tmp_path, capsys):
    assert training.main(["--judgments", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path / "results"),
                          "--artifact", str(tmp_path / "artifacts" / "ranker.joblib")]) == 2
    assert "ERROR:" in capsys.readouterr().err
