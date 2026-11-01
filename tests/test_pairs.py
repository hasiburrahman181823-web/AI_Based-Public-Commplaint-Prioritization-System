"""Pair labels, strict source validation, bounded sampling, and split provenance."""

from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

from ml_training.config import SCHEMA_COLUMNS
from ml_training.scripts.generate_pairs import (
    PAIR_COLUMNS,
    PAIR_TYPES,
    PairValidationError,
    generate_pairs,
    hard_negative_eligible,
    load_splits,
    main,
    text_style,
    write_pairs,
)


def row(identifier, group, **overrides):
    result = dict(zip(SCHEMA_COLUMNS, [identifier, group, "water unavailable today", "en", "water", "high", "natural", "Dhaka", "2026-01-01T12:00:00Z"]))
    result.update(overrides)
    return result


def splits_with(rows):
    return {name: pd.DataFrame(rows if name == "train" else [], columns=list(SCHEMA_COLUMNS) + ["text_style", "is_derived", "derivation", "parent_id"]) for name in ("train", "val", "test")}


def quiet_pairs(splits, cap=10, seed=42):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return generate_pairs(splits, max_pairs_per_type=cap, seed=seed)


def write_inputs(directory, splits):
    directory.mkdir(parents=True)
    for split, frame in splits.items():
        frame.to_csv(directory / f"{split}.csv", index=False)


def test_cross_language_positives_require_shared_group_and_distinct_languages():
    data = splits_with([
        row("en1", "g1"), row("bn1", "g1", lang="bn", text="পানি নেই"),
        row("bl1", "g1", lang="banglish", text="pani nai"), row("en2", "g2"),
    ])
    pairs, _ = quiet_pairs(data)
    positive = pairs.loc[pairs.pair_type.eq("positive_cross_language")]
    assert len(positive) == 3
    assert positive.label.eq(1).all()
    assert positive.original_id_a.eq(positive.original_id_b).all()
    assert set(zip(positive.id_a, positive.id_b)) == {("bl1", "bn1"), ("bl1", "en1"), ("bn1", "en1")}


def test_manual_paraphrases_attach_to_existing_split_and_have_priority(tmp_path):
    directory = tmp_path / "multilingual"
    write_inputs(directory, splits_with([row("base", "g1"), row("bn", "g1", lang="bn")]))
    paraphrases = tmp_path / "paraphrases.csv"
    pd.DataFrame([row("manual", "g1", text="No water is available", split="train")]).to_csv(paraphrases, index=False)
    data = load_splits(directory, paraphrases)
    pairs, _ = quiet_pairs(data)
    manual_pairs = pairs.loc[pairs.pair_type.eq("positive_paraphrase")]
    assert len(manual_pairs) == 2
    assert manual_pairs.label.eq(1).all()
    assert set(zip(manual_pairs.id_a, manual_pairs.id_b)) == {("base", "manual"), ("bn", "manual")}
    assert data["train"].loc[data["train"].id.eq("manual"), "derivation"].item() == "manual_paraphrase"
    assert manual_pairs.derivation_b.eq("manual_paraphrase").all()
    assert manual_pairs.is_derived_b.eq("true").all()
    assert len(pairs.loc[pairs.pair_type.eq("positive_cross_language")]) == 1


@pytest.mark.parametrize("group,split,error", [("unknown", "train", "unknown original_id"), ("g1", "test", "belongs to train")])
def test_rejects_unknown_or_moved_manual_groups(tmp_path, group, split, error):
    directory = tmp_path / "multilingual"
    write_inputs(directory, splits_with([row("base", "g1")]))
    paraphrases = tmp_path / "manual.csv"
    pd.DataFrame([row("manual", group, split=split)]).to_csv(paraphrases, index=False)
    with pytest.raises(PairValidationError, match=error):
        load_splits(directory, paraphrases)


def test_hard_negatives_require_known_metadata_difference_and_same_style():
    a = row("a", "a")
    assert hard_negative_eligible(a, row("b", "b", location="Chattogram"))
    assert hard_negative_eligible(a, row("b", "b", timestamp="2026-01-02T00:00:00Z", location=""))
    assert not hard_negative_eligible(a, row("b", "b", location="  dHaKa  ", timestamp="2026-01-01T23:59:00Z"))
    assert not hard_negative_eligible(a, row("b", "b", location="", timestamp=""))
    assert not hard_negative_eligible(a, row("b", "a", location="Chattogram"))
    assert not hard_negative_eligible(a, row("b", "b", location="Chattogram", category="roads"))
    assert not hard_negative_eligible(a, row("b", "b", location="Chattogram", lang="bn"))
    assert not hard_negative_eligible(a, row("b", "b", location="Chattogram", text=" ".join(["word"] * 13)))


def test_hard_negative_dates_are_compared_in_utc():
    # Both represent January 1 UTC despite different source calendar dates.
    same = splits_with([
        row("a", "a", timestamp="2026-01-02T01:00:00+06:00", location=""),
        row("b", "b", timestamp="2026-01-01T22:00:00Z", location=""),
    ])
    pairs, _ = quiet_pairs(same)
    assert pairs.empty
    different = splits_with([
        row("a", "a", timestamp="2026-01-02T01:00:00+06:00", location=""),
        row("b", "b", timestamp="2026-01-02T00:00:00Z", location=""),
    ])
    pairs, _ = quiet_pairs(different)
    assert pairs.pair_type.tolist() == ["hard_negative"]
    assert pairs.iloc[0].time_a == "2026-01-01T19:00:00Z"


def test_explicit_styles_match_case_and_spaces_and_do_not_match_inferred_styles():
    a = row("a", "a", text_style="  FORMAL   REPORT ")
    b = row("b", "b", text_style="formal report", location="Chattogram", lang="bn")
    assert hard_negative_eligible(a, b)
    assert not hard_negative_eligible(a, row("c", "c", location="Chattogram"))
    assert text_style(row("x", "x", text=" ".join(["w"] * 12)))[-1] == "short"
    assert text_style(row("x", "x", text=" ".join(["w"] * 13)))[-1] == "medium"
    assert text_style(row("x", "x", text=" ".join(["w"] * 30)))[-1] == "medium"
    assert text_style(row("x", "x", text=" ".join(["w"] * 31)))[-1] == "long"


def test_easy_negatives_require_different_category_and_group():
    pairs, _ = quiet_pairs(splits_with([
        row("a", "g1", category="water"), row("b", "g1", category="roads"), row("c", "g2", category="roads"),
    ]))
    easy = pairs.loc[pairs.pair_type.eq("easy_negative")]
    assert set(zip(easy.id_a, easy.id_b)) == {("a", "c")}
    assert easy.label.eq(0).all()


def test_reproducible_capped_unique_and_split_isolated():
    all_splits = {}
    for split in ("train", "val", "test"):
        records = []
        for group in range(20):
            for language in ("en", "bn", "banglish"):
                records.append(row(f"{split}-{group}-{language}", f"{split}-{group}", lang=language, category=str(group % 2), location=f"town-{group}"))
        all_splits[split] = pd.DataFrame(records)
    first, summary = quiet_pairs(all_splits, cap=5, seed=123)
    second, other_summary = quiet_pairs({name: frame.iloc[::-1] for name, frame in all_splits.items()}, cap=5, seed=123)
    pd.testing.assert_frame_equal(first, second)
    assert summary == other_summary
    assert first.pair_id.is_unique
    assert first.id_a.lt(first.id_b).all()
    assert first.groupby(["split", "pair_type"]).size().le(5).all()
    assert sum(first.pair_type.eq("hard_negative")) == 15
    assert sum(first.pair_type.eq("easy_negative")) == 15
    for split in ("train", "val", "test"):
        selected = first.loc[first.split.eq(split)]
        assert selected.id_a.str.startswith(split).all()
        assert selected.id_b.str.startswith(split).all()
        assert all(attempts <= 200 for attempts in summary["splits"][split]["attempts"].values())


def test_underfilled_candidates_warn_and_have_exact_summary():
    data = splits_with([row("a", "g1"), row("b", "g1", lang="bn")])
    with pytest.warns(UserWarning, match="bounded attempts"):
        pairs, summary = generate_pairs(data, max_pairs_per_type=3)
    assert len(pairs) == 1
    assert summary["splits"]["train"]["sampled_candidate_counts"]["positive_cross_language"] == 1
    assert summary["total_pairs"] == 1
    assert summary["label_status"] == "candidate_suggestions_require_human_annotation"


@pytest.mark.parametrize("column,value,error", [("id", "", "empty"), ("original_id", "", "empty"), ("text", " ", "empty"), ("category", "", "empty"), ("lang", "fr", "lang"), ("timestamp", "not-a-date", "timestamp"), ("is_derived", "maybe", "is_derived")])
def test_rejects_invalid_rows(column, value, error):
    with pytest.raises(PairValidationError, match=error):
        quiet_pairs(splits_with([row("a", "g1", **{column: value})]))


def test_rejects_global_id_duplicates_and_original_group_leakage():
    data = splits_with([row("a", "g1")])
    data["val"] = pd.DataFrame([row("a", "g2")])
    with pytest.raises(PairValidationError, match="globally unique"):
        quiet_pairs(data)
    data["val"] = pd.DataFrame([row("b", "g1")])
    with pytest.raises(PairValidationError, match="Split leakage"):
        quiet_pairs(data)


@pytest.mark.parametrize("cap", [0, -1, 2.5, True])
def test_rejects_invalid_caps(cap):
    with pytest.raises(PairValidationError, match="positive integer"):
        generate_pairs(splits_with([]), max_pairs_per_type=cap)


def test_strict_csv_parsing_rejects_extra_fields_and_missing_schema(tmp_path):
    directory = tmp_path / "inputs"
    write_inputs(directory, splits_with([row("a", "g1")]))
    (directory / "train.csv").write_text("text,category,severity\ntext,water,high\n", encoding="utf-8")
    with pytest.raises(PairValidationError, match="Missing required columns"):
        load_splits(directory)
    header = ",".join(SCHEMA_COLUMNS)
    (directory / "train.csv").write_text(header + "\na,g1,text,en,water,high,source,Dhaka,,EXTRA\n", encoding="utf-8")
    with pytest.raises(PairValidationError, match="fields; expected"):
        load_splits(directory)


def test_output_roundtrip_overwrite_guard_and_empty_split_headers(tmp_path):
    pairs, summary = quiet_pairs(splits_with([row("0001", "g1"), row("0002", "g1", lang="bn")]))
    output = tmp_path / "pairs"
    files = write_pairs(pairs, summary, output)
    assert len(files) == 5
    loaded = pd.read_csv(output / "pairs.csv", dtype=str, keep_default_na=False)
    assert list(loaded.columns) == list(PAIR_COLUMNS)
    assert loaded.id_a.item() == "0001"
    assert loaded.label.item() == "1"
    assert pd.read_csv(output / "test" / "pairs.csv").empty
    assert json.loads((output / "summary.json").read_text())["total_pairs"] == 1
    with pytest.raises(PairValidationError, match="already exists"):
        write_pairs(pairs, summary, output)
    assert len(write_pairs(pairs, summary, output, overwrite=True)) == 5


def test_cli_prints_candidate_status_and_reports_missing_input(tmp_path, capsys):
    directory = tmp_path / "inputs"
    write_inputs(directory, splits_with([row("a", "g1"), row("b", "g1", lang="bn")]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert main(["--input-dir", str(directory), "--output-dir", str(tmp_path / "out"), "--max-pairs-per-type", "2"]) == 0
    assert "candidate suggestions" in capsys.readouterr().out
    assert main(["--input-dir", str(tmp_path / "missing")]) == 2
    assert "CSV not found" in capsys.readouterr().err
