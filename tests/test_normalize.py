"""Behavioral tests for shared cleaning and Banglish normalization."""

import json

import pytest

from ml.normalize import clean_text, load_variants, normalize_banglish


def test_clean_nfkc_digits_latin_case_and_spaces():
    assert clean_text("  ＷＡＴＥＲ\tনেই  ১২৩\n৪৫৬৭৮৯০  ") == "water নেই 123 4567890"


def test_remove_http_https_and_www_urls():
    text = "আগে https://example.com/a?x=১২ HTTP://EXAMPLE.COM www.site.test পরে"
    assert clean_text(text) == "আগে পরে"


def test_cleaning_preserves_bangla_punctuation_and_empty_text():
    assert clean_text("পানি নেই।") == "পানি নেই।"
    assert clean_text(" \n\t ") == ""


@pytest.mark.parametrize("value", [None, 12, float("nan")])
def test_non_string_input_is_rejected(value):
    with pytest.raises(TypeError, match="text must be a string"):
        clean_text(value)


def test_banglish_repetition_variants_and_punctuation():
    assert normalize_banglish("Paniiii PANY, NEI! BIDDUT nai.") == "pani pani, nai! bidyut nai."


def test_regular_double_letters_are_preserved():
    assert normalize_banglish("school good coffee book") == "school good coffee book"


def test_bangla_letters_are_not_collapsed():
    assert normalize_banglish("নাাা ৯৯৯ পানি") == "নাাা 999 পানি"


@pytest.mark.parametrize(
    "text",
    ["Paniiii NEI", "pany biddut raasta", "Shomossha AChhe", "স্কুল ১২ www.example.com", ""],
)
def test_normalization_is_idempotent(text):
    once = normalize_banglish(text)
    assert normalize_banglish(once) == once


def test_custom_dictionary_and_alias_chains(tmp_path):
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"jol": "pany", "pany": "pani"}), encoding="utf-8")
    assert normalize_banglish("JOL! pany", path) == "pani! pani"
    assert load_variants(path) == {"jol": "pani", "pany": "pani"}


def test_custom_dictionary_replaces_default(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    assert normalize_banglish("PANY", path) == "pany"


def test_dictionary_edits_are_loaded_and_callers_cannot_mutate_cache(tmp_path):
    path = tmp_path / "custom.json"
    path.write_text('{"jol": "pani"}', encoding="utf-8")
    variants = load_variants(path)
    variants["jol"] = "invalid"
    assert normalize_banglish("jol", path) == "pani"
    path.write_text('{"jol": "water"}', encoding="utf-8")
    assert normalize_banglish("jol", path) == "water"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("invalid json", "Cannot load Banglish variants"),
        ("[]", "must be a JSON object"),
        ('{"pany": 2}', "must map strings to strings"),
        ('{"PANY": "pani"}', "single lowercase Latin words"),
        ('{"a": "b", "b": "a"}', "Cyclic"),
        ('{"paniii": "pani", "paniiii": "jol"}', "Conflicting"),
    ],
)
def test_bad_dictionary_has_clear_error(tmp_path, content, message):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_variants(path)


def test_missing_dictionary_has_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot load Banglish variants"):
        load_variants(tmp_path / "missing.json")
