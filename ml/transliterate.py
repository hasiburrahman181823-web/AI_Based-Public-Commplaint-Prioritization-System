"""Small, deterministic Bengali-to-Latin transliterator for offline baselines.

This is a spelling rule baseline, not a learned phonetic transliterator or a
translation system. Inherent-vowel deletion, conjuncts, dialects and names can
need human correction. Rules are explicit so the same input is reproducible.
"""

from __future__ import annotations

import unicodedata

TRANSLITERATOR_VERSION = "bn-rules-v1"

VOWELS = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u",
    "ঋ": "ri", "ঌ": "li", "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
}
CONSONANTS = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "ph", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh",
    "স": "s", "হ": "h", "ড়": "r", "ঢ়": "rh", "য়": "y",
    "ৰ": "r", "ৱ": "w",
}
MATRAS = {
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ৄ": "rri", "ৢ": "li", "ৣ": "lli", "ে": "e", "ৈ": "oi",
    "ো": "o", "ৌ": "ou",
}
SIGNS = {"ং": "ng", "ঃ": "h", "ঁ": "n", "ৎ": "t", "ঽ": "'"}
_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_JOINERS = "\u200c\u200d"


def transliterate_bangla(text: str) -> str:
    """Return a reproducible Latin-script approximation; preserve punctuation.

    Bangla digits become ASCII digits and existing Latin letters become lower
    case. Matras replace a consonant's inherent vowel, virama suppresses it,
    and a bare nonfinal consonant gets ``o``. Ordinary terminal consonants have
    no added vowel. These deliberately simple rules do not cover all Bengali
    pronunciations. Examples: ``পানি নাই`` -> ``pani nai`` and
    ``বিদ্যুৎ`` -> ``bidyut``.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", text).translate(_DIGITS).lower()
    pieces: list[str] = []
    index = 0
    previous_virama = False
    while index < len(text):
        char = text[index]
        if char in _JOINERS:
            index += 1
            continue
        pair = text[index:index + 2]
        # Unicode normalization decomposes these three letters into nukta pairs.
        consonant = pair if pair in CONSONANTS else char
        if consonant in CONSONANTS:
            latin = CONSONANTS[consonant]
            if char == "য" and consonant == "য" and previous_virama:
                latin = "y"
            pieces.append(latin)
            index += len(consonant)
            while index < len(text) and text[index] in _JOINERS:
                index += 1
            following = text[index] if index < len(text) else ""
            if following in MATRAS:
                pieces.append(MATRAS[following])
                index += 1
                previous_virama = False
            elif following == "্":
                index += 1
                previous_virama = True
            else:
                # Nasal signs keep an inherent vowel: বাংলা -> bangla.
                if following in CONSONANTS or following in SIGNS:
                    pieces.append("o")
                previous_virama = False
            continue
        pieces.append(VOWELS.get(char, MATRAS.get(char, SIGNS.get(char, char))))
        previous_virama = char == "্"
        index += 1
    return "".join(pieces)
