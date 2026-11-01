"""Seeded, synthetic Banglish spelling perturbations for paired robustness tests.

Only whole ASCII alphabetic words are considered. Bengali text, numeric tokens,
URLs and surrounding punctuation/whitespace are preserved. Noise is a stress
test, not evidence that a manually collected natural-language test set behaves
the same way.
"""

from __future__ import annotations

import random
import re

_TOKEN = re.compile(r"https?://\S+|www\.\S+|(?<![\w])[A-Za-z]+(?![\w])", re.IGNORECASE)


def add_banglish_noise(text: str, rng: random.Random, rate: float = 0.15) -> str:
    """Edit each selected ASCII token once using the caller's seeded RNG.

    A selected token receives one expressive-letter elongation or one phonetic
    replacement. Elongation may be reversed by the normalizer; the robustness
    workflow must apply exactly its ordinary preprocessing after perturbation.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
        raise ValueError("rate must be a number between 0 and 1")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")
    if rate == 0:
        return text

    def edit(match):
        token = match.group()
        if token.lower().startswith(("http://", "https://", "www.")) or rng.random() >= rate:
            return token
        replacements = []
        lower = token.lower()
        for source, target in (("sh", "s"), ("bh", "v"), ("ph", "f"), ("i", "ee"), ("u", "oo")):
            start = lower.find(source)
            if start >= 0:
                replacement = target.upper() if token[start:start + len(source)].isupper() else target
                replacements.append(token[:start] + replacement + token[start + len(source):])
        if replacements and rng.random() < 0.5:
            return rng.choice(replacements)
        index = rng.randrange(len(token))
        return token[:index] + token[index] * rng.randint(3, 5) + token[index + 1:]

    return _TOKEN.sub(edit, text)


def perturb_texts(texts, seed: int = 42, rate: float = 0.15) -> list[str]:
    """Perturb an ordered sequence reproducibly, with one local RNG per call."""
    rng = random.Random(seed)
    return [add_banglish_noise(text, rng, rate=rate) for text in texts]
