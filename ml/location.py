"""Gazetteer-based multilingual location extraction with conservative fuzziness.

The JSON schema is ``{schema_version: 1, areas: [{id, name, parent_id,
aliases: [str, ...]}, ...]}``. IDs are stable identifiers; names and aliases may
be extended. An area hierarchy supports compatibility checks, never geographic
kilometre distances. Fuzzy matching requires the optional/core rapidfuzz package.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

DEFAULT_GAZETTEER_PATH = Path(__file__).with_name("location_gazetteer.json")
_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def normalize_location_text(text: str) -> str:
    """Normalize separators and script digits while retaining Bengali marks."""
    if not isinstance(text, str):
        raise TypeError("Location text must be a string")
    text = unicodedata.normalize("NFKC", text).translate(_DIGITS).lower()
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)
    text = "".join(" " if unicodedata.category(char)[0] in ("P", "S") else char for char in text)
    text = re.sub(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", text)
    return " ".join(text.split())


class LocationGazetteer:
    """Resolve known place mentions to stable IDs without forcing ambiguity.

    Explicit location metadata takes precedence. Conflicting text is flagged
    rather than quietly discarded. Multiple unrelated mentions return unknown;
    ``candidate_ids`` preserves the evidence for review. Exact parent aliases
    are suppressed when a more specific descendant is also mentioned.
    """

    def __init__(self, path: str | Path | None = None, fuzzy_threshold: float = 90):
        if isinstance(fuzzy_threshold, bool) or not 0 < fuzzy_threshold <= 100:
            raise ValueError("fuzzy_threshold must be greater than 0 and at most 100")
        self.path = Path(path) if path is not None else DEFAULT_GAZETTEER_PATH
        self.fuzzy_threshold = float(fuzzy_threshold)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read location gazetteer {self.path}: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != 1 or not isinstance(raw.get("areas"), list) or not raw["areas"]:
            raise ValueError("Gazetteer must contain schema_version=1 and a nonempty areas list")
        self.areas: dict[str, dict] = {}
        alias_owner: dict[str, str] = {}
        seen_aliases: set[tuple[str, str]] = set()
        self.aliases: list[tuple[str, str, tuple[str, ...]]] = []
        for area in raw["areas"]:
            if not isinstance(area, dict) or not isinstance(area.get("id"), str) or not area["id"].strip():
                raise ValueError("Each gazetteer area needs a nonempty string id")
            identifier = area["id"]
            if identifier in self.areas:
                raise ValueError(f"Duplicate gazetteer ID: {identifier}")
            parent = area.get("parent_id")
            if parent is not None and (not isinstance(parent, str) or not parent):
                raise ValueError(f"Invalid parent_id for {identifier}")
            aliases = area.get("aliases")
            if not isinstance(aliases, list) or not aliases or any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
                raise ValueError(f"Area {identifier} needs a nonempty list of string aliases")
            self.areas[identifier] = {"name": area.get("name", identifier), "parent_id": parent}
            for alias in aliases:
                normalized = normalize_location_text(alias)
                if not normalized:
                    raise ValueError(f"Empty normalized location alias in {identifier}")
                previous = alias_owner.setdefault(normalized, identifier)
                if previous != identifier:
                    raise ValueError(f"Ambiguous normalized alias {normalized!r}: {previous} and {identifier}")
                if (normalized, identifier) not in seen_aliases:
                    self.aliases.append((normalized, identifier, tuple(normalized.split())))
                    seen_aliases.add((normalized, identifier))
        self.ancestors: dict[str, list[str]] = {}
        for identifier in self.areas:
            chain, current = [], identifier
            while (parent := self.areas[current]["parent_id"]) is not None:
                if parent not in self.areas:
                    raise ValueError(f"Unknown parent_id {parent!r} in gazetteer")
                if parent == identifier or parent in chain:
                    raise ValueError(f"Cyclic location hierarchy at {identifier}")
                chain.append(parent)
                current = parent
            self.ancestors[identifier] = chain

    def _compatible(self, left: str, right: str) -> bool:
        return left == right or left in self.ancestors[right] or right in self.ancestors[left]

    @lru_cache(maxsize=8192)
    def _matches(self, normalized: str) -> tuple[tuple, ...]:
        if not normalized:
            return ()
        tokens = normalized.split()
        hits: list[tuple[str, float, str, str, int, int]] = []
        for alias, identifier, parts in self.aliases:
            width = len(parts)
            for start in range(len(tokens) - width + 1):
                if tuple(tokens[start:start + width]) == parts:
                    hits.append((identifier, 100.0, "exact", alias, start, start + width))
        # Fuzziness operates on bounded token windows, not a whole long report.
        # Numbers must agree exactly to avoid merging Mirpur 10 and Mirpur 11.
        try:
            from rapidfuzz.fuzz import ratio
        except ImportError as exc:
            if hits:
                ratio = None
            else:
                raise RuntimeError("Fuzzy location extraction requires rapidfuzz; install requirements.txt") from exc
        if ratio is not None:
            for alias, identifier, parts in self.aliases:
                if len(alias.replace(" ", "")) < 5:
                    continue
                alias_numbers = re.findall(r"\d+", alias)
                best = None
                for width in range(max(1, len(parts) - 1), len(parts) + 2):
                    for start in range(len(tokens) - width + 1):
                        window = " ".join(tokens[start:start + width])
                        if re.findall(r"\d+", window) != alias_numbers:
                            continue
                        score = float(ratio(alias, window))
                        if score >= self.fuzzy_threshold and (best is None or score > best[1]):
                            best = (identifier, score, "fuzzy", alias, start, start + width)
                if best and best[1] < 100:
                    hits.append(best)
        # Prefer the precise child over a broad parent that also occurs inside
        # its alias. This preserves multiple unrelated child mentions.
        identifiers = {hit[0] for hit in hits}
        suppressed = {ancestor for child in identifiers for ancestor in self.ancestors[child] if ancestor in identifiers}
        return tuple(hit for hit in hits if hit[0] not in suppressed)

    def _resolve(self, text: str) -> dict:
        hits = self._matches(normalize_location_text(text))
        best_by_id = {}
        for hit in hits:
            old = best_by_id.get(hit[0])
            if old is None or (hit[1], hit[2] == "exact", len(hit[3])) > (old[1], old[2] == "exact", len(old[3])):
                best_by_id[hit[0]] = hit
        ranked = sorted(best_by_id.values(), key=lambda hit: (-hit[1], -len(hit[3]), hit[0]))
        candidates = sorted(best_by_id)
        chosen = ranked[0] if len(ranked) == 1 else None
        if len(ranked) > 1:
            top = ranked[0]
            overlap = all(max(top[4], other[4]) < min(top[5], other[5]) for other in ranked[1:])
            if overlap and top[1] - ranked[1][1] >= 3:
                chosen = top
        if chosen is None:
            return {"canonical_id": None, "parent_id": None, "ancestor_ids": [], "matched_alias": None,
                    "method": "unknown", "confidence": 0.0, "candidate_ids": candidates,
                    "ambiguous": len(candidates) > 1, "conflict": False}
        identifier, score, method, alias, _, _ = chosen
        return {"canonical_id": identifier, "parent_id": self.areas[identifier]["parent_id"],
                "ancestor_ids": list(self.ancestors[identifier]), "matched_alias": alias,
                "method": method, "confidence": score / 100, "candidate_ids": candidates,
                "ambiguous": False, "conflict": False}

    def extract(self, text: str, explicit_location: str = "") -> dict:
        """Resolve metadata and text, retaining explicit/text disagreement flags."""
        if not isinstance(text, str) or not isinstance(explicit_location, str):
            raise TypeError("Complaint and explicit location must be strings")
        textual = self._resolve(text)
        if not explicit_location.strip():
            return dict(textual, source="text", explicit_unresolved=False)
        explicit = self._resolve(explicit_location)
        if explicit["canonical_id"] is not None:
            identifier = explicit["canonical_id"]
            text_evidence = [textual["canonical_id"]] if textual["canonical_id"] is not None else textual["candidate_ids"]
            conflict = any(not self._compatible(identifier, candidate) for candidate in text_evidence)
            result = dict(explicit, source="explicit", explicit_unresolved=False)
            result["candidate_ids"] = sorted(set(explicit["candidate_ids"]) | set(textual["candidate_ids"]))
            result["conflict"] = conflict
            if conflict:
                result["method"] = "conflict"
            return result
        if explicit["ambiguous"]:
            return dict(explicit, source="explicit", explicit_unresolved=True)
        return dict(textual, source="text", explicit_unresolved=True)


def location_pair_features(a: dict, b: dict) -> dict[str, float]:
    """Encode canonical compatibility; ``distance`` is categorical, not km."""
    left, right = a.get("canonical_id"), b.get("canonical_id")
    missing = left is None or right is None
    compatible = not missing and (left == right or left in b.get("ancestor_ids", []) or right in a.get("ancestor_ids", []))
    return {
        "loc_match": float(not missing and left == right),
        "loc_distance_flag": float(not missing and not compatible),
        "loc_missing": float(missing),
        "loc_conflict": float(bool(a.get("conflict")) or bool(b.get("conflict"))),
        "loc_compatible": float(compatible),
    }
