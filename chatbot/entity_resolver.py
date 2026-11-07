"""Resolve institution mentions to canonical institute_key values."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import unicodedata
import pandas as pd
from rapidfuzz import fuzz, process


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def expand_common_abbreviations(value: str) -> str:
    """Expand common Indian higher-education abbreviations before fuzzy matching."""
    text = normalize_name(value)
    replacements = [
        (r"\biit\b", "indian institute of technology"),
        (r"\bnit\b", "national institute of technology"),
        (r"\biisc\b", "indian institute of science"),
        (r"\biim\b", "indian institute of management"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


class InstitutionResolver:
    def __init__(self, df: pd.DataFrame):
        required = {
            "institute_key", "name", "display_name",
            "name_in_source", "city", "state"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset lacks institution resolver columns: {sorted(missing)}")

        source = df.dropna(subset=["institute_key"]).copy()
        self.institutions: dict[str, dict] = {}
        self.alias_to_keys: dict[str, set[str]] = defaultdict(set)

        for key, group in source.groupby("institute_key", sort=False):
            def first_value(column: str) -> str | None:
                values = group[column].dropna().astype(str).str.strip()
                return values.mode().iloc[0] if not values.empty else None

            institution = {
                "institute_key": str(key),
                "name": first_value("name"),
                "display_name": first_value("display_name"),
                "city": first_value("city"),
                "state": first_value("state"),
            }
            self.institutions[str(key)] = institution

            aliases = set()
            for column in ("name", "display_name", "name_in_source"):
                aliases.update(
                    str(v).strip()
                    for v in group[column].dropna().unique()
                    if str(v).strip()
                )
            for alias in aliases:
                normalized = normalize_name(alias)
                if normalized:
                    self.alias_to_keys[normalized].add(str(key))

        self.aliases = list(self.alias_to_keys.keys())

    def _apply_location(self, keys: set[str], state: str | None,
                        city: str | None) -> set[str]:
        result = set(keys)
        if state:
            state_norm = normalize_name(state)
            result = {
                key for key in result
                if normalize_name(self.institutions[key].get("state") or "") == state_norm
            }
        if city:
            city_norm = normalize_name(city)
            result = {
                key for key in result
                if normalize_name(self.institutions[key].get("city") or "") == city_norm
            }
        return result

    def _candidate_payload(self, key: str, score: float) -> dict:
        item = dict(self.institutions[key])
        item["confidence"] = round(float(score), 1)
        return item

    def resolve(self, query: str, state: str | None = None,
                city: str | None = None, limit: int = 5,
                min_score: float = 88.0, ambiguity_gap: float = 5.0) -> dict:
        query_norm = normalize_name(query)
        expanded = expand_common_abbreviations(query)
        if not query_norm:
            return {"status": "not_found", "query": query, "institution": None, "candidates": []}

        # Prefer exact aliases; location can disambiguate campuses sharing a name.
        exact_keys = self.alias_to_keys.get(query_norm, set())
        if not exact_keys and expanded != query_norm:
            exact_keys = self.alias_to_keys.get(expanded, set())
        exact_keys = self._apply_location(set(exact_keys), state, city)
        if len(exact_keys) == 1:
            key = next(iter(exact_keys))
            return {
                "status": "matched", "match_type": "exact", "query": query,
                "institution": self._candidate_payload(key, 100), "candidates": []
            }
        if len(exact_keys) > 1:
            candidates = [self._candidate_payload(k, 100) for k in sorted(exact_keys)]
            return {
                "status": "ambiguous", "match_type": "exact", "query": query,
                "institution": None, "candidates": candidates[:limit]
            }

        # Very short abbreviations (for example, "SRM" or "IIT") are too
        # ambiguous for fuzzy matching. Require an exact alias or more context.
        compact_query = query_norm.replace(" ", "")
        if len(compact_query) <= 3:
            return {
                "status": "needs_clarification",
                "match_type": "short_query",
                "query": query,
                "institution": None,
                "candidates": [],
                "message": (
                    "This institution abbreviation is too broad to resolve safely. "
                    "Provide a longer name, city, or state."
                ),
            }

        # Fuzzy matching is advisory. Never silently select a close campus match.
        search_query = expanded or query_norm
        matches = process.extract(
            search_query, self.aliases, scorer=fuzz.WRatio, limit=30
        )
        best_by_key: dict[str, float] = {}
        for alias, score, _ in matches:
            for key in self.alias_to_keys[alias]:
                current = best_by_key.get(key, 0)
                best_by_key[key] = max(current, float(score))

        candidate_keys = self._apply_location(set(best_by_key), state, city)
        ranked = sorted(
            ((key, best_by_key[key]) for key in candidate_keys),
            key=lambda item: item[1], reverse=True
        )
        candidates = [self._candidate_payload(k, s) for k, s in ranked[:limit]]

        if not ranked or ranked[0][1] < min_score:
            return {
                "status": "not_found", "match_type": "fuzzy", "query": query,
                "institution": None, "candidates": candidates
            }

        second_score = ranked[1][1] if len(ranked) > 1 else -1
        if len(ranked) > 1 and ranked[0][1] - second_score < ambiguity_gap:
            return {
                "status": "ambiguous", "match_type": "fuzzy", "query": query,
                "institution": None, "candidates": candidates
            }

        key, score = ranked[0]
        return {
            "status": "matched", "match_type": "fuzzy", "query": query,
            "institution": self._candidate_payload(key, score),
            "candidates": candidates[1:]
        }


if __name__ == "__main__":
    from data_loader import load_dataset
    data = load_dataset()
    resolver = InstitutionResolver(data)
    for sample in ["IIT Dharwad", "IIT Madras", "NIT Karnataka", "SRM"]:
        print(f"\n{sample}: {resolver.resolve(sample)}")
