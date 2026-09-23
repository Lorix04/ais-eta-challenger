"""M16B target-free canonical AIS destination resolver.

The resolver intentionally uses only the observed destination string and a
versioned external/curated lookup table.  No reference ETA, target, outcome,
position, MMSI identity or split label participates in destination resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
import re
import unicodedata

import pandas as pd

M16B_RESOLVER_VERSION = "m16b-canonical-destination-v1-20260921"
M16B_FUZZY_THRESHOLD = 0.94
M16B_FUZZY_MARGIN = 0.04

UNKNOWN_TOKENS = {"", "UNKNOWN", "N A", "NA", "N/A", "NONE", "NULL", "NOT AVAILABLE"}
FOR_ORDER_TOKENS = {"FOR ORDER", "FOR ORDERS", "ORDER", "ORDERS", "FO"}
NON_SPECIFIC_TOKENS = {"ATLANTIC OCEAN", "MARE", "WORKSITE", "IN PORT"}
AMBIGUOUS_GEOGRAPHIES = {"MALTA"}


def ascii_upper(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.upper().strip()


def normalize_text(value: object) -> str:
    text = ascii_upper(value)
    text = text.replace("→", ">").replace("=>", ">").replace("->", ">")
    text = re.sub(r">+", ">", text)
    text = re.sub(r"[,_;|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", ascii_upper(value))


def route_terminal(normalized: str) -> tuple[str, bool]:
    if ">" not in normalized:
        return normalized, False
    parts = [p.strip() for p in normalized.split(">") if p.strip()]
    return (parts[-1] if parts else normalized), True


@dataclass(frozen=True)
class DestinationResolution:
    raw_destination: str
    normalized_destination: str
    terminal_segment: str
    canonical_destination: str
    canonical_unlocode: str | None
    canonical_port_name: str | None
    canonical_country: str | None
    canonical_lat: float | None
    canonical_lon: float | None
    resolution_method: str
    resolution_confidence: float
    is_resolved_port: bool
    is_non_specific: bool
    is_route_expression: bool
    catalog_version: str

    def to_dict(self) -> dict:
        return asdict(self)


class DestinationResolver:
    def __init__(self, catalog: pd.DataFrame):
        required = {"unlocode", "port_name", "country", "lat", "lon", "aliases", "source_url", "source_note"}
        missing = required - set(catalog.columns)
        if missing:
            raise ValueError(f"M16B catalog missing columns: {sorted(missing)}")
        self.catalog = catalog.copy()
        self.catalog["unlocode"] = self.catalog["unlocode"].astype(str).map(compact_token)
        if self.catalog["unlocode"].duplicated().any():
            raise ValueError("M16B catalog has duplicate UN/LOCODE values")

        self.by_code: dict[str, dict] = {}
        self.alias_to_code: dict[str, str] = {}
        self.fuzzy_aliases: list[tuple[str, str]] = []
        for row in self.catalog.to_dict(orient="records"):
            code = row["unlocode"]
            self.by_code[code] = row
            aliases = [row["port_name"], code]
            aliases.extend(str(row.get("aliases") or "").split(";"))
            for alias in aliases:
                norm = normalize_text(alias)
                if not norm:
                    continue
                key = compact_token(norm) if re.fullmatch(r"[A-Z]{2}\s?[A-Z0-9]{3}", norm) else norm
                existing = self.alias_to_code.get(key)
                if existing and existing != code:
                    raise ValueError(f"ambiguous M16B exact alias {key}: {existing}/{code}")
                self.alias_to_code[key] = code
                if len(norm) >= 5 and not re.fullmatch(r"[A-Z]{2}\s?[A-Z0-9]{3}", norm):
                    self.fuzzy_aliases.append((norm, code))

    @classmethod
    def from_csv(cls, path: str | Path) -> "DestinationResolver":
        return cls(pd.read_csv(path))

    def _catalog_resolution(self, raw: str, normalized: str, terminal: str, route: bool, code: str, method: str, conf: float) -> DestinationResolution:
        row = self.by_code[code]
        lat = None if pd.isna(row["lat"]) else float(row["lat"])
        lon = None if pd.isna(row["lon"]) else float(row["lon"])
        return DestinationResolution(
            raw, normalized, terminal, code, code, str(row["port_name"]), str(row["country"]),
            lat, lon, method, conf, True, False, route, M16B_RESOLVER_VERSION,
        )

    def resolve(self, value: object) -> DestinationResolution:
        raw = "" if value is None else str(value)
        normalized = normalize_text(value)
        terminal, route = route_terminal(normalized)
        terminal_norm = normalize_text(terminal)
        terminal_compact = compact_token(terminal_norm)

        if terminal_norm in UNKNOWN_TOKENS or terminal_compact in {compact_token(x) for x in UNKNOWN_TOKENS}:
            return DestinationResolution(raw, normalized, terminal_norm, "UNKNOWN", None, None, None, None, None,
                                         "unknown", 1.0, False, True, route, M16B_RESOLVER_VERSION)

        if terminal_norm in FOR_ORDER_TOKENS or any(x in normalized for x in ("FOR ORDER", "FOR ORDERS")):
            return DestinationResolution(raw, normalized, terminal_norm, "FOR_ORDERS", None, None, None, None, None,
                                         "for_orders", 1.0, False, True, route, M16B_RESOLVER_VERSION)

        if terminal_norm in NON_SPECIFIC_TOKENS:
            return DestinationResolution(raw, normalized, terminal_norm, "NON_SPECIFIC", None, None, None, None, None,
                                         "non_specific", 1.0, False, True, route, M16B_RESOLVER_VERSION)

        if terminal_norm in AMBIGUOUS_GEOGRAPHIES:
            return DestinationResolution(raw, normalized, terminal_norm, f"AMBIGUOUS:{terminal_norm}", None, None, None, None, None,
                                         "ambiguous_geography", 1.0, False, False, route, M16B_RESOLVER_VERSION)

        if terminal_compact in self.by_code:
            method = "route_terminal_unlocode" if route else "exact_unlocode"
            return self._catalog_resolution(raw, normalized, terminal_norm, route, terminal_compact, method, 1.0)

        exact_key = terminal_compact if re.fullmatch(r"[A-Z]{2}\s?[A-Z0-9]{3}", terminal_norm) else terminal_norm
        if exact_key in self.alias_to_code:
            method = "route_terminal_alias" if route else "curated_alias"
            return self._catalog_resolution(raw, normalized, terminal_norm, route, self.alias_to_code[exact_key], method, 0.99)

        # Conservative fuzzy match only against human-readable curated aliases.
        scores: dict[str, float] = {}
        if len(terminal_norm) >= 5:
            for alias, code in self.fuzzy_aliases:
                score = SequenceMatcher(None, terminal_norm, alias).ratio()
                scores[code] = max(scores.get(code, 0.0), score)
        if scores:
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            best_code, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            if best_score >= M16B_FUZZY_THRESHOLD and (best_score - second_score) >= M16B_FUZZY_MARGIN:
                method = "route_terminal_fuzzy" if route else "conservative_fuzzy"
                return self._catalog_resolution(raw, normalized, terminal_norm, route, best_code, method, float(best_score))

        # Formatting-only canonicalization for UN/LOCODE-like values not in the
        # compact curated catalog.  These are never asserted to be verified ports.
        if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{3}", terminal_compact):
            return DestinationResolution(raw, normalized, terminal_norm, terminal_compact, terminal_compact, None, terminal_compact[:2], None, None,
                                         "unlocode_like_unverified", 0.70, False, False, route, M16B_RESOLVER_VERSION)

        unresolved_key = terminal_norm or "UNKNOWN"
        return DestinationResolution(raw, normalized, terminal_norm, f"TEXT:{unresolved_key}", None, None, None, None, None,
                                     "unresolved_text", 0.0, False, False, route, M16B_RESOLVER_VERSION)

    def transform(self, series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame([self.resolve(v).to_dict() for v in series.tolist()], index=series.index)
