"""
Text normalization for business names, addresses and countries.

Normalization is intentionally conservative: it lowercases, strips accents,
normalizes punctuation/whitespace, expands common address abbreviations and
business-suffix variants, and unifies "&"/"and" — but it never drops tokens
that could carry matching signal (e.g. numbers, unit identifiers).
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Common street/address abbreviations -> expanded form.
# Matched as whole tokens only (word boundaries), case-insensitive.
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "pkwy": "parkway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "mt": "mount",
    "ft": "fort",
    "sq": "square",
    "ter": "terrace",
    "pl": "place",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "po": "po",  # "po box" kept as-is, handled separately
}

# Common business-entity suffix normalization. Longer / multi-word patterns
# are applied first via regex so "private limited" collapses before the
# single-word table would otherwise leave "private" and "limited" separate.
BUSINESS_SUFFIX_PHRASES = [
    (r"\bprivate limited\b", "pvt ltd"),
    (r"\blimited liability company\b", "llc"),
    (r"\blimited liability partnership\b", "llp"),
    (r"\bcorporation\b", "corp"),
    (r"\bincorporated\b", "inc"),
    (r"\bcompany\b", "co"),
    (r"\blimited\b", "ltd"),
]


def _strip_accents(text: str) -> str:
    """Unicode-normalize and strip diacritics (e.g. 'Café' -> 'cafe')."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _basic_clean(text: str) -> str:
    """Lowercase, strip accents, normalize punctuation and whitespace."""
    text = _strip_accents(text)
    text = text.lower()
    # Unify "&" with "and"
    text = text.replace("&", " and ")
    # Replace punctuation (except alnum, whitespace) with a space
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Collapse multiple whitespace into one
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _apply_business_suffixes(text: str) -> str:
    for pattern, replacement in BUSINESS_SUFFIX_PHRASES:
        text = re.sub(pattern, replacement, text)
    return text


def _apply_address_abbreviations(text: str) -> str:
    tokens = text.split(" ")
    expanded = [ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in tokens]
    return " ".join(expanded)


def normalize_business_name(raw_name: object) -> str:
    """Normalize a business name string. Returns '' for missing values."""
    if raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)):
        return ""
    text = str(raw_name).strip()
    if not text:
        return ""
    text = _basic_clean(text)
    text = _apply_business_suffixes(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(raw_address: object) -> str:
    """Normalize an address string. Returns '' for missing values."""
    if raw_address is None or (isinstance(raw_address, float) and pd.isna(raw_address)):
        return ""
    text = str(raw_address).strip()
    if not text:
        return ""
    text = _basic_clean(text)
    text = _apply_address_abbreviations(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_country(raw_country: object) -> str:
    """
    Normalize a country string. Treated as an OPEN-SET string — no hard-coded
    list of valid countries, no mapping to ISO codes (that would require an
    external reference table). Just consistent casing/formatting.
    """
    if raw_country is None or (isinstance(raw_country, float) and pd.isna(raw_country)):
        return ""
    text = str(raw_country).strip()
    if not text:
        return ""
    text = _strip_accents(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_postal_code(raw_address: object) -> str:
    """
    Best-effort extraction of a postal/PIN-like code from a raw address
    string: a standalone run of 4-8 digits (optionally with one internal
    hyphen, e.g. US ZIP+4). Returns '' if none found. This is a heuristic
    signal only, used for an extra blocking/feature rule — not a source of
    truth.
    """
    if raw_address is None or (isinstance(raw_address, float) and pd.isna(raw_address)):
        return ""
    text = str(raw_address)
    matches = re.findall(r"\b\d{4,8}(?:-\d{3,4})?\b", text)
    if not matches:
        return ""
    # Prefer the last match — postal codes usually trail the address.
    return matches[-1].replace("-", "")


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with normalized_* columns added, keeping the
    original business_name / business_address / country columns intact.
    """
    out = df.copy()
    out["normalized_business_name"] = out["business_name"].map(normalize_business_name)
    out["normalized_business_address"] = out["business_address"].map(normalize_address)
    out["normalized_country"] = out["country"].map(normalize_country)
    out["postal_code"] = out["business_address"].map(extract_postal_code)
    return out
