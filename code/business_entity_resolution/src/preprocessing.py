"""
preprocessing.py

Cleans up the raw text fields (name, address, country) so that things
like "Corp" and "Corporation" or "Rd" and "Road" compare as equal
instead of looking like completely different strings.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Common address word shortcuts -> full word.
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
    "po": "po",
}

# Business suffix patterns. Longer phrases go first so "private limited"
# gets caught before we'd otherwise leave "private" and "limited" separate.
BUSINESS_SUFFIX_PATTERNS = [
    (r"\bprivate limited\b", "pvt ltd"),
    (r"\blimited liability company\b", "llc"),
    (r"\blimited liability partnership\b", "llp"),
    (r"\bcorporation\b", "corp"),
    (r"\bincorporated\b", "inc"),
    (r"\bcompany\b", "co"),
    (r"\blimited\b", "ltd"),
]


def _strip_accents(text: str) -> str:
    """Turn accented letters into plain ones, e.g. 'Cafe' from 'Café'."""
    normalized = unicodedata.normalize("NFKD", text)
    result = ""
    for ch in normalized:
        if not unicodedata.combining(ch):
            result += ch
    return result


def _basic_clean(text: str) -> str:
    """Lowercase, remove accents, clean up punctuation and spacing."""
    text = _strip_accents(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _apply_business_suffixes(text: str) -> str:
    for pattern, replacement in BUSINESS_SUFFIX_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def _apply_address_abbreviations(text: str) -> str:
    words = text.split(" ")
    new_words = []
    for word in words:
        if word in ADDRESS_ABBREVIATIONS:
            new_words.append(ADDRESS_ABBREVIATIONS[word])
        else:
            new_words.append(word)
    return " ".join(new_words)


def normalize_business_name(raw_name) -> str:
    """Clean up a business name. Returns '' if there's nothing there."""
    if raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)):
        return ""

    text = str(raw_name).strip()
    if not text:
        return ""

    text = _basic_clean(text)
    text = _apply_business_suffixes(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(raw_address) -> str:
    """Clean up an address. Returns '' if there's nothing there."""
    if raw_address is None or (isinstance(raw_address, float) and pd.isna(raw_address)):
        return ""

    text = str(raw_address).strip()
    if not text:
        return ""

    text = _basic_clean(text)
    text = _apply_address_abbreviations(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_country(raw_country) -> str:
    """
    Clean up a country name. We don't try to match it against a fixed
    list of countries - just clean formatting so the same country name
    written differently still matches.
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


def extract_postal_code(raw_address) -> str:
    """
    Try to pull a postal/PIN code out of the address (a run of 4-8
    digits). This is just a rough guess, not guaranteed to be right,
    but it's a useful extra signal for blocking/matching.
    """
    if raw_address is None or (isinstance(raw_address, float) and pd.isna(raw_address)):
        return ""

    text = str(raw_address)
    matches = re.findall(r"\b\d{4,8}(?:-\d{3,4})?\b", text)
    if not matches:
        return ""

    # postal codes are usually near the end of the address, so take the last match
    last_match = matches[-1]
    return last_match.replace("-", "")


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized_* columns to a dataframe, keeping the original columns too."""
    df = df.copy()
    df["normalized_business_name"] = df["business_name"].map(normalize_business_name)
    df["normalized_business_address"] = df["business_address"].map(normalize_address)
    df["normalized_country"] = df["country"].map(normalize_country)
    df["postal_code"] = df["business_address"].map(extract_postal_code)
    return df