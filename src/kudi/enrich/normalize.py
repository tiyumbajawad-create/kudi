"""Merchant string normalization: uppercase -> strip processor prefixes
-> strip phone/store/geo noise -> collapse whitespace -> alias
resolution against a curated canonical list (design doc §5.1). This
stage alone dramatically improves everything downstream -- it's the
difference between a classifier seeing 'SQ *BLUE BOTTLE COF 4155551234
CA' a thousand slightly-different ways and seeing 'BLUE BOTTLE COFFEE'
consistently.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from kudi.enrich.merchant_aliases import CANONICAL_MERCHANTS, DOMAIN_ALIASES

ALIAS_MATCH_THRESHOLD = 85.0

# rapidfuzz's scorers are case-sensitive by default; match everything in
# upper case and map back to the properly-cased canonical name.
_CANONICAL_UPPER_TO_NAME = {name.upper(): name for name in CANONICAL_MERCHANTS}
_CANONICAL_UPPER_LIST = list(_CANONICAL_UPPER_TO_NAME.keys())

# Processor / payment-rail prefixes that precede the actual merchant name.
# NOTE: Uber/Lyft/DoorDash are themselves merchants in this catalog (not
# a prefix in front of some other business), so only the app-specific
# suffix after the merchant name is stripped -- the merchant name itself
# must survive.
_PROCESSOR_PREFIXES = [
    r"SQ \*",
    r"TST\* ?",
    r"PAYPAL \*",
    r"\*(TRIP|EATS|RIDE) ",
]
_PREFIX_RE = re.compile("|".join(_PROCESSOR_PREFIXES), re.IGNORECASE)

_DOMAIN_SUFFIX_RE = re.compile(r"\.COM\b", re.IGNORECASE)

_PHONE_RE = re.compile(r"\b\d{10}\b")
_STORE_ID_RE = re.compile(r"#\d+")
_TRUNCATION_RE = re.compile(r"\*[A-Z0-9]{2,4}\b")
_TRAILING_DIGITS_RE = re.compile(r"\b\d{3,6}\b")

_STATES = {
    "GA",
    "TX",
    "CO",
    "WA",
    "ID",
    "FL",
    "NY",
    "CA",
    "US",
}
_CITY_WORDS = {
    "ATLANTA",
    "AUSTIN",
    "DENVER",
    "SEATTLE",
    "BOISE",
    "TAMPA",
}

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TRAIL_RE = re.compile(r"[\s*./]+$")


def _strip_geo_tokens(text: str) -> str:
    words = [w for w in text.split() if w not in _STATES and w not in _CITY_WORDS]
    return " ".join(words)


def clean_descriptor(raw: str) -> str:
    """The deterministic regex cascade, without alias resolution.
    Exposed separately because the classifier wants this cleaned text as
    a feature even when no canonical alias is confidently found."""
    text = raw.upper()
    text = _PREFIX_RE.sub("", text)
    text = _DOMAIN_SUFFIX_RE.sub("", text)
    text = text.replace("/", " ")
    text = _PHONE_RE.sub("", text)
    text = _STORE_ID_RE.sub("", text)
    text = _TRUNCATION_RE.sub("", text)
    text = _TRAILING_DIGITS_RE.sub("", text)
    text = _strip_geo_tokens(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _PUNCT_TRAIL_RE.sub("", text)
    return text


def _resolve_domain_alias(raw_upper: str) -> str | None:
    for token, canonical in DOMAIN_ALIASES.items():
        if token in raw_upper:
            return canonical
    return None


def normalize_merchant(raw: str) -> str:
    """Full cascade: clean, then resolve against the canonical alias
    list via fuzzy matching. Falls back to the cleaned (but unresolved)
    string when nothing matches confidently -- still far more useful to
    the classifier than the raw descriptor."""
    domain_hit = _resolve_domain_alias(raw.upper())
    if domain_hit:
        return domain_hit

    cleaned = clean_descriptor(raw)
    if not cleaned:
        return cleaned

    match = process.extractOne(
        cleaned,
        _CANONICAL_UPPER_LIST,
        scorer=fuzz.token_set_ratio,
        score_cutoff=ALIAS_MATCH_THRESHOLD,
    )
    if match:
        return _CANONICAL_UPPER_TO_NAME[match[0]]
    return cleaned
