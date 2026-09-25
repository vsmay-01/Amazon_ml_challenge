"""
normalization.py -- Multi-view name / address / country normalization.

Per the design doc (Section 13): normalization must remove formatting
variation without destroying identity-bearing signals, and multiple views
are maintained rather than destructively rewriting the original.

No external geocoding or business-data lookup is used anywhere here, per
the challenge's hard compliance gates.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

LEGAL_SUFFIXES = [
    "inc", "incorporated", "llc", "l l c", "ltd", "limited", "corp",
    "corporation", "co", "company", "plc", "llp", "lp", "pvt", "private",
    "pty", "gmbh", "sa", "srl", "bv", "ag", "kk", "sarl",
]

ADDRESS_ABBREVIATIONS = {
    "street": "st", "st.": "st", "avenue": "ave", "ave.": "ave",
    "boulevard": "blvd", "blvd.": "blvd", "road": "rd", "rd.": "rd",
    "drive": "dr", "dr.": "dr", "lane": "ln", "ln.": "ln",
    "suite": "ste", "ste.": "ste", "apartment": "apt", "apt.": "apt",
    "floor": "fl", "fl.": "fl", "building": "bldg", "bldg.": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
}


def _unicode_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def _basic_clean(text: str) -> str:
    text = text.lower().strip()
    text = _unicode_normalize(text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_legal_suffix(name_norm: str) -> str:
    tokens = name_norm.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def token_sorted(text_norm: str) -> str:
    return " ".join(sorted(text_norm.split()))


def char_ngrams(text_norm: str, n: int = 3) -> set[str]:
    t = text_norm.replace(" ", "")
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def abbreviate_address_tokens(addr_norm: str) -> str:
    tokens = addr_norm.split()
    out = [ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in tokens]
    return " ".join(out)


@dataclass
class NameViews:
    raw: str
    norm: str
    no_suffix: str
    token_sorted: str
    char_ngrams: set = field(default_factory=set)


@dataclass
class AddressViews:
    raw: str
    norm: str
    tokens: list = field(default_factory=list)
    char_ngrams: set = field(default_factory=set)
    abbreviated: str = ""


def normalize_name(raw_name: str) -> NameViews:
    norm = _basic_clean(raw_name)
    no_suffix = strip_legal_suffix(norm)
    return NameViews(
        raw=raw_name,
        norm=norm,
        no_suffix=no_suffix,
        token_sorted=token_sorted(no_suffix),
        char_ngrams=char_ngrams(no_suffix),
    )


def normalize_address(raw_address: str) -> AddressViews:
    norm = _basic_clean(raw_address)
    abbreviated = abbreviate_address_tokens(norm)
    return AddressViews(
        raw=raw_address,
        norm=norm,
        tokens=norm.split(),
        char_ngrams=char_ngrams(norm),
        abbreviated=abbreviated,
    )


def normalize_country(raw_country: str) -> str:
    """
    Treat country as an open-set string attribute (no whitelist): the training
    distribution covers US/India but the test set also contains France, so we
    must not hard-code a US/India-only mapping. We only do light cleanup.
    """
    return _basic_clean(raw_country)
