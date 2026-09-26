"""Phase 2 — Text representation.

Builds *multiple parallel representations* of a name/address string rather than
one canonical cleaned string, per the plan's reconciliation of four external
reviews: Unicode normalization alone misses transliteration noise, and ASCII
folding destroys information needed to keep genuinely different non-Latin
businesses distinguishable.

Representations produced, all derived solely from the input string (no
external data, no lookups, no network):
  - normalized:      NFKC + casefold + Latin-diacritic fold + punctuation/whitespace
                     cleanup; non-Latin scripts preserved. Names: look-alike digits
                     fixed ("M0dern"), www/com/NULL dropped. Addresses: leading zeros
                     stripped, street types / state names / renamed cities mapped to
                     one form (hand-authored tables below)
  - tokens:          whitespace-split tokens of `normalized`
  - suffix tokens:   legal-suffix tokens detected and separated out (US/India/France),
                     kept as a *flag*, never silently discarded from the record's identity
  - char_ngrams:     word-boundary-padded character 3-5 grams over `normalized`
  - script_runs:     contiguous same-script substrings (Latin, 9 Indic scripts, digits)
  - dominant_script: the script covering the most characters
  - skeleton:        a hand-authored Indic -> Latin *rough* transliteration for all 9
                      major Indic scripts, built from the public Unicode block layout
                      (not a third-party library)
  - phonetic codes:  Soundex-like consonant-class codes of skeleton words, so a
                      native-script name can meet its English spelling

Changing any of this changes model inputs: bump REPR_VERSION, retrain, and
the inference script refuses models trained on an older representation.

`anyascii` is deliberately NOT used in this module by default -- see plan.md
Phase 2 for the reconciliation across reviews. It stays installed only as an
optional experimental comparison signal (`anyascii_repr`, opt-in, never called
by `build_name_repr`/`build_address_repr`), not part of the graded pipeline.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import regex  # third-party: needed for \p{Script=...}, stdlib `re` doesn't support it

# Version of the text representation; stored with each trained model.
# v3: NFKC/casefold only, 4 Indic scripts. v4: diacritic fold, name/address
# cleanup + canonical tables, 9 Indic scripts, phonetic codes. v5: same text,
# but feature semantics changed (digit Jaccard -1 when a side has no digits),
# so v4 models must not run on this code. v6: addr_key uses the street token
# (was "18|18"), postcodes keep leading zeros, unit numbers never taken as the
# house number.
REPR_VERSION = "v6"

# ---------------------------------------------------------------------------
# Legal suffix normalization -- conservative, not English-only (reviewer #1/#4
# correction: don't hardcode to only the two training countries; France
# appears only in test).
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    # US-style
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "llp", "pllc", "ltd", "limited",
    # India-style
    "pvt", "private",
    # France-style
    "sarl", "sas", "sasu", "sa", "eurl", "sci",
}

# same legal form, different spelling ("ltd"/"limited")
SUFFIX_FAMILY = {"limited": "ltd", "incorporated": "inc", "corporation": "corp", "company": "co", "private": "pvt"}

_WS_RE = re.compile(r"\s+")
_AMPERSAND_RE = re.compile(r"(?<=\s)&(?=\s)|^&(?=\s)|(?<=\s)&$")

# Categories to strip as punctuation/symbols. Deliberately NOT using regex \w
# for this: Python's \w excludes Unicode combining marks (category Mc/Mn --
# e.g. Devanagari/Tamil dependent vowel signs and virama/pulli), so a naive
# `[^\w\s]` punctuation strip silently shreds Indic-script words into single
# letters. Caught by testing this module against real EDA sample rows before
# building anything on top of it -- see the smoke test.
_STRIP_CATEGORY_PREFIXES = ("P", "S")  # Punctuation, Symbol


def _strip_punctuation(s: str) -> str:
    return "".join(
        " " if unicodedata.category(ch).startswith(_STRIP_CATEGORY_PREFIXES) else ch
        for ch in s
    )


_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"))  # zero-width chars, soft hyphen
_LATIN_LIGATURES = str.maketrans({"œ": "oe", "æ": "ae", "ø": "o", "đ": "d", "ł": "l", "ı": "i"})


def _fold_latin_diacritics(s: str) -> str:
    """Remove Latin diacritics only (combining marks U+0300-U+036F): the
    S2/S3 sources inject fake accents ("Çlub", "FRÀNCE") that S1 never has.
    Indic vowel signs / virama live in their own blocks and are untouched."""
    if s.isascii():
        return s
    d = unicodedata.normalize("NFD", s.translate(_LATIN_LIGATURES))
    return unicodedata.normalize("NFC", "".join(ch for ch in d if not 0x300 <= ord(ch) <= 0x36F))


def normalize_unicode(s: str) -> str:
    """NFKC normalize + casefold + Latin-diacritic fold + punctuation->space
    + whitespace collapse.

    Non-Latin scripts are preserved (no ASCII transliteration here), and
    Indic combining marks (vowel signs, virama/pulli) are preserved too --
    this is the representation everything else is built from.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold().translate(_ZERO_WIDTH)
    s = _fold_latin_diacritics(s)
    s = _AMPERSAND_RE.sub(" and ", s)
    s = _strip_punctuation(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# Placeholder values some sources use for a missing field.
_NULL_TOKENS = {"null", "none", "nan", "nil"}

# Digits used as look-alike letters inside names ("M0dern", "5tudio", "Hatt0n").
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})
_NAME_DROP_TOKENS = {"www", "com", "http", "https"} | _NULL_TOKENS


def _fix_name_token(t: str) -> str:
    """Exactly one look-alike digit inside an otherwise alphabetic token with
    >= 3 letters -> letter. Ordinals ("2nd", "21st", "10th") and codes ("3m",
    "b2b") have < 3 letters or > 1 digit, so they are left alone."""
    n_digits = sum(ch.isdigit() for ch in t)
    if n_digits != 1 or len(t) - 1 < 3 or not t.isalnum() or not t.isascii():
        return t
    return t.translate(_LEET)


def clean_name_tokens(normalized: str) -> list[str]:
    return [_fix_name_token(t) for t in normalized.split() if t not in _NAME_DROP_TOKENS]


# ---------------------------------------------------------------------------
# Address canonicalization: hand-authored variant tables (street types, state
# names in full / native script -> the short code the other sources use,
# renamed cities). Applied token-wise with up to 3-token phrases.
# ---------------------------------------------------------------------------
_ADDR_SINGLE = {
    # street types (US) -- expanded to the long form S1 uses
    "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "rd": "road", "dr": "drive",
    "ln": "lane", "cir": "circle", "blvd": "boulevard", "bd": "boulevard", "pl": "place",
    "pkwy": "parkway", "hwy": "highway", "trl": "trail", "ter": "terrace", "sq": "square", "cres": "crescent",
    "ctr": "center", "centre": "center", "apt": "unit", "ste": "suite", "ft": "fort", "twp": "township",
    # these abbreviations double as state codes, so the SHORT form is canonical
    "court": "ct", "floor": "fl", "flr": "fl", "mount": "mt",
    # French (unambiguous everywhere; ambiguous ones are in _ADDR_COUNTRY["France"])
    "che": "chemin", "imp": "impasse", "fbg": "faubourg", "rte": "route",
    # US states (full -> code)
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky",
    "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
    # Indian states (full / native script -> the codes source 3 uses)
    "maharashtra": "mh", "महाराष्ट्र": "mh", "delhi": "dl", "दिल्ली": "dl", "karnataka": "ka", "ಕರ್ನಾಟಕ": "ka",
    "தமிழ்நாடு": "tn", "gujarat": "gj", "ગુજરાત": "gj", "পশ্চিমবঙ্গ": "wb", "telangana": "tg", "తెలంగాణ": "tg",
    "haryana": "hr", "हरियाणा": "hr", "rajasthan": "rj", "राजस्थान": "rj", "kerala": "kl", "keralam": "kl",
    "കേരളം": "kl", "bihar": "br", "बिहार": "br", "ఆంధ్రప్రదేశ్": "ap", "punjab": "pb", "ਪੰਜਾਬ": "pb",
    "odisha": "od", "orissa": "od", "ଓଡ଼ିଶା": "od",
    # renamed cities
    "bengaluru": "bangalore", "gurugram": "gurgaon", "bombay": "mumbai", "calcutta": "kolkata",
    "madras": "chennai", "ahmadabad": "ahmedabad",
}
_ADDR_MULTI = {
    ("new", "hampshire"): "nh", ("new", "jersey"): "nj", ("new", "mexico"): "nm", ("new", "york"): "ny",
    ("north", "carolina"): "nc", ("north", "dakota"): "nd", ("south", "carolina"): "sc",
    ("south", "dakota"): "sd", ("west", "virginia"): "wv", ("rhode", "island"): "ri",
    ("district", "of", "columbia"): "dc",
    ("uttar", "pradesh"): "up", ("उत्तर", "प्रदेश"): "up", ("tamil", "nadu"): "tn", ("west", "bengal"): "wb",
    ("madhya", "pradesh"): "mp", ("मध्य", "प्रदेश"): "mp", ("andhra", "pradesh"): "ap",
    ("n", "a"): "",  # "N/A" placeholder
}
# Country-specific overrides. France (test only): S2/S3 abbreviate "Saint" as "st" (US: street), write
# allee/chemin/cours/quai as all/ch/crs/q, add "N°", and give the departement (Nord, Pas-de-Calais,
# Gironde, Loire-Atlantique) where S1 gives the region -- found by comparing France S1 vs S2 address
# tokens on the unlabeled test files. Both admin levels map to one region code.
_ADDR_COUNTRY = {
    "France": {
        "st": "saint", "ste": "sainte", "r": "rue", "all": "allee", "ch": "chemin", "crs": "cours", "q": "quai",
        "no": "", "n": "", "nord": "reg_hdf", "gironde": "reg_naq",  # "N°"/"nº" -> n / no
    },
}
_ADDR_MULTI_COUNTRY = {
    "France": {
        ("hauts", "de", "france"): "reg_hdf", ("pas", "de", "calais"): "reg_hdf",
        ("nouvelle", "aquitaine"): "reg_naq",
        ("pays", "de", "la", "loire"): "reg_pdl", ("loire", "atlantique"): "reg_pdl",
    },
}
_ADDR_TABLES = {
    c: ({**_ADDR_SINGLE, **_ADDR_COUNTRY.get(c, {})}, {**_ADDR_MULTI, **_ADDR_MULTI_COUNTRY.get(c, {})})
    for c in set(_ADDR_COUNTRY) | set(_ADDR_MULTI_COUNTRY)
}
_ADDR_MULTI_MAX = max(len(k) for tbl in [_ADDR_MULTI, *_ADDR_MULTI_COUNTRY.values()] for k in tbl)


def canonicalize_address_tokens(tokens: list[str], country: str = "") -> list[str]:
    single, multi = _ADDR_TABLES.get(country, (_ADDR_SINGLE, _ADDR_MULTI))
    out: list[str] = []
    i = 0
    while i < len(tokens):
        for n in range(min(_ADDR_MULTI_MAX, len(tokens) - i), 1, -1):
            rep = multi.get(tuple(tokens[i : i + n]))
            if rep is not None:
                if rep:
                    out.append(rep)
                i += n
                break
        else:
            t = tokens[i]
            if t.isdigit():
                t = t.lstrip("0") or "0"  # "0018 rue" == "18 rue"
            elif t not in _NULL_TOKENS:
                t = single.get(t, t)
            else:
                t = ""
            if t:
                out.append(t)
            i += 1
    return out


# canonical street types (after canonicalize_address_tokens) and unit words
_STREET_TYPES = {
    "street", "avenue", "road", "drive", "lane", "circle", "boulevard", "place", "parkway", "highway", "trail",
    "terrace", "square", "crescent", "ct", "way", "pike", "loop", "row", "path", "plaza", "marg", "salai", "gali",
    "cross", "main", "nagar", "rue", "chemin", "impasse", "route", "allee", "quai", "faubourg",
}
_UNIT_WORDS = {"fl", "unit", "suite", "apt", "flat", "pmb", "box", "lot", "po", "room", "shop", "plot"}


def address_number_key(addr_norm: str) -> tuple[str, str]:
    """(house number, street word), wherever they sit -- sources reorder
    fields ("napoleon 18 meadowlark lane oh"). Prefer a digit run (<= 5
    digits, so 6-digit PINs are skipped) followed within 4 tokens by a street
    type; otherwise the first such digit run not preceded by a unit word
    ("fl 0", "unit 4", "pmb 9424"). Street word = next non-digit token."""
    toks = addr_norm.split()
    fallback = ("", "")
    for i, t in enumerate(toks):
        if not (t.isdigit() and len(t) <= 5) or (i > 0 and toks[i - 1] in _UNIT_WORDS):
            continue  # "fl 4 100 main street" -> 100, not the floor
        street = next((u for u in toks[i + 1:] if not u.isdigit()), "")
        if any(x in _STREET_TYPES for x in toks[i + 1:i + 5]):
            return t, street
        if not fallback[0]:
            fallback = (t, street)
    return fallback


def name_core(name_norm: str) -> str:
    """Name without legal suffixes and spaces ("advanced-columbia inc" and
    "Advanced Columbia Incorporated" -> "advancedcolumbia")."""
    return "".join(t for t in name_norm.split() if t not in LEGAL_SUFFIXES)


def tokenize(normalized: str) -> list[str]:
    return normalized.split() if normalized else []


def strip_legal_suffixes(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Return (tokens_without_suffixes, suffix_tokens_found).

    Only reliably catches Latin-script suffix tokens (Inc/Ltd/Pvt/SARL/...).
    Non-Latin equivalents (e.g. a suffix written in Bengali script) are not
    caught here by design -- that signal is instead carried by the character
    n-gram / transliteration-skeleton representations feeding Phase 4
    features, which is the conservative, no-hardcoded-dictionary approach.
    """
    kept, found = [], []
    for t in tokens:
        if t in LEGAL_SUFFIXES:
            found.append(t)
        else:
            kept.append(t)
    return kept, found


@lru_cache(maxsize=50_000)
def char_ngrams(normalized: str, n_min: int = 3, n_max: int = 5) -> set[str]:
    """Word-boundary-padded character n-grams, n in [n_min, n_max].

    Padding with a single leading/trailing space (char_wb-style) lets n-grams
    at word edges carry boundary information without needing per-token
    padding for every token.

    Cached: `features.py` calls this fresh per candidate pair (6x per pair --
    3 fields x 2 sides), but a Source 1 entity's name/address is identical
    across every one of its ~35-125 candidate pairs, so this is genuinely
    redundant work without caching. The returned set is never mutated by any
    caller (verified -- only read via `&`/`|` in `features.py`), so sharing
    the cached object across callers is safe. Each worker process gets its
    own cache (multiprocessing, not shared memory) -- 50k entries x ~100
    short strings each is a bounded, modest per-worker memory cost.
    """
    if not normalized:
        return set()
    padded = f" {normalized} "
    grams: set[str] = set()
    for n in range(n_min, n_max + 1):
        if len(padded) < n:
            continue
        for i in range(len(padded) - n + 1):
            grams.add(padded[i : i + n])
    return grams


# ---------------------------------------------------------------------------
# Script detection -- verified working against real Bengali sample from EDA.
# ---------------------------------------------------------------------------
_INDIC_SCRIPTS = ("Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada", "Malayalam")
_SCRIPT_ORDER = ("Latin",) + _INDIC_SCRIPTS
_SCRIPT_TEST = {name: regex.compile(rf"\p{{Script={name}}}+") for name in _SCRIPT_ORDER}
_RUN_SPLIT_RE = regex.compile(
    "|".join(rf"\p{{Script={name}}}+" for name in _SCRIPT_ORDER) + r"|[0-9]+|\S+"
)


def _classify_run(text: str) -> str:
    if text.isdigit():
        return "Digit"
    for name in _SCRIPT_ORDER:
        if _SCRIPT_TEST[name].fullmatch(text):
            return name
    return "Other"


def script_runs(normalized: str) -> list[tuple[str, str]]:
    """Segment into contiguous (script_name, run_text) tuples, in order."""
    if not normalized:
        return []
    return [(_classify_run(m.group()), m.group()) for m in _RUN_SPLIT_RE.finditer(normalized)]


def dominant_script(normalized: str) -> str:
    """The script covering the most characters, or 'None' for an empty string."""
    counts: dict[str, int] = {}
    for script, text in script_runs(normalized):
        if script == "Digit":
            continue  # digits don't indicate a script family
        counts[script] = counts.get(script, 0) + len(text)
    if not counts:
        return "None"
    return max(counts, key=counts.get)


def script_match(normalized_a: str, normalized_b: str) -> bool:
    da, db = dominant_script(normalized_a), dominant_script(normalized_b)
    return da == db and da != "None"


# ---------------------------------------------------------------------------
# Hand-authored transliteration skeleton (Devanagari / Bengali / Tamil -> Latin).
#
# This is NOT a linguistically accurate transliteration scheme -- it is a
# rough character-level skeleton built directly from each script's public
# Unicode code-chart layout (independent vowels, dependent vowel signs,
# consonants, virama/pulli, digits), authored here rather than imported from
# any third-party library. The goal (per the plan) is only to make a
# transliterated pair collide more in character-n-gram feature space than
# they would with no mapping at all -- not to produce publishable
# transliteration. Unmapped characters pass through unchanged (never crash,
# never silently drop content).
# ---------------------------------------------------------------------------

_DEVANAGARI_MAP = {
    # independent vowels
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u",
    "ऊ": "uu", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
    "औ": "au", "ऄ": "a",
    # dependent vowel signs (matras)
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    # anusvara / visarga / candrabindu
    "ं": "n", "ः": "h", "ँ": "n",
    # virama (suppresses inherent vowel; no output of its own)
    "्": "",
    # consonants
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
    # digits
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

_BENGALI_MAP = {
    "অ": "a", "আ": "aa", "ই": "i", "ঈ": "ii", "উ": "u",
    "ঊ": "uu", "ঋ": "ri", "এ": "e", "ঐ": "ai", "ও": "o",
    "ঔ": "au",
    "া": "aa", "ি": "i", "ী": "ii", "ু": "u", "ূ": "uu",
    "ৃ": "ri", "ে": "e", "ৈ": "ai", "ো": "o", "ৌ": "au",
    "ং": "n", "ঃ": "h", "ঁ": "n",
    "্": "",
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "c", "ছ": "ch", "জ": "j", "ঝ": "jh", "ঞ": "ny",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "ph", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "y", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh",
    "স": "s", "হ": "h",
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
}

_TAMIL_MAP = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ii", "உ": "u",
    "ஊ": "uu", "எ": "e", "ஏ": "ee", "ஐ": "ai", "ஒ": "o",
    "ஓ": "oo", "ஔ": "au",
    "ா": "aa", "ி": "i", "ீ": "ii", "ு": "u", "ூ": "uu",
    "ெ": "e", "ே": "ee", "ை": "ai", "ொ": "o", "ோ": "oo",
    "ௌ": "au",
    "்": "",  # pulli (virama)
    "க": "k", "ங": "ng", "ச": "c", "ஞ": "ny", "ட": "t",
    "ண": "n", "த": "t", "ந": "n", "ன": "n", "ப": "p",
    "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v",
    "ழ": "zh", "ள": "l", "ற": "r", "ஸ": "s", "ஷ": "sh",
    "ஹ": "h",
    # Grantha-derived extra consonants used for loanwords (e.g. transliterated
    # English business names)
    "ஜ": "j", "ஶ": "sh",
    "௦": "0", "௧": "1", "௨": "2", "௩": "3", "௪": "4",
    "௫": "5", "௬": "6", "௭": "7", "௮": "8", "௯": "9",
}

_GUJARATI_MAP = {
    "અ": "a", "આ": "aa", "ઇ": "i", "ઈ": "ii", "ઉ": "u",
    "ઊ": "uu", "ઋ": "ri", "એ": "e", "ઐ": "ai", "ઓ": "o",
    "ઔ": "au",
    "ા": "aa", "િ": "i", "ી": "ii", "ુ": "u", "ૂ": "uu",
    "ૃ": "ri", "ે": "e", "ૈ": "ai", "ો": "o", "ૌ": "au",
    "ં": "n", "ઃ": "h", "ઁ": "n",
    "્": "",
    "ક": "k", "ખ": "kh", "ગ": "g", "ઘ": "gh", "ઙ": "ng",
    "ચ": "c", "છ": "ch", "જ": "j", "ઝ": "jh", "ઞ": "ny",
    "ટ": "t", "ઠ": "th", "ડ": "d", "ઢ": "dh", "ણ": "n",
    "ત": "t", "થ": "th", "દ": "d", "ધ": "dh", "ન": "n",
    "પ": "p", "ફ": "ph", "બ": "b", "ભ": "bh", "મ": "m",
    "ય": "y", "ર": "r", "લ": "l", "વ": "v", "શ": "sh",
    "ષ": "sh", "સ": "s", "હ": "h", "ળ": "l",
    "૦": "0", "૧": "1", "૨": "2", "૩": "3", "૪": "4",
    "૫": "5", "૬": "6", "૭": "7", "૮": "8", "૯": "9",
}

_DEVANAGARI_MAP.update({
    "ॉ": "o", "ॅ": "e", "ऑ": "o", "ऍ": "e", "़": "", "ॐ": "om",
    # short e/o and extra letters: unused in Hindi, but their aligned slots
    # are common in the Dravidian scripts
    "ऎ": "e", "ऒ": "o", "ॆ": "e", "ॊ": "o", "ऩ": "n", "ऱ": "r", "ऴ": "zh",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
})


def _aligned_map(base: int, explicit: dict[str, str] | None = None) -> dict[str, str]:
    """The Indic Unicode blocks share Devanagari's (ISCII-derived) layout, so
    the character at the same offset from each block start is the same
    letter. Derive a map for any such block from the Devanagari one; explicit
    per-script entries (e.g. Tamil's zh/r/n letters) take precedence."""
    out = {chr(base + ord(ch) - 0x900): lat for ch, lat in _DEVANAGARI_MAP.items()
           if len(ch) == 1 and 0x900 <= ord(ch) < 0x980}
    out.update(explicit or {})
    return out


_TRANSLIT_MAPS = {
    "Devanagari": _DEVANAGARI_MAP,
    "Bengali": _aligned_map(0x980, _BENGALI_MAP),
    "Gurmukhi": _aligned_map(0xA00, {"ੰ": "n", "ੱ": ""}),
    "Gujarati": _aligned_map(0xA80, _GUJARATI_MAP),
    "Oriya": _aligned_map(0xB00),
    "Tamil": _aligned_map(0xB80, {**_TAMIL_MAP, "ஃ": ""}),
    "Telugu": _aligned_map(0xC00),
    "Kannada": _aligned_map(0xC80),
    "Malayalam": _aligned_map(0xD00, {"ൺ": "n", "ൻ": "n", "ർ": "r", "ൽ": "l", "ൾ": "l", "ൿ": "k"}),
}


# Soundex-like consonant classes, used to compare a transliteration skeleton
# with the English spelling of the same name (vowels, h, y and doubled
# classes vary between the two, the consonant skeleton mostly does not):
# "laxmi"/"lkshmi" -> 425, "private"/"praaiveet" -> 1613.
_PHONETIC_CLASS = str.maketrans(
    "bfpvwcgjkqsxzdtlmnr", "1111122222222334556", "aeiouyh"
)


def phonetic_code(token: str) -> str:
    if not token.isascii() or not token.isalpha():
        return ""
    code = token.replace("tion", "sn").translate(_PHONETIC_CLASS)  # English -tion ~ "shn"
    return "".join(ch for i, ch in enumerate(code) if i == 0 or ch != code[i - 1])


def phonetic_tokens(skeleton: str) -> list[str]:
    """Phonetic codes of a skeleton's tokens (codes shorter than 2 dropped)."""
    return [c for c in (phonetic_code(t) for t in skeleton.split()) if len(c) >= 2]


def transliterate_skeleton(normalized: str) -> str:
    """Rough Latin skeleton of any Devanagari/Bengali/Tamil runs in the string.

    Latin/digit/other runs pass through unchanged. Unmapped characters within
    a mapped script pass through unchanged too (never dropped, never crashes).
    """
    if not normalized:
        return ""
    out_parts: list[str] = []
    for script, text in script_runs(normalized):
        table = _TRANSLIT_MAPS.get(script)
        if table is None:
            out_parts.append(text)
            continue
        if script == "Malayalam":
            text = text.replace("റ്റ", "ട്ട")  # Malayalam doubled rra is pronounced "tt"
        out_parts.append("".join(table.get(ch, ch) for ch in text))
    return " ".join(p for p in out_parts if p)


# ---------------------------------------------------------------------------
# Optional experimental comparison signal -- NOT used by build_*_repr below.
# See plan.md Phase 2 for the four-review reconciliation on why this stays
# opt-in rather than part of the graded pipeline by default.
# ---------------------------------------------------------------------------
def anyascii_repr(s: str) -> str | None:
    """Experimental only. Returns None if anyascii isn't installed/enabled."""
    try:
        import anyascii as _anyascii
    except ImportError:
        return None
    if not s:
        return ""
    return _anyascii.anyascii(s)


# ---------------------------------------------------------------------------
# Combined per-field representations
# ---------------------------------------------------------------------------
@dataclass
class NameRepr:
    raw: str
    normalized: str
    tokens: list[str]
    suffix_tokens: list[str]
    tokens_no_suffix: list[str]
    dominant_script: str
    skeleton: str


@dataclass
class AddressRepr:
    raw: str
    normalized: str
    tokens: list[str]
    dominant_script: str
    skeleton: str
    leading_digits: str  # candidate street number
    digit_tokens: list[str]  # all standalone digit runs, leading zeros stripped
    first_word: str = ""  # first non-digit token (street word of a leading-number address)
    postcodes: tuple = ()  # 5/6-digit runs with leading zeros KEPT ("07001", "01000")


def build_name_repr(raw: str) -> NameRepr:
    """Note: does NOT compute char_ngrams here, even though the module-level
    `char_ngrams()` function exists and this dataclass used to carry it. That
    field was silently unused by the real pipeline (features.py always calls
    `char_ngrams()` fresh, on-demand, per candidate pair -- a much smaller
    set than "every raw record") and precomputing/storing it for every one of
    ~11.7M raw train+test records was pure wasted memory. Removing it is what
    fixed a real MemoryError crash during a full-scale run -- caught by
    actually running at scale, not by inspection."""
    tokens = clean_name_tokens(normalize_unicode(raw))
    normalized = " ".join(tokens)
    tokens_no_suffix, suffix_tokens = strip_legal_suffixes(tokens)
    return NameRepr(
        raw=raw,
        normalized=normalized,
        tokens=tokens,
        suffix_tokens=suffix_tokens,
        tokens_no_suffix=tokens_no_suffix,
        dominant_script=dominant_script(normalized),
        skeleton=transliterate_skeleton(normalized),
    )


_DIGIT_RUN_RE = re.compile(r"\d+")


def build_address_repr(raw: str, country: str = "") -> AddressRepr:
    base = normalize_unicode(raw)
    tokens = canonicalize_address_tokens(tokenize(base), country)
    normalized = " ".join(tokens)
    digit_tokens = [d.lstrip("0") or "0" for d in _DIGIT_RUN_RE.findall(normalized)]
    leading_digits = digit_tokens[0] if tokens and tokens[0].isdigit() else ""
    return AddressRepr(
        raw=raw,
        normalized=normalized,
        tokens=tokens,
        dominant_script=dominant_script(normalized),
        skeleton=transliterate_skeleton(normalized),
        leading_digits=leading_digits,
        digit_tokens=digit_tokens,
        first_word=next((t for t in tokens if not t.isdigit()), ""),
        postcodes=tuple(d for d in _DIGIT_RUN_RE.findall(base) if len(d) in (5, 6)),
    )


def build_address_repr_pair(args: tuple[str, str]) -> AddressRepr:
    """(address, country) -> AddressRepr; module-level so the process pool can pickle it."""
    return build_address_repr(*args)
