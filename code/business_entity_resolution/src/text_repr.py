"""Phase 2 — Text representation.

Builds *multiple parallel representations* of a name/address string rather than
one canonical cleaned string, per the plan's reconciliation of four external
reviews: Unicode normalization alone misses transliteration noise, and ASCII
folding destroys information needed to keep genuinely different non-Latin
businesses distinguishable.

Representations produced, all derived solely from the input string (no
external data, no lookups, no network):
  - normalized:      NFKC + casefold + punctuation/whitespace cleanup, script preserved
  - tokens:          whitespace-split tokens of `normalized`
  - suffix tokens:   legal-suffix tokens detected and separated out (US/India/France),
                     kept as a *flag*, never silently discarded from the record's identity
  - char_ngrams:     word-boundary-padded character 3-5 grams over `normalized`
  - script_runs:     contiguous same-script substrings (Latin/Devanagari/Bengali/Tamil/digits)
  - dominant_script: the script covering the most characters
  - skeleton:        a hand-authored Devanagari/Bengali/Tamil -> Latin *rough* transliteration,
                      built from the public Unicode block layout (not a third-party library) --
                      see the plan's dependency-audit note on why this is the safe default
                      over bundled transliteration libraries.

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


def normalize_unicode(s: str) -> str:
    """NFKC normalize + casefold + punctuation->space + whitespace collapse.

    Script is preserved (no ASCII folding), and combining marks (vowel signs,
    virama/pulli) are preserved too -- this is the representation everything
    else is built from.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = _AMPERSAND_RE.sub(" and ", s)
    s = _strip_punctuation(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


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
_SCRIPT_ORDER = ("Latin", "Devanagari", "Bengali", "Tamil", "Gujarati")
_SCRIPT_TEST = {name: regex.compile(rf"\p{{Script={name}}}+") for name in _SCRIPT_ORDER}
_RUN_SPLIT_RE = regex.compile(
    r"\p{Script=Latin}+|\p{Script=Devanagari}+|\p{Script=Bengali}+|\p{Script=Tamil}+"
    r"|[0-9]+|\S+"
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

_TRANSLIT_MAPS = {
    "Devanagari": _DEVANAGARI_MAP,
    "Bengali": _BENGALI_MAP,
    "Tamil": _TAMIL_MAP,
    "Gujarati": _GUJARATI_MAP,
}


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
    digit_tokens: list[str]  # all standalone digit runs (postal-code-like candidates)


def build_name_repr(raw: str) -> NameRepr:
    """Note: does NOT compute char_ngrams here, even though the module-level
    `char_ngrams()` function exists and this dataclass used to carry it. That
    field was silently unused by the real pipeline (features.py always calls
    `char_ngrams()` fresh, on-demand, per candidate pair -- a much smaller
    set than "every raw record") and precomputing/storing it for every one of
    ~11.7M raw train+test records was pure wasted memory. Removing it is what
    fixed a real MemoryError crash during a full-scale run -- caught by
    actually running at scale, not by inspection."""
    normalized = normalize_unicode(raw)
    tokens = tokenize(normalized)
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


def build_address_repr(raw: str) -> AddressRepr:
    normalized = normalize_unicode(raw)
    tokens = tokenize(normalized)
    digit_tokens = _DIGIT_RUN_RE.findall(normalized)
    leading_digits = digit_tokens[0] if tokens and tokens[0].isdigit() else ""
    return AddressRepr(
        raw=raw,
        normalized=normalized,
        tokens=tokens,
        dominant_script=dominant_script(normalized),
        skeleton=transliterate_skeleton(normalized),
        leading_digits=leading_digits,
        digit_tokens=digit_tokens,
    )
