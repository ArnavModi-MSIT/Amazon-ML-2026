"""Phase 4 -- Pairwise similarity features.

Computes a fixed-width numeric feature vector for a (Source1, candidate)
pair, using only the two records' own fields (name/address/country) as
required by the no-external-data rule. Everything here is derived from the
representation columns `blocking.add_representations` already produces, plus
a `target_source` flag distinguishing S2 from S3 candidates (EDA showed the
two sources have different match-rate profiles).

Reconciled per the plan from four external reviews:
  - character n-gram Jaccard on both raw normalized text AND the
    transliteration skeleton (the skeleton pass matters for cross-script
    pairs -- same lesson learned the hard way in blocking.py)
  - rapidfuzz for all edit-distance-family metrics (C-backed, fast at scale)
  - address features favor token/digit-level agreement over generic
    full-string edit distance (component-level signal is more informative)
  - a cross-field feature: name tokens appearing in the *other* record's
    address (some source records bury the business name in the address field)
  - country-match as an explicit strong feature, not left for the model to
    discover from a single categorical column
"""
from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, LCSseq

from . import text_repr as tr


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0  # both-empty: no evidence of a difference; treat as neutral/agreeing
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _containment(a: set, b: set) -> float:
    """Fraction of `a` found in `b` -- asymmetric, catches one name being a
    subset/substring of the other (e.g. "Acme" vs "Acme Robotics Inc")."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _safe_ratio(fn, s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fn(s1, s2)


def _len_ratio(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    la, lb = len(s1), len(s2)
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _script_fractions(normalized: str) -> dict[str, float]:
    """Proportion of characters (by length) belonging to each broad category."""
    runs = tr.script_runs(normalized)
    total = sum(len(text) for _, text in runs) or 1
    latin = sum(len(text) for script, text in runs if script == "Latin")
    digit = sum(len(text) for script, text in runs if script == "Digit")
    non_latin = total - latin - digit
    return {"pct_latin": latin / total, "pct_digit": digit / total, "pct_non_latin": non_latin / total}


def build_name_features(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """`a`/`b` are row dicts with the representation columns from
    `blocking.add_representations` (name_norm, name_skeleton, name_tokens_no_suffix, ...).
    """
    a_norm, b_norm = a["name_norm"], b["name_norm"]
    a_skel, b_skel = a["name_skeleton"], b["name_skeleton"]
    a_tokens, b_tokens = set(a["name_tokens_no_suffix"]), set(b["name_tokens_no_suffix"])
    a_ngrams, b_ngrams = tr.char_ngrams(a_norm), tr.char_ngrams(b_norm)
    a_skel_ngrams, b_skel_ngrams = tr.char_ngrams(a_skel), tr.char_ngrams(b_skel)

    feats = {
        "name_char_ngram_jaccard": _jaccard(a_ngrams, b_ngrams),
        "name_skeleton_ngram_jaccard": _jaccard(a_skel_ngrams, b_skel_ngrams),
        "name_token_jaccard": _jaccard(a_tokens, b_tokens),
        "name_token_containment_a_in_b": _containment(a_tokens, b_tokens),
        "name_token_containment_b_in_a": _containment(b_tokens, a_tokens),
        "name_shared_token_count": float(len(a_tokens & b_tokens)),
        "name_levenshtein_ratio": _safe_ratio(fuzz.ratio, a_norm, b_norm) / 100.0,
        "name_token_sort_ratio": _safe_ratio(fuzz.token_sort_ratio, a_norm, b_norm) / 100.0,
        "name_partial_ratio": _safe_ratio(fuzz.partial_ratio, a_norm, b_norm) / 100.0,
        "name_jaro_winkler": _safe_ratio(JaroWinkler.normalized_similarity, a_norm, b_norm),
        "name_lcs_ratio": _safe_ratio(LCSseq.normalized_similarity, a_norm, b_norm),
        "name_skeleton_levenshtein_ratio": _safe_ratio(fuzz.ratio, a_skel, b_skel) / 100.0,
        "name_len_ratio": _len_ratio(a_norm, b_norm),
        "name_token_count_diff": float(abs(len(a["name_tokens_no_suffix"]) - len(b["name_tokens_no_suffix"]))),
        "name_suffix_match": float(bool(set(a.get("name_suffix_tokens", [])) & set(b.get("name_suffix_tokens", [])))),
        "name_suffix_present_both": float(bool(a.get("name_suffix_tokens")) and bool(b.get("name_suffix_tokens"))),
        "script_match": float(a["dominant_script"] == b["dominant_script"] and a["dominant_script"] != "None"),
        # Explicit exact-key-match flags -- the fuzzy string-similarity
        # features above are already ~1.0 for these cases, but an explicit
        # boolean gives the tree model a clean, single-feature split instead
        # of an implicit near-1.0 threshold in a continuous feature (found
        # via the "Entity Resolution in Practice" paper's point that a
        # candidate this strong shouldn't rely purely on the classifier
        # rediscovering it from continuous similarity scores).
        "name_key_exact_match": float(bool(a.get("name_key")) and a.get("name_key") == b.get("name_key")),
        # First distinctive token is often the brand ("acme" in "acme robotics inc").
        "name_first_token_match": float(
            bool(a["name_tokens_no_suffix"]) and bool(b["name_tokens_no_suffix"])
            and a["name_tokens_no_suffix"][0] == b["name_tokens_no_suffix"][0]
        ),
    }
    a_script_pct = _script_fractions(a_norm)
    b_script_pct = _script_fractions(b_norm)
    feats["pct_non_latin_diff"] = abs(a_script_pct["pct_non_latin"] - b_script_pct["pct_non_latin"])
    return feats


def build_address_features(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    a_norm, b_norm = a["addr_norm"], b["addr_norm"]
    a_missing, b_missing = (not a_norm), (not b_norm)
    a_tokens, b_tokens = set(a_norm.split()), set(b_norm.split())
    a_ngrams, b_ngrams = tr.char_ngrams(a_norm), tr.char_ngrams(b_norm)
    a_digits, b_digits = set(a["addr_digit_tokens"]), set(b["addr_digit_tokens"])
    # 5-digit (US zip / French postal code) or 6-digit (India PIN) tokens --
    # separates "same postcode" from a coincidentally shared street number.
    a_codes = {t for t in a_digits if len(t) in (5, 6)}
    b_codes = {t for t in b_digits if len(t) in (5, 6)}
    postcode = -1.0 if not (a_codes and b_codes) else float(bool(a_codes & b_codes))

    return {
        "addr_postcode_match": postcode,
        "addr_char_ngram_jaccard": _jaccard(a_ngrams, b_ngrams),
        "addr_token_jaccard": _jaccard(a_tokens, b_tokens),
        "addr_levenshtein_ratio": _safe_ratio(fuzz.ratio, a_norm, b_norm) / 100.0,
        "addr_token_sort_ratio": _safe_ratio(fuzz.token_sort_ratio, a_norm, b_norm) / 100.0,
        "addr_len_ratio": _len_ratio(a_norm, b_norm),
        "addr_leading_digits_match": float(
            bool(a["addr_leading_digits"]) and a["addr_leading_digits"] == b["addr_leading_digits"]
        ),
        "addr_digit_token_jaccard": _jaccard(a_digits, b_digits),
        "addr_missing_either": float(a_missing or b_missing),
        "addr_missing_both": float(a_missing and b_missing),
        "addr_key_exact_match": float(bool(a.get("addr_key")) and a.get("addr_key") == b.get("addr_key")),
    }


def build_cross_field_features(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Name tokens found in the *other* record's address -- catches records
    that bury the business name inside the address field."""
    a_name_tokens = set(a["name_tokens_no_suffix"])
    b_name_tokens = set(b["name_tokens_no_suffix"])
    a_addr_tokens = set(a["addr_norm"].split())
    b_addr_tokens = set(b["addr_norm"].split())
    return {
        "name_a_in_addr_b": _containment(a_name_tokens, b_addr_tokens),
        "name_b_in_addr_a": _containment(b_name_tokens, a_addr_tokens),
    }


def build_pair_features(a: dict[str, Any], b: dict[str, Any], target_source: int) -> dict[str, float]:
    """Full feature vector for one (Source1 row, candidate row) pair.

    `target_source`: 0 if `b` is from Source 2, 1 if from Source 3 -- lets a
    single jointly-trained model account for the two sources' differing
    noise/match-rate profiles (EDA: S2-only vs S3-only match counts differ).
    """
    feats: dict[str, float] = {}
    feats.update(build_name_features(a, b))
    feats.update(build_address_features(a, b))
    feats.update(build_cross_field_features(a, b))
    feats["country_match"] = float(a["country"] == b["country"])
    feats["target_source"] = float(target_source)
    return feats


def build_pair_features_from_tuple(args: tuple[dict[str, Any], dict[str, Any], int]) -> dict[str, float]:
    """Module-level (picklable) wrapper around `build_pair_features` for use
    with `parallel_utils.parallel_map` -- Windows' 'spawn' multiprocessing
    start method requires a plain module-level function, not a lambda."""
    a, b, target_source = args
    return build_pair_features(a, b, target_source)


FEATURE_NAMES: list[str] = sorted(
    set(build_pair_features(
        {"name_norm": "", "name_skeleton": "", "name_tokens_no_suffix": (), "name_suffix_tokens": [],
         "dominant_script": "None", "addr_norm": "", "addr_leading_digits": "", "addr_digit_tokens": [],
         "country": "", "name_key": "", "addr_key": ""},
        {"name_norm": "", "name_skeleton": "", "name_tokens_no_suffix": (), "name_suffix_tokens": [],
         "dominant_script": "None", "addr_norm": "", "addr_leading_digits": "", "addr_digit_tokens": [],
         "country": "", "name_key": "", "addr_key": ""},
        target_source=0,
    ).keys())
)
