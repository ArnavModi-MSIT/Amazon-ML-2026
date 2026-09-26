"""Word-level transliteration dictionary learned from the training ground truth.

The hand-built skeleton (text_repr.transliterate_skeleton) maps Indic
characters to rough Latin, which rarely reproduces the English spelling S1
uses ("क्रिएटिव" -> "krietiv", not "creative"). Many native-script S2/S3
names are word-for-word renderings of the S1 name, so aligning the tokens of
true pairs with the same token count gives direct word mappings
("प्राइवेट" -> "private", "टेक्नोलॉजी" -> "technology"). Only the provided
training files are used (no external data); pairs of the held-out validation
entities are excluded so validation stays honest. Idea from a public
competition repo (learned translit dictionary), re-implemented here.

The dictionary is saved next to the model; text_repr loads it in every
process via the BER_TRANSLIT environment variable, so training and
inference use the identical mapping.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from . import text_repr as tr

_INDIC = re.compile(r"[ऀ-෿]")
ENV_VAR = "BER_TRANSLIT"


def _tokens(raw: str) -> list[str]:
    return tr.clean_name_tokens(tr.normalize_unicode(tr.clean_name_raw(raw)))  # same tokens as build_name_repr


def build(s1: pd.DataFrame, pools: list[pd.DataFrame], true_matches: dict, exclude_s1: set,
          min_count: int = 2, min_share: float = 0.6) -> dict[str, str]:
    """`true_matches`: {source1_entity_id: set(matched ids)} (io_utils.build_true_matches)."""
    s1_name = dict(zip(s1["entity_id"], s1["business_name"]))
    pool = pd.concat([p[["entity_id", "business_name"]] for p in pools], ignore_index=True)
    pool = pool[pool["business_name"].str.contains(_INDIC, na=False)]
    pool_name = dict(zip(pool["entity_id"], pool["business_name"]))
    counts: dict[str, Counter] = defaultdict(Counter)
    for a, b in ((a, b) for a, ms in true_matches.items() for b in ms):
        if a in exclude_s1 or b not in pool_name or a not in s1_name:
            continue
        nt, lt = _tokens(pool_name[b]), _tokens(s1_name[a])
        if len(nt) != len(lt) or not nt:
            continue  # word-for-word renderings only
        for x, y in zip(nt, lt):
            if _INDIC.search(x) and y.isascii() and y.isalpha():
                counts[x][y] += 1
    out = {}
    for x, c in counts.items():
        y, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_share:
            out[x] = y
    return out


def save(mapping: dict[str, str], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, sort_keys=True)


def activate(path) -> None:
    """Make every process (this one and pool workers spawned afterwards) use the dictionary."""
    os.environ[ENV_VAR] = str(path)
    tr.load_translit(str(path))
