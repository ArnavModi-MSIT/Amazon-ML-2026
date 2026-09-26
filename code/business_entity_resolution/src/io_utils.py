"""Loading and writing the challenge's tab-separated files.

All reads/writes go through here so the sep="\\t" / dtype / column-order rules
are enforced in exactly one place.
"""
from pathlib import Path

import pandas as pd

from . import config


def load_source(path: Path) -> pd.DataFrame:
    """Load one *_source{1,2,3}.tsv file with the expected dtypes."""
    df = pd.read_csv(
        path,
        sep=config.SEP,
        dtype={
            "entity_id": "string",
            "business_name": "string",
            "business_address": "string",
            "country": "string",
        },
        keep_default_na=False,  # an empty address/name is data, not NaN
        na_values=[],
    )
    missing = set(config.SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")
    return df


def load_ground_truth(path: Path = config.TRAIN_GROUND_TRUTH) -> pd.DataFrame:
    """Load train_ground_truth.tsv. matched_entity_ids stays a raw string;
    split it with `split_id_list` when you need the actual list.
    """
    df = pd.read_csv(
        path,
        sep=config.SEP,
        dtype={"source1_entity_id": "string", "matched_entity_ids": "string"},
        keep_default_na=False,
        na_values=[],
    )
    missing = set(config.GROUND_TRUTH_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")
    return df


def split_id_list(cell: str) -> list[str]:
    """'S2-1,S3-2' -> ['S2-1', 'S3-2']; '' -> []."""
    if not cell:
        return []
    return cell.split(",")


def build_true_matches(ground_truth: pd.DataFrame) -> dict:
    """{source1_entity_id: set(matched_entity_ids)}, one entry per row.

    Shared by pipeline.py and sampling.py (both need this same mapping) --
    lives here rather than in either to avoid a circular import between them.
    Uses zip() over two Series rather than `.iterrows()`, which constructs a
    full pandas Series object per row and is dramatically slower at
    millions-of-rows scale (train_ground_truth.tsv is 2.2M rows).
    """
    return {
        s1_id: set(split_id_list(cell))
        for s1_id, cell in zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"])
    }


def write_id_list_file(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    """Write a matching_results.tsv / candidate_pairs.tsv-shaped file.

    `df` must already have exactly `columns`, one row per Source-1 entity,
    with the id-list column comma-joined (empty string for no matches).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(path, sep=config.SEP, index=False)
