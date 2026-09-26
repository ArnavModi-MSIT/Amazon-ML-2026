"""Phase 0 — Exploratory Data Analysis.

Profiles all 7 challenge files and writes a human-readable report to
reports/eda_report.md. Run before writing any pipeline code.

Usage (from student_resource/code/business_entity_resolution/):
    python -m src.eda
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from . import config, io_utils

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


def _section(title: str) -> str:
    return f"\n## {title}\n"


def profile_source_file(name: str, path: Path, lines: list[str]) -> pd.DataFrame:
    t0 = time.time()
    df = io_utils.load_source(path)
    elapsed = time.time() - t0

    lines.append(_section(name))
    lines.append(f"- path: `{path.relative_to(config.PROJECT_ROOT)}`")
    lines.append(f"- rows: {len(df):,}  (loaded in {elapsed:.1f}s)")
    lines.append(f"- columns: {list(df.columns)}")

    dup_ids = df["entity_id"].duplicated().sum()
    lines.append(f"- duplicate entity_id count: {dup_ids:,}")

    prefixes = df["entity_id"].str.split("-", n=1).str[0].value_counts()
    lines.append(f"- entity_id prefixes: {prefixes.to_dict()}")

    for col in ("business_name", "business_address"):
        empty = (df[col].str.len() == 0).sum()
        lens = df[col].str.len()
        lines.append(
            f"- `{col}`: empty={empty:,} ({empty / len(df):.2%}), "
            f"len min/median/mean/max = {lens.min()}/{lens.median():.0f}/{lens.mean():.1f}/{lens.max()}"
        )

    exact_dup_names = df["business_name"].str.lower().duplicated().sum()
    lines.append(
        f"- exact case-insensitive duplicate `business_name` values: {exact_dup_names:,} "
        f"({exact_dup_names / len(df):.2%}) — expected to be common (e.g. chain names), "
        "not itself evidence of a data issue"
    )

    country_counts = df["country"].value_counts()
    lines.append(f"- country distribution:\n\n```\n{country_counts.to_string()}\n```")

    lines.append("- 5 random sample rows:\n")
    sample = df.sample(min(5, len(df)), random_state=config.RANDOM_SEED)
    lines.append("```\n" + sample.to_string(index=False) + "\n```")

    return df


def profile_ground_truth(lines: list[str], train_source1: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    gt = io_utils.load_ground_truth()
    elapsed = time.time() - t0

    lines.append(_section("train_ground_truth.tsv"))
    lines.append(f"- rows: {len(gt):,}  (loaded in {elapsed:.1f}s)")

    s1_ids = set(train_source1["entity_id"])
    gt_ids = set(gt["source1_entity_id"])
    lines.append(f"- ground truth covers every train_source1 entity: {gt_ids == s1_ids}")
    if gt_ids != s1_ids:
        lines.append(f"  - in gt but not source1: {len(gt_ids - s1_ids):,}")
        lines.append(f"  - in source1 but not gt: {len(s1_ids - gt_ids):,}")

    match_counts = gt["matched_entity_ids"].apply(
        lambda cell: 0 if not cell else cell.count(",") + 1
    )
    lines.append("- match-count distribution per Source 1 entity (this is the class-balance"
                  " reality: most weight sits wherever the mode is, and singletons at 0"
                  " matter a lot for scoring):\n")
    dist = match_counts.value_counts().sort_index()
    dist_display = dist.head(15)
    lines.append("```\nmatches -> count\n" + dist_display.to_string() + "\n")
    if len(dist) > 15:
        lines.append(f"... ({len(dist) - 15} more distinct match-counts above 14, "
                      f"max={match_counts.max()})\n")
    lines.append("```")

    n_singletons = (match_counts == 0).sum()
    lines.append(
        f"- singleton rate (0 matches): {n_singletons:,} / {len(gt):,} = {n_singletons / len(gt):.2%}"
    )

    # which source(s) do matches come from?
    s2_only = gt["matched_entity_ids"].str.contains("S2-") & ~gt["matched_entity_ids"].str.contains("S3-")
    s3_only = gt["matched_entity_ids"].str.contains("S3-") & ~gt["matched_entity_ids"].str.contains("S2-")
    both = gt["matched_entity_ids"].str.contains("S2-") & gt["matched_entity_ids"].str.contains("S3-")
    lines.append(
        f"- of entities with >=1 match: S2-only={s2_only.sum():,}, S3-only={s3_only.sum():,}, "
        f"both S2 and S3={both.sum():,}"
    )

    return gt


def estimate_naive_pair_count(lines: list[str], sizes: dict[str, int]) -> None:
    lines.append(_section("Why blocking is mandatory"))
    s1, s2, s3 = sizes["source1"], sizes["source2"], sizes["source3"]
    naive_pairs = s1 * s2 + s1 * s3
    lines.append(
        f"- Naive all-pairs comparison for this split: {s1:,} x {s2:,} + {s1:,} x {s3:,} "
        f"= {naive_pairs:,} candidate pairs. Even at a generous 10M comparisons/sec for a cheap "
        f"similarity check, that's ~{naive_pairs / 10_000_000 / 3600:.1f} hours of pure compute — "
        "brute force is not viable; blocking must cut this down by several orders of magnitude."
    )


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# EDA Report — Business Entity Resolution\n",
                         f"Generated by `src/eda.py`.\n"]

    for split_name, split_dir, files in (
        ("TRAIN", config.TRAIN_DIR, [
            ("train_source1", config.TRAIN_SOURCE1),
            ("train_source2", config.TRAIN_SOURCE2),
            ("train_source3", config.TRAIN_SOURCE3),
        ]),
        ("TEST", config.TEST_DIR, [
            ("test_source1", config.TEST_SOURCE1),
            ("test_source2", config.TEST_SOURCE2),
            ("test_source3", config.TEST_SOURCE3),
        ]),
    ):
        lines.append(f"\n# {split_name}\n")
        dfs = {}
        for name, path in files:
            print(f"Loading {name} ...", file=sys.stderr)
            dfs[name] = profile_source_file(name, path, lines)

        if split_name == "TRAIN":
            gt = profile_ground_truth(lines, dfs["train_source1"])
        estimate_naive_pair_count(
            lines,
            {
                "source1": len(dfs[f"{split_name.lower()}_source1"]),
                "source2": len(dfs[f"{split_name.lower()}_source2"]),
                "source3": len(dfs[f"{split_name.lower()}_source3"]),
            },
        )

    report_path = config.REPORTS_DIR / "eda_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
