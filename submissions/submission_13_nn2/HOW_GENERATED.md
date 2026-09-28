# Submission 13 — final MiniLM blend

**Leaderboard score:** 0.976331

**Held-out test-like macro F0.5:** 0.9844

This is the repository's best-scoring submission. It uses the v11 LightGBM candidate scores blended with a multilingual MiniLM cross-encoder trained for two epochs on all 1.32 million generated training pairs.

## Decision rule

- Blend: `sigmoid(0.8 * logit(p_v11) + 0.2 * logit(p_nn2))`
- Base threshold: 0.80
- France threshold: 0.8935, calibrated to the tested 1.8% France pair reduction
- Final constraint: global one-to-one assignment for Source 2 and Source 3 records

The candidate set is unchanged from submission #07 and averages 13.29 records per Source 1 entity.

## Files

- `candidate_pairs.tsv.gz.part00` and `.part01`: one gzip stream split into GitHub-safe chunks. Concatenate the parts before decompressing.
- `matching_results.tsv.gz`: the final #13 match output.

Reassemble on Linux/macOS:

```bash
cat candidate_pairs.tsv.gz.part00 candidate_pairs.tsv.gz.part01 > candidate_pairs.tsv.gz
gzip -dk candidate_pairs.tsv.gz
gzip -dk matching_results.tsv.gz
```

Reassemble on Windows PowerShell:

```powershell
python -c "from pathlib import Path; Path('candidate_pairs.tsv.gz').write_bytes(Path('candidate_pairs.tsv.gz.part00').read_bytes() + Path('candidate_pairs.tsv.gz.part01').read_bytes())"
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('candidate_pairs.tsv.gz','rb'),open('candidate_pairs.tsv','wb'))"
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('matching_results.tsv.gz','rb'),open('matching_results.tsv','wb'))"
```

## Checksums of decompressed files

| File | MD5 |
|---|---|
| `candidate_pairs.tsv` | `9f44ed23a4668eb0826a1741ccdb74d3` |
| `matching_results.tsv` | `935c7f5c464d0066c7e0d40a868047c5` |

Both files passed the submission validator with ID checks enabled.
