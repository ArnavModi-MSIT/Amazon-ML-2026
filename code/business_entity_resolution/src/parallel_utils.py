"""Shared parallel-map helper for the CPU-bound row-wise/pairwise work in
text_repr.py, blocking.py, features.py, sampling.py and pipeline.py.

Windows uses the 'spawn' multiprocessing start method (no fork), so worker
functions must be plain module-level functions -- no lambdas or closures --
and any script driving this must guard its entry point with
`if __name__ == "__main__":` (all of this project's `_smoke_test_*.py`
scripts already run as `python -m src.xxx`, whose implicit module execution
satisfies this).

CRITICAL lesson learned the hard way: a naive `parallel_map` that creates a
*new* ProcessPoolExecutor on every call caused a real system-wide freeze
("the paging file is too small for this operation to complete" -- a Windows
virtual-memory exhaustion error, not just a Python exception). Root cause:
`blocking.add_representations` calls `parallel_map` twice (name + address)
and runs 3x per pipeline call (S1/S2/S3), so a single `train_and_evaluate`/
`run_inference` call was spawning 6 separate batches of `n_jobs` fresh
processes -- each of which independently re-imports the entire heavy
scientific stack (pandas/numpy/scipy/sklearn/lightgbm) under Windows' spawn
model. That's dozens of concurrent heavy imports competing for memory/disk
I/O in a short window, enough to exhaust the page file and stall the whole
OS. The fix: create ONE pool and reuse it across every parallel step in a
pipeline run via `parallel_pool()`, instead of one-per-call.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Lowered from `cpu_count - 1` after the page-file-exhaustion incident above --
# safety margin against repeating it, since the root cause (page file sized
# too small for a burst of concurrent heavy imports) is a system-level
# constraint we can't fully control from here. Half the logical cores still
# gives meaningful parallelism without the same peak memory spike.
DEFAULT_N_JOBS = max(1, (os.cpu_count() or 2) // 2)

# Below this many items, ProcessPoolExecutor startup overhead (spawning
# processes, re-importing modules on Windows) costs more than it saves.
MIN_ITEMS_FOR_PARALLEL = 2000

# Hard cap on chunksize regardless of total item count -- see the MemoryError
# incident this fixed: at ~4.9M items / 15 jobs the naive formula computed a
# chunksize of ~81,000, meaning each IPC round-trip had to pickle/unpickle
# ~81,000 result objects at once, and a worker died mid-`recv()` trying to
# deserialize one oversized chunk. Capping chunksize keeps each IPC payload
# bounded no matter how large `items` is.
MAX_CHUNKSIZE = 2000


# BLAS/OpenMP thread-oversubscription guard (flagged by external review, not
# an original consideration): numpy/scipy/scikit-learn are commonly linked
# against OpenBLAS/MKL, which by default spawn their OWN internal thread pool
# per process, sized to the CPU count. Without this, `n_jobs` worker
# processes could each ALSO spin up several BLAS threads internally, turning
# "8 workers" into effectively dozens of competing OS threads -- compounding
# exactly the kind of resource pressure that caused the page-file-exhaustion
# freeze. Setting these to 1 per worker process is safe here: the actual
# parallelism in this pipeline is across processes (rows/pairs), not within
# a single BLAS call. Must be set before the child processes are spawned --
# children inherit the parent's environment at spawn time.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")


@contextmanager
def parallel_pool(n_jobs: int = DEFAULT_N_JOBS) -> Iterator["ProcessPoolExecutor | None"]:
    """Create ONE process pool for reuse across an entire pipeline run.

    Yields None when n_jobs<=1 (caller's `parallel_map` calls will then run
    sequentially in-process, no pool needed at all).
    """
    if n_jobs <= 1:
        yield None
        return
    pool = ProcessPoolExecutor(max_workers=n_jobs)
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)


def parallel_map(
    func: Callable[[T], R],
    items: list[T],
    pool: "ProcessPoolExecutor | None" = None,
    n_jobs: int = DEFAULT_N_JOBS,
    chunksize: int | None = None,
) -> list[R]:
    """Like `[func(x) for x in items]`, spread across a process pool.

    Pass `pool=` (from `parallel_pool()`) to reuse an existing pool across
    multiple calls -- always prefer this in a real pipeline run. Without
    `pool`, falls back to creating a transient one-off pool (fine for a
    single standalone call, e.g. in a small script or test, but never do
    this repeatedly in a loop -- that's exactly what caused the page-file
    exhaustion incident documented at the top of this module).

    Falls back to a plain single-process loop when `n_jobs<=1` (or no pool
    and `n_jobs<=1`) or the input is too small for process-pool overhead to
    be worth it.
    """
    if len(items) < MIN_ITEMS_FOR_PARALLEL:
        return [func(x) for x in items]

    if pool is not None:
        workers = pool._max_workers
        if chunksize is None:
            chunksize = min(MAX_CHUNKSIZE, max(1, len(items) // (workers * 4)))
        return list(pool.map(func, items, chunksize=chunksize))

    if n_jobs <= 1:
        return [func(x) for x in items]
    if chunksize is None:
        chunksize = min(MAX_CHUNKSIZE, max(1, len(items) // (n_jobs * 4)))
    with ProcessPoolExecutor(max_workers=n_jobs) as one_off_pool:
        return list(one_off_pool.map(func, items, chunksize=chunksize))
