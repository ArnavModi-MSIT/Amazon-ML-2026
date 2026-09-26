"""Lightweight progress + resource logging.

Added after the page-file-exhaustion freeze taught us we had zero visibility
into what was running before it crashed: `python -m` output is fully
block-buffered when redirected to a file (not a terminal), so nothing
printed until the process exited or froze -- by which point it's too late
to know which stage was responsible. `log()` here always flushes
immediately, and reports memory across the WHOLE process tree (parent + all
worker children), since the freeze was about aggregate system memory
pressure, not any single process's RSS.
"""
from __future__ import annotations

import time

import psutil

_start = time.time()
_proc = psutil.Process()


def _tree_stats() -> tuple[float, int]:
    """(total RSS in MB across this process + all children, process count)."""
    total = _proc.memory_info().rss
    n = 1
    try:
        children = _proc.children(recursive=True)
    except psutil.Error:
        children = []
    for c in children:
        try:
            total += c.memory_info().rss
            n += 1
        except psutil.Error:
            continue  # child exited between listing and reading -- ignore
    return total / (1024**2), n


def log(msg: str) -> None:
    """Print `msg` with elapsed time, total process-tree RSS, and process
    count -- flushed immediately so it's visible in real time even when
    stdout is redirected to a file, not just at process exit."""
    elapsed = time.time() - _start
    rss_mb, n_proc = _tree_stats()
    print(f"[{elapsed:7.1f}s | {rss_mb:8.0f}MB RSS | {n_proc:3d} procs] {msg}", flush=True)
