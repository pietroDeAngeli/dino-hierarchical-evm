"""
Lightweight phase-level profiling for the EVM fitting pipeline.

Usage:
    from prof_utils import timed, PROF, report

    with timed("build_dmat", n_points=len(X)):
        ...

At the end of the run, call report() to print an aggregated breakdown
by phase, and a CSV dump per-call for finer-grained analysis (e.g. to
see how cost scales with node population).

Designed to be import-safe even if torch/cuda isn't synchronized:
call torch.cuda.synchronize() inside timed() when CUDA is in use, so
that GPU-async ops are actually measured (otherwise you'd just be
timing kernel *launch*, not execution).
"""
import time
import csv
import contextlib
from collections import defaultdict

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


PROF = defaultdict(list)  # phase_name -> list of (elapsed_seconds, metadata dict)

# When False, report() and dump_csv() are silent no-ops. This prevents the
# per-call profiling flood when something in the predict/eval loop calls
# report() after every single prediction. Turn it on once, right before the
# single final report you actually want (see enable_reporting()).
_REPORTING_ENABLED = False


def enable_reporting(enabled=True):
    """Toggle whether report()/dump_csv() actually emit output."""
    global _REPORTING_ENABLED
    _REPORTING_ENABLED = enabled


def _sync():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def timed(phase, **metadata):
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        elapsed = time.perf_counter() - t0
        PROF[phase].append((elapsed, metadata))


def report(top_n=20):
    if not _REPORTING_ENABLED:
        return
    print("\n" + "=" * 80)
    print("PROFILING REPORT (aggregated by phase)")
    print("=" * 80)
    rows = []
    for phase, calls in PROF.items():
        total = sum(c[0] for c in calls)
        n = len(calls)
        avg = total / n if n else 0.0
        rows.append((phase, total, n, avg))
    rows.sort(key=lambda r: -r[1])
    print(f"{'phase':<25}{'total_s':>12}{'n_calls':>10}{'avg_ms':>12}")
    for phase, total, n, avg in rows:
        print(f"{phase:<25}{total:>12.2f}{n:>10d}{avg*1000:>12.3f}")

    print("\nTop slow individual calls (any phase), with metadata:")
    flat = []
    for phase, calls in PROF.items():
        for elapsed, meta in calls:
            flat.append((elapsed, phase, meta))
    flat.sort(key=lambda r: -r[0])
    for elapsed, phase, meta in flat[:top_n]:
        print(f"  {elapsed*1000:9.2f} ms  {phase:<20} {meta}")
    print("=" * 80 + "\n")


def dump_csv(path):
    if not _REPORTING_ENABLED:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "elapsed_s", "metadata"])
        for phase, calls in PROF.items():
            for elapsed, meta in calls:
                writer.writerow([phase, elapsed, meta])
    print(f"Wrote per-call profiling data to {path}")


def reset():
    PROF.clear()
