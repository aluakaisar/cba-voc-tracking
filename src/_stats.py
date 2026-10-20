"""
The two statistics this project needs from scipy, implemented on numpy alone.

Why: scipy is a 30MB compiled wheel whose version has to agree with numpy's,
and on older Python versions that agreement is easy to miss. Asking pip to
find a working numpy/scipy/pandas triple is the single most common way to
lose an afternoon to a dependency resolver. Spearman's rho and a Theil-Sen
fit are twenty lines each, so the dependency is not worth its cost here.

Both functions are checked against their scipy equivalents in
tests/test_stats.py, including the tie-handling that is easy to get wrong.
"""

from __future__ import annotations

import numpy as np


def rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks from 1, with ties sharing the average of the ranks they span.

    This matches scipy.stats.rankdata's default 'average' method. Tie handling
    matters: with ordinary ordinal ranks, two periods holding the same IMTP
    would be given an arbitrary order, which then shows up as a spurious trend.
    """
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    # Average the ranks within each run of equal values.
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation coefficient.

    Pearson correlation computed on the ranks rather than the values, which is
    what makes it a measure of monotonic association rather than linear
    association. Returns NaN when either input has no variation, because
    constant ranks have no correlation to measure.
    """
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def theilslopes(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Theil-Sen robust line fit, returning (slope, intercept).

    The slope is the median of the slopes of every pair of points. Taking a
    median rather than a least-squares average is what makes it resistant:
    a single wild point corrupts a handful of the pairwise slopes but cannot
    move their median far.

    The intercept follows scipy's convention, median(y) - slope * median(x),
    rather than the least-squares intercept.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 2:
        return float("nan"), float("nan")

    # All pairwise slopes, skipping pairs with equal x (vertical, undefined).
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    dy = y[j] - y[i]
    ok = dx != 0
    if not ok.any():
        return float("nan"), float("nan")

    slope = float(np.median(dy[ok] / dx[ok]))
    intercept = float(np.median(y) - slope * np.median(x))
    return slope, intercept
