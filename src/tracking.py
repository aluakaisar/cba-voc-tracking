"""
Quality-determinant tracking metrics.

Implementation of the quality tracking framework in:

    Barravecchia, F., Mastrogiacomo, L., & Franceschini, F. (2023).
    Product quality tracking based on digital Voice-of-Customers.
    Total Quality Management & Business Excellence, 34(11-12), 1386-1409.
    https://doi.org/10.1080/14783363.2023.2177147

Every function here operates on a *topical prevalence* matrix TP, the output of
a topic model: one row per review, one column per topic, each row summing to 1.
Cell (j, d) is the share of review j attributable to topic d.

Note on the sign convention for Spearman's rho
-----------------------------------------------
Section 4.3 of the paper prints the classification rule as
"if rS < +0.4 then decreasing", which should read -0.4, and Table 8 then
labels increasing determinants as rS < -0.4, i.e. with the signs swapped
relative to the text. The worked example in Table 6 is unambiguous
(a declining profile yields rho = -0.964), so this module uses the
conventional sign: rho > 0 means the topic is gaining share over time.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Sequence

import numpy as np
import pandas as pd

import _stats

TrendLabel = Literal["increasing", "constant", "decreasing"]
AnomalyKind = Literal["global_outlier", "temporary_change", "structural_change"]

# Paper's section 4.3 threshold, following Myers et al. (2013).
RHO_NEUTRAL_BAND = 0.4


# ---------------------------------------------------------------------------
# Core indicators (paper sections 4.2 - 4.4)
# ---------------------------------------------------------------------------


def validate_tp(tp: pd.DataFrame, tol: float = 1e-6) -> None:
    """Raise if TP is not a valid topical prevalence matrix.

    Each row must sum to 1 (eq. between sections 2.3 and 4.2). A violation
    almost always means the topic model returned hard assignments or that
    rows were filtered after normalisation.
    """
    if tp.empty:
        raise ValueError("TP matrix is empty")
    if (tp.values < -tol).any():
        raise ValueError("TP contains negative prevalences")
    row_sums = tp.sum(axis=1)
    bad = row_sums[(row_sums - 1.0).abs() > tol]
    if len(bad):
        raise ValueError(
            f"{len(bad)} of {len(tp)} rows do not sum to 1 "
            f"(e.g. index {bad.index[0]!r} sums to {bad.iloc[0]:.6f})"
        )


def mtp(tp: pd.DataFrame) -> pd.Series:
    """Mean Topic Proportion, paper eq. (1).

    One number per topic: its average prevalence across every review in the
    window. Sums to 1 across topics, eq. (2).

    This is the whole-window summary. It is deliberately included because the
    paper's central demonstration is that MTP can be near-identical for topics
    whose period-by-period behaviour is completely different, so MTP alone is
    not a tracking indicator.
    """
    return tp.mean(axis=0).rename("MTP")


def imtp(tp: pd.DataFrame, periods: Sequence) -> pd.DataFrame:
    """Interval Mean Topical Prevalence, paper eq. (3).

    Same average as MTP, taken within each sampling period instead of over the
    whole window. Returns a (period x topic) frame; each row sums to 1, eq. (4).

    Parameters
    ----------
    tp : (n_reviews x n_topics) topical prevalence matrix.
    periods : length-n_reviews sequence of period labels, one per review.
              Use ``pandas.PeriodIndex`` values (e.g. monthly or quarterly)
              so the result sorts chronologically.
    """
    if len(periods) != len(tp):
        raise ValueError(
            f"periods has length {len(periods)} but TP has {len(tp)} rows"
        )
    grouped = tp.groupby(pd.Index(periods, name="period"), sort=True).mean()
    return grouped


def period_counts(periods: Sequence) -> pd.Series:
    """Number of reviews per sampling period.

    Always report this alongside IMTP. A period built from very few reviews
    produces an IMTP value with wide sampling error, and the trend statistics
    below give it exactly as much weight as a period built from thousands.
    """
    return (
        pd.Series(1, index=pd.Index(periods, name="period"))
        .groupby(level=0, sort=True)
        .sum()
        .rename("n_reviews")
    )


def spearman_rho(series: pd.Series) -> float:
    """Spearman rank correlation between sampling period order and IMTP.

    Paper eq. (5). Measures only whether the profile is monotonic, not by how
    much it moves. Returns NaN for a series shorter than 3 points or with no
    variation (a perfectly flat profile has no ranks to correlate).
    """
    y = pd.Series(series).dropna()
    if len(y) < 3 or y.nunique() == 1:
        return float("nan")
    x = np.arange(len(y), dtype=float)
    return _stats.spearmanr(x, y.to_numpy(dtype=float))


def classify_trend(rho: float, band: float = RHO_NEUTRAL_BAND) -> TrendLabel:
    """Label a profile from its rho, paper section 4.3.

    rho >  band  -> increasing
    rho < -band  -> decreasing
    otherwise    -> constant  (includes NaN, which is treated as no trend)
    """
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return "constant"
    if rho > band:
        return "increasing"
    if rho < -band:
        return "decreasing"
    return "constant"


def imtp_slope(series: pd.Series) -> float:
    """IMTP Average Slope, paper eq. (6).

    Slope of the least-squares line through the IMTP points, in share per
    sampling period. This is what rho cannot tell you: two profiles can both
    be 'increasing' while moving at very different rates.
    """
    y = pd.Series(series).dropna()
    if len(y) < 2:
        return float("nan")
    t = np.arange(len(y), dtype=float)
    t_dev = t - t.mean()
    y_dev = y.to_numpy(dtype=float) - y.to_numpy(dtype=float).mean()
    return float((t_dev * y_dev).sum() / (t_dev**2).sum())


# ---------------------------------------------------------------------------
# Level 1: emergence / disappearance (paper section 4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopicSetComparison:
    """Qualitative comparison of two windows' topic sets."""

    continuing: tuple[str, ...]
    emerged: tuple[str, ...]
    disappeared: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def compare_topic_sets(
    earlier: Sequence[str], later: Sequence[str]
) -> TopicSetComparison:
    """Which topics persisted, appeared, or vanished between two windows.

    This is the only comparison available when each window was modelled
    separately (the paper's 'local time window'), because topic indices from
    two independent model runs are not comparable. Matching is on the labels
    you assigned, so the labelling has to be done consistently by hand.
    """
    a, b = set(earlier), set(later)
    return TopicSetComparison(
        continuing=tuple(sorted(a & b)),
        emerged=tuple(sorted(b - a)),
        disappeared=tuple(sorted(a - b)),
    )


# ---------------------------------------------------------------------------
# Level 4: anomaly detection (paper section 4.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anomaly:
    topic: str
    period: object
    kind: AnomalyKind
    z_score: float
    imtp: float
    baseline: float

    def to_dict(self) -> dict:
        return asdict(self)


def _robust_sigma(
    values: np.ndarray, rel_floor: float = 0.02, tol: float = 1e-12
) -> float:
    """Scale estimate for period-to-period noise that survives level shifts.

    Estimated from successive differences rather than from the values
    themselves. A step change inflates the spread of the values but leaves
    almost every difference untouched, so a standard deviation of the levels
    would hide the very shift we are trying to find.

    Three estimators are tried in order, because each can degenerate:

    1. MAD of the differences, rescaled to a sigma. Most resistant to
       outliers, but returns ~0 whenever more than half the differences are
       identical, which happens on flat or near-flat profiles.
    2. IQR of the differences, rescaled. Survives up to a quarter of the
       points being extreme and copes with the flat case above.
    3. Plain standard deviation. Inflated by the anomalies themselves, so it
       is a last resort only.

    Each is compared against ``tol`` rather than zero, because a median
    subtraction on decimal data leaves float residue on the order of 1e-17,
    which is not a real dispersion but is not exactly zero either.

    Finally floored relative to the series mean, so an exactly constant
    profile cannot produce sigma = 0 and divide by it.
    """
    diffs = np.diff(values)
    if len(diffs) == 0:
        return float("nan")

    centred = diffs - np.median(diffs)
    scale = np.sqrt(2)  # differencing doubles the variance

    mad = float(np.median(np.abs(centred)))
    sigma = mad / 0.6745 / scale

    if sigma <= tol:
        q1, q3 = np.percentile(centred, [25, 75])
        sigma = float(q3 - q1) / 1.349 / scale

    if sigma <= tol:
        sigma = float(np.std(diffs, ddof=0)) / scale

    floor = rel_floor * float(np.mean(np.abs(values)))
    return max(sigma, floor, 1e-9)


def detect_anomalies(
    series: pd.Series,
    topic: str = "",
    z_threshold: float = 3.0,
    persistence: int = 2,
    shift_ratio: float = 0.5,
    min_baseline: int = 3,
    baseline_window: int = 6,
) -> list[Anomaly]:
    """Flag and classify departures from a topic's established behaviour.

    The paper names three shapes (section 4.5) but leaves the detector open,
    pointing at statistical control charts as one option. This follows that
    logic in two stages.

    Each period is compared against a forecast made from the periods before
    it: a Theil-Sen line fitted to the trailing ``baseline_window``, projected
    one step forward. An anomaly is a period the recent past failed to
    predict.

    Forecasting from a trailing window rather than detrending the whole series
    is the load-bearing choice here, and it took three attempts to get right.
    Residuals from one line fitted to the entire series do not work: a line
    through a step change absorbs the step, leaving residuals positive before
    the break and negative after, so a permanent shift reads as two unrelated
    blips. Comparing each period to a trailing *median* does not work either:
    on a rising series the median lags the drift, so the trend alone
    manufactures a deviation of two sigma before any noise, and every rising
    topic becomes a wall of false alarms. Detrending first by a single
    estimated drift fails from both ends: the median of successive
    differences is unbiased by a step but so noisy that its error accumulates
    into a spurious ramp of about four noise sigma over a dozen periods, while
    Theil-Sen over the whole series is tight on a genuine trend but badly
    wrong on a step, and detrending by a wrong slope invents shifts that were
    never there.

    A local forecast has none of those failure modes. A steady trend is
    predicted accurately by extrapolating its own recent history, so it raises
    nothing. A step is invisible to the window that precedes it, so it raises
    a large residual exactly once.

    The local fit is Theil-Sen rather than least squares so that a spike
    sitting inside the trailing window does not tilt the forecast for the
    periods after it.

    Classification extends the same forecast line forward and asks what the
    residual does next:

        global_outlier     - one period out, straight back on forecast
        temporary_change   - out, then decays back over the following periods
        structural_change  - the residual stays shifted

    ``z_threshold`` defaults to 3.0, following Shewhart control-chart practice
    rather than the 2.5 that looks reasonable in isolation. At 2.5 sigma a
    two-sided test fires on about 1.2% of in-control periods, so across 20
    topics over 24 months roughly six false alarms are expected by chance
    alone. At 3.0 sigma that drops to under one.

    Note that the threshold applies to the *forecast* residual, whose spread
    is wider than the raw period-to-period noise (the fitted line carries its
    own error), so a raw 3.3-sigma wobble sits at roughly 2.8 sigma here and
    will not be flagged. That is the intended behaviour, not a miss.

    Two limits worth stating in any write-up. Sigma is estimated from the same
    series being tested, so one enormous break drags the threshold up and can
    mask smaller ones. And the first ``min_baseline`` periods can never be
    flagged, because there is nothing yet to forecast them from. On a short
    series this is a shortlist for human inspection, not a significance test.
    """
    y = pd.Series(series).dropna()
    min_fit = max(min_baseline, 3)
    if len(y) < min_fit + 2:
        return []

    values = y.to_numpy(dtype=float)
    sigma = _robust_sigma(values)
    if not np.isfinite(sigma) or sigma <= 0:
        return []

    # A one-step-ahead forecast from an n-point fit is less certain than a
    # single observation: it carries the line's own error as well as the
    # period's. For a 6-point window that inflation is about 1.4x.
    def _forecast_sigma(n: int) -> float:
        t = np.arange(n, dtype=float)
        ss = float(((t - t.mean()) ** 2).sum())
        lever = 1.0 + 1.0 / n + ((n - t.mean()) ** 2) / ss if ss > 0 else 2.0
        return sigma * float(np.sqrt(lever))

    def _project(window: np.ndarray, ahead: int) -> float:
        t = np.arange(len(window), dtype=float)
        slope, intercept = _stats.theilslopes(window, t)
        return float(intercept + slope * (len(window) - 1 + ahead))

    out: list[Anomaly] = []
    skip_until = -1
    for i in range(min_fit, len(values)):
        if i <= skip_until:
            continue

        window = values[max(0, i - baseline_window) : i]
        if len(window) < min_fit:
            continue

        forecast = _project(window, 1)
        resid = values[i] - forecast
        limit = z_threshold * _forecast_sigma(len(window))
        if abs(resid) <= limit:
            continue

        after = values[i + 1 : i + 1 + persistence]
        if len(after) == 0:
            kind: AnomalyKind = "global_outlier"
        else:
            after_resid = np.array(
                [values[i + 1 + k] - _project(window, 2 + k) for k in range(len(after))]
            )
            same_side = np.sign(after_resid) == np.sign(resid)
            held = np.abs(after_resid) > abs(resid) * shift_ratio
            if np.all(same_side) and np.all(held):
                kind = "structural_change"
            elif same_side[0] and held[0]:
                kind = "temporary_change"
            else:
                kind = "global_outlier"

        out.append(
            Anomaly(
                topic=topic,
                period=y.index[i],
                kind=kind,
                z_score=float(resid / _forecast_sigma(len(window))),
                imtp=float(values[i]),
                baseline=forecast,
            )
        )

        # After a declared level shift, the pre-shift history is no longer a
        # valid baseline. A window straddling the break fits a steep slope
        # through two flat levels and forecasts well past the new one, so the
        # periods just after a structural change would each be flagged again
        # for failing to keep climbing. Skip far enough for the window to
        # clear the break entirely. The cost is that a second event inside
        # ``baseline_window`` periods of a shift is missed; the alternative is
        # reporting one shift several times over.
        skip_until = i + (baseline_window - 1 if kind == "structural_change" else persistence)
    return out


# ---------------------------------------------------------------------------
# Convenience: the whole level 2-4 summary in one frame
# ---------------------------------------------------------------------------


def track(
    tp: pd.DataFrame,
    periods: Sequence,
    band: float = RHO_NEUTRAL_BAND,
    z_threshold: float = 2.5,
) -> dict[str, object]:
    """Run the full tracking analysis and return every intermediate result.

    Returns a dict with:
        imtp       (period x topic) frame, eq. (3)
        mtp        Series, eq. (1)
        counts     reviews per period
        summary    per-topic MTP, rho, trend label, slope, first/last IMTP
        anomalies  list of Anomaly records across all topics
    """
    validate_tp(tp)
    imtp_df = imtp(tp, periods)
    mtp_s = mtp(tp)
    counts = period_counts(periods)

    rows = []
    anomalies: list[Anomaly] = []
    for topic in imtp_df.columns:
        col = imtp_df[topic]
        rho = spearman_rho(col)
        rows.append(
            {
                "topic": topic,
                "MTP": mtp_s.get(topic, float("nan")),
                "spearman_rho": rho,
                "trend": classify_trend(rho, band=band),
                "imtp_slope": imtp_slope(col),
                "imtp_first": col.iloc[0],
                "imtp_last": col.iloc[-1],
            }
        )
        anomalies.extend(
            detect_anomalies(col, topic=str(topic), z_threshold=z_threshold)
        )

    summary = (
        pd.DataFrame(rows)
        .set_index("topic")
        .sort_values("imtp_slope", ascending=False)
    )
    return {
        "imtp": imtp_df,
        "mtp": mtp_s,
        "counts": counts,
        "summary": summary,
        "anomalies": anomalies,
    }
