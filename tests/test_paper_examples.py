"""
Validation of tracking.py against the published worked examples.

Every number asserted here is printed in Barravecchia, Mastrogiacomo &
Franceschini (2023). If these pass, the implementation reproduces the paper's
own arithmetic, which is the only external check available for this method.

Run with:  pytest -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracking import (  # noqa: E402
    classify_trend,
    compare_topic_sets,
    detect_anomalies,
    imtp,
    imtp_slope,
    mtp,
    spearman_rho,
    track,
    validate_tp,
)

TOPICS = ["determinant_A", "determinant_B", "determinant_C"]

# Table 4: nine reviews, three sampling periods (Jan/Feb/Mar 2022).
TABLE_4 = pd.DataFrame(
    [
        [0.80, 0.15, 0.05],
        [0.10, 0.70, 0.20],
        [0.80, 0.15, 0.05],
        [0.25, 0.70, 0.05],
        [0.45, 0.15, 0.40],
        [0.35, 0.10, 0.55],
        [0.15, 0.65, 0.20],
        [0.20, 0.10, 0.70],
        [0.10, 0.30, 0.60],
    ],
    columns=TOPICS,
    index=pd.to_datetime(
        [
            "2022-01-03",
            "2022-01-15",
            "2022-01-17",
            "2022-02-11",
            "2022-02-16",
            "2022-02-18",
            "2022-03-09",
            "2022-03-13",
            "2022-03-22",
        ]
    ),
)

TABLE_4_PERIODS = TABLE_4.index.to_period("M")


class TestTable4:
    """Topical prevalence matrix and the MTP indicator, eq. (1) and (2)."""

    def test_every_review_sums_to_one(self):
        assert np.allclose(TABLE_4.sum(axis=1), 1.0)

    def test_validate_tp_accepts_the_paper_matrix(self):
        validate_tp(TABLE_4)  # must not raise

    def test_mtp_matches_published_values(self):
        got = mtp(TABLE_4)
        # Paper prints 0.36 / 0.33 / 0.31 (2 dp).
        assert got["determinant_A"] == pytest.approx(3.2 / 9)
        assert got["determinant_B"] == pytest.approx(3.0 / 9)
        assert got["determinant_C"] == pytest.approx(2.8 / 9)
        assert [round(v, 2) for v in got] == [0.36, 0.33, 0.31]

    def test_mtp_sums_to_one(self):
        """Paper eq. (2)."""
        assert mtp(TABLE_4).sum() == pytest.approx(1.0)


class TestTable5:
    """IMTP indicator, eq. (3) and (4)."""

    def test_imtp_matches_published_values(self):
        got = imtp(TABLE_4, TABLE_4_PERIODS)
        assert got.shape == (3, 3)

        # Published: period 1 -> 0.57 / 0.33 / 0.10
        #            period 2 -> 0.35 / 0.32 / 0.33
        #            period 3 -> 0.15 / 0.35 / 0.50
        expected = [
            [1.70 / 3, 1.00 / 3, 0.30 / 3],
            [1.05 / 3, 0.95 / 3, 1.00 / 3],
            [0.45 / 3, 1.05 / 3, 1.50 / 3],
        ]
        assert np.allclose(got.to_numpy(), expected)
        assert [[round(v, 2) for v in row] for row in got.to_numpy()] == [
            [0.57, 0.33, 0.10],
            [0.35, 0.32, 0.33],
            [0.15, 0.35, 0.50],
        ]

    def test_each_period_sums_to_one(self):
        """Paper eq. (4)."""
        assert np.allclose(imtp(TABLE_4, TABLE_4_PERIODS).sum(axis=1), 1.0)

    def test_imtp_reveals_what_mtp_hides(self):
        """The paper's central claim, asserted as a test.

        All three determinants have MTP within 0.05 of each other, yet A falls
        by 0.42 and C rises by 0.40 across the same three periods.
        """
        m = mtp(TABLE_4)
        assert m.max() - m.min() < 0.05

        i = imtp(TABLE_4, TABLE_4_PERIODS)
        assert i["determinant_A"].iloc[-1] - i["determinant_A"].iloc[0] < -0.40
        assert i["determinant_C"].iloc[-1] - i["determinant_C"].iloc[0] > 0.39
        assert abs(
            i["determinant_B"].iloc[-1] - i["determinant_B"].iloc[0]
        ) < 0.03


class TestTable6:
    """Spearman's rho, eq. (5). Seven periods, declining determinant A."""

    PROFILE = pd.Series(
        [0.08, 0.07, 0.062, 0.055, 0.056, 0.045, 0.035],
        index=pd.RangeIndex(1, 8, name="period"),
    )

    def test_rho_matches_published_value(self):
        assert spearman_rho(self.PROFILE) == pytest.approx(-0.964, abs=5e-4)

    def test_rho_equals_hand_computation(self):
        """1 - 6*sum(d^2) / (T*(T^2-1)) with sum(d^2) = 110, T = 7."""
        assert spearman_rho(self.PROFILE) == pytest.approx(
            1 - (6 * 110) / (7 * (7**2 - 1))
        )

    def test_not_minus_one_because_of_the_single_inversion(self):
        """Period 5 (0.056) sits above period 4 (0.055).

        That one out-of-order pair is the entire reason the paper reports
        -0.964 rather than -1.
        """
        assert self.PROFILE.iloc[4] > self.PROFILE.iloc[3]
        assert spearman_rho(self.PROFILE) > -1.0

        monotonic = self.PROFILE.sort_values(ascending=False)
        assert spearman_rho(monotonic) == pytest.approx(-1.0)

    def test_classified_as_decreasing(self):
        assert classify_trend(spearman_rho(self.PROFILE)) == "decreasing"


class TestTable7:
    """IMTP Average Slope, eq. (6). Seven periods, rising determinant B."""

    PROFILE = pd.Series(
        [0.1, 0.25, 0.2, 0.34, 0.35, 0.4, 0.38],
        index=pd.RangeIndex(1, 8, name="period"),
    )

    def test_slope_matches_published_value(self):
        assert imtp_slope(self.PROFILE) == pytest.approx(0.0461, abs=5e-5)

    def test_slope_equals_hand_computation(self):
        """Numerator 1.29, denominator 28."""
        assert imtp_slope(self.PROFILE) == pytest.approx(1.29 / 28)

    def test_mean_period_and_mean_imtp(self):
        """Paper prints t-bar = 4 and IMTP-bar = 0.29."""
        assert self.PROFILE.index.to_numpy().mean() == 4
        assert round(self.PROFILE.mean(), 2) == 0.29

    def test_slope_separates_two_increasing_profiles(self):
        """Section 4.4: both rise, one 2.4x faster.

        Paper reports 0.0189 for determinant A and 0.0461 for B.
        """
        slow = pd.Series([0.10, 0.12, 0.13, 0.16, 0.18, 0.21, 0.22])
        assert classify_trend(spearman_rho(slow)) == "increasing"
        assert classify_trend(spearman_rho(self.PROFILE)) == "increasing"
        assert imtp_slope(self.PROFILE) > 2 * imtp_slope(slow)


class TestSignConvention:
    """Guards the sign inconsistency between the paper's text and Table 8.

    Section 4.3 prints "if rS < +0.4 then decreasing" (should be -0.4) and
    Table 8 labels increasing determinants "rS < -0.4". Table 6's worked
    example settles it: a declining profile gives rho = -0.964.
    """

    def test_rising_profile_gives_positive_rho(self):
        rising = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        assert spearman_rho(rising) == pytest.approx(1.0)
        assert classify_trend(spearman_rho(rising)) == "increasing"

    def test_falling_profile_gives_negative_rho(self):
        falling = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        assert spearman_rho(falling) == pytest.approx(-1.0)
        assert classify_trend(spearman_rho(falling)) == "decreasing"

    def test_neutral_band_is_inclusive(self):
        assert classify_trend(0.4) == "constant"
        assert classify_trend(-0.4) == "constant"
        assert classify_trend(0.41) == "increasing"
        assert classify_trend(-0.41) == "decreasing"


class TestRestaurantCaseStudy:
    """Section 5.1: a new determinant forces the others down.

    The paper observes that when a 'Covid rules' topic emerged, every
    continuing determinant's IMTP fell. That is arithmetic, not a change in
    customer opinion: eq. (4) fixes the row total at 1.
    """

    def test_new_topic_mechanically_reduces_the_others(self):
        before = pd.DataFrame(
            [[0.4, 0.35, 0.25]], columns=["food", "service", "price"]
        )
        after = pd.DataFrame(
            [[0.3, 0.26, 0.19, 0.25]],
            columns=["food", "service", "price", "covid_rules"],
        )
        validate_tp(before)
        validate_tp(after)
        for topic in ["food", "service", "price"]:
            assert after[topic].iloc[0] < before[topic].iloc[0]

    def test_emergence_is_detected(self):
        cmp = compare_topic_sets(
            ["food", "service", "price"],
            ["food", "service", "price", "covid_rules"],
        )
        assert cmp.emerged == ("covid_rules",)
        assert cmp.disappeared == ()
        assert cmp.continuing == ("food", "price", "service")

    def test_disappearance_is_detected(self):
        cmp = compare_topic_sets(["food", "queue"], ["food", "delivery"])
        assert cmp.emerged == ("delivery",)
        assert cmp.disappeared == ("queue",)


def noisy(levels, sd=0.01, seed=0):
    """A profile at the given levels with realistic period-to-period wobble.

    Synthetic profiles built from a repeating pattern (0.10, 0.11, 0.09, ...)
    are a bad test bed: more than half their successive differences are
    identical, which degenerates every robust scale estimator. Real IMTP
    series wobble irregularly, so the fixtures below do too.
    """
    rng = np.random.default_rng(seed)
    return pd.Series(np.asarray(levels, dtype=float) + rng.normal(0, sd, len(levels)))


class TestAnomalyShapes:
    """Section 4.5: the three named anomaly shapes."""

    def test_global_outlier(self):
        """One period out, then straight back to the old level."""
        s = noisy([0.20] * 4 + [0.55] + [0.20] * 4)
        hits = detect_anomalies(s, z_threshold=3.0)
        assert [a.kind for a in hits] == ["global_outlier"]
        assert hits[0].period == 4
        assert hits[0].imtp == pytest.approx(0.55, abs=0.04)
        assert hits[0].baseline == pytest.approx(0.20, abs=0.02)

    def test_structural_change(self):
        """Level moves and stays moved."""
        s = noisy([0.10] * 6 + [0.40] * 6, seed=1)
        hits = detect_anomalies(s, z_threshold=3.0)
        assert hits, "a level that quadruples and holds must be flagged"
        assert hits[0].kind == "structural_change"
        assert hits[0].period == 6

    def test_temporary_change(self):
        """Out, then decays back rather than snapping back."""
        s = noisy([0.10, 0.10, 0.10, 0.10, 0.45, 0.30, 0.18, 0.11, 0.10], seed=2)
        hits = detect_anomalies(s, z_threshold=3.0)
        assert hits
        assert hits[0].kind == "temporary_change"
        assert hits[0].period == 4

    def test_quiet_series_flags_nothing(self):
        """Seed chosen so the fixture's largest draw is 1.3 sigma.

        Seed 3 happens to contain a 3.3-sigma draw, and a 3-sigma control
        chart is supposed to flag that, so it would be testing the fixture
        rather than the detector.
        """
        s = noisy([0.20] * 12, seed=7)
        assert detect_anomalies(s, z_threshold=3.0) == []

    def test_a_genuine_outlier_in_quiet_noise_is_caught(self):
        """The converse guard: a real jump out of quiet noise is not ignored."""
        s = noisy([0.20] * 6 + [0.35] + [0.20] * 5, seed=10)
        hits = detect_anomalies(s, z_threshold=3.0)
        assert [a.kind for a in hits] == ["global_outlier"]
        assert hits[0].period == 6

    def test_structural_change_is_reported_once(self):
        """A window straddling the break forecasts past the new level.

        Without a long enough skip after a declared shift, each period after
        it gets flagged again for failing to keep climbing.
        """
        s = noisy([0.10] * 6 + [0.40] * 6, seed=1)
        hits = detect_anomalies(s, z_threshold=3.0)
        assert len(hits) == 1
        assert hits[0].kind == "structural_change"

    def test_steady_trend_is_not_an_anomaly(self):
        """A determinant drifting up for two years is a trend, not a spike.

        Levels 2 and 3 exist to describe this; level 4 must not double-report
        it, or every rising topic becomes a wall of false alarms.
        """
        s = noisy(np.linspace(0.10, 0.30, 24), seed=4)
        assert classify_trend(spearman_rho(s)) == "increasing"
        assert detect_anomalies(s, z_threshold=3.0) == []

    def test_spike_is_reported_once_not_twice(self):
        """A spike has a large change going up and coming back down.

        Only the event itself should be reported, not its recovery.
        """
        s = noisy([0.15] * 5 + [0.60] + [0.15] * 5, seed=5)
        hits = detect_anomalies(s, z_threshold=3.0)
        assert len(hits) == 1
        assert hits[0].period == 5

    def test_too_short_series_is_skipped(self):
        assert detect_anomalies(pd.Series([0.1, 0.9, 0.1])) == []

    def test_a_line_fit_would_have_missed_the_step(self):
        """Regression guard for the bug this detector was rewritten to fix.

        Fitting one line through a step change leaves residuals that are
        positive before the break and negative after it, so the shift reads as
        two unrelated blips. The trailing-baseline comparison sees it as one
        structural change.
        """
        s = pd.Series([0.10] * 6 + [0.40] * 6)
        t = np.arange(len(s), dtype=float)
        slope, intercept = np.polyfit(t, s.to_numpy(float), 1)
        resid = s.to_numpy(float) - (slope * t + intercept)
        assert resid[5] < 0 < resid[6]  # sign flips across the break

        kinds = [a.kind for a in detect_anomalies(s, z_threshold=2.0)]
        assert "structural_change" in kinds


class TestGuards:
    """Input validation, so a broken topic model fails loudly."""

    def test_rows_must_sum_to_one(self):
        bad = pd.DataFrame([[0.5, 0.2]], columns=["a", "b"])
        with pytest.raises(ValueError, match="do not sum to 1"):
            validate_tp(bad)

    def test_negative_prevalence_rejected(self):
        bad = pd.DataFrame([[1.2, -0.2]], columns=["a", "b"])
        with pytest.raises(ValueError, match="negative"):
            validate_tp(bad)

    def test_period_length_must_match(self):
        with pytest.raises(ValueError, match="length"):
            imtp(TABLE_4, TABLE_4_PERIODS[:5])

    def test_flat_profile_returns_nan_rho_and_constant_label(self):
        flat = pd.Series([0.25] * 6)
        assert np.isnan(spearman_rho(flat))
        assert classify_trend(spearman_rho(flat)) == "constant"


class TestTrackEndToEnd:
    def test_track_reproduces_the_paper_example(self):
        res = track(TABLE_4, TABLE_4_PERIODS)

        assert list(res["counts"]) == [3, 3, 3]
        assert res["mtp"].sum() == pytest.approx(1.0)
        assert np.allclose(res["imtp"].sum(axis=1), 1.0)

        summary = res["summary"]
        assert summary.loc["determinant_A", "imtp_slope"] < 0
        assert summary.loc["determinant_C", "imtp_slope"] > 0
        assert summary.index[0] == "determinant_C"  # steepest riser first

    def test_track_rejects_an_invalid_tp_matrix(self):
        bad = TABLE_4.copy()
        bad.iloc[0, 0] = 0.9
        with pytest.raises(ValueError):
            track(bad, TABLE_4_PERIODS)
