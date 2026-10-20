"""
Checks the hand-rolled statistics against scipy's, where scipy is available.

The point of src/_stats.py is to let the analysis run without scipy installed,
since pinning a compatible numpy/scipy/pandas triple is a reliable way to lose
an afternoon. But "without scipy" must not mean "without agreeing with scipy",
so these tests compare the two implementations wherever scipy is present and
skip quietly where it is not.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import _stats  # noqa: E402


@pytest.fixture(scope="module")
def sp():
    """scipy.stats, or skip the test."""
    return pytest.importorskip("scipy.stats", reason="scipy not installed")


SERIES = [
    [0.08, 0.07, 0.062, 0.055, 0.056, 0.045, 0.035],      # the paper's Table 6
    [0.1, 0.25, 0.2, 0.34, 0.35, 0.4, 0.38],              # the paper's Table 7
    [0.2, 0.2, 0.2, 0.3, 0.1, 0.1],                       # heavy ties
    [1.0, 2.0, 3.0, 4.0, 5.0],                            # perfectly monotonic
    [5.0, 1.0, 4.0, 2.0, 3.0],                            # no trend
    [0.1, 0.1, 0.1, 0.1, 0.9],                            # one outlier
]


class TestRankdata:
    def test_matches_scipy(self, sp):
        for s in SERIES:
            assert np.allclose(_stats.rankdata(np.array(s)), sp.rankdata(s))

    def test_ties_share_the_average_rank(self):
        # Two values tied for ranks 1 and 2 both become 1.5.
        assert np.allclose(_stats.rankdata(np.array([5.0, 5.0, 9.0])), [1.5, 1.5, 3.0])

    def test_three_way_tie(self):
        assert np.allclose(_stats.rankdata(np.array([2.0, 2.0, 2.0])), [2.0, 2.0, 2.0])


class TestSpearman:
    def test_matches_scipy(self, sp):
        x = np.arange(7, dtype=float)
        for s in SERIES:
            mine = _stats.spearmanr(x[: len(s)], np.array(s))
            theirs = sp.spearmanr(x[: len(s)], s).statistic
            assert mine == pytest.approx(theirs, abs=1e-12)

    def test_reproduces_the_paper(self):
        """Table 6: a declining profile over seven periods gives -0.964."""
        y = np.array([0.08, 0.07, 0.062, 0.055, 0.056, 0.045, 0.035])
        rho = _stats.spearmanr(np.arange(7, dtype=float), y)
        assert rho == pytest.approx(1 - (6 * 110) / (7 * 48))
        assert rho == pytest.approx(-0.964, abs=5e-4)

    def test_monotonic_extremes(self):
        x = np.arange(5, dtype=float)
        assert _stats.spearmanr(x, x) == pytest.approx(1.0)
        assert _stats.spearmanr(x, x[::-1]) == pytest.approx(-1.0)

    def test_constant_input_is_nan(self):
        x = np.arange(5, dtype=float)
        assert np.isnan(_stats.spearmanr(x, np.ones(5)))

    def test_ties_do_not_invent_a_trend(self):
        """Ordinal ranks would order tied values arbitrarily and fake a trend."""
        y = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        assert np.isnan(_stats.spearmanr(np.arange(6, dtype=float), y))


class TestTheilSen:
    def test_slope_matches_scipy(self, sp):
        for s in SERIES:
            x = np.arange(len(s), dtype=float)
            mine, _ = _stats.theilslopes(np.array(s), x)
            theirs = sp.theilslopes(s, x)[0]
            assert mine == pytest.approx(theirs, abs=1e-12)

    def test_intercept_matches_scipy(self, sp):
        for s in SERIES:
            x = np.arange(len(s), dtype=float)
            _, mine = _stats.theilslopes(np.array(s), x)
            theirs = sp.theilslopes(s, x)[1]
            assert mine == pytest.approx(theirs, abs=1e-12)

    def test_exact_line_recovered(self):
        x = np.arange(10, dtype=float)
        slope, intercept = _stats.theilslopes(3.0 * x + 7.0, x)
        assert slope == pytest.approx(3.0)
        assert intercept == pytest.approx(7.0)

    def test_resists_a_single_wild_point(self):
        """One corrupted period should barely move the fit.

        This is the property the anomaly detector depends on: a spike sitting
        inside the trailing window must not tilt the forecast for the periods
        after it.

        The outlier goes near the end of the window, not the middle. A point at
        the centroid of x has zero leverage on a least-squares slope: it shifts
        the intercept and leaves the slope untouched, so putting it there would
        make least squares look robust and prove nothing.
        """
        x = np.arange(11, dtype=float)
        clean = 2.0 * x + 1.0
        dirty = clean.copy()
        dirty[9] = 500.0

        robust, _ = _stats.theilslopes(dirty, x)
        least_squares = float(np.polyfit(x, dirty, 1)[0])

        assert robust == pytest.approx(2.0, abs=0.5)
        assert abs(least_squares - 2.0) > 10.0  # dragged an order of magnitude

    def test_centroid_outlier_has_no_leverage_on_a_slope(self):
        """The companion fact, asserted so the point above is not folklore."""
        x = np.arange(11, dtype=float)
        dirty = 2.0 * x + 1.0
        dirty[5] = 500.0  # x = 5 is exactly the mean of x
        assert float(np.polyfit(x, dirty, 1)[0]) == pytest.approx(2.0)

    def test_degenerate_inputs(self):
        assert np.isnan(_stats.theilslopes(np.array([1.0]), np.array([0.0]))[0])
        # All x identical: every pairwise slope is undefined.
        assert np.isnan(_stats.theilslopes(np.array([1.0, 2.0]), np.array([3.0, 3.0]))[0])


class TestNoScipyImport:
    def test_tracking_module_does_not_import_scipy(self):
        """Guards the whole point of this module."""
        source = (Path(__file__).resolve().parents[1] / "src" / "tracking.py").read_text()
        assert "scipy" not in source
