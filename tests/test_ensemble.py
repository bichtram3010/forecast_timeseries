import numpy as np

from m5.ensemble import combine, search_weights, simplex_grid


def test_simplex_grid():
    grid = simplex_grid(3, 0.1)
    assert grid.shape == (66, 3)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0)
    assert (grid >= 0).all()


def test_combine():
    out = combine({"a": np.ones((2, 2)), "b": np.zeros((2, 2))}, {"a": 0.3, "b": 0.7})
    np.testing.assert_allclose(out, 0.3)


def test_search_picks_single_best_model():
    actual = np.full((2, 4), 10.0)
    fcs = {"good": actual.copy(), "bad": actual * 2, "worse": actual * 3}
    weights, score = search_weights(fcs, lambda f: float(np.abs(f - actual).mean()))
    assert weights == {"bad": 0.0, "good": 1.0, "worse": 0.0}
    assert score == 0.0


def test_search_finds_a_blend():
    actual = np.full((2, 4), 10.0)
    fcs = {"low": actual - 2, "high": actual + 2}
    weights, score = search_weights(fcs, lambda f: float(np.abs(f - actual).mean()))
    assert weights == {"high": 0.5, "low": 0.5}
    assert score < 1e-6
