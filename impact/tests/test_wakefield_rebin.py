import numpy as np
import pytest
from beamphysics.wakefields import TabularWakefield

from impact.wakefields import (
    check_wakefield_dz_names,
    rebin_tabular_wakefield,
    wakefield_dz_for,
)


def igf_table(delta: float, n: int) -> TabularWakefield:
    """Bin-averaged table of w(s) = 1/sqrt(s), with the first row doubled."""
    F = 2 * np.sqrt((np.arange(n) + 0.5) * delta)
    K = np.diff(F, prepend=0.0) / delta
    K[0] *= 2
    return TabularWakefield(-delta * np.arange(n)[::-1], K[::-1], kind="linear")


def rows(table: TabularWakefield) -> np.ndarray:
    return table.W_data[::-1]


def impact_wake_potential(table_rows: np.ndarray, density: np.ndarray, h: float):
    """The read-in wake convolution of IMPACT-T and IMPACT-Z on a mesh of width h."""
    green = np.array(table_rows[: len(density)], dtype=float)
    green[0] *= 0.5
    n = len(density)
    return np.array([np.sum(green[: n - k] * density[k:]) for k in range(n)]) * h


@pytest.fixture(scope="module")
def gaussian_bunch():
    sigma, h = 1e-3, 40e-6
    z = np.arange(-6 * sigma, 6 * sigma + h / 2, h)
    density = np.exp(-(z**2) / 2 / sigma**2) / np.sqrt(2 * np.pi) / sigma
    # Exact potential of w(s) = 1/sqrt(s); s = u^2 removes the singularity.
    u = np.linspace(0, np.sqrt(14 * sigma), 20001)
    exact = np.array(
        [
            np.trapezoid(2 * np.exp(-((zz + u**2) ** 2) / 2 / sigma**2), u)
            / np.sqrt(2 * np.pi)
            / sigma
            for zz in z
        ]
    )
    weight = density / density.sum()
    loss = np.sum(weight * exact)
    spread = np.sqrt(np.sum(weight * (exact - loss) ** 2))
    return h, density, weight, loss, spread


def loss_and_spread_errors(table_rows, gaussian_bunch):
    h, density, weight, loss, spread = gaussian_bunch
    V = impact_wake_potential(table_rows, density, h)
    L = np.sum(weight * V)
    S = np.sqrt(np.sum(weight * (V - L) ** 2))
    return L / loss - 1, S / spread - 1


@pytest.mark.parametrize("delta", [1e-6, 5e-6, 10e-6, 20e-6, 40e-6, 40e-6 / 3.7])
def test_rebin_matches_the_exact_wake_potential(gaussian_bunch, delta: float) -> None:
    h, density = gaussian_bunch[:2]
    n = len(density)
    table = igf_table(delta, int(np.ceil(n * h / delta)) + 2)

    rebinned = rebin_tabular_wakefield(table, dz=h, zmax=(n - 1) * h)
    loss_error, spread_error = loss_and_spread_errors(rows(rebinned), gaussian_bunch)
    assert abs(loss_error) < 1e-3
    assert abs(spread_error) < 2e-3


def test_resampling_a_fine_table_is_wrong(gaussian_bunch) -> None:
    # The reason for the rebin: reading a 1 um table at the 40 um mesh points.
    h, density = gaussian_bunch[:2]
    table = igf_table(1e-6, 40 * len(density) + 2)
    resampled = table.wake(-h * np.arange(len(density)))
    loss_error, _ = loss_and_spread_errors(resampled, gaussian_bunch)
    assert loss_error > 0.5


@pytest.mark.parametrize("multiple", [1, 3, 5])
def test_rebin_is_exact_at_odd_multiples(multiple: int) -> None:
    delta = 2e-6
    h = multiple * delta
    rebinned = rebin_tabular_wakefield(igf_table(delta, 300), dz=h, zmax=100 * h)
    np.testing.assert_allclose(rows(rebinned)[:50], rows(igf_table(h, 50)), rtol=1e-12)


def test_rebin_keeps_the_table_without_dz() -> None:
    table = igf_table(1e-6, 100)
    np.testing.assert_array_equal(rows(rebin_tabular_wakefield(table)), rows(table))


def test_wake_is_zero_beyond_the_table() -> None:
    table = igf_table(1e-6, 100)
    extended = rebin_tabular_wakefield(table, zmax=199e-6)
    assert extended.default_zmax == pytest.approx(199e-6)
    np.testing.assert_array_equal(rows(extended)[:100], rows(table))
    assert not np.any(rows(extended)[100:])


def test_row_limit_shortens_the_range_not_the_spacing() -> None:
    table = igf_table(1e-6, 2000)
    shortened = rebin_tabular_wakefield(table, max_rows=1000)
    assert len(shortened.z_data) == 1000
    np.testing.assert_array_equal(rows(shortened), rows(table)[:1000])

    with pytest.warns(UserWarning, match="shorter than wakefield_zmax"):
        rebin_tabular_wakefield(table, zmax=1.5e-3, max_rows=1000)


def test_rebin_rejects_bad_input() -> None:
    z = np.array([-4.0, -2.0, -1.0, 0.0]) * 1e-6
    with pytest.raises(ValueError, match="uniform"):
        rebin_tabular_wakefield(TabularWakefield(z, np.ones(4)))
    with pytest.raises(ValueError, match="positive"):
        rebin_tabular_wakefield(igf_table(1e-6, 10), dz=0)


def test_wakefield_dz_lookup() -> None:
    assert wakefield_dz_for(None, "P1") is None
    assert wakefield_dz_for(1e-6, "P1") == 1e-6
    assert wakefield_dz_for({"P1": 2e-6}, "P1") == 2e-6
    assert wakefield_dz_for({"P1": 2e-6}, "P2") is None

    check_wakefield_dz_names({"P1": 2e-6}, {"P1", "P2"})
    with pytest.raises(ValueError, match="P9"):
        check_wakefield_dz_names({"P9": 2e-6}, {"P1", "P2"})
