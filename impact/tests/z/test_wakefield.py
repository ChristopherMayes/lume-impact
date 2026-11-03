from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
from beamphysics.wakefields import ResistiveWallPseudomode, TabularWakefield
from pytao import SubprocessTao as Tao
from pytao.model import BeamInit

from ...wakefields import rebin_tabular_wakefield
from ...z import ImpactZ, ImpactZInput
from ...z.input import Drift, RfcavityStructureWakefield, ToggleSpaceCharge

WAKE = ResistiveWallPseudomode.from_material("copper-slac-pub-10707", radius=0.0045)

HEADER = """
no_digested
beginning[beta_a] = 1
beginning[beta_b] = 1
parameter[geometry] = open
parameter[e_tot] = 100e6
parameter[particle] = electron
"""

# Two pipes with the same wake per meter, a plain drift, and a pipe carrying the
# same wake lumped (scale_with_length = F), so all three share one table.
LATTICE = f"""{HEADER}
p1: pipe, L = 1, sr_wake = {WAKE.to_bmad()}
d1: drift, L = 0.5
p2: p1
p3: pipe, L = 2, sr_wake = {WAKE.to_bmad(scale_with_length=False)}
lat: line = (p1, d1, p2, p3)
use, lat
"""

# The same wake as a z_long table at 0.5 um spacing, per meter and lumped.
ZLONG = TabularWakefield.from_wakefield(WAKE, zmax=2e-4, n=401)
ZLONG_BMAD = ZLONG.to_bmad()
ZLONG_LATTICE = f"""{HEADER}
p1: pipe, L = 1, sr_wake = {ZLONG_BMAD}
d1: drift, L = 0.5
p3: pipe, L = 2, sr_wake = {ZLONG.to_bmad(scale_with_length=False)}
lat: line = (p1, d1, p3)
use, lat
"""


def make_tao(tmp_path: pathlib.Path, contents: str) -> Tao:
    lattice = tmp_path / "lat.bmad"
    lattice.write_text(contents)
    return Tao(lattice_file=str(lattice), noplot=True)


def set_beam(tao: Tao, n_particle: int = 5000, sig_z: float = 10e-6) -> None:
    BeamInit(
        bunch_charge=1e-9,
        n_particle=n_particle,
        a_norm_emit=1e-9,
        b_norm_emit=1e-9,
        sig_z=sig_z,
        sig_pz=1e-6,
    ).set(tao)


@pytest.fixture
def wake_input(tmp_path: pathlib.Path) -> ImpactZInput:
    with make_tao(tmp_path, LATTICE) as tao:
        set_beam(tao)
        # No beam has been tracked, so there are no initial particles.
        return ImpactZInput.from_tao(tao)


def wake_switches(lattice, name: str):
    """Indices of the wake on/off switches and the element they bracket."""
    names = [ele.name for ele in lattice]
    return (
        names.index(f"{name}_WAKE_ON"),
        names.index(name),
        names.index(f"{name}_WAKE_OFF"),
    )


def test_wake_elements_bracket_each_wake(wake_input: ImpactZInput) -> None:
    lattice = wake_input.lattice
    for name in ("P1", "P2", "P3"):
        on, ele, off = wake_switches(lattice, name)
        assert on < ele < off
        assert lattice[on].enable_wakefield > 0
        assert lattice[off].enable_wakefield < 0
        assert not any(
            isinstance(e, RfcavityStructureWakefield) for e in lattice[on + 1 : off]
        )

    assert not any(ele.name.startswith("D1_WAKE") for ele in lattice)


def test_identical_wakes_share_a_table(wake_input: ImpactZInput) -> None:
    switches = [
        ele for ele in wake_input.lattice if isinstance(ele, RfcavityStructureWakefield)
    ]
    assert len(switches) == 6
    assert {ele.file_id for ele in switches} == {500.0}
    assert list(wake_input.file_data) == ["500"]


def test_lumped_wake_is_spread_over_the_element(wake_input: ImpactZInput) -> None:
    scale = {
        ele.name: ele.scale
        for ele in wake_input.lattice
        if isinstance(ele, RfcavityStructureWakefield)
    }
    assert scale["P1_WAKE_ON"] == 1.0
    assert scale["P3_WAKE_ON"] == pytest.approx(0.5)


def test_table_range_comes_from_the_wake_model(wake_input: ImpactZInput) -> None:
    s = wake_input.file_data["500"][:, 0]
    assert s[0] == 0.0
    # Tao reports the mode parameters to about nine significant digits.
    assert s[-1] == pytest.approx(WAKE.default_zmax, rel=1e-6)


@pytest.mark.parametrize("factor", [0.1, 3.0])
def test_wakefield_zmax_only_extends_the_table(
    tmp_path: pathlib.Path, factor: float
) -> None:
    zmax = factor * WAKE.default_zmax
    with make_tao(tmp_path, LATTICE) as tao:
        set_beam(tao)
        I = ImpactZInput.from_tao(tao, wakefield_zmax=zmax)
    s = I.file_data["500"][:, 0]
    assert s[-1] == pytest.approx(max(zmax, WAKE.default_zmax), rel=1e-6)


def test_wakes_keep_the_current_with_space_charge_off(
    wake_input: ImpactZInput,
) -> None:
    assert wake_input.bunch_charge == pytest.approx(1e-9)
    assert isinstance(wake_input.lattice[0], ToggleSpaceCharge)
    assert not wake_input.lattice[0].enable

    wake_input.space_charge_off()
    assert wake_input.bunch_charge == pytest.approx(1e-9)
    assert all(
        not ele.enable
        for ele in wake_input.lattice
        if isinstance(ele, ToggleSpaceCharge)
    )


def test_space_charge_off_without_wakes_zeroes_the_current() -> None:
    input = ImpactZInput(lattice=[Drift(length=1.0)])
    input.bunch_charge = 1e-9
    input.space_charge_off()
    assert input.average_current == 0.0


@pytest.mark.parametrize(
    "sr_wake, match",
    [
        pytest.param(
            ZLONG_BMAD.replace("z_long = {", "z_long = {smoothing_sigma = 1e-6, "),
            "smoothing_sigma",
            id="smoothing",
        ),
        pytest.param(
            ZLONG_BMAD.replace(
                "z_long = {", "z_long = {position_dependence = x_leading, "
            ),
            "position_dependence",
            id="position-dependence",
        ),
        pytest.param(
            ZLONG_BMAD.replace("z_long = {", "z_long = {time_based = T, "),
            "Time-based",
            id="time-based",
        ),
        pytest.param(
            ZLONG_BMAD.replace(
                "z_long = {",
                "longitudinal = {1e15, 26e3, 1.4e5, 0.25, none}, z_long = {",
            ),
            "both",
            id="table-and-modes",
        ),
        pytest.param(
            "{transverse = {1e14, 26e3, 1.4e5, 0.25, none, leading}}",
            "transverse",
            id="transverse",
        ),
    ],
)
def test_unsupported_wakes_raise(
    tmp_path: pathlib.Path, sr_wake: str, match: str
) -> None:
    contents = f"""{HEADER}
p1: pipe, L = 1, sr_wake = {sr_wake}
lat: line = (p1)
use, lat
"""
    with make_tao(tmp_path, contents) as tao:
        set_beam(tao)
        with pytest.raises(NotImplementedError, match=match):
            ImpactZInput.from_tao(tao)


def test_energy_loss_matches_bmad(tmp_path: pathlib.Path) -> None:
    with make_tao(tmp_path, LATTICE) as tao:
        set_beam(tao)
        tao.track_beam()
        P0 = tao.particles("BEGINNING")
        P1 = tao.particles("END")
        I = ImpactZ.from_tao(tao)

    I.run()
    P2 = I.output.particles["final_particles"]
    bmad_loss = P0["mean_energy"] - P1["mean_energy"]
    impact_loss = P0["mean_energy"] - P2["mean_energy"]
    assert impact_loss == pytest.approx(bmad_loss, rel=0.03)


def test_short_table_warns_at_run_time(tmp_path: pathlib.Path) -> None:
    with make_tao(tmp_path, LATTICE) as tao:
        # The bunch is far longer than the natural range of the wake.
        set_beam(tao, n_particle=1000, sig_z=200e-6)
        tao.track_beam()
        I = ImpactZ.from_tao(tao)

    with pytest.warns(UserWarning, match="longer than a wakefield table"):
        I.run()


def test_long_table_does_not_warn(tmp_path: pathlib.Path) -> None:
    with make_tao(tmp_path, LATTICE) as tao:
        set_beam(tao, n_particle=1000)
        tao.track_beam()
        I = ImpactZ.from_tao(tao)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        I.run()
    assert not any("wakefield table" in str(w.message) for w in caught)
    assert np.all(np.isfinite(I.output.particles["final_particles"]["energy"]))


def zlong_input(tmp_path: pathlib.Path, **kwargs) -> ImpactZInput:
    with make_tao(tmp_path, ZLONG_LATTICE) as tao:
        set_beam(tao)
        return ImpactZInput.from_tao(tao, **kwargs)


def wake_tables(input: ImpactZInput) -> dict[str, np.ndarray]:
    """The wake table of each element, by element name."""
    return {
        ele.name.removesuffix("_WAKE_ON"): input.file_data[str(int(ele.file_id))]
        for ele in input.lattice
        if isinstance(ele, RfcavityStructureWakefield) and ele.enable_wakefield > 0
    }


def test_zlong_table_keeps_its_rows(tmp_path: pathlib.Path) -> None:
    input = zlong_input(tmp_path)
    assert list(input.file_data) == ["500"]
    table = input.file_data["500"]
    # Tao reports the table to about nine significant digits.
    np.testing.assert_allclose(table[:, 0], -ZLONG.z_data[::-1], rtol=1e-6, atol=1e-15)
    np.testing.assert_allclose(table[:, 1], ZLONG.W_data[::-1], rtol=1e-6)

    scale = {
        ele.name: ele.scale
        for ele in input.lattice
        if isinstance(ele, RfcavityStructureWakefield)
    }
    assert scale["P1_WAKE_ON"] == 1.0
    assert scale["P3_WAKE_ON"] == pytest.approx(0.5)


def test_zlong_table_is_rebinned(tmp_path: pathlib.Path) -> None:
    table = zlong_input(tmp_path, wakefield_dz=2e-6).file_data["500"]
    expected = rebin_tabular_wakefield(ZLONG, dz=2e-6)
    np.testing.assert_allclose(np.diff(table[:, 0]), 2e-6, rtol=1e-9)
    np.testing.assert_allclose(table[:, 1], expected.W_data[::-1], rtol=1e-6)


def test_wakefield_dz_per_element(tmp_path: pathlib.Path) -> None:
    tables = wake_tables(zlong_input(tmp_path, wakefield_dz={"P1": 2e-6}))
    assert np.diff(tables["P1"][:2, 0]) == pytest.approx(2e-6)
    assert np.diff(tables["P3"][:2, 0]) == pytest.approx(0.5e-6)

    with pytest.raises(ValueError, match="P9"):
        zlong_input(tmp_path, wakefield_dz={"P9": 2e-6})


def test_zlong_energy_loss_matches_bmad(tmp_path: pathlib.Path) -> None:
    with make_tao(tmp_path, ZLONG_LATTICE) as tao:
        set_beam(tao)
        tao.track_beam()
        P0 = tao.particles("BEGINNING")
        P1 = tao.particles("END")
        I = ImpactZ.from_tao(tao)

    I.run()
    P2 = I.output.particles["final_particles"]
    bmad_loss = P0["mean_energy"] - P1["mean_energy"]
    impact_loss = P0["mean_energy"] - P2["mean_energy"]
    assert impact_loss == pytest.approx(bmad_loss, rel=0.03)
