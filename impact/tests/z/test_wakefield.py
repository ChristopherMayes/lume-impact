from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
from beamphysics.wakefields import ResistiveWallPseudomode, TabularWakefield
from pytao import SubprocessTao as Tao
from pytao.model import BeamInit

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


def make_tao(tmp_path: pathlib.Path, contents: str) -> Tao:
    lattice = tmp_path / "lat.bmad"
    lattice.write_text(contents)
    return Tao(lattice_file=str(lattice), noplot=True)


def set_beam(tao: Tao, n_particle: int = 5000) -> None:
    BeamInit(
        bunch_charge=1e-9,
        n_particle=n_particle,
        a_norm_emit=1e-9,
        b_norm_emit=1e-9,
        sig_z=10e-6,
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
            TabularWakefield.from_wakefield(WAKE, zmax=2e-4, n=401).to_bmad(),
            "z_long",
            id="tabulated",
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
        set_beam(tao, n_particle=1000)
        tao.track_beam()
        I = ImpactZ.from_tao(tao, wakefield_zmax=5e-6)

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
