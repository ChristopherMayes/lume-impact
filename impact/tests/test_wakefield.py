import os

import numpy as np
import pytest
from beamphysics.interfaces.impact import create_impact_t_wakefield_data
from beamphysics.wakefields import ResistiveWallPseudomode
from pytao import SubprocessTao as Tao
from pytao.model import BeamInit

from impact import Impact
from impact.lattice import ele_line
from impact.parsers import parse_ele

WAKE = ResistiveWallPseudomode.from_material("copper-slac-pub-10707", radius=0.0045)


def reparse(ele, keys):
    """Write an element line and parse it back, keeping only keys."""
    ele = {**ele, "name": "w"}
    new = parse_ele(ele_line(ele))
    return {k: new[k] for k in keys}, {k: ele[k] for k in keys}


def table_impact(zmax=1e-3, n=1000):
    """A 1 m drift with a read-in resistive wall wake and space charge off."""
    data, length = create_impact_t_wakefield_data(WAKE, zmax=zmax, n=n)
    I = Impact()
    I.header["Np"] = 2000
    I.total_charge = 1e-9
    I.stop = 1.0
    I.input["fieldmaps"]["fort.300"] = {"info": {"format": "rfdata"}, "data": data}
    # Impact-T ignores elements after the stop.
    *eles, stop = I.lattice
    I.input["lattice"] = [
        {"type": "spacecharge", "name": "spacecharge_off", "s": 0, "is_on": False},
        *eles,
        {
            "type": "wakefield",
            "name": "pipe_wakefield",
            "s_begin": 0.0,
            "s": 1.0,
            "method": "from_file",
            "filename": "fort.300",
            "table_length": length,
        },
        stop,
    ]
    I.ele_bookkeeper()
    return I


def test_wakefield_from_file_line_round_trip():
    line = "0 1 300 -6 1 1 0.5 1.5 0 -1 0.0002 /"
    ele = parse_ele(line)
    assert ele["method"] == "from_file"
    assert ele["filename"] == "fort.300"
    assert ele["s_begin"] == 0.5
    assert ele["s"] == 1.5
    assert ele["table_length"] == 0.0002

    new, old = reparse(ele, ("s_begin", "s", "method", "filename", "table_length"))
    assert new == old


def test_wakefield_analytical_line_round_trip():
    ele = parse_ele("0 -1 0 -6 1 1 0.0 1.0 0.0116 0.0292 0.035 /")
    assert ele["method"] == "analytical"
    new, old = reparse(ele, ("s_begin", "s", "method", "iris_radius", "gap", "period"))
    assert new == old


def test_wakefield_table_input_and_archive(tmp_path):
    I = table_impact()
    I.write_input(path=str(tmp_path))

    J = Impact(input_file=str(tmp_path / "ImpactT.in"))
    assert J.ele["pipe_wakefield"]["filename"] == "fort.300"
    np.testing.assert_allclose(
        J.input["fieldmaps"]["fort.300"]["data"],
        I.input["fieldmaps"]["fort.300"]["data"],
    )

    afile = tmp_path / "archive.h5"
    I.archive(str(afile))
    K = Impact.from_archive(str(afile))
    np.testing.assert_array_equal(
        K.input["fieldmaps"]["fort.300"]["data"],
        I.input["fieldmaps"]["fort.300"]["data"],
    )
    assert (
        K.ele["pipe_wakefield"]["table_length"]
        == I.ele["pipe_wakefield"]["table_length"]
    )


def test_run_keeps_wakefield_table():
    I = table_impact()
    # The mesh cell must be well below the 44 um period of this wake.
    I.header["Nz"] = 256
    I.run()
    assert os.path.getsize(os.path.join(I.path, "fort.300")) > 0

    P = I.particles["initial_particles"]
    z = P.z - P["mean_z"]
    edges = np.linspace(z.min(), z.max(), 801)
    dz = edges[1] - edges[0]
    density, _ = np.histogram(z, edges)
    density = density / (density.sum() * dz)
    expected = np.sum(WAKE.convolve_density(density * P.charge, dz) * density) * dz

    energy = I.output["stats"]["mean_kinetic_energy"]
    assert energy[0] - energy[-1] == pytest.approx(expected, rel=0.05)


LATTICE = f"""
no_digested
beginning[beta_a] = 1
beginning[beta_b] = 1
parameter[geometry] = open
parameter[e_tot] = 100e6
parameter[particle] = electron
p1: pipe, L = 1, sr_wake = {WAKE.to_bmad()}
p2: pipe, L = 0.5
p3: p1
lat: line = (p1, p2, p3)
use, lat
"""


@pytest.fixture
def wake_tao(tmp_path):
    lattice = tmp_path / "wake.bmad"
    lattice.write_text(LATTICE)
    with Tao(lattice_file=str(lattice), noplot=True) as tao:
        beam_init = BeamInit(
            bunch_charge=1e-9,
            n_particle=5000,
            a_norm_emit=1e-9,
            b_norm_emit=1e-9,
            sig_z=10e-6,
            sig_pz=1e-6,
        )
        beam_init.set(tao)
        yield tao


def test_from_tao_wakefield_elements(wake_tao):
    I = Impact.from_tao(wake_tao, wakefield_zmax=2e-4)

    wakes = [ele for ele in I.lattice if ele["type"] == "wakefield"]
    assert [(ele["s_begin"], ele["s"]) for ele in wakes] == [(0.0, 1.0), (1.5, 2.5)]
    # Identical wakes share a table.
    assert {ele["filename"] for ele in wakes} == {"fort.300"}
    assert I.input["fieldmaps"]["fort.300"]["data"].shape[1] == 3

    assert I.total_charge == pytest.approx(1e-9)
    assert I.ele["spacecharge_off"]["is_on"] is False


def test_from_tao_wakefield_matches_bmad(wake_tao):
    wake_tao.track_beam()
    P0 = wake_tao.particles("BEGINNING")
    P1 = wake_tao.particles("END")

    I = Impact.from_tao(wake_tao, wakefield_zmax=2e-4)
    P = P0.copy()
    P.drift_to_t()
    I.initial_particles = P
    I.run()
    P2 = I.particles["final_particles"]

    bmad_loss = P0["mean_energy"] - P1["mean_energy"]
    impact_loss = P0["mean_energy"] - P2["mean_energy"]
    assert impact_loss == pytest.approx(bmad_loss, rel=0.03)
