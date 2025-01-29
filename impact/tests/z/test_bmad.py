from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pmd_beamphysics import ParticleGroup, single_particle
from pmd_beamphysics.units import mec2
from pytao import SubprocessTao as Tao

from ...z import ImpactZ, ImpactZInput
from .conftest import z_tests

lattice_root = z_tests / "bmad"

lattices = pytest.mark.parametrize(
    "lattice", [pytest.param(fn, id=fn.name) for fn in lattice_root.glob("*.bmad")]
)


@lattices
def test_from_tao(lattice: pathlib.Path) -> None:
    with Tao(lattice_file=lattice, noplot=True) as tao:
        print(ImpactZInput.from_tao(tao))


def set_initial_particles(
    tao: Tao, P0: ParticleGroup, path: pathlib.Path | None = None
) -> None:
    path = path or pathlib.Path(".")

    fn = path / "initial_particles.h5"
    P0.write(str(fn))
    tao.cmds(
        [
            f"set beam_init position_file = {fn}",
            f"set beam_init n_particle = {len(P0)}",
            f"set beam_init bunch_charge = {P0.charge}",
            "set beam_init saved_at = *",
            "set global track_type = single",
            "set global track_type = beam",
        ]
    )


@lattices
def test_compare_sxy(tmp_path: pathlib.Path, lattice: pathlib.Path) -> None:
    energy = 10e6

    pz = np.sqrt(energy**2 - mec2**2)

    # Should give beta_x = 19569511.835591838 m
    P0 = single_particle(x=1e-3, pz=pz)

    with Tao(lattice_file=lattice, noplot=True) as tao:
        set_initial_particles(tao, P0, path=tmp_path)
        input = ImpactZInput.from_tao(tao)
        I = ImpactZ(input)
        output = I.run()

        zP0 = output.particles["initial_particles"]

        # Check that Impact-Z wrote the same particles that we are using
        assert zP0 == P0

        # P1 = output.particles["final_particles"]

        z = output.stats["z"]
        x = output.stats["mean_x"]
        y = output.stats["mean_y"]

        x_tao = np.asarray(tao.bunch_comb("x"))
        y_tao = np.asarray(tao.bunch_comb("y"))
        s_tao = np.asarray(tao.bunch_comb("s"))

        x_tao_interp = np.interp(z, s_tao, x_tao)
        y_tao_interp = np.interp(z, s_tao, y_tao)

        _fig, (ax0, ax1) = plt.subplots(2, figsize=(12, 8))
        ax0.plot(z, x, label="Impact-Z")
        ax0.plot(s_tao, x_tao, "--", label="Tao")
        ax0.set_ylabel(r"$x$ (m)")

        ax1.plot(z, y, label="Impact-Z")
        ax1.plot(s_tao, y_tao, "--", label="Tao")
        ax1.set_ylabel(r"$y$ (m)")
        ax1.set_xlabel(r"$s$ (m)")

        plt.legend()
        plt.show()

        np.testing.assert_allclose(actual=x, desired=x_tao_interp)
        np.testing.assert_allclose(actual=y, desired=y_tao_interp)
