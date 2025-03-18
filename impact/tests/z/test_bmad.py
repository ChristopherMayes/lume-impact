from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Generator
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pmd_beamphysics import ParticleGroup, single_particle
from pmd_beamphysics.units import mec2
from pytao import SubprocessTao as Tao

import impact.z as IZ

from ...z import ImpactZ, ImpactZInput, ImpactZParticles
from ...z.constants import IntegratorType
from .conftest import z_tests

lattice_root = z_tests / "bmad"

lattice_markers = {
    "elements.bmad": pytest.mark.xfail(reason="Unsupported elements"),
    # "csr_bench.bmad": pytest.mark.xfail(reason="Additional setup required"),
}
comparison_markers = {}
lattices = pytest.mark.parametrize(
    "lattice",
    [
        pytest.param(fn, id=fn.name, marks=lattice_markers.get(fn.name, []))
        for fn in lattice_root.glob("*.bmad")
    ],
)


@contextlib.contextmanager
def tao_with_lattice(
    tmp_path: pathlib.Path, contents: str, name: str = "lattice.lat"
) -> Generator[Tao]:
    lattice_path = tmp_path / name
    with open(lattice_path, "w") as fp:
        print(dedent(contents.rstrip()), file=fp)

    with Tao(lattice_file=lattice_path, noplot=True) as tao:
        yield tao


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


comparison_lattices = [
    "dipole.bmad",
    "drift.bmad",
    "octupole.bmad",
    "quad.bmad",
    "sextupole.bmad",
    "solenoid.bmad",
    "decapole.bmad",
    "lcavity.bmad",
    "optics_matching.bmad",
    "lcavity_rf.bmad",
]


positron_lattices = [
    "optics_matching.bmad",
]


@pytest.fixture(
    params=[IntegratorType.linear_map, IntegratorType.runge_kutta],
    ids=["linear_map", "runge_kutta"],
)
def integrator_type(request: pytest.FixtureRequest) -> IntegratorType:
    return request.param


def check_weighted_initial_particles(
    expected: ParticleGroup, actual: ParticleGroup
) -> None:
    if len(expected) != 1:
        assert expected == actual
        return

    # TODO/NOTE: zeroed weight for np=1
    weighted_actual = actual.copy()
    weighted_actual.weight = expected.weight

    assert weighted_actual == expected


@pytest.mark.parametrize(
    "lattice",
    [
        pytest.param(lattice_root / fn, id=fn, marks=comparison_markers.get(fn, []))
        for fn in comparison_lattices
    ],
)
def test_compare_sxy(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    integrator_type: IntegratorType,
    lattice: pathlib.Path,
) -> None:
    if (
        lattice.name == "solenoid.bmad"
        and integrator_type == IntegratorType.runge_kutta
    ):
        pytest.skip("Not yet working?")
    energy = 10e6
    pz = np.sqrt(energy**2 - mec2**2)

    species = "positron" if lattice.name in positron_lattices else "electron"
    P0 = single_particle(x=1e-3, pz=pz, species=species)

    with Tao(lattice_file=lattice, noplot=True) as tao:
        set_initial_particles(tao, P0, path=tmp_path)
        input = ImpactZInput.from_tao(tao, integrator_type=integrator_type)

        if input.integrator_type == IntegratorType.runge_kutta:
            if integrator_type == IntegratorType.linear_map:
                pytest.skip("Runge-kutta required")

        input.integrator_type = integrator_type

        x_tao = np.array(tao.bunch_comb("x"))
        y_tao = np.array(tao.bunch_comb("y"))
        s_tao = np.array(tao.bunch_comb("s"))

    input.space_charge_off()

    I = ImpactZ(input)
    output = I.run()

    zP0 = output.particles["initial_particles"]

    # Check that Impact-Z wrote the same particles that we are using
    check_weighted_initial_particles(expected=P0, actual=zP0)

    # P1 = output.particles["final_particles"]

    z = output.stats.z
    x = output.stats.mean_x
    y = output.stats.mean_y

    x_tao_interp = np.interp(z, s_tao, x_tao)
    y_tao_interp = np.interp(z, s_tao, y_tao)

    fig, (ax0, ax1) = plt.subplots(2, figsize=(12, 8))

    fig.suptitle(request.node.name)
    ax0.plot(z, x, label="Impact-Z")
    ax0.plot(s_tao, x_tao, "--", label="Tao")
    ax0.set_ylabel(r"$x$ (m)")

    ax1.plot(z, y, label="Impact-Z")
    ax1.plot(s_tao, y_tao, "--", label="Tao")
    ax1.set_ylabel(r"$y$ (m)")
    ax1.set_xlabel(r"$s$ (m)")

    plt.legend()
    plt.show()

    fig.suptitle(f"{request.node.name} (interp)")
    ax0.plot(z, x, label="Impact-Z")
    ax0.plot(z, x_tao_interp, "--", label="Tao (interpolated)")
    ax0.set_ylabel(r"$x$ (m)")

    ax1.plot(z, y, label="Impact-Z")
    ax1.plot(z, y_tao_interp, "--", label="Tao (interpolated)")
    ax1.set_ylabel(r"$y$ (m)")
    ax1.set_xlabel(r"$s$ (m)")

    plt.legend()
    plt.show()
    np.testing.assert_allclose(actual=x, desired=x_tao_interp, atol=1e-4)
    np.testing.assert_allclose(actual=y, desired=y_tao_interp, atol=1e-4)


def test_check_initial_particles(tmp_path: pathlib.Path) -> None:
    x0 = 0.001
    y0 = 0.002
    z0 = 0
    t0 = 0.003
    px0 = 1e6
    py0 = 2e6
    energy0 = 10e6
    pz0 = np.sqrt(energy0**2 - px0**2 - py0**2 - mec2**2)

    P0 = single_particle(px=px0, py=py0, pz=pz0, x=x0, y=y0, z=z0, t=t0)

    tao = Tao(lattice_file=lattice_root / "drift.bmad", plot="mpl")

    P0.write(tmp_path / "p0.h5")
    tao.cmds(
        [
            f"set beam_init position_file = {tmp_path}/p0.h5",
            f"set beam_init n_particle = {len(P0)}",
            f"set beam_init bunch_charge = {P0.charge}",
            "set beam_init saved_at = beginning d",
            "set global track_type = single",
            "set global track_type = beam",
        ]
    )

    tao.plot("beta", include_layout=False)
    plt.show()

    input = IZ.ImpactZInput.from_tao(tao)

    assert input.initial_particles == P0

    input.space_charge_off()

    I = IZ.ImpactZ(input, use_temp_dir=False, workdir=tmp_path, initial_particles=P0)

    output = I.run(verbose=True)

    assert output is not None
    assert I.output is output

    Pin = output.particles["initial_particles"]

    P0_z_written = ImpactZParticles.from_file(tmp_path / "particle.in")
    P0_written = P0_z_written.to_particle_group(
        reference_frequency=I.input.reference_frequency,
        reference_kinetic_energy=I.input.reference_kinetic_energy,
        phase_reference=I.input.initial_phase_ref,
    )
    check_weighted_initial_particles(expected=Pin, actual=P0)
    assert P0_written == Pin


@pytest.mark.parametrize(
    "kicker",
    [
        "kick: hkicker, l = 0.6, bl_kick=1e-3",
        "kick: vkicker, l = 0.6, bl_kick=1e-3",
        "kick: kicker, l = 0.6, bl_hkick=1e-3",
        "kick: kicker, l = 0.6, bl_vkick=1e-3",
    ],
)
def test_kicker_with_nonzero_field_kick(tmp_path: pathlib.Path, kicker: str) -> None:
    with pytest.raises(NotImplementedError):
        with tao_with_lattice(
            tmp_path=tmp_path,
            contents=f"""\
                no_digested
                beginning[beta_a] = 10.   ! m  a-mode beta function
                beginning[beta_b] = 10.   ! m  b-mode beta function
                beginning[e_tot] = 10e6   ! eV

                parameter[geometry] = open
                parameter[particle] = electron

                {kicker}

                lat: line = (kick)
                use, lat
            """,
        ) as tao:
            ImpactZInput.from_tao(tao)
