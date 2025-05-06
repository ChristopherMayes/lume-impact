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
from .conftest import z_tests, test_failure_artifacts

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


rotation_comparison_lattices = [
    "drift.bmad",
    "octupole.bmad",
    "quad.bmad",
    "sextupole.bmad",
    # "solenoid.bmad",  # -> TODO some xfails
    "decapole.bmad",
    "lcavity.bmad",
    "lcavity_rf.bmad",
]

comparison_lattices_without_rotation = [
    "dipole.bmad",
    "optics_matching.bmad",
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


def compare_sxy(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    integrator_type: IntegratorType,
    lattice: pathlib.Path,
    tilt: float | None = None,
    x_pitch: float | None = None,
    y_pitch: float | None = None,
    x_offset: float | None = None,
    y_offset: float | None = None,
    ele_to_move: int = 1,
):
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

        for attr, adj in [
            ("tilt", tilt),
            ("x_pitch", x_pitch),
            ("y_pitch", y_pitch),
            ("x_offset", x_offset),
            ("y_offset", y_offset),
        ]:
            if adj is not None:
                cmd = f"set ele {ele_to_move} {attr} = {adj}"
                print("!!!", cmd)
                tao.cmd(cmd, raises=True)

        print("\n".join(tao.cmd(f"show ele {ele_to_move}")))

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
    print(I.input)
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

    atol = 1e-4
    x_pass = np.allclose(x, x_tao_interp, atol=atol)
    y_pass = np.allclose(y, y_tao_interp, atol=atol)
    passed = x_pass and y_pass
    x_pass_fail = "Pass" if x_pass else "FAIL"
    y_pass_fail = "Pass" if y_pass else "FAIL"
    pass_fail = "Pass" if passed else "FAIL"

    fig, (ax0, ax1, ax2) = plt.subplots(3, figsize=(12, 8))
    fig.suptitle(f"{request.node.name}\n{pass_fail}")
    ax0.plot(z, x, color="red")
    ax0.plot(s_tao, x_tao, "--", color="blue")
    ax0.scatter(z, x_tao_interp, marker="o", color="purple")

    if not x_pass:
        ax0_right = ax0.twinx()
        delta = x - x_tao_interp
        # Plot the delta values on the right y-axis
        ax0_right.plot(z, delta, color="gray", alpha=0.7)
        ax0_right.set_ylabel("Delta (IZ - Tao)")
        max_abs_delta = max(abs(delta)) if len(delta) > 0 else 1e-6
        ax0_right.set_ylim(-max_abs_delta * 1.1, max_abs_delta * 1.1)

    ax0.set_ylabel(rf"$x$ (m) {x_pass_fail}")

    ax1.plot(z, y, color="red", label="IMPACT-Z")
    ax1.plot(s_tao, y_tao, "--", color="blue", label="Tao")
    ax1.scatter(z, y_tao_interp, marker="o", color="purple", label="Tao (interpolated)")
    ax1.set_ylabel(rf"$y$ (m) {y_pass_fail}")

    if not y_pass:
        ax1_right = ax1.twinx()
        delta = x - x_tao_interp
        # Plot the delta values on the right y-axis
        ax1_right.plot(z, delta, color="gray", alpha=0.7)
        ax1_right.set_ylabel("Delta (IZ - Tao)")
        max_abs_delta = max(abs(delta)) if len(delta) > 0 else 1e-6
        ax1_right.set_ylim(-max_abs_delta * 1.1, max_abs_delta * 1.1)

    ax1.set_xlabel(r"$s$ (m)")
    ax1.legend()

    I.input.plot(ax=ax2)

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(-0.1, s_tao.max() + 0.1)

    plt.show()

    if not x_pass or not y_pass:
        name = request.node.name.replace("/", "_")
        plt.savefig(test_failure_artifacts / f"{name}.png")

    np.testing.assert_allclose(
        actual=x, desired=x_tao_interp, atol=atol, err_msg="X differs"
    )
    np.testing.assert_allclose(
        actual=y, desired=y_tao_interp, atol=atol, err_msg="Y differs"
    )


@pytest.mark.parametrize(
    "lattice",
    [
        pytest.param(lattice_root / fn, id=fn, marks=comparison_markers.get(fn, []))
        for fn in comparison_lattices_without_rotation
    ],
)
def test_compare_sxy(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    integrator_type: IntegratorType,
    lattice: pathlib.Path,
) -> None:
    compare_sxy(
        request=request,
        tmp_path=tmp_path,
        integrator_type=integrator_type,
        lattice=lattice,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "lattice",
    [
        pytest.param(lattice_root / fn, id=fn, marks=comparison_markers.get(fn, []))
        for fn in rotation_comparison_lattices
    ],
)
@pytest.mark.parametrize(
    ("tilt", "x_pitch", "x_offset", "y_pitch", "y_offset"),
    [
        # tilt test cases (others zero)
        pytest.param(np.pi / 4, 0.0, 0.0, 0.0, 0.0, id="tilt=pi/4"),
        pytest.param(-np.pi / 4, 0.0, 0.0, 0.0, 0.0, id="tilt=-pi/4"),
        pytest.param(np.pi / 2, 0.0, 0.0, 0.0, 0.0, id="tilt=pi/2"),
        pytest.param(-np.pi / 2, 0.0, 0.0, 0.0, 0.0, id="tilt=-pi/2"),
        # x_pitch test cases (others zero)
        pytest.param(0.0, 1.0, 0.0, 0.0, 0.0, id="x_pitch=positive"),
        pytest.param(0.0, -1.0, 0.0, 0.0, 0.0, id="x_pitch=negative"),
        # y_pitch test cases (others zero)
        pytest.param(0.0, 0.0, 0.0, 1.0, 0.0, id="y_pitch=positive"),
        pytest.param(0.0, 0.0, 0.0, -1.0, 0.0, id="y_pitch=negative"),
        # x_offset test cases (others zero)
        pytest.param(0.0, 0.0, 0.0001, 0.0, 0.0, id="x_offset=0.0001"),
        pytest.param(0.0, 0.0, -0.0001, 0.0, 0.0, id="x_offset=-0.0001"),
        # y_offset test cases (others zero)
        pytest.param(0.0, 0.0, 0.0, 0.0, 0.0001, id="y_offset=0.0001"),
        pytest.param(0.0, 0.0, 0.0, 0.0, -0.0001, id="y_offset=-0.0001"),
    ],
)
def test_compare_sxy_rotated(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    integrator_type: IntegratorType,
    lattice: pathlib.Path,
    tilt: float,
    x_pitch: float,
    y_pitch: float,
    x_offset: float,
    y_offset: float,
) -> None:
    is_lcavity = lattice.name == "lcavity.bmad"
    pitch_magnitude = 0.000_01 if is_lcavity else 0.001

    # Use sign from value but magnitude based on lattice type
    if x_pitch != 0.0:
        x_pitch = pitch_magnitude if x_pitch > 0 else -pitch_magnitude
    if y_pitch != 0.0:
        y_pitch = pitch_magnitude if y_pitch > 0 else -pitch_magnitude

    compare_sxy(
        request=request,
        tmp_path=tmp_path,
        integrator_type=integrator_type,
        lattice=lattice,
        tilt=tilt,
        x_pitch=x_pitch,
        y_pitch=y_pitch,
        x_offset=x_offset,
        y_offset=y_offset,
        ele_to_move=1,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("tilt", "x_pitch", "x_offset", "y_pitch", "y_offset"),
    [
        # tilt test cases (others zero)
        pytest.param(np.pi / 4, 0.0, 0.0, 0.0, 0.0, id="tilt=pi/4"),
        pytest.param(-np.pi / 4, 0.0, 0.0, 0.0, 0.0, id="tilt=-pi/4"),
        pytest.param(np.pi / 2, 0.0, 0.0, 0.0, 0.0, id="tilt=pi/2"),
        pytest.param(-np.pi / 2, 0.0, 0.0, 0.0, 0.0, id="tilt=-pi/2"),
        # x_pitch test cases (others zero)
        pytest.param(
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            id="x_pitch=positive",
            marks=pytest.mark.xfail(reason="TODO bmad discrepancy", strict=True),
        ),
        pytest.param(
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            id="x_pitch=negative",
            marks=pytest.mark.xfail(reason="TODO bmad discrepancy", strict=True),
        ),
        # y_pitch test cases (others zero)
        pytest.param(
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            id="y_pitch=positive",
            marks=pytest.mark.xfail(reason="TODO bmad discrepancy", strict=True),
        ),
        pytest.param(
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            id="y_pitch=negative",
            marks=pytest.mark.xfail(reason="TODO bmad discrepancy", strict=True),
        ),
        # x_offset test cases (others zero)
        pytest.param(0.0, 0.0, 0.0001, 0.0, 0.0, id="x_offset=0.0001"),
        pytest.param(
            0.0,
            0.0,
            -0.0001,
            0.0,
            0.0,
            id="x_offset=-0.0001",
            marks=pytest.mark.xfail(reason="TODO bmad discrepancy", strict=True),
        ),
        # y_offset test cases (others zero)
        pytest.param(0.0, 0.0, 0.0, 0.0, 0.0001, id="y_offset=0.0001"),
        pytest.param(0.0, 0.0, 0.0, 0.0, -0.0001, id="y_offset=-0.0001"),
    ],
)
def test_compare_sxy_rotated_solenoid(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    integrator_type: IntegratorType,
    tilt: float,
    x_pitch: float,
    y_pitch: float,
    x_offset: float,
    y_offset: float,
) -> None:
    pitch_magnitude = 0.001

    # Use sign from value but magnitude based on lattice type
    if x_pitch != 0.0:
        x_pitch = pitch_magnitude if x_pitch > 0 else -pitch_magnitude
    if y_pitch != 0.0:
        y_pitch = pitch_magnitude if y_pitch > 0 else -pitch_magnitude
    compare_sxy(
        request=request,
        tmp_path=tmp_path,
        integrator_type=integrator_type,
        lattice=lattice_root / "solenoid.bmad",
        tilt=tilt,
        x_pitch=x_pitch,
        y_pitch=y_pitch,
        x_offset=x_offset,
        y_offset=y_offset,
        ele_to_move=1,
    )


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
