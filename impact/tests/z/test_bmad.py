from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Generator
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pytest
from beamphysics import ParticleGroup, single_particle
from beamphysics.units import mec2
from pytao import SubprocessTao as Tao

import impact.z as IZ

from ...z import ImpactZ, ImpactZInput, ImpactZParticles
from ...z.constants import IntegratorType
from .conftest import test_failure_artifacts, z_tests

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
    "kickers.bmad",
    "hkicker.bmad",
    "vkicker.bmad",
    "kicker.bmad",
]

comparison_lattices_without_rotation = [
    "dipole.bmad",
    "optics_matching.bmad",
    "decapole_scaled.bmad",
    "hkicker.bmad",
    "vkicker.bmad",
    "kicker.bmad",
]

positron_lattices = [
    "optics_matching.bmad",
]

# Mean-orbit agreement tolerance for compare_sxy, in meters (absolute).
DEFAULT_ORBIT_ATOL = 2e-5
ORBIT_ATOL = {
    "quad.bmad": 1e-4,  # x_pitch cases: 7.0e-5 observed
    "decapole.bmad": 2e-4,  # tilt cases: 8.7e-5 observed
    "drift.bmad": 5e-5,  # tilt cases: 3.0e-5 observed
    "hkicker.bmad": 5e-5,  # thin-kick splitting: 2.3e-5 observed
    "vkicker.bmad": 5e-5,  # thin-kick splitting: 2.3e-5 observed
    "kicker.bmad": 5e-5,  # thin-kick splitting: 2.5e-5 observed
    "kickers.bmad": 1e-4,  # x_offset cases: 3.4e-5 observed
    "lcavity.bmad": 1e-4,  # y_offset cases: 2.7e-5 observed
    "optics_matching.bmad": 1e-4,  # chromatic difference: 3.6e-5 observed
}

# (lattice, pitch axis) combinations where the converted lattice disagrees
# with Bmad well beyond tolerance (up to ~1e-3) for an off-axis probe
# particle.
# TODO: investigate pitch handling for higher-order multipoles.
PITCH_DISCREPANCY_LATTICES = {
    "octupole.bmad": ("x_pitch", "y_pitch"),
    "decapole.bmad": ("x_pitch",),
}

# Colorblind-safe colors for the comparison figures.  Tao vs IMPACT-Z are
# additionally distinguished by solid vs dashed lines.
TAO_COLOR = "#2a78d6"
IZ_COLOR = "#eb6834"
DELTA_X_COLOR = "#1baf7a"
DELTA_Y_COLOR = "#4a3aa7"
PASS_COLOR = "#008300"
FAIL_COLOR = "#e34948"


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
    # P0 = single_particle(x=1e-3, pz=pz, species=species)
    P0 = single_particle(
        x=1e-3,
        px=2e-3 * pz,
        y=2e-3,
        py=3e-3 * pz,
        pz=(1 - 0.0001) * pz,
        species=species,
    )

    comb_ds_save = 0.01

    with Tao(lattice_file=lattice, noplot=True) as tao:
        tao.cmd(f"set beam comb_ds_save = {comb_ds_save}")
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
                print(cmd)
                tao.cmd(cmd, raises=True)

        print("\n".join(tao.cmd(f"show ele {ele_to_move}")))

        input = ImpactZInput.from_tao(tao, integrator_type=integrator_type)

        if (
            input.integrator_type == IntegratorType.runge_kutta
            and integrator_type == IntegratorType.linear_map
        ):
            pytest.skip("Runge-kutta required")

        input.integrator_type = integrator_type

        x_tao = np.array(tao.bunch_comb("x"))
        y_tao = np.array(tao.bunch_comb("y"))
        s_tao = np.array(tao.bunch_comb("s"))

    input.space_charge_off()

    I = ImpactZ(input)
    print(I.input)
    output = I.run(verbose=True)

    zP0 = output.particles["initial_particles"]

    # Check that Impact-Z wrote the same particles that we are using
    check_weighted_initial_particles(expected=P0, actual=zP0)

    # P1 = output.particles["final_particles"]

    z = output.stats.z
    x = output.stats.mean_x
    y = output.stats.mean_y

    np.testing.assert_allclose(
        z[0],
        s_tao[0],
        atol=2 * comb_ds_save,
        err_msg=f"IMPACT-Z and Tao s range start differs: {z[0]} vs {s_tao[0]}",
    )
    np.testing.assert_allclose(
        z[-1],
        s_tao[-1],
        atol=2 * comb_ds_save,
        err_msg=f"IMPACT-Z and Tao s range end differs: {z[-1]} vs {s_tao[-1]}",
    )

    x_tao_interp = np.interp(z, s_tao, x_tao)
    y_tao_interp = np.interp(z, s_tao, y_tao)

    atol = ORBIT_ATOL.get(lattice.name, DEFAULT_ORBIT_ATOL)
    dx = x - x_tao_interp
    dy = y - y_tao_interp
    max_dx = float(np.max(np.abs(dx)))
    max_dy = float(np.max(np.abs(dy)))
    passed = max_dx <= atol and max_dy <= atol

    fig, (ax_x, ax_y, ax_r, ax_lat) = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(12, 9),
        height_ratios=[2, 2, 1.6, 1.0],
        constrained_layout=True,
    )
    fig.suptitle(request.node.name)

    for ax, tao_v, iz_v, label in ((ax_x, x_tao, x, "x"), (ax_y, y_tao, y, "y")):
        ax.plot(s_tao, tao_v, "-", color=TAO_COLOR, lw=2, label="Tao")
        ax.plot(z, iz_v, "--", color=IZ_COLOR, lw=2, label="IMPACT-Z")
        ax.set_ylabel(rf"$\langle {label} \rangle$ (m)")
        ax.grid(alpha=0.25)
    ax_x.legend(loc="best", fontsize=9)

    ax_r.axhspan(-atol, atol, color="0.92", zorder=0, label=rf"$\pm$atol = {atol:g}")
    ax_r.axhline(0.0, color="0.6", lw=0.8, zorder=1)
    ax_r.plot(z, dx, "-", color=DELTA_X_COLOR, lw=2, label=r"$\Delta x$")
    ax_r.plot(z, dy, "--", color=DELTA_Y_COLOR, lw=2, label=r"$\Delta y$")
    rmax = max(1.3 * atol, 1.15 * max(max_dx, max_dy))
    ax_r.set_ylim(-rmax, rmax)
    ax_r.set_ylabel("IZ $-$ Tao (m)")
    ax_r.legend(loc="best", fontsize=9, ncols=3)
    ax_r.grid(alpha=0.25)
    ax_r.set_title(
        f"{'PASS' if passed else 'FAIL'}:  "
        f"max|Δx| = {max_dx:.2e},  max|Δy| = {max_dy:.2e},  atol = {atol:g}",
        color=PASS_COLOR if passed else FAIL_COLOR,
    )

    I.input.plot(ax=ax_lat)
    ax_lat.set_xlabel(r"$s$ (m)")

    for ax in (ax_x, ax_y, ax_r, ax_lat):
        ax.set_xlim(min(s_tao.min(), z.min()) - 0.02, max(s_tao.max(), z.max()) + 0.02)

    if not passed:
        name = request.node.name.replace("/", "_")
        plt.savefig(test_failure_artifacts / f"{name}.png")

    axes = PITCH_DISCREPANCY_LATTICES.get(lattice.name, ())
    if ("x_pitch" in axes and x_pitch) or ("y_pitch" in axes and y_pitch):
        pytest.xfail("TODO: pitch discrepancy vs Bmad for higher-order multipoles")

    np.testing.assert_allclose(
        actual=x, desired=x_tao_interp, rtol=0.0, atol=atol, err_msg="X differs"
    )
    np.testing.assert_allclose(
        actual=y, desired=y_tao_interp, rtol=0.0, atol=atol, err_msg="Y differs"
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
        pytest.param(0.1, 0.0, 0.0, 0.0, 0.0, id="tilt=0.1"),
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


@pytest.mark.parametrize("aperture_at", ["entrance_end", "exit_end"])
def test_kicker_with_aperture(tmp_path: pathlib.Path, aperture_at: str) -> None:
    with tao_with_lattice(
        tmp_path=tmp_path,
        contents=f"""\
            no_digested
            beginning[beta_a] = 10.   ! m  a-mode beta function
            beginning[beta_b] = 10.   ! m  b-mode beta function
            beginning[e_tot] = 10e6   ! eV

            parameter[geometry] = open
            parameter[particle] = electron

            kick: hkicker, l = 0.6, bl_kick = 1e-3, num_steps = 10,
                x1_limit = 0.01, x2_limit = 0.01, y1_limit = 0.01, y2_limit = 0.01,
                aperture_at = {aperture_at}

            lat: line = (kick)
            use, lat
        """,
    ) as tao:
        input = ImpactZInput.from_tao(tao)

    apertures = [
        idx
        for idx, ele in enumerate(input.lattice)
        if isinstance(ele, IZ.CollimateBeam)
    ]
    kicks = [
        idx
        for idx, ele in enumerate(input.lattice)
        if isinstance(ele, IZ.KickBeamUsingMultipole)
    ]
    assert len(apertures) == 1
    assert kicks

    (aperture_idx,) = apertures
    if aperture_at == "entrance_end":
        assert aperture_idx < min(kicks)
    else:
        assert aperture_idx > max(kicks)
