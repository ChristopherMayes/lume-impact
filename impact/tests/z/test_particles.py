from __future__ import annotations
import pathlib
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import c_light

from ...z.particles import ImpactZParticles


def gaussian_data(
    n_particle: int = 100,
    charge: float = 1e-9,
    p0: float = 1e9,
    mean: np.ndarray | None = None,
    sigma_mat: np.ndarray | None = None,
    t_ref: float = 0.0,
):
    """
    Makes Gaussian particle data from a Bmad-style sigma matrix.

    Parameters
    ----------
    n_particle: int, default=100
        Number of particles.
    charge : float, default=1e-9
        Charge in C.
    p0 : float
        Reference momentum in eV/c
    mean : np.ndarray of shape (6,), optional
        Mean positions. Default = None gives zeros
    sigma_mat : np.ndarray of shape (6,6), optional
        Sigma matrix in Bmad units. If default, this is the identity * 1e-3

    Returns
    -------
    dict
        ParticleGroup-compatible data dictionary:
        >>> ParticleGroup(data=gaussian_data())
    """
    if mean is None:
        mean = np.zeros(6)

    if sigma_mat is None:
        cov = np.eye(6) * 1e-3
    else:
        cov = sigma_mat

    dat = np.random.multivariate_normal(mean, cov, size=n_particle)
    x = dat[:, 0]
    px = dat[:, 1]
    y = dat[:, 2]
    py = dat[:, 3]

    # z = -beta * c * (t - t_ref)
    # avoid calculating beta => lazy mode -- high energy, z = -c * (t - t_ref)
    # users will need to call drift_to_z() if they have an alternative representation
    z = dat[:, 4]
    t = -z / c_light + t_ref
    pz = dat[:, 5]

    data = {
        "x": x,
        "px": px * p0,
        "y": y,
        "py": py * p0,
        "z": np.zeros(n_particle),
        "pz": (1 + pz) * p0,
        "t": t,
        "weight": charge / n_particle,
        "status": np.ones(n_particle),
        "species": "electron",
    }

    return data


def test_round_trip():
    P1 = ParticleGroup(data=gaussian_data())

    reference_freq = 1300000000.0
    ref_kinetic = 1.0

    izp = ImpactZParticles.from_particle_group(
        P1,
        reference_frequency=reference_freq,
        reference_kinetic_energy=ref_kinetic,
    )
    P2 = izp.to_particle_group(
        reference_frequency=reference_freq,
        reference_kinetic_energy=ref_kinetic,
        phase_reference=0.0,
    )

    for key in [
        "x",
        "px",
        "y",
        "py",
        "pz",
        "z",
        "t",
        "status",
        "weight",
        "id",
    ]:
        assert_allclose(
            np.asarray(P1[key]),
            np.asarray(P2[key]),
            err_msg=f"Round-tripped key {key!r} not allclose",
        )

    assert P1 == P2


def test_round_trip_to_file(tmp_path: pathlib.Path):
    P1 = ParticleGroup(data=gaussian_data())

    reference_freq = 1300000000.0
    ref_kinetic = 1.0

    izp = ImpactZParticles.from_particle_group(
        P1,
        reference_frequency=reference_freq,
        reference_kinetic_energy=ref_kinetic,
    )

    fn = tmp_path / "part.in"
    izp.write_impact(fn)

    izp1 = ImpactZParticles.from_file(fn)
    assert izp == izp1

    P2 = izp1.to_particle_group(
        reference_frequency=reference_freq,
        reference_kinetic_energy=ref_kinetic,
        phase_reference=0.0,
    )
    assert P1 == P2


def test_read_empty_file() -> None:
    P = ImpactZParticles.from_contents("")
    assert len(P.impactz_x) == 0


@pytest.mark.parametrize(
    "raw_contents",
    [
        pytest.param(
            """\
 2
 1 2 3 4 5 6 7 8 9
 11 12 13 14 15 16 17 18 19
""",
            id="leading_spaces_and_header",
        ),
        pytest.param(
            """\
 1 2 3 4 5 6 7 8 9
 11 12 13 14 15 16 17 18 19
""",
            id="leading_spaces_no_header",
        ),
        pytest.param(
            """\
2
1 2 3 4 5 6 7 8 9
11 12 13 14 15 16 17 18 19
""",
            id="no_spaces_and_header",
        ),
        pytest.param(
            """\
1 2 3 4 5 6 7 8 9
11 12 13 14 15 16 17 18 19
""",
            id="no_spaces_no_header",
        ),
    ],
)
def test_read_leading_spaces_and_header(raw_contents: str) -> None:
    P = ImpactZParticles.from_contents(raw_contents)

    expected_raw_particles = [
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
        ],
        [
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
        ],
    ]

    assert list(P.by_row(unwrap_numpy=True)) == expected_raw_particles


@pytest.mark.parametrize(
    "raw_contents",
    [
        pytest.param(
            """\
200 0. 0.
  0.1117306E-01  0.7051202E-03  0.8310899E-02 -0.4291156E-03 -0.8144438E+00  0.6477521E-03 -0.1064629E-08 -0.9713664E-15  0.8938300E+05
  0.4550570E-02  0.7720120E-04  0.7668217E-02 -0.3586523E-03 -0.2172263E+01  0.7569714E-03 -0.1064629E-08 -0.9713664E-15  0.8938800E+05
 -0.1176394E-01 -0.2410100E-03  0.7567829E-02 -0.1866356E-03 -0.7357012E+00  0.6863160E-03 -0.1064629E-08 -0.9713664E-15  0.8938200E+05
""",
            id="example1_particle.in",
        ),
    ],
)
def test_smoke_read_example_particles(raw_contents: str):
    P = ImpactZParticles.from_contents(raw_contents)
    rows = list(P.by_row())
    assert len(rows) == 3
    np.testing.assert_allclose(
        rows[0],
        (
            0.1117306e-01,
            0.7051202e-03,
            0.8310899e-02,
            -0.4291156e-03,
            -0.8144438e00,
            0.6477521e-03,
            -0.1064629e-08,
            -0.9713664e-15,
            0.8938300e05,
        ),
    )

    np.testing.assert_allclose(
        rows[1],
        (
            0.4550570e-02,
            0.7720120e-04,
            0.7668217e-02,
            -0.3586523e-03,
            -0.2172263e01,
            0.7569714e-03,
            -0.1064629e-08,
            -0.9713664e-15,
            0.8938800e05,
        ),
    )

    np.testing.assert_allclose(
        rows[2],
        (
            -0.1176394e-01,
            -0.2410100e-03,
            0.7567829e-02,
            -0.1866356e-03,
            -0.7357012e00,
            0.6863160e-03,
            -0.1064629e-08,
            -0.9713664e-15,
            0.8938200e05,
        ),
    )
