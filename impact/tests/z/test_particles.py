from __future__ import annotations
import numpy as np
from numpy.testing import assert_allclose

from pmd_beamphysics import ParticleGroup
from ...z.particles import ImpactZParticles


def gaussian_data(
    n_particle: int = 100,
    charge: float = 1e-9,
    p0: float = 1e9,
    mean: np.ndarray | None = None,
    sigma_mat: np.ndarray | None = None,
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
    z = dat[:, 4]
    pz = dat[:, 5]

    data = {
        "x": x,
        "px": px * p0,
        "y": y,
        "py": py,
        "z": z,
        "pz": (1 + pz) * p0,
        "t": np.zeros(n_particle),
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
    )

    P2.z = P1["z"]
    for key in [
        "x",
        "px",
        "y",
        "py",
        # "z",
        "pz",
        "t",
        "status",
        "weight",
        "id",
    ]:
        assert_allclose(
            np.asarray(P1[key]),
            np.asarray(P2[key]),
            err_msg=f"Round-tripped key {key!r} not allclose",
            rtol=1e-2,
        )

    # TODO
    # assert P1 == P2
