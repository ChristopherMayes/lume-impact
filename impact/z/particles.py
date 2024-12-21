from __future__ import annotations

import pathlib
from typing import Generator, NamedTuple

import numpy as np
from pmd_beamphysics import ParticleGroup
from pydantic import Field

from pmd_beamphysics.particles import c_light

from ..particles import SPECIES_MASS
from .parsers import fix_line
from .types import AnyPath, BaseModel, NDArray


class Particle(NamedTuple):
    impactz_x: float  # col 1
    impactz_px: float  # col 2
    impactz_y: float  # col 3
    impactz_py: float  # col 4
    impactz_phase: float  # col 5
    impactz_pz: float  # col 6
    impactz_charge_to_mass_ratio: float  # col 7
    impactz_weight: float  # col 8
    impactz_id: int  # col 9

    # 1. x: multiply by c_light/omega to get x in  meters
    # 2 px: multiply by the particle's rest mass (0.511... e6 for electrons) to be momentum px in eV/c
    # 3. y: (same as 1)
    # 4. py (same as 2
    # 5. phase: multiply by -1/omega to get t time in seconds
    # 6.  we need the reference energy at the element to know how to convert to for pz
    # 7: Use this mass in 2
    # 8: this is the weight
    # 9: cast to int and use as id


class ImpactZParticles(BaseModel):
    impactz_x: NDArray
    impactz_px: NDArray
    impactz_y: NDArray
    impactz_py: NDArray
    impactz_phase: NDArray
    impactz_pz: NDArray
    impactz_charge_to_mass_ratio: NDArray
    impactz_weight: NDArray
    impactz_id: NDArray
    filename: pathlib.Path | None = Field(default=None, exclude=True)

    @classmethod
    def from_contents(
        cls, contents: str, filename: AnyPath | None = None
    ) -> ImpactZParticles:
        """
        Load main input from its file contents.

        Parameters
        ----------
        contents : str
            The contents of the main input file.
        filename : AnyPath or None, optional
            The filename, if known.

        Returns
        -------
        ImpactZParticles
        """

        contents = fix_line(contents)
        dtype = np.dtype(
            {
                "names": (
                    "impactz_x",
                    "impactz_px",
                    "impactz_y",
                    "impactz_py",
                    "impactz_phase",
                    "impactz_pz",
                    "impactz_charge_to_mass_ratio",
                    "impactz_weight",
                    "impactz_id",
                ),
                "formats": [np.float64] * 8 + [np.int64],
            }
        )
        (x, px, y, py, phase, pz, charge_to_mass_ratio, weight, id) = np.loadtxt(
            contents.splitlines(),
            unpack=True,
            dtype=dtype,
            skiprows=1,  # TODO: this may be 0 or 1 depending on input particles or output particles
        )
        return ImpactZParticles(
            impactz_x=x,
            impactz_px=px,
            impactz_y=y,
            impactz_py=py,
            impactz_phase=phase,
            impactz_pz=pz,
            impactz_charge_to_mass_ratio=charge_to_mass_ratio,
            impactz_weight=weight,
            impactz_id=id,
            filename=pathlib.Path(filename) if filename else None,
        )

    @classmethod
    def from_file(cls, filename: AnyPath) -> ImpactZParticles:
        """
        Load a main input file from disk.

        Parameters
        ----------
        filename : AnyPath
            The filename to load.

        Returns
        -------
        ImpactZParticles
        """
        with open(filename) as fp:
            contents = fp.read()
        return cls.from_contents(contents, filename=filename)

    def to_particle_group(
        self,
        reference_frequency: float,
        reference_kinetic_energy: float,
        species: str = "electron",
    ) -> ParticleGroup:
        """
        Convert ImpactZ particles to ParticleGroup.
        """

        mc2 = SPECIES_MASS[species]

        if species == "electron":
            assert np.allclose(-1.0 / self.impactz_charge_to_mass_ratio, mc2)

        omega = 2 * np.pi * reference_frequency

        x = self.impactz_x * c_light / omega
        px = self.impactz_px * mc2

        y = self.impactz_y * c_light / omega
        py = self.impactz_py * mc2

        E = reference_kinetic_energy + (1.0 - self.impactz_pz) * mc2
        pz = np.sqrt(E**2 - self.impactz_px**2 - self.impactz_py**2 * mc2**2)
        t = self.impactz_phase / omega  # TODO maybe minus sign as well?
        weight = np.abs(self.impactz_weight)
        weight[np.where(weight == 0.0)] = 1e-20

        data = {
            "x": x,
            "px": px,
            "y": y,
            "py": py,
            "z": np.zeros_like(self.impactz_x),
            "pz": pz,
            "t": t,
            "weight": weight,
            "species": species,
            "status": np.ones_like(self.impactz_x),
        }
        return ParticleGroup(data=data)

    @property
    def rows(self) -> Generator[tuple[float, ...], None, None]:
        for row in zip(
            (
                self.impactz_x,
                self.impactz_px,
                self.impactz_y,
                self.impactz_py,
                self.impactz_phase,
                self.impactz_pz,
                self.impactz_charge_to_mass_ratio,
                self.impactz_weight,
                self.impactz_id,
            )
        ):
            yield tuple(float(v) for v in row)

    def write_impact(self, fn: AnyPath) -> None:
        with open(fn, "w") as fp:
            print("Writing particles to", fn)
            print(len(self.x), file=fp)

            for row in self.rows:
                print(" ".join(f"{v:g}" for v in row), file=fp)
