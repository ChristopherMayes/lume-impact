from __future__ import annotations

import pathlib
from typing import NamedTuple

import numpy as np
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.interfaces.impact import impact_particles_to_particle_data
from pydantic import Field

from .parsers import parse_input_line
from .types import AnyPath, BaseModel


class Particle(NamedTuple):
    x: float
    GBx: float
    y: float
    GBy: float
    z: float
    GBz: float


class ImpactZParticles(BaseModel):
    particles: list[Particle]
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

        particles = []
        for lineno, line in enumerate(contents.splitlines()[1:], start=2):
            parts = parse_input_line(line)
            if len(parts) < 6:
                raise ValueError(
                    f"Particles data on line {lineno} insufficient ({len(parts)} is less than 6)"
                )
            particles.append(Particle(*parts[:6]))

        return ImpactZParticles(
            particles=particles,
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
        mc2: float = 0.0,
        species: str = "",
        time: float = 0.0,
        macrocharge: float = 0.0,
        cathode_kinetic_energy_ref: float | None = None,
        verbose: bool = False,
    ) -> ParticleGroup:
        """
        Convert impact particles ParticleGroup.

        particle_charge is the charge in units of |e|

        At the cathode, Impact-T translates z to t = z / (beta*c) for emission,
        where (beta*c) is the velocity calculated from kinetic energy:
            header['Bkenergy'] in eV.
        This is purely a conversion factor.

        If cathode_kinetic_energy_ref is given, z will be parsed appropriately to t, and z will be set to 0.

        Otherwise, particles will be set to the same time.
        """
        particles = {
            "x": np.asarray([particle.x for particle in self.particles]),
            "y": np.asarray([particle.y for particle in self.particles]),
            "z": np.asarray([particle.z for particle in self.particles]),
            "GBx": np.asarray([particle.GBx for particle in self.particles]),
            "GBy": np.asarray([particle.GBy for particle in self.particles]),
            "GBz": np.asarray([particle.GBz for particle in self.particles]),
        }

        data = impact_particles_to_particle_data(
            particles,
            mc2=mc2,
            species=species,
            time=time,
            macrocharge=macrocharge,
            cathode_kinetic_energy_ref=cathode_kinetic_energy_ref,
            verbose=verbose,
        )
        return ParticleGroup(data=data)

    def write_impact(self, fn: AnyPath) -> None:
        with open(fn, "w") as fp:
            print("Writing particles to", fn)
            print(len(self.particles), file=fp)
            for particle in self.particles:
                extended_particle = list(particle) + [0.0, 0.0, 0.0]  # extra data?
                print(" ".join(f"{v:g}" for v in extended_particle), file=fp)
