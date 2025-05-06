from __future__ import annotations

import io
import logging
import pathlib
from typing import Generator, NamedTuple

import numpy as np
from pmd_beamphysics import ParticleGroup
import polars as pl
from pydantic import Field

from scipy.constants import e

from pmd_beamphysics.particles import c_light
from pmd_beamphysics.species import MASS_OF, charge_state, mass_of

from .parsers import fix_line
from .types import AnyPath, BaseModel, NDArray


logger = logging.getLogger(__name__)


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


def detect_species(charge_to_mass_ratio: float) -> str:
    deltas: dict[str, float] = {}
    for species, mass_eV in MASS_OF.items():
        state = charge_state(species)
        ratio = state / mass_eV
        deltas[species] = abs(charge_to_mass_ratio - ratio)

    return min(deltas, key=deltas.get)


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
    species: str = "electron"
    filename: pathlib.Path | None = Field(default=None, exclude=True)

    @staticmethod
    def empty(filename: str | pathlib.Path | None = None):
        empty = np.zeros(0)
        return ImpactZParticles(
            impactz_x=empty,
            impactz_px=empty,
            impactz_y=empty,
            impactz_py=empty,
            impactz_phase=empty,
            impactz_pz=empty,
            impactz_charge_to_mass_ratio=empty,
            impactz_weight=empty,
            impactz_id=empty,
            filename=pathlib.Path(filename) if filename else None,
        )

    @classmethod
    def from_contents(
        cls,
        contents: str,
        filename: AnyPath | None = None,
        apply_exponent_fix: bool = False,
        species: str | None = None,
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

        if apply_exponent_fix:
            contents = fix_line(contents.strip())

        if not contents:
            return cls.empty(filename)

        num_cols = 9
        schema = dict.fromkeys([str(col) for col in range(num_cols)], pl.Float64)
        with io.StringIO(contents) as fp:
            while True:
                first_line = fp.readline()
                if not first_line:
                    return cls.empty(filename)

                if first_line.strip():
                    break

            if len(first_line.strip().split()) < num_cols:
                # The first line may be the number of particles (with maybe another
                # couple unknown values after - because why not), depending on if
                # it's an input file or an output file
                pass
            else:
                fp.seek(0)

            start_pos = fp.tell()

            try:
                (x, px, y, py, phase, pz, charge_to_mass_ratio, weight, id) = (
                    pl.read_csv(
                        fp,
                        separator=" ",
                        has_header=False,
                        schema=schema,
                    )
                    .to_numpy()
                    .T
                )
            except pl.exceptions.ComputeError:
                fp.seek(start_pos)
                lines = fp.read().splitlines()
                (x, px, y, py, phase, pz, charge_to_mass_ratio, weight, id) = (
                    np.loadtxt(
                        lines,
                        unpack=True,
                        dtype=np.float64,
                        usecols=range(num_cols),
                        ndmin=1,
                    )
                )

        if not species:
            if charge_to_mass_ratio.ndim == 0:
                # TODO: only cmayes hits this scenario in lcavity-bmad?
                charge_to_mass_ratio = np.asarray([charge_to_mass_ratio])

            if len(charge_to_mass_ratio):
                species = detect_species(charge_to_mass_ratio[0])
                if species != "electron":
                    logger.warning(f"Detected species: {species}")
            else:
                logger.warning(
                    f"Charge to mass ratio is empty; assuming electrons ({filename=})"
                )
                species = "electron"

        return ImpactZParticles(
            impactz_x=x,
            impactz_px=px,
            impactz_y=y,
            impactz_py=py,
            impactz_phase=phase,
            impactz_pz=pz,
            impactz_charge_to_mass_ratio=charge_to_mass_ratio,
            impactz_weight=weight,
            impactz_id=np.asarray(id, dtype=int),  # may be stored as float?
            species=species,
            filename=pathlib.Path(filename) if filename else None,
        )

    def by_row(self, *, unwrap_numpy: bool = True):
        arrays = [
            self.impactz_x,
            self.impactz_px,
            self.impactz_y,
            self.impactz_py,
            self.impactz_phase,
            self.impactz_pz,
            self.impactz_charge_to_mass_ratio,
            self.impactz_weight,
            self.impactz_id,
        ]
        if unwrap_numpy:
            for item in zip(*arrays):
                yield [float(v) for v in item]
        else:
            for item in zip(*arrays):
                yield item

    @classmethod
    def from_file(
        cls, filename: AnyPath, species: str | None = None
    ) -> ImpactZParticles:
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
        return cls.from_contents(contents, filename=filename, species=species)

    def to_particle_group(
        self,
        reference_frequency: float,
        reference_kinetic_energy: float,
        phase_reference: float,
    ) -> ParticleGroup:
        """
        Convert ImpactZ particles to ParticleGroup.
        """

        species_mass = mass_of(self.species)

        omega = 2 * np.pi * reference_frequency

        x = self.impactz_x * c_light / omega
        px = self.impactz_px * species_mass

        y = self.impactz_y * c_light / omega
        py = self.impactz_py * species_mass

        E = reference_kinetic_energy + (1.0 - self.impactz_pz) * species_mass

        # E^2 = px^2 + py^2 + pz^2 + (mc^2)^2
        pz = np.sqrt(E**2 - px**2 - py**2 - species_mass**2)
        t = (phase_reference + self.impactz_phase) / omega
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
            "species": self.species,
            "status": np.ones_like(self.impactz_x),
            "id": self.impactz_id,
        }
        return ParticleGroup(data=data)

    @classmethod
    def from_particle_group(
        cls,
        particle_group: ParticleGroup,
        reference_frequency: float,
        reference_kinetic_energy: float,
    ) -> ImpactZParticles:
        if not particle_group.in_z_coordinates:
            raise ValueError(
                "ParticleGroup must have the same Z coordinate. Use `.drift_to_z()`."
            )

        num_particles = len(particle_group)
        omega = 2 * np.pi * reference_frequency
        species_mass = particle_group.mass
        species_charge = particle_group.species_charge

        impactz_x = particle_group.x * omega / c_light
        impactz_px = particle_group.px / species_mass

        impactz_y = particle_group.y * omega / c_light
        impactz_py = particle_group.py / species_mass

        # E = particle_group.energy
        E = np.sqrt(
            particle_group.px**2
            + particle_group.py**2
            + particle_group.pz**2
            + species_mass**2
        )
        impactz_pz = 1.0 - (E - reference_kinetic_energy) / species_mass

        impactz_t = particle_group.t * omega

        if num_particles == 1:
            impactz_weight = np.zeros_like(impactz_x)
        else:
            impactz_weight = np.abs(particle_group.weight) * np.sign(species_charge)

        impactz_charge_to_mass_ratio = np.ones_like(impactz_x) * (
            (species_charge / e) / species_mass
        )
        return cls(
            impactz_x=impactz_x,
            impactz_px=impactz_px,
            impactz_y=impactz_y,
            impactz_py=impactz_py,
            impactz_pz=impactz_pz,
            impactz_phase=impactz_t,
            impactz_weight=impactz_weight,
            impactz_charge_to_mass_ratio=impactz_charge_to_mass_ratio,
            impactz_id=particle_group.id,
        )

    @property
    def rows(self) -> Generator[tuple[float, ...], None, None]:
        for row in zip(
            self.impactz_x,
            self.impactz_px,
            self.impactz_y,
            self.impactz_py,
            self.impactz_phase,
            self.impactz_pz,
            self.impactz_charge_to_mass_ratio,
            self.impactz_weight,
            self.impactz_id,
        ):
            yield Particle(*row)

    def write_impact(self, fn: AnyPath, precision: int = 20) -> None:
        with open(fn, "w") as fp:
            logger.info(f"Writing particles to {fn}")
            print(len(self.impactz_x), file=fp)

            fmt = "{:.%dg}" % precision
            for row in self.rows:
                print(" ".join(fmt.format(v) for v in row), file=fp)


def particle_diff(P1: ParticleGroup, P2: ParticleGroup) -> dict[str, np.ndarray]:
    if len(P1) != len(P2):
        raise ValueError(
            "Particle groups must have the same number of particles to be diffed"
        )

    if not len(P1):
        return {}

    return {
        key: np.asarray([p1v - p2v for p1v, p2v in zip(P1.data[key], P2.data[key])])
        for key in P1.data
        if isinstance(P1.data[key][0], float)
    }
