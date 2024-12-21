from __future__ import annotations

import logging
import pathlib
import typing
from typing import Any, TypeVar
from collections.abc import Generator, Sequence

import numpy as np
import pydantic
import pydantic.alias_generators
from .particles import ImpactZParticles
from pmd_beamphysics.units import pmd_unit
from typing_extensions import override

from . import parsers
from .input import ImpactZInput, WriteFull
from .types import (
    AnyPath,
    BaseModel,
    PydanticParticleGroup,
    PydanticPmdUnit,
    SequenceBaseModel,
)
from .units import Degrees, Meters, MeV, Radians, Unitless, known_unit

try:
    from collections.abc import Mapping
except ImportError:
    pass

if typing.TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class RunInfo(BaseModel):
    """
    Impact-Z run information.

    Attributes
    ----------
    error : bool
        True if an error occurred during the Impact-Z run.
    error_reason : str or None
        Error explanation, if `error` is set.
    run_script : str
        The command-line arguments used to run Impact-Z
    output_log : str
        Impact-Z output log
    start_time : float
        Start time of the process
    end_time : float
        End time of the process
    run_time : float
        Wall clock run time of the process
    """

    error: bool = pydantic.Field(
        default=False, description="`True` if an error occurred during the Impact-Z run"
    )
    error_reason: str | None = pydantic.Field(
        default=None, description="Error explanation, if `error` is set."
    )
    run_script: str = pydantic.Field(
        default="", description="The command-line arguments used to run Impact-Z"
    )
    output_log: str = pydantic.Field(
        default="", repr=False, description="Impact-Z output log"
    )
    start_time: float = pydantic.Field(
        default=0.0, repr=False, description="Start time of the process"
    )
    end_time: float = pydantic.Field(
        default=0.0, repr=False, description="End time of the process"
    )
    run_time: float = pydantic.Field(
        default=0.0, description="Wall clock run time of the process"
    )

    @property
    def success(self) -> bool:
        """`True` if the run was successful."""
        return not self.error


# def _empty_ndarray():
#     return np.zeros(0)


def load_stat_files_from_path(
    workdir: pathlib.Path,
) -> tuple[dict[str, np.ndarray], dict[str, pmd_unit]]:
    stats = {}
    units = {}
    for fnum, cls in file_number_to_cls.items():
        fn = workdir / f"fort.{fnum}"
        if fn.exists():
            stats.update(cls.from_file(fn))
            for key, field in cls.model_fields.items():
                field_units = field.metadata[0]["units"]

                if field_units == "MeV":
                    field_units = known_unit["eV"]
                    stats[key] *= 1e6

                units[key] = field_units

    return stats, units


class ImpactZOutput(Mapping, BaseModel, arbitrary_types_allowed=True):
    """
    IMPACT-Z command output.

    Attributes
    ----------
    run : RunInfo
        Execution information - e.g., how long did it take and what was
        the output from Impact-Z.
    alias : dict[str, str]
        Dictionary of aliased data keys.
    """

    run: RunInfo = pydantic.Field(
        default_factory=RunInfo,
        description="Run-related information - output text and timing.",
    )
    stats: dict[str, np.ndarray] = pydantic.Field(default={}, repr=False)
    alias: dict[str, str] = pydantic.Field(
        default={
            "-cov_x__gammabeta_x": "neg_cov_x__gammabeta_x",
            "mean_z": "z",
        },
    )
    final_particles_raw: ImpactZParticles | None = pydantic.Field(
        default=None, repr=False
    )
    final_particles: PydanticParticleGroup | None = None
    key_to_unit: dict[str, PydanticPmdUnit] = pydantic.Field(default={}, repr=False)

    @override
    def __eq__(self, other: Any) -> bool:
        return BaseModel.__eq__(self, other)

    @override
    def __ne__(self, other: Any) -> bool:
        return BaseModel.__ne__(self, other)

    @override
    def __getitem__(self, key: str) -> Any:
        """Support for Mapping -> easy access to data."""
        # _check_for_unsupported_key(key)

        # parent, array_attr = self._split_parent_and_attr(key)
        # return getattr(parent, array_attr)
        key = self.alias.get(key, key)
        return self.stats[key]

    @override
    def __iter__(self) -> Generator[str]:
        """Support for Mapping -> easy access to data."""
        yield from self.stats

    @override
    def __len__(self) -> int:
        """Support for Mapping -> easy access to data."""
        return len(self.stats)

    def units(self, key: str) -> pmd_unit:
        return self.key_to_unit[self.alias.get(key, key)]

    def stat(self, key: str):
        """
        Array from .output['stats'][key]

        Additional keys are avalable:
            'mean_energy': mean energy
            'Ez': z component of the electric field at the centroid particle
            'Bz'  z component of the magnetic field at the centroid particle
            'cov_{a}__{b}': any symmetric covariance matrix term
        """
        if key in ("Ez", "Bz"):
            raise NotImplementedError()
            # return self.centroid_field(component=key[0:2])

        if key == "mean_energy":
            raise NotImplementedError()
            # return self.stat("mean_kinetic_energy") + self.mc2

        # Allow flipping covariance keys
        if key.startswith("cov_") and key not in self.stats:
            k1, k2 = key[4:].split("__")
            key = f"cov_{k2}__{k1}"

        if key not in self.stats:
            raise ValueError(f"{key} is not available in the output data")

        return self.stats[self.alias.get(key, key)]

    @classmethod
    def from_input_settings(
        cls,
        input: ImpactZInput,
        workdir: pathlib.Path,
        load_fields: bool = False,
        load_particles: bool = False,
        smear: bool = True,
    ) -> ImpactZOutput:
        """
        Load ImpactZ output based on the configured input settings.

        Parameters
        ----------
        load_fields : bool, default=False
            After execution, load all field files.
        load_particles : bool, default=False
            After execution, load all particle files.
        smear : bool, default=True
            If set, for particles, this will smear the phase over the sample
            (skipped) slices, preserving the modulus.

        Returns
        -------
        ImpactZOutput
            The output data.
        """
        stats, units = load_stat_files_from_path(workdir)

        final_particles_raw = None
        final_particles = None
        for ele in input.lattice:
            if isinstance(ele, WriteFull) and final_particles is None:
                final_particles_raw = ImpactZParticles.from_file(
                    workdir / f"fort.{ele.file_id}"
                )
                final_particles = final_particles_raw.to_particle_group(
                    reference_kinetic_energy=input.initial_kinetic_energy,
                    reference_frequency=input.scaling_frequency,
                    # species=...
                )

        return ImpactZOutput(
            stats=stats,
            key_to_unit=units,
            final_particles=final_particles,
            final_particles_raw=final_particles_raw,
        )

    def plot(
        self,
        y: str | Sequence[str] = ("sigma_x", "sigma_y"),
        x: str = "mean_z",
        xlim=None,
        ylim=None,
        ylim2=None,
        y2=[],
        nice=True,
        include_layout=True,
        include_labels=False,
        include_markers=True,
        include_particles=True,
        include_field=True,
        field_t=0,
        include_legend=True,
        return_figure=False,
        tex=True,
        **kwargs,
    ):
        """ """
        from ..plot import plot_stats_with_layout

        if not self.stats:
            # Just plot fieldmaps if there are no stats
            raise NotImplementedError()
            # return plot_layout(
            #     self,
            #     xlim=xlim,
            #     include_markers=include_markers,
            #     include_labels=include_labels,
            #     include_field=include_field,
            #     field_t=field_t,
            #     return_figure=return_figure,
            #     **kwargs,
            # )

        return plot_stats_with_layout(
            self,
            ykeys=y,
            ykeys2=y2,
            xkey=x,
            xlim=xlim,
            ylim=ylim,
            ylim2=ylim2,
            nice=nice,
            tex=tex,
            include_layout=False,  # include_layout,
            include_labels=include_labels,
            include_field=include_field,
            field_t=field_t,
            include_markers=include_markers,
            include_particles=include_particles,
            include_legend=include_legend,
            return_figure=return_figure,
            **kwargs,
        )


file_number_to_cls: dict[str, FortranOutputFileData] = {}
T = TypeVar("T", bound="FortranOutputFileData")


class FortranOutputFileData(SequenceBaseModel):
    def __init_subclass__(cls, file_id: int, **kwargs):
        super().__init_subclass__(**kwargs)

        assert isinstance(file_id, int)
        assert file_id not in file_number_to_cls, f"Duplicate element ID {file_id}"
        file_number_to_cls[file_id] = cls

    @classmethod
    def from_file(cls: type[T], filename: AnyPath) -> dict[str, np.ndarray]:
        data = {attr: [] for attr in cls.model_fields}
        with open(filename) as fp:
            for line in fp.read().splitlines():
                for attr, value in zip(data, parsers.parse_input_line(line)):
                    data[attr].append(value)

        return {key: np.asarray(items) for key, items in data.items()}


class ReferenceParticles(FortranOutputFileData, file_id=18):
    """
    Reference particle information from an output file.

    Parameters
    ----------
    z : float
        Distance in meters (1st column).
    absolute_phase : float
        Absolute phase in radians (2nd column).
    mean_gamma : float
        Mean gamma (3rd column).
    mean_kinetic_energy : float
        Mean kinetic energy in MeV (4th column).
        LUME-ImpactZ converts this automatically to eV.
    mean_beta : float
        Beta (5th column).
    max_r : float
        Rmax in meters, measured from the axis of pipe (6th column).

    Notes
    -----
    Data is written using the following Fortran code:
    write(18,99)z,this%refptcl(5),gam,energy,bet,sqrt(glrmax)*xl
    """

    z: Meters
    absolute_phase: Radians
    mean_gamma: Unitless
    mean_kinetic_energy: MeV
    mean_beta: Unitless
    max_r: Meters


class RmsX(FortranOutputFileData, file_id=24):
    """
    RMS size information in X.

    Attributes
    ----------
    z : float
        Mean z distance (m)
    mean_x : float
        Centroid location (m)
    sigma_x : float
        RMS size (m)
    mean_gammabeta_x : float
        Centroid momentum [rad]
    sigma_gammabeta_x : float
        RMS momentum [rad]
    neg_cov_x__gammabeta_x : float
        Twiss parameter, alpha
    norm_emit_x : float
        normalized RMS emittance [m-rad]

    Notes
    -----
    Data is written using the following Fortran code:
    write(24,100)z,x0*xl,xrms*xl,px0/gambet,pxrms/gambet,-xpx/epx,epx*xl
    """

    z: Meters
    mean_x: Meters
    sigma_x: Meters
    mean_gammabeta_x: Radians
    sigma_gammabeta_x: Radians
    neg_cov_x__gammabeta_x: Meters
    norm_emit_x: Meters  # m-rad


class RmsY(FortranOutputFileData, file_id=25):
    """
    RMS size information in Y.

    Attributes
    ----------
    z : float
        z distance (m)
    mean_y : float
        centroid location (m)
    sigma_y : float
        RMS size (m)
    mean_gammabeta_y : float
        Centroid momentum [rad]
    sigma_gammabeta_y : float
        RMS momentum [rad]
    neg_cov_y__gammabeta_y : float
        Twiss parameter, alpha
    norm_emit_y : float
        normalized RMS emittance [m-rad]

    Notes
    -----
    Data is written using the following Fortran code:
    write(25,100)z,y0*xl,yrms*xl,py0/gambet,pyrms/gambet,-ypy/epy,epy*xl
    """

    z: Meters
    mean_y: Meters
    sigma_y: Meters
    mean_gammabeta_y: Radians
    sigma_gammabeta_y: Radians
    neg_cov_y__gammabeta_y: Meters
    norm_emit_y: Meters  # m-rad


class RmsZ(FortranOutputFileData, file_id=26):
    """
    RMS size information in Z.

    Attributes
    ----------
    z : float
        z distance (m)
    mean_z : float
        centroid location (m)
    sigma_z : float
        RMS size (m)
    mean_gammabeta_z : float
        Centroid momentum [MeV]
    sigma_gammabeta_z : float
        RMS momentum [MeV]
    neg_cov_z__gammabeta_z : float
        Twiss parameter, alpha
    norm_emit_z : float
        normalized RMS emittance [degree-MeV]

    Notes
    -----
    Data is written using the following Fortran code:
    write(26,100)z,z0*xt,zrms*xt,pz0*qmc,pzrms*qmc,-zpz/epz,epz*qmc*xt
    """

    z: Meters
    mean_z: Meters
    sigma_z: Meters
    mean_gammabeta_z: MeV
    sigma_gammabeta_z: MeV
    neg_cov_z__gammabeta_z: Unitless
    norm_emit_z: Meters


class MaxAmplitude(FortranOutputFileData, file_id=27):
    """
    File fort.27: maximum amplitude information

    Parameters
    ----------
    z : float
        z distance (m)
    max_amplitude_x : float
        Maximum X (m)
    max_amplitude_gammabeta_x : float
        Maximum Px (rad)
    max_amplitude_y : float
        Maximum Y (m)
    max_amplitude_gammabeta_y : float
        Maximum Py (rad)
    max_amplitude_phase : float
        Maximum Phase (degree)
    max_amplitude_energy_dev : float
        Maximum Energy deviation (MeV)

    Notes
    -----
    Data is written using the following Fortran code:
    write(27,100)z,glmax(1)*xl,glmax(2)/gambet,glmax(3)*xl,&
                 glmax(4)/gambet,glmax(5)*xt,glmax(6)*qmc
    """

    z: Meters
    max_amplitude_x: Meters
    max_amplitude_gammabeta_x: Radians
    max_amplitude_y: Meters
    max_amplitude_gammabeta_y: Radians
    max_amplitude_phase: Degrees  # really?
    max_amplitude_energy_dev: MeV


class LoadBalanceLossDiagnostic(FortranOutputFileData, file_id=28):
    """
    File fort.28: Load balance and loss diagnostic.

    Attributes
    ----------
    z : float
        z distance (m)
    loadbalance_min_n_particle : float
        Minimum number of particles on a PE
    loadbalance_max_n_particle : float
        Maximum number of particles on a PE
    n_particle : float
        Total number of particles in the bunch

    Notes
    -----
    Data is written using the following Fortran code:
    write(28,101)z,npctmin,npctmax,nptot
    """

    z: Meters
    loadbalance_min_n_particle: Unitless
    loadbalance_max_n_particle: Unitless
    n_particle: Unitless


class BeamDistribution3rd(FortranOutputFileData, file_id=29):
    """
    File fort.29: cubic root of 3rd moments of the beam distribution

    Attributes
    ----------
    z : float
        z distance (m)
    moment3_x : float
        X (m)
    moment3_gammabeta_x : float
        Px (rad)
    moment3_y : float
        Y (m)
    moment3_gammabeta_y : float
        Py (rad)
    moment3_phase : float
        phase (degree)
    moment3_energy_deviation : float
        Energy deviation (MeV)

    Notes
    -----
    Data is written using the following Fortran code:
    write(29,100)z,x03*xl,px03/gambet,y03*xl,py03/gambet,z03*xt,&
                 pz03*qmc
    """

    z: Meters
    moment3_x: Meters
    moment3_gammabeta_x: Radians
    moment3_y: Meters
    moment3_gammabeta_y: Radians
    moment3_phase: Degrees
    moment3_energy_deviation: MeV


class BeamDistribution4th(FortranOutputFileData, file_id=30):
    """
    File fort.30: square root, square root of 4th moments of the beam distribution

    Attributes
    ----------
    z : float
        z distance (m)
    moment4_x : float
        X (m)
    moment4_gammabeta_x : float
        Px (rad)
    moment4_y : float
        Y (m)
    moment4_gammabeta_y : float
        Py (rad)
    moment4_phase : float
        Phase (degree)
    moment4_energy_deviation : float
        Energy deviation (MeV)

    Notes
    -----
    Data is written using the following Fortran code:
    write(30,100)z,x04*xl,px04/gambet,y04*xl,py04/gambet,z04*xt,&
                 pz04*qmc
    """

    z: Meters
    moment4_x: Meters
    moment4_gammabeta_x: Radians
    moment4_y: Meters
    moment4_gammabeta_y: Radians
    moment4_phase: Degrees
    moment4_energy_deviation: MeV


class ParticlesAtChargedState(FortranOutputFileData, file_id=32):
    """
    File fort.32: number of particles for each charge state

    This file contains data about the number of particles for each charge state
    at different z distances.

    Attributes
    ----------
    z : float
        The z distance in meters.
    charge_state_n_particle : int
        The number of particles for each charge state.

    Notes
    -----
    Data is written using the following Fortran code:
    write(32,*)z,nptlist(1:nchrg)
    """

    z: Meters
    charge_state_n_particle: Unitless
