from __future__ import annotations

import logging
import pathlib
import typing
from typing import Any, TypeVar
from collections.abc import Generator, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pydantic
import pydantic.alias_generators

from .particles import ImpactZParticles
from pmd_beamphysics.units import pmd_unit
from typing_extensions import override

from . import archive as _archive, parsers
from .input import HasOutputFile, ImpactZInput, WriteSliceInfo
from .types import (
    AnyPath,
    BaseModel,
    PydanticParticleGroup,
    PydanticPmdUnit,
    SequenceBaseModel,
    NDArray,
)
from .units import (
    AmperesArray,
    Degrees,
    DegreesArray,
    eVArray,
    MeVArray,
    Meters,
    MeV,
    MetersArray,
    Radians,
    RadiansArray,
    Unitless,
    UnitlessArray,
    known_unit,
)

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


def load_stat_files_from_path(
    workdir: pathlib.Path,
    reference_particle_mass: float,
    reference_frequency: float,
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

    try:
        stats["energy_ref"] = stats["kinetic_energy_ref"] + reference_particle_mass
        stats["p0c"] = np.sqrt(stats["energy_ref"] ** 2 - reference_particle_mass**2)

        stats["mean_energy"] = stats["energy_ref"] - stats["neg_delta_mean_energy"]
        stats["mean_px"] = stats["mean_px_over_p0"] * stats["p0c"]
        stats["mean_py"] = stats["mean_py_over_p0"] * stats["p0c"]
        stats["sigma_px"] = stats["sigma_px_over_p0"] * stats["p0c"]
        stats["sigma_py"] = stats["sigma_py_over_p0"] * stats["p0c"]
        stats["sigma_t"] = stats["sigma_phase_deg"] / 360.0 / reference_frequency
        stats["mean_t"] = stats["mean_phase_deg"] / 360.0 / reference_frequency
    except KeyError as ex:
        logger.exception(f"Some expected statistics unavailable? Missing: {ex}")

    return stats, units


def _get_dict_key(
    dct: dict[str | int, Any],
    file_id: int | float,
    name: str,
) -> str | int:
    """Get an unused dictionary key for a file_id/element name."""
    if not name:
        return int(file_id)

    if name not in dct:
        return name

    key = f"{name}.{file_id}"
    if key not in dct:
        return key

    idx = 1
    while key in dct:
        key = f"{name}_{idx}"
        idx += 1
    return key


class ImpactZSlices(BaseModel):
    """
    A class to represent the impact Z slices.

    Attributes
    ----------
    bunch_length : NDArray
        Bunch length coordinate (m).
    particles_per_slice : NDArray
        Number of particles per slice.
    current_per_slice : NDArray
        Current (A) per slice.
    normalized_emittance_x : NDArray
        X normalized emittance (m-rad) per slice.
    normalized_emittance_y : NDArray
        Y normalized emittance (m-rad) per slice.
    dE_E : NDArray
        dE/E.
    uncorrelated_energy_spread : NDArray
        Uncorrelated energy spread (eV) per slice.
    mean_x : NDArray
        <x> (m) of each slice.
    mean_y : NDArray
        <y> (m) of each slice.
    mismatch_factor_x : NDArray
        X mismatch factor.
    mismatch_factor_y : NDArray
        Y mismatch factor.
    """

    bunch_length: MetersArray = np.zeros(0)
    particles_per_slice: NDArray = np.zeros(0)
    current_per_slice: AmperesArray = np.zeros(0)
    normalized_emittance_x: NDArray = np.zeros(0)
    normalized_emittance_y: NDArray = np.zeros(0)
    dE_E: NDArray = np.zeros(0)
    uncorrelated_energy_spread: eVArray = np.zeros(0)
    mean_x: NDArray = np.zeros(0)
    mean_y: NDArray = np.zeros(0)
    mismatch_factor_x: NDArray = np.zeros(0)
    mismatch_factor_y: NDArray = np.zeros(0)

    filename: pathlib.Path | None = pydantic.Field(default=None, exclude=True)

    def debug_plot_all(self, xkey: str, figsize=(12, 12)):
        keys = list(self.model_fields)
        keys = keys[: keys.index("filename")]
        keys.remove(xkey)

        assert len(keys) % 2 == 0
        fig, axs = plt.subplot_mosaic(
            list(list(pair) for pair in zip(keys[::2], keys[1::2])),
            figsize=figsize,
        )

        x = getattr(self, xkey)
        for ykey in keys:
            ax = axs[ykey]
            try:
                y = getattr(self, ykey)
                ax.scatter(x, y)
            except Exception as ex:
                logger.error(f"Failed to plot key: {ykey} {ex.__class__.__name__} {ex}")
            else:
                ax.set_xlabel(xkey)
                ax.set_ylabel(ykey)

        fig.tight_layout()
        return fig, axs

    @classmethod
    def from_contents(
        cls, contents: str, filename: AnyPath | None = None
    ) -> ImpactZSlices:
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
        ImpactZSlices
        """

        contents = parsers.fix_line(contents)
        fields = list(cls.model_fields)[:9]
        dtype = np.dtype(
            {
                "names": fields,
                "formats": [np.float64] * 9,
            }
        )

        lines = contents.splitlines()
        if not lines:
            return ImpactZSlices(
                filename=pathlib.Path(filename) if filename else None,
            )

        arrays = np.loadtxt(
            lines,
            dtype=dtype,
            usecols=range(len(fields)),
            unpack=True,
        )

        data = {field: arr for field, arr in zip(fields, arrays)}
        return ImpactZSlices(
            **data,
            filename=pathlib.Path(filename) if filename else None,
        )

    @classmethod
    def from_file(cls, filename: AnyPath) -> ImpactZSlices:
        """
        Load a main input file from disk.

        Parameters
        ----------
        filename : AnyPath
            The filename to load.

        Returns
        -------
        ImpactZSlices
        """
        with open(filename) as fp:
            contents = fp.read()
        return cls.from_contents(contents, filename=filename)


def _empty_ndarray():
    return np.zeros(0)


def _split_extra(cls: type[BaseModel], dct: dict) -> dict[str, Any]:
    extra = dct.pop("extra", {})
    assert isinstance(extra, dict)
    # Don't let computed fields make it into 'extra':
    for fld in cls.model_computed_fields:
        dct.pop(fld, None)
    return {key: dct.pop(key) for key in set(dct) - set(cls.model_fields)}


class OutputStats(BaseModel):
    beta_ref: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Beta of the reference particle."
    )
    charge_state_n_particle: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="The number of particles for each charge state.",
    )
    gamma_ref: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Reference particle gamma."
    )
    kinetic_energy_ref: MeVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Reference particle kinetic energy. (eV)",
    )
    loadbalance_max_n_particle: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Maximum number of particles on a processing element (PE).",
    )
    loadbalance_min_n_particle: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Minimum number of particles on a processing element (PE).",
    )
    max_amplitude_energy_dev: MeVArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Maximum energy deviation in MeV."
    )
    max_amplitude_gammabeta_x: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Maximum Px (in radians)."
    )
    max_amplitude_gammabeta_y: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Maximum Py (in radians)."
    )
    max_amplitude_phase: DegreesArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Maximum phase (in degrees)."
    )
    max_amplitude_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Maximum X (in meters)."
    )
    max_amplitude_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Maximum Y (in meters)."
    )
    max_r: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Maximum radius (Rmax) in meters, measured from the axis of the pipe.",
    )
    mean_px_over_p0: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean $px / p0$ (rad).",
    )
    mean_py_over_p0: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean $py / p0$ (rad).",
    )
    mean_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Centroid location in the x-direction (meters).",
    )
    mean_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Centroid location in the y-direction (meters).",
    )
    mean_phase_deg: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean phase (degrees)",
    )
    moment3_energy_deviation: MeVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for energy deviation (eV).",
    )
    moment3_px_over_p0: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for Px (rad).",
    )
    moment3_py_over_p0: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for Py (rad).",
    )
    moment3_phase: DegreesArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for phase (degree).",
    )
    moment3_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for x (meters).",
    )
    moment3_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for y (meters).",
    )
    moment4_energy_deviation: MeVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Fourth-order central moment for energy deviation (eV).",
    )
    moment4_px_over_p0: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Fourth-order central moment for Px (rad).",
    )
    moment4_py_over_p0: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Fourth-order central moment for Py (rad).",
    )
    moment4_phase: DegreesArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Fourth-order central moment for phase (degree).",
    )
    moment4_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Fourth-order central moment for x (meters).",
    )
    moment4_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Fourth-order central moment for y (meters).",
    )
    n_particle: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Total number of particles in the bunch.",
    )
    neg_delta_mean_energy: MeVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Negative delta mean energy (eV).",
        repr=False,
    )
    twiss_alpha_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss parameter alpha for x-direction.",
    )
    twiss_alpha_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss parameter alpha for y-direction.",
    )
    twiss_alpha_z: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss parameter alpha for z-direction.",
    )
    norm_emit_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Normalized RMS emittance in x-direction (m-rad).",
    )
    norm_emit_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Normalized RMS emittance in y-direction (m-rad).",
    )
    norm_emit_z: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Normalized RMS emittance in z-direction (degree-MeV).",
    )
    phase_ref: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Absolute phase in radians."
    )
    sigma_px_over_p0: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma $px / p0$ (rad).",
    )
    sigma_py_over_p0: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma $py / p0$ (rad).",
    )
    sigma_energy: MeVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma energy (eV).",
    )
    sigma_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="RMS size in the x-direction (meters).",
    )
    sigma_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="RMS size in the y-direction (meters).",
    )
    sigma_phase_deg: DegreesArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma phase (degrees).",
    )
    z: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Z position (meters)"
    )

    # Calculated stats
    energy_ref: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Energy reference (eV) (computed)",
    )
    mean_energy: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean energy (eV) (computed)",
    )
    p0c: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Momentum reference (eV) (computed)",
    )
    mean_px: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean px (eV) (computed)",
    )
    mean_py: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean py (eV) (computed)",
    )
    mean_t: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean time (s) (computed)",
    )
    sigma_px: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma px (eV) (computed)",
    )
    sigma_py: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma py (eV) (computed)",
    )
    sigma_t: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="RMS size in time (rad) (computed)",
    )

    units: dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
    extra: dict[str, NDArray] = pydantic.Field(
        default_factory=dict,
        description=(
            "Additional Impact-Z output data.  This is a future-proofing mechanism "
            "in case Impact-Z changes and LUME-Impact is not yet ready for it."
        ),
    )

    @classmethod
    def from_stats_files(
        cls,
        workdir: pathlib.Path,
        reference_particle_mass: float,
        reference_frequency: float,
    ) -> OutputStats:
        stats, units = load_stat_files_from_path(
            workdir,
            reference_particle_mass=reference_particle_mass,
            reference_frequency=reference_frequency,
        )

        extra = _split_extra(cls, stats)
        return OutputStats(units=units, extra=extra, **stats)


file_number_to_cls: dict[int, type[FortranOutputFileData]] = {}
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
                parsed = parsers.parse_input_line(line)
                for attr, value in zip(data, parsed.data):
                    data[attr].append(value)

        return {key: np.asarray(items) for key, items in data.items()}


class ReferenceParticles(FortranOutputFileData, file_id=18):
    """
    Reference particle information from an output file.

    Parameters
    ----------
    z : float
        Distance in meters (1st column).
    phase_ref : float
        Absolute phase in radians (2nd column).
    gamma_ref : float
        Reference particle gamma (3rd column).
    kinetic_energy_ref : float
        Reference particle kinetic energy in MeV (4th column).
        LUME-ImpactZ converts this automatically to eV.
    beta_ref : float
        Beta (5th column).
    max_r : float
        Rmax in meters, measured from the axis of pipe (6th column).

    Notes
    -----
    Data is written using the following Fortran code:
    write(18,99)z,this%refptcl(5),gam,energy,bet,sqrt(glrmax)*xl
    """

    z: Meters
    phase_ref: Radians
    gamma_ref: Unitless
    kinetic_energy_ref: MeV
    beta_ref: Unitless
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
    mean_px_over_p0 : float
        Mean $px / p0$ (rad)
    sigma_px_over_p0 : float
        Sigma $px / p0$ (rad)
    twiss_alpha_x : float
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
    mean_px_over_p0: Radians
    sigma_px_over_p0: Radians
    twiss_alpha_x: Meters
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
    mean_py_over_p0 : float
        Mean $py / p0$ [rad]
    sigma_py_over_p0 : float
        $py / p0$ [rad]
    twiss_alpha_y : float
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
    mean_py_over_p0: Radians
    sigma_py_over_p0: Radians
    twiss_alpha_y: Meters
    norm_emit_y: Meters  # m-rad


class RmsZ(FortranOutputFileData, file_id=26):
    """
    RMS size information in Z.

    Attributes
    ----------
    z : float
        z distance (m)
    mean_phase_deg : float
        Mean phase (degrees)
    sigma_phase_deg : float
        RMS phase in degrees.
    neg_delta_mean_energy : float
        Negative delta mean energy, used to convert to mean energy [eV]
        where `neg_delta_mean_energy = (kinetic_energy_ref - mean_energy) + reference_particle_mass `
        In the file, this is stored as MeV and LUME-Impact converts to eV automatically.
    sigma_energy : float
        RMS momentum [eV]
        In the file, this is stored as MeV and LUME-Impact converts to eV automatically.
    twiss_alpha_z : float
        Twiss parameter, alpha
    norm_emit_z : float
        normalized RMS emittance [degree-MeV]

    Notes
    -----
    Data is written using the following Fortran code:
    write(26,100)z,z0*xt,zrms*xt,pz0*qmc,pzrms*qmc,-zpz/epz,epz*qmc*xt
    """

    z: Meters
    mean_phase_deg: Degrees
    sigma_phase_deg: Degrees
    neg_delta_mean_energy: MeV
    sigma_energy: MeV
    twiss_alpha_z: Unitless
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
        Maximum Energy deviation [eV]
        In the file, this is stored as MeV and LUME-Impact converts to eV automatically.

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
    moment3_px_over_p0 : float
        Px (rad)
    moment3_y : float
        Y (m)
    moment3_py_over_p0 : float
        Py (rad)
    moment3_phase : float
        phase (degree)
    moment3_energy_deviation : float
        Energy deviation [eV]
        In the file, this is stored as MeV and LUME-Impact converts to eV automatically.

    Notes
    -----
    Data is written using the following Fortran code:
    write(29,102)z,ravg*xl,rrms*xl,r90*xl,r95*xl,r99*xl,sqrt(rrmax)*xl
    """

    z: Meters
    moment3_x: Meters
    moment3_px_over_p0: Radians
    moment3_y: Meters
    moment3_py_over_p0: Radians
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
    moment4_px_over_p0 : float
        Px (rad)
    moment4_y : float
        Y (m)
    moment4_py_over_p0 : float
        Py (rad)
    moment4_phase : float
        Phase (degree)
    moment4_energy_deviation : float
        Energy deviation [eV]
        In the file, this is stored as MeV and LUME-Impact converts to eV automatically.

    Notes
    -----
    Data is written using the following Fortran code:
    write(30,100)z,x04*xl,px04/gambet,y04*xl,py04/gambet,z04*xt,&
                 pz04*qmc
    """

    z: Meters
    moment4_x: Meters
    moment4_px_over_p0: Radians
    moment4_y: Meters
    moment4_py_over_p0: Radians
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


class ImpactZOutput(Mapping, BaseModel):
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
    stats: OutputStats = OutputStats()
    alias: dict[str, str] = pydantic.Field(
        default={
            "-cov_x__gammabeta_x": "twiss_alpha_x",
        },
    )
    particles_raw: dict[str | int, ImpactZParticles] = pydantic.Field(
        default={},
        repr=False,
    )
    particles: dict[str | int, PydanticParticleGroup] = pydantic.Field(
        default={},
        repr=False,
    )
    slices: dict[int, ImpactZSlices] = pydantic.Field(
        default={},
        repr=False,
    )
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
        try:
            return getattr(self.stats, key)
        except AttributeError:
            raise KeyError(key)

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

        # Allow flipping covariance keys
        if key.startswith("cov_") and key not in self:
            k1, k2 = key[4:].split("__")
            key = f"cov_{k2}__{k1}"

        if key not in self:
            raise ValueError(f"{key} is not available in the output data")

        return self[self.alias.get(key, key)]

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

        stats = OutputStats.from_stats_files(
            workdir,
            # reference_kinetic_energy=input.reference_kinetic_energy,
            reference_particle_mass=input.reference_particle_mass,
            reference_frequency=input.reference_frequency,
        )

        units = stats.units.copy()
        particles_raw = {}
        particles = {}
        slices = {}

        z_start = 0.0
        z_end = 0.0

        for ele in input.lattice:
            z_start = z_end
            z_end += ele.length
            z_end_idx = np.argmin(np.abs(stats.z - z_start))

            if isinstance(ele, WriteSliceInfo):
                key = _get_dict_key(slices, ele.file_id, ele.name)
                slices[ele.file_id] = ImpactZSlices.from_file(
                    workdir / f"fort.{ele.file_id}"
                )
            elif ele.class_information().has_output_file and isinstance(
                ele, HasOutputFile
            ):
                raw = ImpactZParticles.from_file(workdir / f"fort.{ele.file_id}")

                key = _get_dict_key(particles_raw, ele.file_id, ele.name)
                particles_raw[key] = raw
                phase_ref = stats.phase_ref[z_end_idx]
                kinetic_energy = stats.kinetic_energy_ref[z_end_idx]
                particles[key] = raw.to_particle_group(
                    reference_kinetic_energy=kinetic_energy,
                    reference_frequency=input.reference_frequency,
                    phase_reference=phase_ref,
                )

        return ImpactZOutput(
            stats=stats,
            key_to_unit=units,
            particles=particles,
            particles_raw=particles_raw,
            slices=slices,
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

    def debug_plot_all(self, figsize=(12, 64)):
        keys = list(self)
        fig, axs = plt.subplot_mosaic(
            list(list(pair) for pair in zip(keys[::2], keys[1::2])),
            figsize=figsize,
        )

        for key in keys:
            try:
                self.plot(key, ax=axs[key])
            except Exception as ex:
                logger.error(f"Failed to plot key: {key} {ex.__class__.__name__} {ex}")

        fig.tight_layout()
        return fig, axs

    def archive(self, h5: h5py.Group) -> None:
        """
        Dump outputs into the given HDF5 group.

        Parameters
        ----------
        h5 : h5py.Group
            The HDF5 file in which to write the information.
        """
        _archive.store_in_hdf5_file(h5, self)

    @classmethod
    def from_archive(cls, h5: h5py.Group) -> ImpactZOutput:
        """
        Loads output from the given HDF5 group.

        Parameters
        ----------
        h5 : str or h5py.File
            The key to use when restoring the data.
        """
        loaded = _archive.restore_from_hdf5_file(h5)
        if not isinstance(loaded, ImpactZOutput):
            raise ValueError(
                f"Loaded {loaded.__class__.__name__} instead of a "
                f"ImpactZOutput instance.  Was the HDF group correct?"
            )
        return loaded
