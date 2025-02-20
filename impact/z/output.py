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

from .constants import DiagnosticType
from . import archive as _archive, parsers
from .input import HasOutputFile, ImpactZInput, WriteSliceInfo
from .plot import plot_stats_with_layout
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
    Meter_Rad,
    Meter_RadArray,
    SecondsArray,
    eVArray,
    eV_c_Array,
    Meters,
    MeV,
    MetersArray,
    Radians,
    RadiansArray,
    Unitless,
    UnitlessArray,
    known_unit,
    pmd_MeV,
)

try:
    from collections.abc import Mapping
except ImportError:
    pass

if typing.TYPE_CHECKING:
    import matplotlib.axes


logger = logging.getLogger(__name__)
file_number_to_cls: dict[DiagnosticType, dict[int, type[FortranOutputFileData]]] = {}
T = TypeVar("T", bound="FortranOutputFileData")


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
    *,
    reference_particle_mass: float,
    reference_frequency: float,
    diagnostic_type: DiagnosticType,
) -> tuple[dict[str, np.ndarray], dict[str, pmd_unit]]:
    stats = {}
    units = {}
    for fnum, cls in file_number_to_cls[diagnostic_type].items():
        fn = workdir / f"fort.{fnum}"
        if fn.exists():
            stats.update(cls.from_file(fn))
            for key, field in cls.model_fields.items():
                field_units = _units_from_metadata(field.metadata)
                if field_units == pmd_MeV:
                    field_units = known_unit["eV"]
                    stats[key] *= 1e6

                units[key] = field_units

    try:
        stats["energy_ref"] = stats["kinetic_energy_ref"] + reference_particle_mass
        stats["p0c"] = np.sqrt(stats["energy_ref"] ** 2.0 - reference_particle_mass**2)

        stats["mean_energy"] = stats["energy_ref"] - stats["neg_mean_rel_energy"]
        stats["mean_gamma"] = stats["mean_energy"] / reference_particle_mass

        stats["mean_px"] = stats["mean_px_over_p0"] * stats["p0c"]
        stats["mean_py"] = stats["mean_py_over_p0"] * stats["p0c"]
        stats["sigma_px"] = stats["sigma_px_over_p0"] * stats["p0c"]
        stats["sigma_py"] = stats["sigma_py_over_p0"] * stats["p0c"]
        stats["sigma_t"] = stats["sigma_phase_deg"] / 360.0 / reference_frequency

        stats["t_ref"] = stats["phase_ref"] / (2.0 * np.pi * reference_frequency)
        stats["mean_t_rel"] = stats["mean_phase_deg"] / 360.0 / reference_frequency
        stats["mean_t"] = stats["mean_t_rel"] + stats["t_ref"]

        betagamma = np.sqrt(stats["mean_gamma"] ** 2 - 1.0)

        if np.any(stats["norm_emit_x"] != 0.0):
            stats["twiss_beta_x"] = (
                stats["sigma_x"] ** 2 * betagamma / stats["norm_emit_x"]
            )
        if np.any(stats["norm_emit_y"] != 0.0):
            stats["twiss_beta_y"] = (
                stats["sigma_y"] ** 2 * betagamma / stats["norm_emit_y"]
            )
    except KeyError as ex:
        logger.warning(f"Some expected statistics unavailable? Missing: {ex}")

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


def _units_from_metadata(md):
    if not md:
        return

    for value in md:
        if isinstance(value, dict) and "units" in value:
            return value["units"]


class OutputStats(BaseModel):
    r"""
    Output statistics.

    Attributes
    ----------
    beta_ref : ndarray
        Beta of the reference particle.
    charge_state_n_particle : ndarray
        The number of particles for each charge state.
    gamma_ref : ndarray
        Reference particle gamma.
    kinetic_energy_ref : ndarray
        Reference particle kinetic energy. (eV)
    loadbalance_max_n_particle : ndarray
        Maximum number of particles on a processing element (PE).
    loadbalance_min_n_particle : ndarray
        Minimum number of particles on a processing element (PE).
    max_amplitude_energy_dev : ndarray
        Maximum energy deviation (eV).
    max_amplitude_gammabeta_x : ndarray
        Maximum Px (in radians).
    max_amplitude_gammabeta_y : ndarray
        Maximum Py (in radians).
    max_amplitude_phase : ndarray
        Maximum phase (in degrees).
    max_amplitude_x : ndarray
        Maximum X (in meters).
    max_amplitude_y : ndarray
        Maximum Y (in meters).
    max_r : ndarray
        Maximum radius (Rmax) in meters, measured from the axis of the pipe.
    mean_phase_deg : ndarray
        Mean phase (degrees)
    mean_px_over_p0 : ndarray
        Mean $px / p0$ (unitless).
    mean_py_over_p0 : ndarray
        Mean $py / p0$ (unitless).
    mean_x : ndarray
        Centroid location in the x-direction (meters).
    mean_y : ndarray
        Centroid location in the y-direction (meters).
    moment3_energy : ndarray
        Third-order central moment for energy deviation (eV).
    moment3_phase : ndarray
        Third-order central moment for phase (degree).
    moment3_px_over_p0 : ndarray
        Third-order central moment for Px (rad).
    moment3_py_over_p0 : ndarray
        Third-order central moment for Py (rad).
    moment3_x : ndarray
        Third-order central moment for x (meters).
    moment3_y : ndarray
        Third-order central moment for y (meters).
    moment4_energy : ndarray
        Fourth-order central moment for energy deviation (eV).
    moment4_phase : ndarray
        Fourth-order central moment for phase (degree).
    moment4_px_over_p0 : ndarray
        Fourth-order central moment for Px over p0 (unitless).
    moment4_py_over_p0 : ndarray
        Fourth-order central moment for Py over p0 (unitless).
    moment4_x : ndarray
        Fourth-order central moment for x (meters).
    moment4_y : ndarray
        Fourth-order central moment for y (meters).
    n_particle : ndarray
        Total number of particles in the bunch.
    neg_mean_rel_energy : ndarray
        Negative delta mean energy (eV).
    norm_emit_x : ndarray
        Normalized RMS emittance in x-direction (m-rad).
    norm_emit_y : ndarray
        Normalized RMS emittance in y-direction (m-rad).
    norm_emit_z : ndarray
        Normalized RMS emittance in z-direction (degree-eV).
    phase_ref : ndarray
        Absolute phase in radians.
    sigma_energy : ndarray
        Sigma energy (eV).
    sigma_phase_deg : ndarray
        Sigma phase (degrees).
    sigma_px_over_p0 : ndarray
        Sigma $px / p0$ (unitless).
    sigma_py_over_p0 : ndarray
        Sigma $py / p0$ (unitless).
    sigma_x : ndarray
        RMS size in the x-direction (meters).
    sigma_y : ndarray
        RMS size in the y-direction (meters).
    twiss_alpha_x : ndarray
        Twiss parameter alpha for x-direction (unitless).
    twiss_alpha_y : ndarray
        Twiss parameter alpha for y-direction (unitless).
    twiss_alpha_z : ndarray
        Twiss parameter alpha for z-direction (unitless).
    z : ndarray
        Z position (meters)
    max_abs_x : ndarray
        Maximum horizontal displacement from the beam axis: $\max(|x|)$ (meters)
    max_abs_px_over_p0 : ndarray
        Maximum $x$-plane transverse momentum $\max(|p_x/p_0|)$ (unitless)
    max_abs_y : ndarray
        Maximum vertical displacement from the beam axis $\max(|y|)$ (meters)
    max_abs_py_over_p0 : ndarray
        Maximum $y$-plane transverse momentum $\max(|p_y/p_0|)$ (unitless)
    max_phase : ndarray
        Maximum deviation in phase (degrees)
    max_energy_dev : ndarray
        Maximum deviation in particle energy (eV)
    mean_r : ndarray
        Mean radius (meters)
    sigma_r : ndarray
        RMS radius (meters)
    mean_r_90percent : ndarray
        90 percent mean radius (meters)
    mean_r_95percent : ndarray
        95 percent mean radius (meters)
    mean_r_99percent : ndarray
        99 percent mean radius (meters)
    max_r_rel : ndarray
        Maximum radius (meters)
    max_abs_gammabeta_x : ndarray
        Maximum $x$-plane transverse momentum $\max(|\gamma\beta_x|)$ (dimensionless)
    max_abs_gammabeta_y : ndarray
        Maximum $x$-plane transverse momentum $\max(|\gamma\beta_y|)$ (dimensionless)
    max_gamma_rel : ndarray
        Maximum deviation in relativistic gamma $\max(|\gamma - \gamma_0)|)$ (dimensionless)
    norm_emit_x_90percent : ndarray
        90% normalied RMS emittance (meter-rad)
    norm_emit_x_95percent : ndarray
        90% normalied RMS emittance (meter-rad)
    norm_emit_x_99percent : ndarray
        90% normalied RMS emittance (meter-rad)
    norm_emit_y_90percent : ndarray
        90% normalied RMS emittance (meter-rad)
    norm_emit_y_95percent : ndarray
        90% normalied RMS emittance (meter-rad)
    norm_emit_y_99percent : ndarray
        90% normalied RMS emittance (meter-rad)
    norm_emit_z_90percent : ndarray
        90% normalied RMS emittance (meter-rad)
    norm_emit_z_95percent : ndarray
        90% normalied RMS emittance (meter-rad)
    norm_emit_z_99percent : ndarray
        90% normalied RMS emittance (meter-rad)
    energy_ref : ndarray
        Energy reference (eV) (computed)
    mean_energy : ndarray
        Mean energy (eV) (computed)
    mean_gamma : ndarray
        Mean gamma (computed)
    mean_px : ndarray
        Mean px (eV/c) (computed)
    mean_py : ndarray
        Mean py (eV/c) (computed)
    mean_t : ndarray
        Mean time (s) (computed)
    mean_t_rel : ndarray
        Mean time relative (s) (computed)
    p0c : ndarray
        Momentum reference (eV) (computed)
    sigma_px : ndarray
        Sigma px (eV/c) (computed)
    sigma_py : ndarray
        Sigma py (eV/c) (computed)
    sigma_t : ndarray
        RMS size in time (rad) (computed)
    t_ref : ndarray
        Reference time (sec) (computed)
    twiss_beta_x : ndarray
        Twiss beta x (m) (computed)
    twiss_beta_y : ndarray
        Twiss beta y (m) (computed)
    units : dict[str, pmd_beamphysics.units.pmd_unit]
        Mapping of attribute name to pmd_unit.
    extra : dict[str, numpy.ndarray]
        Additional Impact-Z output data.  This is a future-proofing mechanism
        in case Impact-Z changes and LUME-Impact is not yet ready for it.
    """

    # Statistics largely unmodified from the data files
    # Some modifications include:
    #   1. Units changed to SI units (e.g., MeV->eV as noted)
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
    kinetic_energy_ref: eVArray = pydantic.Field(
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
    max_amplitude_energy_dev: eVArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Maximum energy deviation (eV)."
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
    mean_phase_deg: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean phase (degrees)",
    )
    mean_px_over_p0: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean $px / p0$ (unitless).",
    )
    mean_py_over_p0: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean $py / p0$ (unitless).",
    )
    mean_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Centroid location in the x-direction (meters).",
    )
    mean_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Centroid location in the y-direction (meters).",
    )
    moment3_energy: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for energy deviation (eV).",
    )
    moment3_phase: DegreesArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for phase (degree).",
    )
    moment3_px_over_p0: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for Px (rad).",
    )
    moment3_py_over_p0: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for Py (rad).",
    )
    moment3_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for x (meters).",
    )
    moment3_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Third-order central moment for y (meters).",
    )
    moment4_energy: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Fourth-order central moment for energy deviation (eV).",
    )
    moment4_phase: DegreesArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Fourth-order central moment for phase (degree).",
    )
    moment4_px_over_p0: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Fourth-order central moment for Px over p0 (unitless).",
    )
    moment4_py_over_p0: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Fourth-order central moment for Py over p0 (unitless).",
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
    neg_mean_rel_energy: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Negative delta mean energy (eV).",
        repr=False,
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
        description="Normalized RMS emittance in z-direction (degree-eV).",
    )
    phase_ref: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Absolute phase in radians."
    )
    sigma_energy: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma energy (eV).",
    )
    sigma_phase_deg: DegreesArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma phase (degrees).",
    )
    sigma_px_over_p0: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma $px / p0$ (unitless).",
    )
    sigma_py_over_p0: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma $py / p0$ (unitless).",
    )
    sigma_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="RMS size in the x-direction (meters).",
    )
    sigma_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="RMS size in the y-direction (meters).",
    )
    twiss_alpha_x: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss parameter alpha for x-direction (unitless).",
    )
    twiss_alpha_y: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss parameter alpha for y-direction (unitless).",
    )
    twiss_alpha_z: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss parameter alpha for z-direction (unitless).",
    )
    z: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Z position (meters)"
    )

    # Max amplitude standard
    max_abs_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Maximum horizontal displacement from the beam axis: $\max(|x|)$ (meters)",
    )
    max_abs_px_over_p0: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Maximum $x$-plane transverse momentum $\max(|p_x/p_0|)$ (unitless)",
    )
    max_abs_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Maximum vertical displacement from the beam axis: $\max(|y|)$ (meters)",
    )
    max_abs_py_over_p0: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Maximum $y$-plane transverse momentum $\max(|p_y/p_0|)$ (unitless)",
    )
    max_phase: DegreesArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Maximum deviation in phase (deg)",
    )
    max_energy_dev: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Maximum deviation in particle energy (eV)",
    )

    # File 29 - beam dist 3rd extended
    mean_r: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Mean radius (meters)"
    )
    sigma_r: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray, description="RMS radius (meters)"
    )
    mean_r_90percent: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray, description="90 percent mean radius (meters)"
    )
    mean_r_95percent: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray, description="95 percent mean radius (meters)"
    )
    mean_r_99percent: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray, description="99 percent mean radius (meters)"
    )
    max_r_rel: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Maximum radius (meters)"
    )

    # Max amplitude extended
    # max_abs_x: MetersArray = pydantic.Field(
    #     default_factory=_empty_ndarray,
    #     description=r"Maximum horizontal displacement from the beam axis  $\max(|x|)$ (meters)",
    # )
    max_abs_gammabeta_x: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Maximum $x$-plane transverse momentum $\max(|\gamma\beta_x|)$ (dimensionless)",
    )
    max_abs_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Maximum vertical displacement from the beam axis $\max(|y|)$ (meters)",
    )
    max_abs_gammabeta_y: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Maximum $x$-plane transverse momentum $\max(|\gamma\beta_y|)$ (dimensionless)",
    )
    max_phase: DegreesArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Maximum deviation in phase (degrees)",
    )
    max_gamma_rel: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Maximum deviation in relativistic gamma $\max(|\gamma - \gamma_0)|)$ (dimensionless)",
    )

    norm_emit_x_90percent: Meter_RadArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="90% normalied RMS emittance (meter-rad)",
    )
    norm_emit_x_95percent: Meter_RadArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="90% normalied RMS emittance (meter-rad)",
    )
    norm_emit_x_99percent: Meter_RadArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="90% normalied RMS emittance (meter-rad)",
    )
    norm_emit_y_90percent: Meter_RadArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="90% normalied RMS emittance (meter-rad)",
    )
    norm_emit_y_95percent: Meter_RadArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="90% normalied RMS emittance (meter-rad)",
    )
    norm_emit_y_99percent: Meter_RadArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="90% normalied RMS emittance (meter-rad)",
    )
    norm_emit_z_90percent: Meter_RadArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="90% normalied RMS emittance (meter-rad)",
    )
    norm_emit_z_95percent: Meter_RadArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="90% normalied RMS emittance (meter-rad)",
    )
    norm_emit_z_99percent: Meter_RadArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="90% normalied RMS emittance (meter-rad)",
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
    mean_gamma: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean gamma (computed)",
    )
    mean_px: eV_c_Array = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean px (eV/c) (computed)",
    )
    mean_py: eV_c_Array = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean py (eV/c) (computed)",
    )
    mean_t: SecondsArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean time (s) (computed)",
    )
    mean_t_rel: UnitlessArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Mean time relative (s) (computed)",
    )
    p0c: eVArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Momentum reference (eV) (computed)",
    )
    sigma_px: eV_c_Array = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma px (eV/c) (computed)",
    )
    sigma_py: eV_c_Array = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Sigma py (eV/c) (computed)",
    )
    sigma_t: SecondsArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="RMS size in time (rad) (computed)",
    )
    t_ref: RadiansArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Reference time (sec) (computed)",
    )
    twiss_beta_x: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss beta x (m) (computed)",
    )
    twiss_beta_y: MetersArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss beta y (m) (computed)",
    )

    units: dict[str, PydanticPmdUnit] = pydantic.Field(
        default_factory=dict,
        repr=False,
        description="Mapping of attribute name to pmd_unit.",
    )
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
        *,
        reference_particle_mass: float,
        reference_frequency: float,
        diagnostic_type: DiagnosticType,
    ) -> OutputStats:
        stats, units = load_stat_files_from_path(
            workdir,
            reference_frequency=reference_frequency,
            diagnostic_type=diagnostic_type,
            reference_particle_mass=reference_particle_mass,
        )

        extra = _split_extra(cls, stats)
        return OutputStats(units=units, extra=extra, **stats)


class FortranOutputFileData(SequenceBaseModel):
    """
    Base class for representing a single row of file data from `fort.{file_id}`.

    Subclasses of this are used to:
    1. Match output file IDs with the type of data they contain
    2. Give names to each column
    3. Track units (by way of annotated attributes)
    """

    def __init_subclass__(
        cls,
        file_id: int,
        diagnostic_types: tuple[DiagnosticType, ...] | DiagnosticType = (
            DiagnosticType.standard,
            DiagnosticType.extended,
        ),
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        assert isinstance(file_id, int)
        assert file_id not in file_number_to_cls, f"Duplicate element ID {file_id}"

        if isinstance(diagnostic_types, DiagnosticType):
            diagnostic_types = (diagnostic_types,)

        for diagnostic_type in diagnostic_types:
            file_number_to_cls.setdefault(diagnostic_type, {})
            file_number_to_cls[diagnostic_type][file_id] = cls

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

    Attributes
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
        Mean $px / p0$ (unitless)
    sigma_px_over_p0 : float
        Sigma $px / p0$ (unitless)
    twiss_alpha_x : float
        Twiss parameter, alpha
    norm_emit_x : float
        normalized RMS emittance [m-rad]
    norm_emit_x_90percent : float
        90% normalized RMS horizontal emittance (meter-rad)
        Only available in `diagnostic_types=extended` mode.
    norm_emit_x_95percent : float
        95% normalized RMS horizontal emittance (meter-rad)
        Only available in `diagnostic_types=extended` mode.
    norm_emit_x_99percent : float
        99% normalized RMS horizontal emittance (meter-rad)
        Only available in `diagnostic_types=extended` mode.
    """

    z: Meters
    mean_x: Meters
    sigma_x: Meters
    mean_px_over_p0: Unitless
    sigma_px_over_p0: Unitless
    twiss_alpha_x: Meters
    norm_emit_x: Meters  # m-rad

    # Available in 'extended' diagnostic_type output:
    norm_emit_x_90percent: Meter_Rad = 0.0
    norm_emit_x_95percent: Meter_Rad = 0.0
    norm_emit_x_99percent: Meter_Rad = 0.0


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
        Mean $py / p0$ [unitless]
    sigma_py_over_p0 : float
        $py / p0$ [unitless]
    twiss_alpha_y : float
        Twiss parameter, alpha
    norm_emit_y : float
        normalized RMS emittance [m-rad]
    norm_emit_y_90percent : float
        90% normalized RMS vertical emittance (meter-rad)
        Only available in `diagnostic_type=extended` mode.
    norm_emit_y_95percent : float
        95% normalized RMS vertical emittance (meter-rad)
        Only available in `diagnostic_type=extended` mode.
    norm_emit_y_99percent : float
        99% normalized RMS vertical emittance (meter-rad)
        Only available in `diagnostic_type=extended` mode.
    """

    z: Meters
    mean_y: Meters
    sigma_y: Meters
    mean_py_over_p0: Unitless
    sigma_py_over_p0: Unitless
    twiss_alpha_y: Meters
    norm_emit_y: Meters  # m-rad

    # Available in 'extended' diagnostic_type output:
    norm_emit_y_90percent: Meter_Rad = 0.0
    norm_emit_y_95percent: Meter_Rad = 0.0
    norm_emit_y_99percent: Meter_Rad = 0.0


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
    neg_mean_rel_energy : float
        Negative delta mean energy, used to convert to mean energy [eV]
        where `neg_mean_rel_energy = (kinetic_energy_ref - mean_energy) + reference_particle_mass `
        In the file, this is stored as MeV and LUME-Impact converts to eV automatically.
    sigma_energy : float
        RMS momentum [eV]
        In the file, this is stored as MeV and LUME-Impact converts to eV automatically.
    twiss_alpha_z : float
        Twiss parameter, alpha
    norm_emit_z : float
        normalized RMS emittance [degree-MeV]
    norm_emit_z_90percent : float
        90% normalized RMS longitudinal emittance (degree-MeV)
        Only available in `diagnostic_type=extended` mode.
    norm_emit_z_95percent : float
        95% normalized RMS longitudinal emittance (degree-MeV)
        Only available in `diagnostic_type=extended` mode.
    norm_emit_z_99percent : float
        99% normalized RMS longitudinal emittance (degree-MeV)
        Only available in `diagnostic_type=extended` mode.
    """

    z: Meters
    mean_phase_deg: Degrees
    sigma_phase_deg: Degrees
    neg_mean_rel_energy: MeV
    sigma_energy: MeV
    twiss_alpha_z: Unitless
    norm_emit_z: Meters

    # Available in 'extended' diagnostic_type output:
    norm_emit_z_90percent: Meter_Rad = 0.0
    norm_emit_z_95percent: Meter_Rad = 0.0
    norm_emit_z_99percent: Meter_Rad = 0.0


class MaxAmplitudeStandard(
    FortranOutputFileData,
    file_id=27,
    diagnostic_types=DiagnosticType.standard,
):
    r"""
    File fort.27: maximum amplitude information (standard)

    Attributes
    ----------
    z : float
        Longitudinal position along the beamline (meters)
    max_abs_x : float
        Maximum horizontal displacement from the beam axis: $\max(|x|)$ (meters)
    max_abs_px_over_p0 : float
        Maximum $x$-plane transverse momentum $\max(|p_x/p_0|)$ (unitless)
    max_abs_y : float
        Maximum vertical displacement from the beam axis: $\max(|y|)$ (meters)
    max_abs_py_over_p0 : float
        Maximum $y$-plane transverse momentum $\max(|p_y/p_0|)$ (unitless)
    max_phase : float
        Maximum deviation in phase (deg)
    max_energy_dev : float
        Maximum deviation in particle energy (MeV)
    """

    z: Meters
    max_abs_x: Meters
    max_abs_px_over_p0: Unitless
    max_abs_y: Meters
    max_abs_py_over_p0: Unitless
    max_phase: Degrees
    max_energy_dev: MeV


class MaxAmplitudeExtended(
    FortranOutputFileData,
    file_id=27,
    diagnostic_types=DiagnosticType.extended,
):
    r"""
    File fort.27: maximum amplitude information (extended)

    Attributes
    ----------
    z : float
        longitudinal position along the beamline (meters)
    max_abs_x : float
         Maximum horizontal displacement from the beam axis  $\max(|x|)$ (meters)
    max_abs_gammabeta_x : float
         Maximum $x$-plane transverse momentum $\max(|\gamma\beta_x|)$ (dimensionless)
    max_abs_y : float
         Maximum vertical displacement from the beam axis $\max(|y|)$ (meters)
    max_abs_gammabeta_y : float
         Maximum $x$-plane transverse momentum $\max(|\gamma\beta_y|)$ (dimensionless)
    max_phase : float
         Maximum deviation in phase (deg???)
    max_gamma_rel : float
        Maximum deviation in relativistic gamma $\max(|\gamma - \gamma_0)|)$ (dimensionless)
    """

    z: Meters
    max_abs_x: Meters
    max_abs_gammabeta_x: Unitless
    max_abs_y: Meters
    max_abs_gammabeta_y: Unitless
    max_phase: Degrees  # really?
    max_gamma_rel: Unitless


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
    """

    z: Meters
    loadbalance_min_n_particle: Unitless
    loadbalance_max_n_particle: Unitless
    n_particle: Unitless


class BeamDistribution3rdStandard(
    FortranOutputFileData,
    file_id=29,
    diagnostic_types=DiagnosticType.standard,
):
    r"""
    File fort.29: cubic root of 3rd moments of the beam distribution

    Attributes
    ----------
    z : float
        z distance (m)
    moment3_x : float
        Cubic root of the third moment of the horizontal position $M_3(x)$ (meters)
    moment3_px_over_p0 : float
        Cubic root of the third moment of the horizontal momentum $M_3(p_x/p_0)$ (unitless)
    moment3_y : float
        Cubic root of the third moment of the vertical position $M_3(y)$ (meters)
    moment3_py_over_p0 : float
        Cubic root of the third moment of the vertical momentum $M_3(p_y/p_0)$ (unitless)
    moment3_phase : float
        Cubic root of the third moment of the phase $M_3(\phi)$ (deg)
    moment3_energy : float
        Cubic root of the energy $M_3(E)$ (MeV)
        In the file, this is stored as MeV and LUME-Impact converts to eV automatically.
    """

    z: Meters
    moment3_x: Meters
    moment3_px_over_p0: Radians
    moment3_y: Meters
    moment3_py_over_p0: Radians
    moment3_phase: Degrees
    moment3_energy: MeV


class BeamDistribution3rdExtended(
    FortranOutputFileData,
    file_id=29,
    diagnostic_types=DiagnosticType.extended,
):
    """
    File fort.29: contains radius moments of the beam distribution.

    Attributes
    ----------
    z : float
        z distance (m)
    mean_r : float
        Mean radius (meters)
    sigma_r : float
        RMS radius (meters)
    mean_r_90percent : float
        90 percent mean radius (meters)
    mean_r_95percent : float
        95 percent mean radius (meters)
    mean_r_99percent : float
        99 percent mean radius (meters)
    max_r_rel : float
        Maximum radius (meters)
    """

    z: Meters
    mean_r: Meters
    sigma_r: Meters
    mean_r_90percent: Meters
    mean_r_95percent: Meters
    mean_r_99percent: Meters
    max_r_rel: Meters


class BeamDistribution4th(
    FortranOutputFileData,
    diagnostic_types=DiagnosticType.standard,
    file_id=30,
):
    r"""
    File fort.30 with diagnostic_type=1 contains the cubic root of the third moments
    of the beam distribution.

    Here $ M_4(x) \equiv\left< (x-\left< x \right>)^4 \right>^{1/4} $,
    averaging over all particles.

    Attributes
    ----------
    z : float
        Longitudinal position along the beamline (meters)
    moment4_x : float
        Fourth root of the third moment of the horizontal position $M_4(x)$ (meters)
    moment4_px_over_p0 : float
        Fourth root of the third moment of the horizontal momentum $M_4(p_x/p_0)$ (rad)
    moment4_y : float
        Fourth root of the third moment of the vertical position $M_4(y)$ (meters)
    moment4_py_over_p0 : float
        Fourth root of the third moment of the vertical momentum $M_4(p_y/p_0)$ (rad)
    moment4_phase : float
        Fourth root of the third moment of the phase $M_4(\phi)$ (deg)
    moment4_energy : float
        Fourth root of the energy $M_4(E)$ (MeV)
        In the file, this is stored as MeV and LUME-Impact converts to eV automatically.
    """

    z: Meters
    moment4_x: Meters
    moment4_px_over_p0: Radians
    moment4_y: Meters
    moment4_py_over_p0: Radians
    moment4_phase: Degrees  # ?
    moment4_energy: MeV


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
            "mean_z": "z",
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
    reference_frequency: float = 0.0
    reference_species: str = pydantic.Field(
        default="",
        description="Reference particle species",
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
        Statistics array from `.stats`.
        """
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
    ) -> ImpactZOutput:
        """
        Load ImpactZ output based on the configured input settings.

        Returns
        -------
        ImpactZOutput
        """

        species = input.reference_species
        stats = OutputStats.from_stats_files(
            workdir,
            reference_frequency=input.reference_frequency,
            diagnostic_type=input.diagnostic_type,
            reference_particle_mass=input.reference_particle_mass,
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

        for key, fld in OutputStats.model_fields.items():
            unit = _units_from_metadata(fld.metadata)
            if unit:
                units[key] = unit

        return cls(
            stats=stats,
            key_to_unit=units,
            particles=particles,
            particles_raw=particles_raw,
            reference_frequency=input.reference_frequency,
            slices=slices,
            reference_species=species,
        )

    def plot(
        self,
        y: str | Sequence[str] = ("sigma_x", "sigma_y"),
        x: str = "z",
        *,
        y2: str | Sequence[str] = (),
        input: ImpactZInput | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        ylim2: tuple[float, float] | None = None,
        nice: bool = True,
        tex: bool = True,
        include_layout: bool = True,
        include_labels: bool = True,
        include_markers: bool = True,
        include_particles: bool = True,
        include_legend: bool = True,
        return_figure: bool = False,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ):
        """ """

        if "ykeys2" in kwargs:
            y2 = kwargs.pop("ykeys2")

        if input is None:
            # Should we warn?
            include_layout = False

        return plot_stats_with_layout(
            self,
            ykeys=y,
            ykeys2=y2,
            xkey=x,
            xlim=xlim,
            ylim=ylim,
            ylim2=ylim2,
            input=input,
            nice=nice,
            tex=tex,
            include_layout=include_layout,
            include_labels=include_labels,
            # include_field=include_field,
            include_markers=include_markers,
            include_particles=include_particles,
            include_legend=include_legend,
            return_figure=return_figure,
            ax=ax,
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
