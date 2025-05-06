from __future__ import annotations

import logging
import math
import pathlib
import tempfile
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    NamedTuple,
    Sequence,
    TypedDict,
    Union,
    cast,
)


import matplotlib.pyplot as plt
import numpy as np
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.particles import c_light
from pmd_beamphysics.species import charge_state, mass_of
from pytao import Tao, TaoCommandError
from typing_extensions import Literal, TypeAlias

from ..constants import (
    BoundaryType,
    DistributionType,
    GPUFlag,
    IntegratorType,
    MultipoleType,
    DiagnosticType,
    WigglerType,
)

from ...interfaces.bmad import ele_info, tao_unique_names
from .. import Drift, ImpactZInput
from ..fieldmaps import make_solenoid_rfcavity_rfdata_simple
from ..input import (
    CCL,
    AnyInputElement,
    CollimateBeam,
    Dipole,
    InputElementMetadata,
    IntegratorTypeSwitch,
    Multipole,
    Quadrupole,
    RotateBeam,
    Solenoid,
    SolenoidWithRFCavity,
    SuperconductingCavity,
    ToggleSpaceCharge,
    Wiggler,
    WriteFull,
)

if TYPE_CHECKING:
    from .. import ImpactZ


logger = logging.getLogger(__name__)
Which = Literal["model", "base", "design"]
DRIFT_ELEMENT_KEYS = {
    "drift",
    "pipe",
    "monitor",
    "instrument",
    "ecollimator",
    "rcollimator",
}


class UnusableElementError(Exception): ...


class UnsupportedElementError(Exception): ...


TaoInfoDict: TypeAlias = dict[str, Union[str, float, int]]


def ele_methods(tao: Tao, ele: str | int, which: str = "model") -> TaoInfoDict:
    return cast(TaoInfoDict, tao.ele_methods(ele, which=which))


def ele_head(tao: Tao, ele: str | int, which: str = "model") -> TaoInfoDict:
    return cast(TaoInfoDict, tao.ele_head(ele, which=which))


def ele_csr_enabled(
    ele_methods_info: TaoInfoDict,
    global_csr_flag: bool = False,
) -> bool:
    """
    Determine if Coherent Synchrotron Radiation (CSR) is enabled for an element.

    Parameters
    ----------
    ele_methods_info : TaoInfoDict
        Dictionary containing element methods information, from `ele_methods`
    global_csr_flag : bool, optional
        Tao's global CSR flag that must be True for CSR to be enabled, by
        default False.

    Returns
    -------
    bool
    """
    if not global_csr_flag:
        return False

    csr_method = ele_methods_info.get("csr_method", "")
    if not csr_method:
        return False

    return str(csr_method).lower() == "1_dim"


def ele_space_charge_enabled(
    ele_methods_info: TaoInfoDict,
    global_csr_flag: bool = False,
) -> bool:
    if not global_csr_flag:
        return False

    space_charge_method = ele_methods_info.get("space_charge_method", "Off")
    return str(space_charge_method).lower() != "off"


def get_element_index(
    tao: Tao,
    ele: str | int,
) -> int:
    """
    Get the index of a specified element from Tao.

    Parameters
    ----------
    tao : pytao.Tao
        The pytao Tao instance.
    ele : str
        The element as a string.

    Returns
    -------
    int
    """
    head = ele_head(tao, ele)
    return int(head["ix_ele"])


def get_index_to_name(
    tao: Tao,
    track_start: str | int | None = None,
    track_end: str | int | None = None,
    ix_uni: int = 1,
    ix_branch: int = 0,
) -> dict[int, str]:
    """
    Get a mapping from element indices to element names.

    Parameters
    ----------
    tao : Tao
        The Tao object to query.
    track_start : str or int or None, optional
        The start element index or name. If None, uses the first element.
    track_end : str or int or None, optional
        The end element index or name. If None, uses the last element.

    Returns
    -------
    dict[int, str]
        A dictionary mapping element indices to element names, filtered by the specified range.
    """
    idx_to_name = tao_unique_names(tao, ix_uni=ix_uni, ix_branch=ix_branch)

    indices = cast(
        list[str],
        tao.lat_list("*", "ele.ix_ele", ix_uni=str(ix_uni), ix_branch=str(ix_branch)),
    )
    ix_first = int(indices[0])
    ix_last = int(indices[-1])

    ix_start = get_element_index(tao, track_start) if track_start else ix_first
    ix_end = get_element_index(tao, track_end) if track_end else ix_last
    return {
        ix_ele: name
        for ix_ele, name in idx_to_name.items()
        if ix_start <= ix_ele <= ix_end
    }


def get_ele_indices_by_pattern(
    tao: Tao,
    patterns: Sequence[str] | str,
) -> list[int]:
    if isinstance(patterns, str):
        patterns = [patterns]

    def all_ids():
        for pattern in patterns:
            ids = cast(
                Iterable[int],  # NDArray[int]
                tao.lat_list(pattern, "ele.ix_ele", flags="-array_out -no_slaves"),
            )
            yield from list(ids)

    return sorted(set(all_ids()))


def export_particles(tao: Tao, ele_id: str | int):
    """
    Export particles for a given element to an HDF5 file.

    Parameters
    ----------
    tao : Tao
    ele_id : str
        The element for which the particles are to be exported.

    Returns
    -------
    ParticleGroup
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = pathlib.Path(tmpdir) / "particles.h5"
        logger.debug(f"Writing {ele_id} particles to: {fn}")
        tao.cmd(f"write beam -at {ele_id} {fn}")
        return ParticleGroup(h5=str(fn))


def print_ele_info(ele_id: str | int, ele_info: dict[str, Any]) -> None:
    print()
    print(f"Element: {ele_id}")
    print("-----------------")
    for key, value in sorted(ele_info.items()):
        if key.startswith("units#") or key.startswith("has#"):
            continue
        units = ele_info.get(
            f"units#{key}",
            "(units?)" if isinstance(value, float) else "",
        )
        print(f"{key}: {value} {units}")


def get_element_radius(*limits: float, default=0.03) -> float:
    """
    Calculate the maximum radius from the given limits.

    Parameters
    ----------
    *limits : float
        Variable-length argument list of floats representing possible
        limits for the radius limits. At least one limit must be
        provided unless relying on the default value.
    default : float, optional
        The default radius to return if none of the provided limits
        are greater than this value. The default is 0.03 m.

    Returns
    -------
    float
        The maximum value from the provided limits or the default value
        if no valid maximum is found.
    """
    value = max(limits)
    return value or default


class EleMultipoles(TypedDict):
    multipoles_on: bool
    scale_multipoles: bool
    data: list[dict[str, float | int]]


class MultipoleOrder(IntEnum):
    dipole = 1
    quadrupole = 2
    octupole = 3
    decapole = 4


class MultipoleInfo(NamedTuple):
    order: MultipoleOrder
    Bn: float


def get_multipole_info(tao: Tao, ele_id: str | int) -> MultipoleInfo | None:
    """
    Retrieves multipole information for a specified element in Tao.

    Parameters
    ----------
    tao : Tao
        The Tao interface object.
    ele_id : str or int
        The element identifier, either as a string name or integer index.

    Returns
    -------
    MultipoleInfo or None

    Raises
    ------
    ValueError
        If more than one multipole is found, or other unsupported scenarios are
        encountered.
    """
    info = cast(EleMultipoles, tao.ele_multipoles(ele_id))
    data = info["data"]
    if not data or not info["multipoles_on"]:
        return None

    if len(data) > 1:
        raise ValueError("Only one multipole allowed")

    if info["scale_multipoles"]:
        raise ValueError("scale_multipoles not supported")

    d0 = data[0]
    if d0["An"] != 0.0:
        raise ValueError("An of 0 only supported for multipoles for now")

    order = MultipoleOrder(d0["index"])
    return MultipoleInfo(order=order, Bn=d0["Bn"])  # May need another factor


CavityClass: TypeAlias = Union[
    type[SuperconductingCavity], type[SolenoidWithRFCavity], type[CCL]
]


def get_cavity_class(tracking_method: str, cavity_type: str) -> CavityClass:
    """
    Determine the appropriate cavity class based on tracking method and cavity type.

    Parameters
    ----------
    tracking_method : str
        The tracking method to use for beam dynamics calculation.
        Supported values include 'bmad_standard', 'runge_kutta', 'time_runge_kutta'.
    cavity_type : str
        The type of RF cavity.
        Supported values are 'standing_wave' and 'traveling_wave'.

    Returns
    -------
    type[SuperconductingCavity] | type[SolenoidWithRFCavity] | type[CCL]
        The appropriate cavity class implementation based on the provided tracking method
        and cavity type.

    Raises
    ------
    NotImplementedError
        If no mapping exists for the given combination of tracking method and cavity type.
    """
    cavity_type = cavity_type.lower()
    tracking_method = tracking_method.lower()

    if cavity_type == "standing_wave":
        if tracking_method in {"bmad_standard"}:
            return CCL
            # return SuperconductingCavity
        if tracking_method in {"runge_kutta", "time_runge_kutta"}:
            return SolenoidWithRFCavity
    elif cavity_type == "traveling_wave":
        if tracking_method in {"bmad_standard"}:
            return CCL
    raise NotImplementedError(
        f"No mapping of cavity type for {tracking_method=} {cavity_type=}"
    )


# def pad_solenoid_with_rf_cavity(ele: SolenoidWithRFCavity, n_cells: int):
#     if ele.length > (ele.rf_wavelength/2. * n_cells):


def single_element_from_tao_info(
    ele_id: str | int,
    *,
    info: TaoInfoDict,
    ele_methods_info: TaoInfoDict,
    multipole_info: MultipoleInfo | None,
    name: str = "",
    global_csr_flag: bool = False,
    species: str = "electron",
    integrator_type: IntegratorType = IntegratorType.linear_map,
    ref_time_start: float | None = None,
    has_superpositions: bool = False,
) -> tuple[AnyInputElement, np.ndarray | None] | None:
    """
    Convert a Tao element into its corresponding basic Impact input element.

    This does not include collimation elements, integrator type changes, and so on.
    For that functionality, see `element_from_tao`.

    Parameters
    ----------
    ele_id : str or int
        Element identifier.
    info : dict
        Dictionary from `ele_info`.
    ele_methods_info : dict
        Dictionary from `ele_methods`.
    multipole_info : MultipoleInfo or None
        Multipole information for the element, if available.
    name : str, optional
        Name of the element, by default "".
    global_csr_flag : bool, optional
        Whether CSR is enabled globally in Tao or not, by default False.
    species : str, optional
        Particle species, by default "electron".
    integrator_type : IntegratorType, optional
        Type of integrator, by default `IntegratorType.linear_map`.
    ref_time_start : float or None, optional
        Reference time start, by default None.
    has_superpositions : bool, optional
        Whether the element has superpositions, by default False.

    Returns
    -------
    tuple[AnyInputElement, np.ndarray or None] or None
        Tuple of the input element and associated RF data, or None if
        no IMPACT-Z element should be generated (i.e., for markers and such).
    """
    key = str(info["key"]).lower()

    length = float(info["L"])
    x1_limit = float(info.get("X1_LIMIT", 0.0))
    x2_limit = float(info.get("X2_LIMIT", 0.0))
    y1_limit = float(info.get("Y1_LIMIT", 0.0))
    y2_limit = float(info.get("Y2_LIMIT", 0.0))
    rotation_error_x = float(info.get("X_PITCH_TOT", 0.0))
    rotation_error_y = float(info.get("Y_PITCH_TOT", 0.0))
    rotation_error_z = -float(info.get("TILT_TOT", 0.0))
    num_steps = int(info.get("NUM_STEPS", 10))
    radius = get_element_radius(x1_limit, x2_limit, y1_limit, y2_limit, default=0.03)
    csr = ele_csr_enabled(ele_methods_info, global_csr_flag)
    space_charge = ele_space_charge_enabled(ele_methods_info, global_csr_flag)
    metadata: InputElementMetadata = {"bmad_csr": csr, "bmad_sc": space_charge}

    if all(key in info for key in ("Y_PITCH_TOT", "X_OFFSET_TOT", "Y_OFFSET_TOT")):
        offset_x = (
            float(info["X_OFFSET_TOT"])
            + math.sin(float(info["X_PITCH_TOT"])) * length / 2.0
        )
        offset_y = (
            float(info["Y_OFFSET_TOT"])
            - math.sin(float(info["Y_PITCH_TOT"])) * length / 2.0
        )
    else:
        offset_x = 0.0
        offset_y = 0.0

    if key in DRIFT_ELEMENT_KEYS:
        return Drift(
            length=length,
            name=name,
            steps=num_steps,
            map_steps=num_steps,
            radius=1.0,  # no such thing in bmad, right?
            metadata=metadata,
        ), None

    if key in {"hkicker", "vkicker", "kicker"}:
        kick = max(  # integrated field kick in m-T
            (
                np.abs(info.get("BL_KICK", 0.0)),
                np.abs(info.get("BL_VKICK", 0.0)),
                np.abs(info.get("BL_HKICK", 0.0)),
            )
        )
        if kick > 0.0:
            raise NotImplementedError(
                "Kickers with integrated field kick are not supported (bl_kick, bl_vkick, bl_hkick)"
            )
        return Drift(
            length=length,
            name=name,
            steps=num_steps,
            map_steps=num_steps,
            radius=1.0,
            metadata=metadata,
        ), None

    if key == "sbend":
        angle = float(info["ANGLE"])

        if angle == 0.0:
            # Dipoles in Impact-Z don't work if they have zero angle
            return Drift(
                length=length,
                name=name,
                steps=num_steps,
                map_steps=num_steps,
                radius=1.0,
                metadata=metadata,
            ), None

        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for SBend")

        fringe_type = info["FRINGE_TYPE"]
        if fringe_type != "Full":
            logger.warning(
                f"Element #{ele_id} has a fringe type of {fringe_type!r}; to match with "
                f"Impact-Z, this should be 'Full'"
            )

        return Dipole(
            name=name,
            length=length,
            steps=num_steps,
            map_steps=num_steps,
            angle=angle,  # rad
            k1=float(info["K1"]),
            input_switch=201.0 if csr else 0.0,
            hgap=float(info["HGAP"]),
            e1=float(info["E1"]),
            e2=float(info["E2"]),
            entrance_curvature=0.0,
            exit_curvature=0.0,
            fint=float(info["FINT"]),
            # misalignment_error_x=info["X_OFFSET_TOT"],
            # misalignment_error_y=info["Y_OFFSET_TOT"],
            # rotation_error_x=rotation_error_x,
            # rotation_error_y=rotation_error_y,
            # rotation_error_z=rotation_error_z,
            metadata=metadata,
        ), None

    if key in {"sextupole", "octupole", "thick_multipole"}:
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for multipoles")

        if key in {"sextupole", "octupole"}:
            multipole_type = MultipoleType[key]
            # confirmed by Ji as T/m^n (1/30/2025)
            field_strength_key = {
                "sextupole": "B2_GRADIENT",
                "octupole": "B3_GRADIENT",
                # "decapole": "k4",
                # "dodecapole": "k5",
            }[key]
            field_strength = float(info[field_strength_key])
        else:
            if multipole_info is None:
                raise RuntimeError("thick_multipole has no ele:multipoles information")

            multipole_type = MultipoleType.decapole
            b4_gradient = (
                (4.0 * 3.0 * 2.0)  # 4!
                * charge_state(species)
                * multipole_info.Bn
                * float(info["P0C"])
                / c_light
                / length
            )
            field_strength = b4_gradient

        return Multipole(
            name=name,
            length=length,
            steps=num_steps,
            map_steps=num_steps,
            multipole_type=multipole_type,
            field_strength=field_strength,
            file_id=-1,  # TODO?
            radius=radius,
            misalignment_error_x=offset_x,
            misalignment_error_y=offset_y,
            rotation_error_x=rotation_error_x,
            rotation_error_y=rotation_error_y,
            rotation_error_z=rotation_error_z,
            metadata=metadata,
        ), None

    if key == "wiggler":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Wiggler")

        n_period = int(info["N_PERIOD"])
        num_steps = max((num_steps, 10 * n_period))

        return Wiggler(
            name=name,
            length=length,
            steps=num_steps,
            map_steps=num_steps,
            wiggler_type=WigglerType.planar,
            max_field_strength=float(info["B_MAX"]),
            period=float(info["L_PERIOD"]),
            kx=float(info["KX"]),
            file_id=-1,  # TODO?
            radius=radius,
            misalignment_error_x=offset_x,
            misalignment_error_y=offset_y,
            rotation_error_x=rotation_error_x,
            rotation_error_y=rotation_error_y,
            rotation_error_z=rotation_error_z,
            metadata=metadata,
        ), None

    if key == "quadrupole":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Quadrupole")

        k1 = {
            IntegratorType.linear_map: float(info["K1"]),
            IntegratorType.runge_kutta: float(info["B1_GRADIENT"]),
        }[integrator_type]
        return Quadrupole(
            name=name,
            length=length,
            steps=num_steps,
            map_steps=num_steps,
            # The gradient of the quadrupole magnetic field, measured in Tesla per meter.
            k1=k1,
            # file_id : float
            #     An ID for the input gradient file. Determines profile behavior:
            #     if greater than 0, a fringe field profile is read; if less than -10,
            #     a linear transfer map of an undulator is used; if between -10 and 0,
            #     it's the k-value linear transfer map; if equal to 0, it uses the linear
            #     transfer map with the gradient.
            file_id=-1,
            # The radius of the quadrupole, measured in meters.
            radius=radius,  # TODO is this the aperture radius?
            misalignment_error_x=offset_x,
            misalignment_error_y=offset_y,
            rotation_error_x=rotation_error_x,
            rotation_error_y=rotation_error_y,
            rotation_error_z=rotation_error_z,
            metadata=metadata,
        ), None
    if key == "solenoid":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Solenoid")

        return Solenoid(
            name=name,
            length=length,
            steps=num_steps,
            map_steps=num_steps,
            Bz0=float(info["BS_FIELD"]),
            file_id=-1,  # TODO?
            radius=radius,  # TODO arbitrary
            misalignment_error_x=offset_x,
            misalignment_error_y=offset_y,
            rotation_error_x=rotation_error_x,
            rotation_error_y=rotation_error_y,
            rotation_error_z=rotation_error_z,
            metadata=metadata,
        ), None

    if key == "lcavity":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Lcavity")

        cls = get_cavity_class(
            cavity_type=str(info["CAVITY_TYPE"]).lower(),
            tracking_method=str(ele_methods_info["tracking_method"]).lower(),
        )

        if cls is CCL or cls is SuperconductingCavity:
            if cls is CCL and np.abs(offset_x) > 0:
                logger.warning(f"{offset_x=} for CCL element {name!r} may not work")
            if cls is CCL and np.abs(offset_y) > 0:
                logger.warning(f"{offset_y=} for CCL element {name!r} may not work")
            return cls(
                name=name,
                length=length,
                steps=num_steps,
                map_steps=num_steps,
                file_id=-1.0,  # TODO: same for all cavity types?
                rf_frequency=float(info["RF_FREQUENCY"]),
                phase_deg=float(info["PHI0"]) * 360.0,
                radius=radius,  # TODO is this the aperture radius?
                field_scaling=float(info["GRADIENT"]),
                misalignment_error_x=offset_x,
                misalignment_error_y=offset_y,
                rotation_error_x=rotation_error_x,
                rotation_error_y=rotation_error_y,
                rotation_error_z=-rotation_error_z,
                metadata=metadata,
            ), None
        if cls is SolenoidWithRFCavity:
            if has_superpositions:
                raise NotImplementedError(
                    f"Superpositions are not yet supported (seen in element {ele_id} {info['name']} of key {info['key']})"
                )

            if ref_time_start is None:
                raise ValueError(f"ref_time_start required for {cls}")

            phi0 = float(info["PHI0"])
            rf_frequency = float(info["RF_FREQUENCY"])
            # rf_wavelength = c_light / rf_frequency
            phi0_autoscale = float(info["PHI0_AUTOSCALE"])
            phi0_ref = rf_frequency * ref_time_start
            n_cell = int(info["N_CELL"])
            L_active = float(info["L_ACTIVE"])
            L_pad = (length - L_active) / 2

            beta0 = float(info["P0C_START"]) / float(info["E_TOT_START"])

            # Extra phi0 due to padding
            phi0_pad = -L_pad / (beta0 * c_light) * rf_frequency

            rf_data = make_solenoid_rfcavity_rfdata_simple(
                rf_frequency=rf_frequency,
                n_cell=n_cell,
                L_pad=L_pad,
            )
            return cls(
                name=name,
                length=length,
                steps=n_cell * 36,  # heuristic that seems to work
                map_steps=num_steps,
                file_id=1.0,
                rf_frequency=rf_frequency,
                phase_deg=(phi0 + phi0_autoscale - phi0_ref + 0.25 + phi0_pad) * 360.0,
                radius=radius,
                field_scaling=-2.0 * float(info["GRADIENT"]) * length / L_active,
                misalignment_error_x=offset_x,
                misalignment_error_y=offset_y,
                rotation_error_x=rotation_error_x,
                rotation_error_y=rotation_error_y,
                rotation_error_z=-rotation_error_z,
                aperture_size_for_wakefield=0.0,
                bz0=0.0,
                gap_size_for_wakefield=0.0,
                length_for_wakefield=0.0,
                metadata=metadata,
            ), rf_data
        raise RuntimeError(f"Unexpected cavity type: {cls=}")

    if length > 0.0:
        raise UnsupportedElementError(key)


def add_aperture(
    element: AnyInputElement,
    aperture_type: str,
    aperture_at: str,
    *,
    x1_limit: float,
    x2_limit: float,
    y1_limit: float,
    y2_limit: float,
    limit_if_unset: float = 1000.0,
) -> tuple[list[AnyInputElement], list[AnyInputElement]]:
    """
    Add apertures to a beam element.

    Parameters
    ----------
    element : InputElement
        The beam element to which apertures should optionally be added.
    aperture_type : str
        Type of aperture. 'rectangular' or 'elliptical' are supported for
        collimation apertures.
    aperture_at : str
        Position of aperture. 'entrance_end' or 'exit_end' are supported for
        collimation apertures.
    x1_limit : float
        Negative x-axis limit of the aperture.
    x2_limit : float
        Positive x-axis limit of the aperture.
    y1_limit : float
        Negative y-axis limit of the aperture.
    y2_limit : float
        Positive y-axis limit of the aperture.
    limit_if_unset : float, optional
        Default limit value to use when a limit is set to 0.0, by default 1000.0.

    Returns
    -------
    list[AnyInputElement]
        Apertures to add at the start of the element.
    list[AnyInputElement]
        Apertures to add after the end of the element.
    """
    radius = get_element_radius(x1_limit, x2_limit, y1_limit, y2_limit, default=0.03)

    if all(value == 0.0 for value in [x1_limit, x2_limit, y1_limit, y2_limit]):
        return [], []

    def ensure_nonzero(limit: float) -> float:
        if limit == 0.0:
            return limit_if_unset
        return limit

    if aperture_type in {"rectangular", "elliptical"}:
        aperture = CollimateBeam(
            name=f"{element.name}_aperture",
            radius=radius,
            xmin=-ensure_nonzero(x1_limit),
            xmax=ensure_nonzero(x2_limit),
            ymin=-ensure_nonzero(y1_limit),
            ymax=ensure_nonzero(y2_limit),
        )
        if aperture_at == "entrance_end":
            return [aperture], []
        if aperture_at == "exit_end":
            return [], [aperture]

    return [], []


def ele_has_superpositions(tao: Tao, ele_id: int | str) -> bool:
    """
    Check if the specified element has superpositions based on its lord-slave data.

    Parameters
    ----------
    tao : Tao
    ele_id : int or str
        The identifier (ID) or name of the element.

    Returns
    -------
    bool
    """
    return any(
        "super" in info["status"].lower()
        for info in cast(list[dict], tao.ele_lord_slave(ele_id))
    )


def should_switch_integrator(ele: AnyInputElement) -> bool:
    """
    Determines if an element should use the Runge-Kutta integrator.

    Parameters
    ----------
    ele : InputElement

    Returns
    -------
    bool
    """
    if isinstance(ele, Multipole):
        # all multipoles must use RK integrator
        return True

    if isinstance(ele, Wiggler):
        # all wigglers must use RK integrator
        return True

    if isinstance(ele, SolenoidWithRFCavity):
        # standing_wave + runge_kutta/time_runge_kutta -> RK integrator
        return True
    return False


def element_from_tao(
    tao: Tao,
    ele_id: str | int,
    which: Which = "model",
    name: str = "",
    global_csr_flag: bool = False,
    species: str = "electron",
    verbose: bool = False,
    include_collimation: bool = True,
    integrator_type: IntegratorType = IntegratorType.linear_map,
    rfdata_file_id: int = 500,
    file_data: dict[str, np.ndarray] | None = None,
) -> tuple[list[AnyInputElement], dict[str, np.ndarray]]:
    """
    Create beam elements from Tao data.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele_id : str or int
        Element ID (name or index) to extract from Tao.
    which : {"model", "base", "design"}, optional
        Which Tao model to use, by default "model".
    name : str, optional
        Name for the element, by default "".
    global_csr_flag : bool, optional
        Whether CSR is enabled globally in Tao or not, by default False.
    species : str, optional
        Particle species, by default "electron".
    verbose : bool, optional
        Enable verbose output of the element's attributes, by default False.
    include_collimation : bool, optional
        Whether to include collimation elements before and after the primary
        element, by default True.
    integrator_type : IntegratorType, optional
        The type of integrator specified for the input file, by default
        `IntegratorType.linear_map`. Depending on the element settings, the
        element may switch the integrator for the duration of this element to
        `IntegratorType.runge_kutta`.
    rfdata_file_id : int, optional
        RF data file ID, by default 500.  This is only used for certain element
        types.
    file_data : dict[str, np.ndarray] or None, optional
        Existing file data for the lattice, by default None.
        This function will reuse other elements' `file_data` when possible from
        this dictionary.

    Returns
    -------
    list[AnyInputElement]
        A list of IMPACT-Z beam elements
    dict[str, np.ndarray]
        And a dictionary of file data.
    """
    try:
        info = ele_info(tao, ele_id=ele_id, which=which)
    except KeyError:
        raise UnusableElementError(str(ele_id))

    if verbose:
        print_ele_info(ele_id, info)

    key = str(info["key"]).lower()

    multipole_info = get_multipole_info(tao, ele_id=ele_id)
    if multipole_info is not None and key != "thick_multipole":
        raise NotImplementedError(
            f"Multipoles not supported for element type key {key!r}"
        )

    ele_methods_info = ele_methods(tao, ele_id, which=which)
    ele_ref_time_start = cast(TaoInfoDict, tao.ele_param(ele_id, "ele.ref_time_start"))
    ref_time_start = float(ele_ref_time_start["ele_ref_time_start"])

    try:
        res = single_element_from_tao_info(
            ele_id=ele_id,
            info=info,
            multipole_info=multipole_info,
            ele_methods_info=ele_methods_info,
            name=name,
            global_csr_flag=global_csr_flag,
            species=species,
            integrator_type=integrator_type,
            ref_time_start=ref_time_start,
            has_superpositions=ele_has_superpositions(tao, ele_id),
        )
    except NotImplementedError as ex:
        raise NotImplementedError(f"Element {name!r} (id={ele_id}): {ex}")  # from None

    if res is None:
        return [], {}

    inner_ele, rfdata = res
    data = {}

    assert "bmad_sc" in inner_ele.metadata
    assert "bmad_csr" in inner_ele.metadata

    leading_elements: list[AnyInputElement] = []
    trailing_elements: list[AnyInputElement] = []

    if rfdata is not None:
        assert isinstance(inner_ele, SolenoidWithRFCavity)

        for existing_id, existing_data in (file_data or {}).items():
            if np.array_equal(existing_data, rfdata):
                logger.debug(
                    f"Element {inner_ele.name} reusing rfdata from {existing_id}"
                )
                inner_ele.file_id = int(existing_id)
                break
        else:
            logger.debug(f"Element {inner_ele.name} new rfdata {rfdata_file_id}")
            inner_ele.file_id = rfdata_file_id
            data[str(rfdata_file_id)] = rfdata

    if should_switch_integrator(inner_ele):
        leading_elements.append(
            IntegratorTypeSwitch(integrator_type=IntegratorType.runge_kutta)
        )
        trailing_elements.insert(
            0, IntegratorTypeSwitch(integrator_type=IntegratorType.linear_map)
        )

    if inner_ele is None:
        return [], data

    if isinstance(inner_ele, Dipole):
        ref_tilt = cast(float, info["REF_TILT"])
        if ref_tilt != 0.0:
            leading_elements.append(RotateBeam(tilt=ref_tilt))
            trailing_elements.insert(0, RotateBeam(tilt=-ref_tilt))

    if include_collimation:
        aperture_leading, aperture_trailing = add_aperture(
            inner_ele,
            x1_limit=float(info.get("X1_LIMIT", 0.0)),
            x2_limit=float(info.get("X2_LIMIT", 0.0)),
            y1_limit=float(info.get("Y1_LIMIT", 0.0)),
            y2_limit=float(info.get("Y2_LIMIT", 0.0)),
            aperture_type=str(info["aperture_type"]).lower(),
            aperture_at=str(info["aperture_at"]).lower(),
        )
        leading_elements = [*leading_elements, *aperture_leading]
        trailing_elements = [*aperture_trailing, *trailing_elements]

    return [
        *leading_elements,
        inner_ele,
        *trailing_elements,
    ], data


def change_lattice_integrator(
    tao: Tao,
    input: ImpactZInput,
    integrator_type: IntegratorType,
    elem_to_tao_id: dict[int, list[AnyInputElement]],
    which: Which = "model",
) -> None:
    for ele_id, elems in elem_to_tao_id.items():
        for elem in elems:
            if isinstance(elem, Quadrupole):
                info = ele_info(tao, ele_id, which=which)
                elem.k1 = {
                    IntegratorType.linear_map: float(info["K1"]),
                    IntegratorType.runge_kutta: float(info["B1_GRADIENT"]),
                }[integrator_type]
                logger.debug(
                    f"Updated Quadrupole {elem.name} k1={elem.k1} for {integrator_type}"
                )


def toggle_space_charge(lattice: list[AnyInputElement]) -> list[AnyInputElement]:
    new_lattice = []
    was_enabled = False
    space_charge_enabled = False
    for ele in lattice:
        if ele.length > 0 and ele.metadata["bmad_sc"] != space_charge_enabled:
            space_charge_enabled = bool(ele.metadata["bmad_sc"])
            was_enabled = True
            new_lattice.append(ToggleSpaceCharge(enable=space_charge_enabled))
        new_lattice.append(ele)

    if not was_enabled:
        # If space charge was never enabled throughout the lattice, disable it entirely from the start:
        new_lattice.insert(0, ToggleSpaceCharge(enable=False))

    return new_lattice


@dataclass
class ConversionState:
    track_start: str
    track_end: str
    reference_frequency: float
    integrator_type: IntegratorType
    ix_uni: int
    ix_branch: int
    which: Which

    idx_to_name: dict[int, str]
    ix_beginning: int
    initial_particles: ParticleGroup | None

    start_head: TaoInfoDict
    start_twiss: dict[str, float]
    start_gen_attr: dict[str, float]
    start_ele_orbit: dict[str, float]
    global_csr_flag: bool

    branch1: dict[str, Any]
    beam_init: dict[str, Any]
    branch_particle: str

    reference_particle_charge: float
    species_mass: float

    reference_kinetic_energy: float
    space_charge_mesh_size: tuple[int, int, int]

    omega: float
    initial_phase_ref: float
    tao_global: dict[str, Any]

    tao_id_to_elems: dict[int, list[AnyInputElement]] = field(default_factory=dict)

    @property
    def n_particle(self) -> int:
        return len(self.initial_particles) if self.initial_particles else 0

    def convert_lattice(
        self,
        tao: Tao,
        verbose: bool = False,
        initial_particles_file_id: int = 100,
        final_particles_file_id: int = 101,
        initial_rfdata_file_id: int = 500,
        initial_write_full_id: int = 200,
        write_beam_eles: str | Sequence[str] = ("monitor::*", "marker::*"),
        include_collimation: bool = True,
    ) -> tuple[list[AnyInputElement], dict[str, np.ndarray]]:
        lattice: list[AnyInputElement] = [
            WriteFull(name="initial_particles", file_id=initial_particles_file_id),
        ]
        tao_id_to_elems: dict[int, list[AnyInputElement]] = {}

        write_at_ids = get_ele_indices_by_pattern(tao, write_beam_eles)
        output_file_id = initial_write_full_id
        rfdata_file_id = initial_rfdata_file_id

        file_data: dict[str, np.ndarray] = {}

        for ele_id, name in self.idx_to_name.items():
            try:
                z_elems, elem_data = element_from_tao(
                    tao,
                    ele_id,
                    which=self.which,
                    name=name,
                    verbose=verbose,
                    species=self.branch_particle.lower(),
                    global_csr_flag=self.global_csr_flag,
                    include_collimation=include_collimation,
                    integrator_type=self.integrator_type,
                    rfdata_file_id=rfdata_file_id,
                    file_data=file_data,
                )
            except UnusableElementError as ex:
                logger.debug("Skipping element: %s (%s)", ele_id, ex)
            else:
                lattice.extend(z_elems)
                tao_id_to_elems[ele_id] = z_elems

                for ele in z_elems:
                    ele.metadata["bmad_id"] = int(ele_id)

                if elem_data:
                    for key, value in elem_data.items():
                        file_data[key] = value
                    rfdata_file_id = max(int(key) for key in elem_data) + 1

                if ele_id in write_at_ids:
                    if lattice and isinstance(lattice[-1], WriteFull):
                        # Don't duplicate WriteFulls
                        pass
                    else:
                        lattice.append(
                            WriteFull(
                                name=f"WRITE_{name}",
                                file_id=output_file_id,
                                metadata={"bmad_id": int(ele_id)},
                            )
                        )
                        output_file_id += 1

        for dipole, drift in zip(lattice, lattice[1:]):
            if isinstance(dipole, Dipole) and isinstance(drift, Drift):
                drift_csr_enabled = cast(bool, drift.metadata["bmad_csr"])
                if dipole.csr_enabled and drift_csr_enabled:
                    dipole.set_csr(enabled=True, following_drift=True)
                elif dipole.csr_enabled and not drift_csr_enabled:
                    dipole.set_csr(enabled=True, following_drift=False)
                elif not dipole.csr_enabled and drift_csr_enabled:
                    raise NotImplementedError(
                        "Dipole without CSR -> Drift with CSR is not supported in Impact-Z"
                    )

        lattice.append(
            WriteFull(name="final_particles", file_id=final_particles_file_id)
        )

        if self.global_csr_flag:
            lattice = toggle_space_charge(lattice)
        else:
            lattice.insert(0, ToggleSpaceCharge(enable=False))

        self.tao_id_to_elems = tao_id_to_elems
        return lattice, file_data

    def to_input(
        self,
        lattice: list[AnyInputElement],
        file_data: dict[str, np.ndarray],
        radius_x: float = 0.0,
        radius_y: float = 0.0,
        ncpu_y: int = 1,
        ncpu_z: int = 1,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
    ) -> ImpactZInput:
        if nx is None:
            nx = self.space_charge_mesh_size[0]
        if ny is None:
            ny = self.space_charge_mesh_size[1]
        if nz is None:
            nz = self.space_charge_mesh_size[2]

        input = ImpactZInput(
            # Line 1
            ncpu_y=ncpu_y,
            ncpu_z=ncpu_z,
            gpu=GPUFlag.disabled,
            # Line 2
            seed=self.tao_global["random_seed"],
            n_particle=self.n_particle,
            integrator_type=self.integrator_type,
            err=1,
            diagnostic_type=DiagnosticType.extended,
            # Line 3
            nx=nx,
            ny=ny,
            nz=nz,
            boundary_type=BoundaryType.trans_open_longi_open,
            radius_x=radius_x,
            radius_y=radius_y,
            z_period_size=0.0,
            # Line 4
            distribution=(
                DistributionType.read
                if self.initial_particles
                else DistributionType.gauss
            ),
            restart=0,
            subcycle=0,
            nbunch=1,
            # particle_list=particle_list,
            # current_list=current_list,
            # charge_over_mass_list=charge_over_mass_list,
            # Twiss
            twiss_alpha_x=self.start_twiss["alpha_a"],
            twiss_alpha_y=self.start_twiss["alpha_b"],
            # Use the default from the class - this must be nonzero.
            # twiss_alpha_z=0.0,  # start_twiss["alpha_z"],
            twiss_beta_x=self.start_twiss["beta_a"],
            twiss_beta_y=self.start_twiss["beta_b"],
            twiss_beta_z=1.0,  # start_twiss["beta_z"],
            twiss_norm_emit_x=1e-6,
            twiss_norm_emit_y=1e-6,
            twiss_norm_emit_z=1e-6,
            # Twiss mismatch
            twiss_mismatch_e_z=1.0,
            twiss_mismatch_x=1.0,
            twiss_mismatch_y=1.0,
            twiss_mismatch_z=1.0,
            twiss_mismatch_px=1.0,
            twiss_mismatch_py=1.0,
            # Twiss offset
            twiss_offset_energy_z=0.0,
            twiss_offset_phase_z=0.0,
            twiss_offset_x=self.start_ele_orbit["x"],
            twiss_offset_y=self.start_ele_orbit["y"],
            twiss_offset_px=self.start_ele_orbit["px"],
            twiss_offset_py=self.start_ele_orbit["py"],
            average_current=0.0,  # TODO users must set this if they want space charge calcs
            reference_kinetic_energy=self.reference_kinetic_energy,
            reference_particle_mass=self.species_mass,
            reference_particle_charge=self.reference_particle_charge,
            reference_frequency=self.reference_frequency,
            initial_phase_ref=self.initial_phase_ref,
            lattice=lattice,
            initial_particles=self.initial_particles,
            # External file data
            file_data=file_data,
        )
        if self.global_csr_flag:
            input.space_charge_on(bunch_charge=float(self.beam_init["bunch_charge"]))

        return input

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        track_start: str | None = None,
        track_end: str | None = None,
        reference_frequency: float = 1300000000.0,
        ix_uni: int = 1,
        ix_branch: int = 0,
        which: Which = "model",
        integrator_type: IntegratorType = IntegratorType.linear_map,
    ) -> ConversionState:
        idx_to_name = get_index_to_name(
            tao,
            track_start=track_start,
            track_end=track_end,
            ix_uni=ix_uni,
            ix_branch=ix_branch,
        )

        ix_beginning = list(idx_to_name)[0]
        ix_end = list(idx_to_name)[-1]
        try:
            initial_particles = export_particles(tao, ix_beginning)
        except TaoCommandError as ex:
            logger.warning(f"Not using initial particles ({ex.errors[-1].message})")
            initial_particles = None

        start_head = ele_head(tao, str(ix_beginning), which=which)
        start_twiss = cast(
            dict[str, float], tao.ele_twiss(str(ix_beginning), which=which)
        )
        start_gen_attr = cast(
            dict[str, float],
            tao.ele_gen_attribs(str(ix_beginning), which=which),
        )
        start_ele_orbit = cast(
            dict[str, float], tao.ele_orbit(ix_beginning, which=which)
        )
        global_csr_flag = cast(dict, tao.bmad_com())["csr_and_space_charge_on"]
        assert isinstance(global_csr_flag, bool)

        beam_init = cast(Dict[str, Any], tao.beam_init(ix_branch, ix_uni=str(ix_uni)))
        branch1 = cast(Dict[str, Any], tao.branch1(ix_uni, ix_branch))
        branch_particle: str = branch1["param_particle"]

        reference_particle_charge = charge_state(branch_particle.lower())
        species_mass = mass_of(branch_particle.lower())

        reference_kinetic_energy = start_gen_attr["E_TOT"] - species_mass

        omega = 2 * np.pi * reference_frequency
        initial_phase_ref = float(start_head["ref_time"]) * omega
        tao_global = cast(dict, tao.tao_global())

        space_charge_com = cast(dict, tao.space_charge_com())

        return cls(
            track_start=idx_to_name[ix_beginning],
            track_end=idx_to_name[ix_end],
            reference_frequency=reference_frequency,
            integrator_type=integrator_type,
            ix_uni=ix_uni,
            ix_branch=ix_branch,
            which=which,
            idx_to_name=idx_to_name,
            ix_beginning=ix_beginning,
            initial_particles=initial_particles,
            start_head=start_head,
            start_twiss=start_twiss,
            start_gen_attr=start_gen_attr,
            start_ele_orbit=start_ele_orbit,
            global_csr_flag=global_csr_flag,
            branch1=branch1,
            beam_init=beam_init,
            branch_particle=branch_particle,
            reference_particle_charge=reference_particle_charge,
            species_mass=species_mass,
            reference_kinetic_energy=reference_kinetic_energy,
            omega=omega,
            initial_phase_ref=initial_phase_ref,
            tao_global=tao_global,
            space_charge_mesh_size=cast(
                tuple[int, int, int],
                tuple(space_charge_com["space_charge_mesh_size"]),
            ),
        )


def plot_impactz_and_tao_stats(impactz: ImpactZ, tao: Tao) -> None:
    """
    Simple function to compare the output of Impact-Z vs Tao's bunch comb.

    Parameters
    ----------
    impactz : ImpactZ
        ImpactZ object.
    tao : Tao
        The Tao object to compare to.

    Returns
    -------
    None
        This function produces plots but does not return any values.
        Retrieve the last figure by way of `plt.gcf()`, if necessary.
    """

    I = impactz
    if I.output is None:
        raise ValueError("No output available on the ImpactZ object.")

    stats = I.output.stats

    mc2 = I.input.reference_particle_mass
    z = stats.z
    x = stats.mean_x
    y = stats.mean_y
    sigma_x = stats.sigma_x
    sigma_y = stats.sigma_y
    sigma_z = stats.sigma_t * c_light
    energy = stats.mean_energy

    def bunch_comb(who: str) -> np.ndarray:
        return cast(np.ndarray, tao.bunch_comb(who))

    x_tao = bunch_comb("x")
    y_tao = bunch_comb("y")
    sigma_x_tao = np.sqrt(bunch_comb("sigma.11"))
    sigma_y_tao = np.sqrt(bunch_comb("sigma.33"))
    sigma_z_tao = np.sqrt(bunch_comb("sigma.55"))
    p_tao = (1 + bunch_comb("pz")) * bunch_comb("p0c")
    energy_tao = np.hypot(p_tao, mc2)
    s_tao_raw = bunch_comb("s")

    s_tao = s_tao_raw - s_tao_raw[0]

    _fig, axes = plt.subplots(7, figsize=(12, 8))

    ax = axes[0]
    ax.plot(z, x * 1e6, label="Impact-Z")
    ax.plot(s_tao, x_tao * 1e6, "--", label="Tao")
    ax.set_ylabel(r"$\left<x\right>$ (µm)")

    ax = axes[1]
    ax.plot(z, sigma_x * 1e6, label="Impact-Z")
    ax.plot(s_tao, sigma_x_tao * 1e6, "--", label="Tao")
    ax.set_ylabel(r"$\sigma_x$ (µm)")

    ax = axes[2]
    ax.plot(z, y * 1e6, label="Impact-Z")
    ax.plot(s_tao, y_tao * 1e6, "--", label="Tao")
    ax.set_ylabel(r"$\left<y\right>$ (µm)")

    ax = axes[3]
    ax.plot(z, sigma_y * 1e6, label="Impact-Z")
    ax.plot(s_tao, sigma_y_tao * 1e6, "--", label="Tao")
    ax.set_ylabel(r"$\sigma_y$ (µm)")

    ax = axes[4]
    ax.plot(z, sigma_z * 1e6, label="Impact-Z")
    ax.plot(s_tao, sigma_z_tao * 1e6, "--", label="Tao")
    ax.set_ylabel(r"$\sigma_z$ (µm)")

    ax = axes[5]
    ax.plot(z, (energy - energy[0]) / 1e6, label="Impact-Z")
    ax.plot(s_tao, (energy_tao - energy_tao[0]) / 1e6, "--", label="Tao")
    ax.set_ylabel(r"$d\left<E\right>$ (MeV)")

    ax.legend()

    ax = axes[-1]
    tao.matplotlib.plot("lat_layout", axes=[ax])

    s_max = s_tao.max()
    for ax in axes[:-1]:
        ax.set_xlim(0, s_max)

    # Tao has the full s range
    axes[-1].set_xlim(s_tao_raw.max() - s_max, s_tao_raw.max())

    ax.set_xlabel(r"$s$ (m)")


def track_tao(
    tao: Tao,
    particles: ParticleGroup,
    track_start: str | int | None = None,
    track_end: str | int | None = None,
    ix_uni: int | str = "",
    ix_branch: int | str = 0,
):
    """
    Helper to track a ParticleGroup in Tao.

    Parameters
    ----------
    tao : Tao
        The Tao object instance.
    particles : ParticleGroup
        The particle group to track.
    track_start : str, int, or None, optional
        The starting element for tracking, by default None.
    track_end : str, int, or None, optional
        The ending element for tracking, by default None.
    ix_uni : int or str, optional
        The universe index, by default "".
    ix_branch : int or str, optional
        The branch index, by default 0.
    """
    cmds = [
        f"set beam_init bunch_charge = {particles.charge}",
        f"set beam_init n_particle = {particles.n_particle}",
    ]

    if track_start is not None:
        cmds.append(f"set beam_init track_start = {track_start}")

    if track_end is not None:
        cmds.append(f"set beam_init track_start = {track_end}")
    else:
        beam = cast(dict, tao.beam(ix_branch, ix_uni=str(ix_uni)))
        track_end = beam["track_end"] or "END"

    tao.cmds(cmds)

    with tempfile.NamedTemporaryFile(suffix=".h5") as temp_file:
        particles.write(temp_file.name)
        tao.cmd(f"set beam_init position_file = {temp_file.name}")
        tao.cmd("set global lattice_calc_on = T")
        tao.track_beam()

    return ParticleGroup(data=tao.bunch_data(track_end))
