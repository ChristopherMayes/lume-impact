from __future__ import annotations

import logging
import math
import pathlib
import tempfile
from enum import IntEnum
from typing import Any, Dict, Iterable, NamedTuple, Sequence, TypeAlias, TypedDict, cast

import numpy as np
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.particles import c_light
from pmd_beamphysics.species import charge_state, mass_of
from pytao import Tao, TaoCommandError
from typing_extensions import Literal

from impact.z.constants import (
    BoundaryType,
    DiagnosticType,
    DistributionZType,
    GPUFlag,
    IntegratorType,
    MultipoleType,
    OutputZType,
)

from ...interfaces.bmad import ele_info, tao_unique_names
from .. import Drift, ImpactZInput
from ..fieldmaps import make_solenoid_rfcavity_rfdata_simple
from ..input import (
    CCL,
    AnyInputElement,
    CollimateBeamWithRectangularAperture,
    Dipole,
    Multipole,
    Quadrupole,
    Solenoid,
    SolenoidWithRFCavity,
    SuperconductingCavity,
    WriteFull,
)

logger = logging.getLogger(__name__)
Which = Literal["model", "base", "design"]


class UnusableElementError(Exception): ...


class UnsupportedElementError(Exception): ...


TaoInfoDict: TypeAlias = dict[str, str | float | int]


def ele_methods(tao: Tao, ele: str | int, which: str = "model") -> TaoInfoDict:
    return cast(TaoInfoDict, tao.ele_methods(ele, which=which))


def ele_head(tao: Tao, ele: str | int, which: str = "model") -> TaoInfoDict:
    return cast(TaoInfoDict, tao.ele_head(ele, which=which))


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
) -> dict[int, str]:
    idx_to_name = tao_unique_names(tao)
    ix_start = get_element_index(tao, track_start) if track_start else 0
    ix_end = get_element_index(tao, track_end) if track_end else max(idx_to_name)
    return {
        ix_ele: name
        for ix_ele, name in idx_to_name.items()
        if ix_start <= ix_ele <= ix_end
    }


def get_ele_indices_by_pattern(
    tao: Tao,
    patterns: list[str] | str,
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


CavityClass: TypeAlias = (
    type[SuperconductingCavity] | type[SolenoidWithRFCavity] | type[CCL]
)


def get_cavity_class(tracking_method: str, cavity_type: str) -> CavityClass:
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
    default_map_steps: int = 10,
    global_csr_flag: bool = False,
    species: str = "electron",
    integrator_type: IntegratorType = IntegratorType.linear_map,
    ref_time_start: float | None = None,
    has_superpositions: bool = False,
) -> tuple[AnyInputElement, np.ndarray | None] | None:
    key = str(info["key"]).lower()

    length = info["L"]
    assert isinstance(length, float)

    x1_limit = float(info.get("X1_LIMIT", 0.0))
    x2_limit = float(info.get("X2_LIMIT", 0.0))
    y1_limit = float(info.get("Y1_LIMIT", 0.0))
    y2_limit = float(info.get("Y2_LIMIT", 0.0))
    rotation_error_x = float(info.get("X_PITCH_TOT", 0.0))
    rotation_error_y = float(info.get("Y_PITCH_TOT", 0.0))
    rotation_error_z = -float(info.get("TILT_TOT", 0.0))
    num_steps = int(info.get("NUM_STEPS", 10))
    radius = get_element_radius(x1_limit, x2_limit, y1_limit, y2_limit, default=0.03)

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

    if key in {"drift", "pipe", "monitor", "instrument"}:
        return Drift(
            length=length,
            name=name,
            steps=num_steps,
            map_steps=default_map_steps,
            radius=1.0,  # no such thing in bmad, right?
        ), None

    if key == "sbend":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for SBend")

        fringe_type = info["FRINGE_TYPE"]
        if fringe_type != "Full":
            logger.warning(
                f"Element #{ele_id} has a fringe type of {fringe_type!r}; to match with "
                f"Impact-Z, this should be 'Full'"
            )

        element_csr_flag = str(ele_methods_info["csr_method"]).lower() == "1_dim"
        csr = global_csr_flag or element_csr_flag
        return Dipole(
            name=name,
            length=length,
            steps=num_steps,
            map_steps=default_map_steps,
            angle=float(info["ANGLE"]),  # rad
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
        ), None

    if key in {"sextupole", "octupole", "thick_multipole"}:
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Solenoid")

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
            map_steps=default_map_steps,
            # The gradient of the quadrupole magnetic field, measured in Tesla per meter.
            multipole_type=multipole_type,
            field_strength=field_strength,
            file_id=-1,  # TODO?
            radius=radius,
            misalignment_error_x=offset_x,
            misalignment_error_y=offset_y,
            rotation_error_x=rotation_error_x,
            rotation_error_y=rotation_error_y,
            rotation_error_z=rotation_error_z,
        ), None

    if key == "quadrupole":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Quadrupole")
        if np.abs(rotation_error_x) > 0.0:
            raise NotImplementedError("X pitch not currently supported for Quadrupole")
        if np.abs(rotation_error_y) > 0.0:
            raise NotImplementedError("Y pitch not currently supported for Quadrupole")

        k1 = {
            IntegratorType.linear_map: float(info["K1"]),
            IntegratorType.runge_kutta: float(info["B1_GRADIENT"]),
        }[integrator_type]
        return Quadrupole(
            name=name,
            length=length,
            steps=num_steps,
            map_steps=default_map_steps,
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
            rotation_error_z=-rotation_error_z,
        ), None
    if key == "solenoid":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Solenoid")

        return Solenoid(
            name=name,
            length=length,
            steps=num_steps,
            map_steps=default_map_steps,
            # The gradient of the quadrupole magnetic field, measured in Tesla per meter.
            Bz0=float(info["BS_FIELD"]),
            file_id=-1,  # TODO?
            radius=radius,  # TODO arbitrary
            misalignment_error_x=offset_x,
            misalignment_error_y=offset_y,
            rotation_error_x=rotation_error_x,
            rotation_error_y=rotation_error_y,
            rotation_error_z=rotation_error_z,
        ), None

    if key == "lcavity":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Lcavity")
        if np.abs(rotation_error_x) > 0.0:
            raise NotImplementedError("X pitch not currently supported for Lcavity")
        if np.abs(rotation_error_y) > 0.0:
            raise NotImplementedError("Y pitch not currently supported for Lcavity")

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
                map_steps=default_map_steps,
                file_id=-1.0,  # TODO: same for all cavity types?
                rf_frequency=float(info["RF_FREQUENCY"]),
                phase_deg=float(info["PHI0"]) * 360.0,
                radius=radius,  # TODO is this the aperture radius?
                field_scaling=float(info["GRADIENT"]),
                misalignment_error_x=offset_x,
                misalignment_error_y=offset_y,
                rotation_error_x=0.0,
                rotation_error_y=0.0,
                rotation_error_z=-rotation_error_z,
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
            rf_wavelength = c_light / rf_frequency
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
                steps=int((length / rf_wavelength) * 180.0),
                map_steps=default_map_steps,
                file_id=1.0,
                rf_frequency=rf_frequency,
                phase_deg=(phi0 + phi0_autoscale - phi0_ref + 0.25 + phi0_pad) * 360.0,
                radius=radius,
                field_scaling=-2.0 * float(info["GRADIENT"]) * length / L_active,
                misalignment_error_x=offset_x,
                misalignment_error_y=offset_y,
                rotation_error_x=0.0,
                rotation_error_y=0.0,
                rotation_error_z=-rotation_error_z,
                aperture_size_for_wakefield=0.0,
                bz0=0.0,
                gap_size_for_wakefield=0.0,
                length_for_wakefield=0.0,
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
) -> list[AnyInputElement]:
    radius = get_element_radius(x1_limit, x2_limit, y1_limit, y2_limit, default=0.03)

    if all(value == 0.0 for value in [x1_limit, x2_limit, y1_limit, y2_limit]):
        return [element]

    if aperture_type in {"rectangular", "elliptical"}:
        aperture = CollimateBeamWithRectangularAperture(
            name=f"{element.name}_aperture",
            radius=radius,
            xmin=x1_limit,
            xmax=x2_limit,
            ymin=y1_limit,
            ymax=y2_limit,
        )
        if aperture_at == "entrance_end":
            return [aperture, element]
        if aperture_at == "exit_end":
            return [element, aperture]

    return [element]


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


def element_from_tao(
    tao: Tao,
    ele_id: str | int,
    which: Which = "model",
    name: str = "",
    default_map_steps: int = 10,
    global_csr_flag: bool = False,
    species: str = "electron",
    verbose: bool = False,
    include_collimation: bool = False,
    integrator_type: IntegratorType = IntegratorType.linear_map,
    rfdata_file_id: int = 500,
) -> tuple[list[AnyInputElement], dict[int, np.ndarray]]:
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

    res = single_element_from_tao_info(
        ele_id=ele_id,
        info=info,
        multipole_info=multipole_info,
        ele_methods_info=ele_methods_info,
        name=name,
        default_map_steps=default_map_steps,
        global_csr_flag=global_csr_flag,
        species=species,
        integrator_type=integrator_type,
        ref_time_start=ref_time_start,
        has_superpositions=ele_has_superpositions(tao, ele_id),
    )
    if res is None:
        return [], {}

    inner_ele, rfdata = res
    data = {}

    if rfdata is not None:
        assert isinstance(inner_ele, SolenoidWithRFCavity)
        inner_ele.file_id = rfdata_file_id
        data[rfdata_file_id] = rfdata

    # if isinstance(inner_ele, SolenoidWithRFCavity):
    #     # TODO no aperture support here
    #     return pad_solenoid_with_rf_cavity(inner_ele)

    if inner_ele is None:
        return [], data

    if not include_collimation:
        return [inner_ele], data

    return add_aperture(
        inner_ele,
        x1_limit=float(info.get("X1_LIMIT", 0.0)),
        x2_limit=float(info.get("X2_LIMIT", 0.0)),
        y1_limit=float(info.get("Y1_LIMIT", 0.0)),
        y2_limit=float(info.get("Y2_LIMIT", 0.0)),
        aperture_type=str(info["aperture_type"]).lower(),
        aperture_at=str(info["aperture_at"]).lower(),
    ), data


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


def input_from_tao(
    tao: Tao,
    track_start: str | None = None,
    track_end: str | None = None,
    *,
    radius_x: float = 0.0,
    radius_y: float = 0.0,
    ncpu_y: int = 1,
    ncpu_z: int = 1,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    which: Which = "model",
    ix_uni: int = 1,
    ix_branch: int = 0,
    reference_frequency: float = 1300000000.0,  # TODO: consider calculating this? it's somewhat arbitrary
    verbose: bool = False,
    initial_particles_file_id: int = 100,
    final_particles_file_id: int = 101,
    initial_rfdata_file_id: int = 500,
    initial_write_full_id: int = 200,
    write_beam_eles: str | Sequence[str] = ("monitor::*", "marker::*"),
    include_collimation: bool = False,
    integrator_type: IntegratorType = IntegratorType.linear_map,
) -> ImpactZInput:
    idx_to_name = get_index_to_name(
        tao,
        track_start=track_start,
        track_end=track_end,
        # ix_uni=ix_uni, ix_branch=ix_branch
    )

    ix_beginning = list(idx_to_name)[0]
    # ix_end = list(idx_to_name)[-1]
    try:
        initial_particles = export_particles(tao, ix_beginning)
    except TaoCommandError as ex:
        logger.warning(f"Not using initial particles ({ex.errors[-1].message})")
        initial_particles = None
        n_particle = 0
    else:
        n_particle = len(initial_particles)

    start_head = ele_head(tao, str(ix_beginning), which=which)
    start_twiss = cast(dict[str, float], tao.ele_twiss(str(ix_beginning), which=which))
    start_gen_attr = cast(
        dict[str, float],
        tao.ele_gen_attribs(str(ix_beginning), which=which),
    )
    start_ele_orbit = cast(dict[str, float], tao.ele_orbit(ix_beginning, which=which))
    global_csr_flag = cast(dict, tao.bmad_com())["csr_and_space_charge_on"]
    assert isinstance(global_csr_flag, bool)

    branch1 = cast(Dict[str, Any], tao.branch1(ix_uni, ix_branch))
    branch_particle: str = branch1["param_particle"]

    reference_particle_charge = charge_state(branch_particle.lower())
    species_mass = mass_of(branch_particle.lower())

    reference_kinetic_energy = start_gen_attr["E_TOT"] - species_mass

    omega = 2 * np.pi * reference_frequency
    initial_phase_ref = float(start_head["ref_time"]) * omega
    tao_global = cast(dict, tao.tao_global())

    lattice: list[AnyInputElement] = [
        WriteFull(name="initial_particles", file_id=initial_particles_file_id),
    ]
    tao_id_to_elems: dict[int, list[AnyInputElement]] = {}

    write_at_ids = get_ele_indices_by_pattern(tao, write_beam_eles)
    output_file_id = initial_write_full_id
    rfdata_file_id = initial_rfdata_file_id

    file_data = {}
    for ele_id, name in idx_to_name.items():
        try:
            z_elems, elem_data = element_from_tao(
                tao,
                ele_id,
                which=which,
                name=name,
                verbose=verbose,
                species=branch_particle.lower(),
                global_csr_flag=global_csr_flag,
                include_collimation=include_collimation,
                integrator_type=integrator_type,
                rfdata_file_id=rfdata_file_id,
            )
        except UnusableElementError as ex:
            logger.debug("Skipping element: %s (%s)", ele_id, ex)
        else:
            lattice.extend(z_elems)
            tao_id_to_elems[ele_id] = z_elems

            if elem_data:
                for key, value in elem_data.items():
                    file_data[str(key)] = value
                rfdata_file_id = max(elem_data) + 1

            if ele_id in write_at_ids:
                if lattice and isinstance(lattice[-1], WriteFull):
                    # Don't duplicate WriteFulls
                    pass
                else:
                    lattice.append(
                        WriteFull(name=f"WRITE_{name}", file_id=output_file_id)
                    )
                    output_file_id += 1

    # TODO
    # combine_reused_rfdata(z_elems)
    lattice.append(WriteFull(name="final_particles", file_id=final_particles_file_id))

    input = ImpactZInput(
        # Line 1
        ncpu_y=ncpu_y,
        ncpu_z=ncpu_z,
        gpu=GPUFlag.disabled,
        # Line 2
        seed=tao_global["random_seed"],
        n_particle=n_particle,
        integrator_type=integrator_type,
        err=1,
        # diagnostic_type=DiagnosticType.at_bunch_centroid,  # DiagnosticType.at_given_time,
        diagnostic_type=DiagnosticType.at_given_time,
        output_z=OutputZType.extended,
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
            DistributionZType.read if initial_particles else DistributionZType.gauss
        ),
        restart=0,
        subcycle=0,
        nbunch=0,
        # I think there are unused:
        # particle_list=[],
        # current_list=[],
        # charge_over_mass_list=[],
        # Twiss
        twiss_alpha_x=start_twiss["alpha_a"],
        twiss_alpha_y=start_twiss["alpha_b"],
        twiss_alpha_z=0.0,  # start_twiss["alpha_z"],
        twiss_beta_x=start_twiss["beta_a"],
        twiss_beta_y=start_twiss["beta_b"],
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
        twiss_offset_x=start_ele_orbit["x"],
        twiss_offset_y=start_ele_orbit["y"],
        twiss_offset_px=start_ele_orbit["px"],
        twiss_offset_py=start_ele_orbit["py"],
        average_current=0.0,  # TODO users must set this if they want space charge calcs
        reference_kinetic_energy=reference_kinetic_energy,
        reference_particle_mass=species_mass,
        reference_particle_charge=reference_particle_charge,
        reference_frequency=reference_frequency,
        initial_phase_ref=initial_phase_ref,
        lattice=lattice,
        initial_particles=initial_particles,
        # External file data
        file_data=file_data,
    )

    if input.multipoles and input.integrator_type == IntegratorType.linear_map:
        logger.warning(
            "Slower integrator type Runge-Kutta selected as "
            "Multipoles in Impact-Z require it to function."
        )
        input.integrator_type = IntegratorType.runge_kutta
        change_lattice_integrator(
            tao,
            input,
            IntegratorType.runge_kutta,
            tao_id_to_elems,
        )

    return input
