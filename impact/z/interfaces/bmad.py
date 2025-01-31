from __future__ import annotations

from enum import IntEnum
import logging
import math
import pathlib
import tempfile

from typing import Any, Dict, NamedTuple, TypeAlias, TypedDict, cast

from pmd_beamphysics.particles import c_light
import numpy as np
from ..input import (
    AnyInputElement,
    Dipole,
    Multipole,
    Quadrupole,
    Solenoid,
    SolenoidWithRFCavity,
    SuperconductingCavity,
    CCL,
    WriteFull,
)

from pmd_beamphysics.species import charge_state, mass_of
from pmd_beamphysics import ParticleGroup
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

from ...interfaces.bmad import tao_unique_names, ele_info
from .. import Drift, ImpactZInput

logger = logging.getLogger(__name__)
Which = Literal["model", "base", "design"]


class UnusableElementError(Exception): ...


class UnsupportedElementError(Exception): ...


def ele_methods(tao: Tao, ele: str | int, which: str = "model") -> dict[str, int | str]:
    return cast(dict[str, int | str], tao.ele_methods(ele, which=which))


def ele_head(tao: Tao, ele: str | int, which: str = "model") -> dict[str, Any]:
    return cast(dict, tao.ele_head(ele, which=which))


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


def get_element_radius(*limits: float, default=1.0) -> float:
    """
    Calculate the maximum quadrupole radius from the given limits.

    Parameters
    ----------
    *limits : float
        Variable-length argument list of floats representing possible
        limits for the radius limits. At least one limit must be
        provided unless relying on the default value.
    default : float, optional
        The default radius to return if none of the provided limits
        are greater than this value. The default is 1.0.

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


class _CavityCommon(TypedDict):
    name: str
    length: int
    steps: int
    map_steps: int
    file_id: float
    rf_frequency: float
    field_scaling: float
    phase_deg: float
    radius: float
    misalignment_error_x: float
    misalignment_error_y: float
    rotation_error_x: float
    rotation_error_y: float
    rotation_error_z: float


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


def element_from_tao(
    tao: Tao,
    ele_id: str | int,
    which: Which = "model",
    name: str = "",
    default_map_steps: int = 10,
    enable_csr: bool = False,
    species: str = "electron",
    verbose: bool = False,
) -> AnyInputElement | None:
    try:
        info = ele_info(tao, ele_id=ele_id, which=which)
    except KeyError:
        raise UnusableElementError(str(ele_id))

    if verbose:
        print_ele_info(ele_id, info)

    key = str(info["key"]).lower()
    length = info["L"]

    assert isinstance(key, str)
    assert isinstance(length, float)

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

    multipole_info = get_multipole_info(tao, ele_id=ele_id)
    if multipole_info is not None and key != "thick_multipole":
        raise NotImplementedError(
            f"Multipoles not supported for element type key {key!r}"
        )

    if key in {"drift", "pipe", "monitor"}:
        return Drift(
            length=length,
            name=name,
            steps=info["NUM_STEPS"],
            map_steps=default_map_steps,
            radius=1.0,  # no such thing in bmad, right?
        )

    if key == "sbend":
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
            steps=info["NUM_STEPS"],
            map_steps=default_map_steps,
            angle=info["ANGLE"],  # rad
            k1=info["K1"],
            input_switch=201.0 if enable_csr else 0.0,  # TODO
            hgap=info["HGAP"],
            e1=info["E1"],
            e2=info["E2"],
            entrance_curvature=0.0,
            exit_curvature=0.0,
            fint=info["FINT"],
            # misalignment_error_x=info["X_OFFSET_TOT"],  # or X_OFFSET?
            # misalignment_error_y=info["Y_OFFSET_TOT"],  # or Y_OFFSET?
            # rotation_error_x=info["X_PITCH_TOT"],  # or X_PITCH?
            # rotation_error_y=info["Y_PITCH_TOT"],  # or Y_PITCH?
            # rotation_error_z=info["REF_TILT_TOT"],
        )

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
            field_strength = info[field_strength_key]
        else:
            if multipole_info is None:
                raise RuntimeError("thick_multipole has no ele:multipoles information")

            multipole_type = MultipoleType.decapole
            b4_gradient = (
                (4.0 * 3.0 * 2.0)  # 4!
                * charge_state(species)
                * multipole_info.Bn
                * info["P0C"]
                / c_light
                / length
            )
            field_strength = b4_gradient

        radius = get_element_radius(
            info["X1_LIMIT"],
            info["X2_LIMIT"],
            info["Y1_LIMIT"],
            info["Y2_LIMIT"],
            default=1,
        )

        return Multipole(
            name=name,
            length=length,
            steps=info["NUM_STEPS"],
            map_steps=default_map_steps,
            # The gradient of the quadrupole magnetic field, measured in Tesla per meter.
            multipole_type=multipole_type,
            field_strength=field_strength,
            file_id=-1,  # TODO?
            radius=radius,
            misalignment_error_x=offset_x,
            misalignment_error_y=offset_y,
            rotation_error_x=info["X_PITCH_TOT"],  # or X_PITCH?
            rotation_error_y=info["Y_PITCH_TOT"],  # or Y_PITCH?
            rotation_error_z=info["TILT_TOT"],
        )

    if key == "quadrupole":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Quadrupole")
        if np.abs(info["X_PITCH_TOT"]) > 0.0:
            raise NotImplementedError("X pitch not currently supported for Quadrupole")
        if np.abs(info["Y_PITCH_TOT"]) > 0.0:
            raise NotImplementedError("Y pitch not currently supported for Quadrupole")

        radius = get_element_radius(
            info["X1_LIMIT"],
            info["X2_LIMIT"],
            info["Y1_LIMIT"],
            info["Y2_LIMIT"],
            default=1,
        )

        return Quadrupole(
            name=name,
            length=length,
            steps=info["NUM_STEPS"],
            map_steps=default_map_steps,
            # The gradient of the quadrupole magnetic field, measured in Tesla per meter.
            k1=info["K1"],  # NOTE: 1/m^2 (this is not actually )
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
            rotation_error_x=info["X_PITCH_TOT"],  # or X_PITCH?
            rotation_error_y=info["Y_PITCH_TOT"],  # or Y_PITCH?
            rotation_error_z=-info["TILT_TOT"],
        )
    if key == "solenoid":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Solenoid")

        radius = get_element_radius(
            info["X1_LIMIT"],
            info["X2_LIMIT"],
            info["Y1_LIMIT"],
            info["Y2_LIMIT"],
            default=1,
        )

        return Solenoid(
            name=name,
            length=length,
            steps=info["NUM_STEPS"],
            map_steps=default_map_steps,
            # The gradient of the quadrupole magnetic field, measured in Tesla per meter.
            Bz0=info["BS_FIELD"],
            file_id=-1,  # TODO?
            radius=radius,  # TODO arbitrary
            misalignment_error_x=offset_x,
            misalignment_error_y=offset_y,
            rotation_error_x=info["X_PITCH_TOT"],  # or X_PITCH?
            rotation_error_y=info["Y_PITCH_TOT"],  # or Y_PITCH?
            rotation_error_z=info["TILT_TOT"],
        )

    if key == "lcavity":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Lcavity")
        if np.abs(info["X_PITCH_TOT"]) > 0.0:
            raise NotImplementedError("X pitch not currently supported for Lcavity")
        if np.abs(info["Y_PITCH_TOT"]) > 0.0:
            raise NotImplementedError("Y pitch not currently supported for Lcavity")

        radius = get_element_radius(
            info["X1_LIMIT"],
            info["X2_LIMIT"],
            info["Y1_LIMIT"],
            info["Y2_LIMIT"],
            default=1,
        )

        method_info = ele_methods(tao, ele_id, which=which)
        cls = get_cavity_class(
            cavity_type=info["CAVITY_TYPE"].lower(),
            tracking_method=cast(str, method_info["tracking_method"]).lower(),
        )

        common = cast(
            _CavityCommon,
            dict(
                name=name,
                length=length,
                steps=info["NUM_STEPS"],
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
                rotation_error_z=-float(info["TILT_TOT"]),
            ),
        )
        if cls is CCL or cls is SuperconductingCavity:
            if cls is CCL and np.abs(offset_x) > 0:
                logger.warning(f"{offset_x=} for CCL element {name!r} may not work")
            if cls is CCL and np.abs(offset_y) > 0:
                logger.warning(f"{offset_y=} for CCL element {name!r} may not work")
            return cls(**common)
        if cls is SolenoidWithRFCavity:
            return cls(
                **common,
                aperture_size_for_wakefield=0.0,
                bz0=0.0,
                gap_size_for_wakefield=0.0,
                length_for_wakefield=0.0,
            )
        raise RuntimeError(f"Unexpected cavity type: {cls=}")

    if length > 0.0:
        raise UnsupportedElementError(key)


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
    initial_particles_file_id: int = 2000,
    final_particles_file_id: int = 2001,
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

    branch1 = cast(Dict[str, Any], tao.branch1(ix_uni, ix_branch))
    branch_particle: str = branch1["param_particle"]

    reference_particle_charge = charge_state(branch_particle.lower())
    species_mass = mass_of(branch_particle.lower())

    reference_kinetic_energy = start_gen_attr["E_TOT"] - species_mass

    omega = 2 * np.pi * reference_frequency
    initial_phase_ref = start_head["ref_time"] * omega
    tao_global = cast(dict, tao.tao_global())

    lattice: list[AnyInputElement] = [
        WriteFull(name="initial_particles", file_id=initial_particles_file_id),
    ]
    for ele_id, name in idx_to_name.items():
        try:
            z_elem = element_from_tao(
                tao,
                ele_id,
                which=which,
                name=name,
                verbose=verbose,
                species=branch_particle.lower(),
            )
        except UnusableElementError as ex:
            logger.debug("Skipping element: %s (%s)", ele_id, ex)
        else:
            if z_elem is not None:
                lattice.append(z_elem)

    lattice.append(WriteFull(name="final_particles", file_id=final_particles_file_id))

    input = ImpactZInput(
        # Line 1
        ncpu_y=ncpu_y,
        ncpu_z=ncpu_z,
        gpu=GPUFlag.disabled,
        # Line 2
        seed=tao_global["random_seed"],
        n_particle=n_particle,
        integrator_type=IntegratorType.linear_map,
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
    )

    if input.multipoles:
        logger.warning(
            "Slower integrator type Runge-Kutta selected as "
            "Multipoles in Impact-Z require it to function."
        )
        input.integrator_type = IntegratorType.runge_kutta

    return input
