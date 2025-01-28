from __future__ import annotations

import logging
import pathlib
import tempfile

from typing import Any, Dict, cast

import numpy as np
from ...particles import SPECIES_MASS
from ..input import AnyInputElement, Dipole, Multipole, Quadrupole, Solenoid, WriteFull
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


def ele_head(tao: Tao, ele: str | int, which: str = "model") -> dict:
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


def element_from_tao(
    tao: Tao,
    ele_id: str | int,
    which: Which = "model",
    name: str = "",
    default_steps: int = 10,
    default_map_steps: int = 10,
    enable_csr: bool = False,
    verbose: bool = True,
):
    try:
        info = ele_info(tao, ele_id=ele_id, which=which)
    except KeyError:
        raise UnusableElementError(str(ele_id))

    if verbose:
        print_ele_info(ele_id, info)
    key = info["key"].lower()

    if key == "drift":
        return Drift(
            length=info["L"],
            name=name,
            steps=info["NUM_STEPS"],
            map_steps=default_map_steps,
            radius=1.0,  # no such thing in bmad, right?
        )

    if key == "sbend":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for SBend")

        return Dipole(
            length=info["L"],
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
    if key == "quadrupole":
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Quadrupole")

        radius = get_element_radius(
            info["X1_LIMIT"],
            info["X2_LIMIT"],
            info["Y1_LIMIT"],
            info["Y2_LIMIT"],
            default=1,
        )
        return Quadrupole(
            length=info["L"],
            steps=info["NUM_STEPS"],
            map_steps=default_map_steps,
            # The gradient of the quadrupole magnetic field, measured in Tesla per meter.
            B1=info["B1_GRADIENT"],  # T/m
            # file_id : float
            #     An ID for the input gradient file. Determines profile behavior:
            #     if greater than 0, a fringe field profile is read; if less than -10,
            #     a linear transfer map of an undulator is used; if between -10 and 0,
            #     it's the k-value linear transfer map; if equal to 0, it uses the linear
            #     transfer map with the gradient.
            file_id=-1,
            # The radius of the quadrupole, measured in meters.
            radius=radius,  # TODO is this the aperture radius?
            misalignment_error_x=info["X_OFFSET_TOT"],  # or X_OFFSET?
            misalignment_error_y=info["Y_OFFSET_TOT"],  # or Y_OFFSET?
            rotation_error_x=info["X_PITCH_TOT"],  # or X_PITCH?
            rotation_error_y=info["Y_PITCH_TOT"],  # or Y_PITCH?
            rotation_error_z=info["TILT_TOT"],
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
            length=info["L"],
            steps=info["NUM_STEPS"],
            map_steps=default_map_steps,
            # The gradient of the quadrupole magnetic field, measured in Tesla per meter.
            Bz0=info["BS_FIELD"],
            file_id=-1,  # TODO?
            radius=radius,  # TODO arbitrary
            misalignment_error_x=info["X_OFFSET_TOT"],  # or X_OFFSET?
            misalignment_error_y=info["Y_OFFSET_TOT"],  # or Y_OFFSET?
            rotation_error_x=info["X_PITCH_TOT"],  # or X_PITCH?
            rotation_error_y=info["Y_PITCH_TOT"],  # or Y_PITCH?
            rotation_error_z=info["TILT_TOT"],
        )

    if key in {"sextupole", "octupole"}:  # , "decapole"}:
        if np.abs(info["Z_OFFSET_TOT"]) > 0.0:
            raise NotImplementedError("Z offset not supported for Solenoid")

        field_strength_key = {
            "sextupole": "K2",
            "octupole": "K3",
            # "decapole": "k4",
            # "dodecapole": "k5",
        }[key]

        radius = get_element_radius(
            info["X1_LIMIT"],
            info["X2_LIMIT"],
            info["Y1_LIMIT"],
            info["Y2_LIMIT"],
            default=1,
        )

        return Multipole(
            length=info["L"],
            steps=info["NUM_STEPS"],
            map_steps=default_map_steps,
            # The gradient of the quadrupole magnetic field, measured in Tesla per meter.
            multipole_type=MultipoleType[key],
            field_strength=info[field_strength_key],
            file_id=-1,  # TODO?
            radius=radius,
            misalignment_error_x=info["X_OFFSET_TOT"],  # or X_OFFSET?
            misalignment_error_y=info["Y_OFFSET_TOT"],  # or Y_OFFSET?
            rotation_error_x=info["X_PITCH_TOT"],  # or X_PITCH?
            rotation_error_y=info["Y_PITCH_TOT"],  # or Y_PITCH?
            rotation_error_z=info["TILT_TOT"],
        )

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

    lattice: list[AnyInputElement] = [
        WriteFull(name="initial_particles", file_id=2000),
    ]
    for ele_id, name in idx_to_name.items():
        try:
            z_elem = element_from_tao(tao, ele_id, which=which, name=name)
        except UnusableElementError as ex:
            logger.debug("Skipping element: %s (%s)", ele_id, ex)
        except UnsupportedElementError as ex:
            logger.warning("Skipping element: %s (%s)", ele_id, ex)
        else:
            lattice.append(z_elem)

    lattice.append(WriteFull(name="final_particles", file_id=2001))

    start_head = ele_head(tao, str(ix_beginning), which=which)
    start_twiss = cast(dict[str, float], tao.ele_twiss(str(ix_beginning), which=which))
    start_gen_attr = cast(
        dict[str, float],
        tao.ele_gen_attribs(str(ix_beginning), which=which),
    )

    branch1 = cast(Dict[str, Any], tao.branch1(ix_uni, ix_branch))
    branch_particle: str = branch1["param_particle"]

    reference_particle_charge = {
        "electron": -1.0,
        "positron": 1.0,
    }.get(branch_particle.lower(), 0.0)

    if initial_particles is not None:
        species_mass = initial_particles.mass
    else:
        try:
            species_mass = SPECIES_MASS[branch_particle.lower()]
        except KeyError:
            species_mass = 0.0
            logger.warning(f"Unsupported branch particle type: {branch_particle}")

    initial_kinetic_energy = start_gen_attr["E_TOT"] - species_mass

    omega = 2 * np.pi * reference_frequency
    initial_phase_ref = start_head["ref_time"] * omega
    tao_global = cast(dict, tao.tao_global())

    return ImpactZInput(
        # Line 1
        ncpu_y=ncpu_y,
        ncpu_z=ncpu_z,
        gpu=GPUFlag.disabled,
        # Line 2
        seed=tao_global["random_seed"],
        n_particle=0,
        integrator_type=IntegratorType.linear,
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
        twiss_offset_x=0.0,
        twiss_offset_y=0.0,
        twiss_offset_px=0.0,
        twiss_offset_py=0.0,
        average_current=0.0,  # TODO users must set this if they want space charge calcs
        initial_kinetic_energy=initial_kinetic_energy,
        reference_particle_mass=species_mass,
        reference_particle_charge=reference_particle_charge,
        reference_frequency=reference_frequency,
        initial_phase_ref=initial_phase_ref,
        lattice=lattice,
        initial_particles=initial_particles,
    )
