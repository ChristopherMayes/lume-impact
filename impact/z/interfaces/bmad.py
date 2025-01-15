from __future__ import annotations

import logging
import pathlib
import tempfile

# import pprint
from typing import Any, Dict, cast

from ...particles import SPECIES_MASS
from ..input import AnyInputElement, WriteFull
from pmd_beamphysics import ParticleGroup
from pytao import Tao, TaoCommandError
from typing_extensions import Literal

from impact.z.constants import (
    BoundaryType,
    DiagnosticType,
    DistributionZType,
    GPUFlag,
    IntegratorType,
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


def element_from_tao(
    tao: Tao,
    ele_id: str | int,
    which: Which = "model",
    name: str = "",
    steps: int = 10,
    map_steps: int = 10,
):
    try:
        einfo = ele_info(tao, ele_id=ele_id, which=which)
    except KeyError:
        raise UnusableElementError(str(ele_id))

    key = einfo["key"].lower()
    if key == "drift":
        # pprint.pprint(einfo)
        return Drift(
            length=einfo["L"],
            name=name,
            steps=steps,
            map_steps=map_steps,
            radius=1.0,  # no such thing in bmad, right?
        )
    # elif key == "sbend":
    raise UnsupportedElementError(key)


def input_from_tao(
    tao: Tao,
    track_start: str | None = None,
    track_end: str | None = None,
    *,
    ncpu_y: int = 1,
    ncpu_z: int = 1,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    which: Which = "model",
    ix_uni: int = 1,
    ix_branch: int = 0,
    reference_frequency: float = 1300000000.0,
) -> ImpactZInput:
    idx_to_name = get_index_to_name(
        tao,
        track_start=track_start,
        track_end=track_end,
        # ix_uni=ix_uni, ix_branch=ix_branch
    )

    ix_beginning = list(idx_to_name)[0]
    # ix_end = list(idx_to_name)[-1]
    print(idx_to_name)
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

    bunch_params_start = cast(Dict[str, float], tao.bunch_params(ix_beginning))
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

    return ImpactZInput(
        # Line 1
        ncpu_y=ncpu_y,
        ncpu_z=ncpu_z,
        gpu=GPUFlag.disabled,
        # Line 2
        seed=0,
        n_particle=0,
        integrator_type=IntegratorType.linear,
        err=1,
        diagnostic_type=DiagnosticType.at_given_time,
        output_z=OutputZType.extended,
        # Line 3
        nx=nx,
        ny=ny,
        nz=nz,
        boundary_type=BoundaryType.trans_open_longi_open,
        radius_x=0.0,
        radius_y=0.0,
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
        twiss_alpha_x=bunch_params_start["twiss_alpha_x"],
        twiss_alpha_y=bunch_params_start["twiss_alpha_y"],
        twiss_alpha_z=bunch_params_start["twiss_alpha_z"],
        twiss_beta_x=bunch_params_start["twiss_beta_x"],
        twiss_beta_y=bunch_params_start["twiss_beta_y"],
        twiss_beta_z=bunch_params_start["twiss_beta_z"],
        twiss_norm_emit_x=bunch_params_start["twiss_norm_emit_x"],
        twiss_norm_emit_y=bunch_params_start["twiss_norm_emit_y"],
        twiss_norm_emit_z=bunch_params_start["twiss_norm_emit_z"],
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
        average_current=1.0,  # TODO
        initial_kinetic_energy=1.0,  # TODO
        reference_particle_mass=species_mass,
        reference_particle_charge=reference_particle_charge,
        reference_frequency=reference_frequency,
        initial_phase_ref=0.0,
        lattice=lattice,
        initial_particles=initial_particles,
    )
