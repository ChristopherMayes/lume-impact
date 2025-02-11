import numpy as np

from pmd_beamphysics.fields.analysis import (
    accelerating_voltage_and_phase,
    track_field_1df,
)
from scipy.optimize import brent

from .fieldmaps import ele_field
from pmd_beamphysics.units import mec2
from scipy.constants import c
from scipy.constants import e as e_charge

from pmd_beamphysics import single_particle


AUTOPHASE_ATTRS = ("theta0_deg", "dtheta0_deg")


def fast_autophase_ele(
    ele,
    attribute="theta0_deg",
    rel_phase_deg=None,
    phase_range=None,
    fmaps=None,
    t0=0,
    pz0=0,
    q0=-1,
    dz_field=None,
    mc2=mec2,
    debug=False,
):
    """
    Finds the accelerating phase of a single element.
    The element can be a ControlGroup (controlling many elements).

    Optionally sets the relative phase to this.


    Parameters
    ----------
    ele: dict-like
        LUME-Impact element

    attribute: str
        Attribute to vary for the phase.
        'theta0_deg' or 'dtheta0_deg'.

    rel_phase_deg: float, optional=None
        if given, will set this to be the relative phase.

    fmaps: dict
        Dict of fieldmaps that contains one for the ele.
        This is generally required.

    t0: float, optional=0
        initial time (s)

    pz0: float, optional=0
        Initial particle momentum (eV/c)

    q0 : float, optional=-1
        initial particle charge (e) (= -1 for electron)

    dz_field: float, default=None => use wavelength/100
        Step size for field sampling

    mc2: float, default = mec2 (electron mass)
        particle mass (eV)

    debug: bool, defaul=False
        If True, return the phasing function

    Returns
    -------
    dict with keys:
        oncrest_phase_deg: float
            maximum accelerating phase (deg)
        rel_phase_deg: float
            Relative phase to oncrest (deg)
        dt: float
            change in time (s)
        dz: float
            change in z (m)
        dpz: float
            change in z momentum (eV/c)
        pz0: float
            initial z momentum (eV/c)
        pz1:
            final z momentum (eV/c)

    """

    if attribute not in AUTOPHASE_ATTRS:
        raise ValueError(
            f"attribute '{attribute}' is not allowed, should be one of: {AUTOPHASE_ATTRS}"
        )

    # Get ele info
    zedge = ele["zedge"]
    L = ele["L"]
    freq = ele["rf_frequency"]

    if freq == 0:
        dc_field = True
        period = 1e9  # DC field
    else:
        dc_field = False
        period = 1 / freq

    if dz_field is None:
        if freq != 0:
            wavelength = c / freq
            nz = int(100 * L / wavelength)
        else:
            nz = 1000
    else:
        nz = int(L / dz_field)

    # Set exactly
    dz_field = L / nz

    # Put in ele frame
    # Field function
    def Ez_f(z, t):
        return ele_field(ele, t=t, z=z + zedge, fmaps=fmaps, component="Ez")

    # Estimate for accelerating phase at v=c
    if debug:
        Z0 = np.linspace(0, L, nz)
        re_Ez0 = np.array([Ez_f(z, 0) for z in Z0])
        im_Ez0 = np.array([Ez_f(z, period / 4) for z in Z0])
        Ez0 = re_Ez0 + 1j * im_Ez0
        _, phase0 = accelerating_voltage_and_phase(Z0, Ez0, freq)
        phase0_deg = phase0 * 180 / np.pi
        # Handle negative sign
        if q0 < 0:
            phase0_deg += 180
        phase0_deg = phase0_deg % 360
        print(f"v=c phase0_deg = {phase0_deg}")

    # Function for brent
    # Max timestep
    max_step = dz_field / c * 10
    save_phase_deg = ele[attribute]

    def phase_f(phase_deg):
        # t = t0 + period * phase_deg / 360
        ele[attribute] = phase_deg
        zf, pzf, tf = track_field_1df(
            Ez_f,
            zstop=L,
            tmax=100 * nz * period,
            pz0=pz0,
            t0=t0,
            q0=q0,
            mc2=mc2,
            max_step=max_step,
        )
        # print(f"{phase_deg%360 :0.2f} {pzf/1e6:0.2f}")
        return zf, pzf, tf - t0

    if debug:
        return phase_f

    # Accelerating phase
    if dc_field:
        acc_phase_deg = 0
    else:
        dc_field = False
        if phase_range is None:
            ptry = np.linspace(-180, 180, 8)[:-1]
            etry = np.array([phase_f(p)[1] for p in ptry])
            ptry0 = ptry[etry.argmax()]
            brack = (ptry0 - 30, ptry0 + 30)

        else:
            brack = phase_range

        acc_phase_deg = brent(lambda x: -phase_f(x)[1], brack=brack)

    # Re-run.
    if rel_phase_deg is not None:
        final_phase_deg = rel_phase_deg + acc_phase_deg
    else:
        # This will restore the saved phase
        final_phase_deg = save_phase_deg

    z1, pz1, dt = phase_f(final_phase_deg)

    #
    found_rel_phase_deg = (final_phase_deg - acc_phase_deg + 180) % 360 - 180

    # print(f"{name}  {acc_phase_deg:.2f} {save_phase_deg:.2f} {found_rel_phase_deg:0.2f} deg")

    # Form output dict
    output = {}
    if dc_field:
        found_rel_phase_deg = 0

    output["oncrest_phase_deg"] = acc_phase_deg % 360
    output["rel_phase_deg"] = found_rel_phase_deg
    output["dpz"] = pz1 - pz0
    output["dz"] = z1
    output["dt"] = dt
    output["pz0"] = pz0
    output["pz1"] = pz1

    return output


def select_autophase_eles(impact_object):
    """
    Returns a on ordered list of elements and groups
    suitable for autophasing.
    """
    group_dict = impact_object.group
    lattice = impact_object.lattice

    autophase_eles = []
    autophase_groups = []
    veto_eles = set()
    for name, g in group_dict.items():
        if "theta0_deg" in g.var_name:  # Allows dtheta0_deg also
            autophase_groups.append(g)
            for ele in g.eles:
                veto_eles.add(ele["name"])
    for ele in lattice:
        if ele["name"] in veto_eles:
            continue
        if "theta0_deg" in ele and "rf_frequency" in ele:
            if ele["rf_field_scale"] == 0:
                continue
            autophase_eles.append(ele)

    all_eles = sorted(autophase_groups + autophase_eles, key=lambda ele: ele["zedge"])

    return all_eles


def fast_autophase_impact(
    impact_object,
    settings: dict = None,
    t0=0,
    pz0=0.0,
    full_output=False,
    verbose=False,
):
    """
    Parameters
    ----------
    impact_object: Impact
        LUME-Impact object with rf elements to phase

    settings: dict, optional=None
        dict of ele_name:rel_phase_deg

    t0: float, optional=0
        Initial particle time (s)

    pz0: float, optional = 0
        Initial particle z momentum (eV/c)

    full_output: bool, optional = False
        type of output to return (see Returns)

    verbose: bool, optional=False
        If True, prints useful info


    Returns
    -------
    if full_output = True retuns a dict of:
            ele_name:info_dict

    Otherwise returns a dict of:
        ele_name:rel_phase_deg
    which is the same format as settings.


    """

    species = impact_object.species
    zstop = impact_object.stop

    particle = single_particle(t=t0, pz=pz0, species=species)
    q0 = particle.species_charge / e_charge

    # Select ones for phasing
    autophase_eles = select_autophase_eles(impact_object)

    # output dict
    output = {}

    # Step through eles
    for ele in autophase_eles:
        name = ele["name"]
        zedge = ele["zedge"]

        if zedge > zstop:
            break

        if particle.z[0] != zedge:
            particle.drift_to_z(zedge)
        t0 = particle.t[0]
        pz0 = particle.pz[0]

        # print(name, t0, pz0)

        if ele["type"] == "group":
            attribute = ele.var_name
        else:
            attribute = "theta0_deg"

        rel_phase_deg = None  # does not set
        if settings:
            if name in settings:
                rel_phase_deg = settings[name]
                if verbose:
                    print(f"Setting {name} relative phase = {rel_phase_deg} deg")

        out = fast_autophase_ele(
            ele,
            attribute=attribute,
            rel_phase_deg=rel_phase_deg,
            fmaps=impact_object.fieldmaps,
            pz0=pz0,
            t0=t0,
            q0=q0,
        )

        rel_phase_deg = out["rel_phase_deg"]

        if verbose:
            print(f"Found {name:10} relative phase = {rel_phase_deg:0.2f} deg")

        particle.pz = particle.pz + out["dpz"]
        particle.z = particle.z + out["dz"]
        particle.t = particle.t + out["dt"]

        d = output[name] = {}
        d.update(out)

    settings = {name: info["rel_phase_deg"] for name, info in output.items()}

    if full_output:
        output["final_particle"] = particle
        output["settings"] = settings
        return output
    else:
        return settings
