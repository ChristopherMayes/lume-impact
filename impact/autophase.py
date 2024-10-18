from scipy.optimize import brent, brentq

from .lattice import ele_bounds


def autophase_and_scale(
    impact_object,
    phase_ele_name=None,
    phase_attribute="theta0_deg",
    scale_ele_name=None,
    scale_attribute="rf_field_scale",
    phase_range=(-180, 180),
    scale_range=(10e6, 100e6),
    initial_particles=None,
    isolate=True,
    metric="mean_energy",
    target=6e6,
    debug=False,
    algorithm="brent2",
    verbose=False,
):
    """

    Autophases and scales and element (or group).

    The bounds of the element are automatically determined. If scale or phase elements are groups,
    this will be taken into account.

    initial_particles should be a ParticleGroup, and will be tracked to the beginning of the bounds.

    if isolate, then any other elements with an rf_field_scale will be disabled.

    If debug, the phase_and_scale function and copied object will be returned.
    This is useful for algorithm development

    """

    def vprint(*a, **k):
        if verbose:
            print(*a, **k)

    # Start with a copy, so nothing gets messed up
    I = impact_object.copy()
    I.verbose = False
    I.configure()
    vprint("Copied initial Impact object. ")

    if not scale_ele_name:
        scale_ele_name = phase_ele_name

    vprint(f"Phasing {phase_ele_name} by changing {phase_attribute}")
    vprint(f"Scaling {scale_ele_name} by changing {scale_attribute}")

    pele = I[phase_ele_name]
    sele = I[scale_ele_name]

    # Get bounds
    these_names = set()
    for name in [phase_ele_name, scale_ele_name]:
        if name in I.group:
            these_names |= set(I.group[name].ele_names)
        elif name in I.ele:
            these_names.add(name)
        else:
            raise ValueError(f"{name} is not an ele or group")
    s0, s1 = ele_bounds([I.ele[name] for name in these_names])
    vprint(f"Bounds: {s0}, {s1} m")

    # Track up to the element
    P0 = initial_particles
    mean_z = P0["mean_z"]
    if mean_z < s0:
        vprint(f"Tracking initial particles to s = {s0}")
        P0 = I.track(P0, s0)
        vprint("Initial particle: ", P0["mean_z"], P0[metric])
    elif mean_z >= s1:
        raise ValueError(
            f"Initial particles start at mean_z: {mean_z} m, past the bounds {s0}, {s1} m"
        )
        vprint("Warning: paritlces are starting inside the element")

    # Disable other fields
    if isolate:
        for ele2 in I.lattice:
            if ele2["name"] in these_names:
                continue
            if "rf_field_scale" in ele2:
                ele2["rf_field_scale"] = 0
                vprint("Disabling", ele2["name"])

    def phase_and_scale(phase, scale):
        pele[phase_attribute] = phase
        sele[scale_attribute] = scale
        try:
            P = I.track(P0, s=s1)
            if P:
                en = P[metric]
                unit = P.units(metric)
            else:
                en = 0
                unit = ""
            vprint(f"Phase: {phase}, Scale: {scale}, {en/1e6} M{unit}")
        except Exception as ex:
            vprint(f"Exception with Phase: {phase}, Scale: {scale},{ex}")
            en = 0
        return en

    if debug:
        return phase_and_scale, I

    if algorithm == "brent2":
        vprint("Default brent2 algorithm")
        phase1, scale1 = autophase_and_scale_brent2(
            phase_and_scale,
            target=target,
            phase_range=phase_range,
            scale_range=scale_range,
            verbose=verbose,
        )

    else:
        vprint("Custom algorithm")
        phase1, scale1 = algorithm(
            phase_and_scale,
            target=target,
            phase_range=phase_range,
            scale_range=scale_range,
        )

    # mod
    phase1 = phase1 % 360

    # Set original object
    impact_object[phase_ele_name][phase_attribute] = phase1
    impact_object[scale_ele_name][scale_attribute] = scale1

    vprint(f"Set Phase: {phase1}, Scale: {scale1}")

    return phase1, scale1


def autophase_and_scale_brent2(
    phase_scale_f,
    target=10e6,
    phase_range=(-180, 180),
    scale_range=(10e6, 100e6),
    verbose=False,
):
    """
    Two step autophase and scale algorithm with brent and brentq.

    """

    s0, s1 = scale_range
    scale0 = s0  # (s0+s1)/2

    brack = phase_range

    phase0 = (
        brent(
            lambda x: -phase_scale_f(x % 360, scale0) / target + 1.0,
            brack=brack,
            maxiter=30,
            tol=1e-3,
            full_output=False,
        )
        % 360
    )
    if verbose:
        print("Step 1 phasing found:", phase0)

    scale0 = brentq(
        lambda x: phase_scale_f(phase0, x) / target - 1.0,
        s0,
        s1,
        maxiter=20,
        rtol=1e-3,
        full_output=False,
    )
    if verbose:
        print("Step 2  scale found:", scale0)
    brack = (phase0 - 1, phase0 + 1)
    phase1 = (
        brent(
            lambda x: -phase_scale_f(x % 360, scale0) / target + 1.0,
            brack=brack,
            maxiter=20,
            tol=1e-6,
            full_output=False,
        )
        % 360
    )
    if verbose:
        print("Step 3 phase found: ", phase1)

    scale1 = brentq(
        lambda x: phase_scale_f(phase1, x) / target - 1.0,
        s0,
        s1,
        maxiter=20,
        rtol=1e-6,
        full_output=False,
    )
    if verbose:
        print("Step 4 scale found: ", scale1)
    # print("Solution")
    # ps_f(phase1, scale1)

    return phase1, scale1


def autophase(
    impact_object,
    ele_name=None,
    attribute="theta0_deg",
    phase_range=(-180, 180),
    initial_particles=None,
    isolate=True,
    metric="mean_energy",
    maximize=True,
    debug=False,
    algorithm="brent",
    s_stop=None,
    verbose=False,
):
    """

    Simplified version of autophase_and_scale

    """

    def vprint(*a, **k):
        if verbose:
            print(*a, **k)

    # Start with a copy, so nothing gets messed up
    I = impact_object.copy()
    I.verbose = False
    I.configure()
    vprint("Copied initial Impact object. ")

    vprint(f"Phasing {ele_name} by changing {attribute}")
    pele = I[ele_name]

    # Get bounds
    these_names = set()
    if ele_name in I.group:
        these_names |= set(I.group[ele_name].ele_names)
    elif ele_name in I.ele:
        these_names.add(ele_name)
    else:
        raise ValueError(f"{ele_name} is not an ele or group")
    s0, s1 = ele_bounds([I.ele[name] for name in these_names])

    if s_stop:
        s1 = s_stop

    vprint(f"Bounds: {s0}, {s1} m")

    # Track up to the element
    P0 = initial_particles
    mean_z = P0["mean_z"]
    if mean_z < s0:
        vprint(f"Tracking initial particles to s = {s0}")
        P0 = I.track(P0, s0)
        vprint("Initial particle: ", P0["mean_z"], P0[metric])
    elif mean_z >= s1:
        raise ValueError(
            f"Initial particles start at mean_z: {mean_z} m, past the bounds {s0}, {s1} m"
        )
        vprint("Warning: paritlces are starting inside the element")

    # Disable other fields
    if isolate:
        for ele2 in I.lattice:
            if ele2["name"] in these_names:
                continue
            if "rf_field_scale" in ele2:
                ele2["rf_field_scale"] = 0
                vprint("Disabling", ele2["name"])

    def phase_f(phase):
        pele[attribute] = phase
        try:
            P = I.track(P0, s=s1)
            if P:
                en = P[metric]
                unit = P.units(metric)
            else:
                en = 0
                unit = ""
            vprint(f"Phase: {phase}, {en/1e6} M{unit}")
        except Exception as ex:
            vprint(f"Exception with Phase: {phase}, {ex}")
            en = 0
        return en

    if debug:
        return phase_f, I

    if maximize:
        alg_sign = -1
    else:
        alg_sign = 1

    if algorithm == "brent":
        vprint("Default brent2 algorithm")
        phase1 = (
            brent(
                lambda x: alg_sign * phase_f(x % 360),
                brack=phase_range,
                maxiter=20,
                tol=1e-6,
                full_output=False,
            )
            % 360
        )

    else:
        vprint("Custom algorithm")
        phase1 = algorithm(phase_f, phase_range=phase_range)

    # mod
    phase1 = phase1 % 360

    # Set original object
    impact_object[ele_name][attribute] = phase1

    vprint(f"Set Phase: {phase1}")

    return phase1
