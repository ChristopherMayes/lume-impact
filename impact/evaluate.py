import numpy as np


def default_impact_merit(I):
    """
    merit function to operate on an evaluated LUME-Impact object I.

    Returns dict of scalar values
    """
    # Check for error
    if I.output["run_info"]["error"]:
        return {"error": True}
    else:
        m = {"error": False}

    # Gather stat output
    for k in I.output["stats"]:
        m["end_" + k] = I.output["stats"][k][-1]

    m["run_time"] = I.output["run_info"]["run_time"]

    P = I.particles["final_particles"]

    # All impact particles read back have status==1
    #
    ntotal = int(I.stat("n_particle").max())
    nlost = ntotal - len(P)

    m["end_n_particle_loss"] = nlost

    # Get live only for stat calcs
    P = P.where(P.status == 1)

    # No live particles
    if len(P) == 0:
        return {"error": True}

    # Special
    m["end_total_charge"] = P["charge"]
    m["end_higher_order_energy_spread"] = P["higher_order_energy_spread"]
    m["end_norm_emit_xy"] = np.sqrt(m["end_norm_emit_x"] * m["end_norm_emit_y"])
    m["end_norm_emit_4d"] = P["norm_emit_4d"]

    # Remove annoying strings
    if "why_error" in m:
        m.pop("why_error")

    return m
