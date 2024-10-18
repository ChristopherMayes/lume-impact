ele_type = {
    0: "drift",
    1: "quadrupole",
    2: "constfoc",
    3: "solenoid",
    4: "dipole",
    5: "multipole",
    101: "drift_tube_linac",
    204: "srf_cavity",
    105: "solrf",
    110: "emfield",
    111: "emfield_cartesian",
    112: "emfield_cylindrical",
    113: "emfield_analytical",
    -1: "offset_beam",
    -2: "write_beam",
    -3: "write_beam_for_restart",
    -4: "change_timestep",
    -5: "rotationally_symmetric_to_3d",
    -6: "wakefield",
    -7: "merge_bins",
    -8: "spacecharge",
    -9: "write_slice_info",
    -11: "collomate",
    -12: "matrix",
    -13: "dielectric_wakefield",
    -15: "point_to_point_spacecharge",
    -16: "heat_beam",
    -17: "rotate_beam",
    -99: "stop",
}
# Inverse dictionary
itype_of = {}
for k in ele_type:
    itype_of[ele_type[k]] = k


# LLine["drift"] = [("zedge", ""), ("radius", "radius (Not used)")]
