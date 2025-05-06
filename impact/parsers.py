import os
import re
import warnings

import numpy as np
import polars as pl
from pmd_beamphysics.species import mass_of
from pmd_beamphysics.units import multiply_units, unit

from . import fieldmaps, tools
from .particles import identify_species
from .tools import parse_float

# -----------------
# Parsing ImpactT input file


# -----------------
# ImpactT Header
# lattice starts at line 10

# Header dicts
HNAMES = {}
HTYPES = {}
HDEFAULTS = {}
# Line 1
HNAMES[1] = ["Npcol", "Nprow"]
HTYPES[1] = [int, int]
HDEFAULTS[1] = [1, 1]

# Line 2
HNAMES[2] = ["Dt", "Ntstep", "Nbunch"]
HTYPES[2] = [float, int, int]
HDEFAULTS[2] = [0, 100000000, 1]  # Dt must be set

# Line 3
HNAMES[3] = ["Dim", "Np", "Flagmap", "Flagerr", "Flagdiag", "Flagimg", "Zimage"]
HTYPES[3] = [int, int, int, int, int, int, float]
HDEFAULTS[3] = [999, 0, 1, 0, 2, 1, 0.02]

# Line 4
HNAMES[4] = ["Nx", "Ny", "Nz", "Flagbc", "Xrad", "Yrad", "Perdlen"]
HTYPES[4] = [int, int, int, int, float, float, float]
HDEFAULTS[4] = [32, 32, 32, 1, 0.015, 0.015, 100.0]

# Line 5
HNAMES[5] = ["Flagdist", "Rstartflg", "Flagsbstp", "Nemission", "Temission"]
HTYPES[5] = [int, int, int, int, float]
HDEFAULTS[5] = [16, 0, 0, 400, 1.4e-11]

# Line 6-8
HNAMES[6] = ["sigx(m)", "sigpx", "muxpx", "xscale", "pxscale", "xmu1(m)", "xmu2"]
HTYPES[6] = [float for i in range(len(HNAMES[6]))]
HDEFAULTS[6] = [0.0 for i in range(len(HNAMES[6]))]
HNAMES[7] = ["sigy(m)", "sigpy", "muxpy", "yscale", "pyscale", "ymu1(m)", "ymu2"]
HTYPES[7] = HTYPES[6]
HDEFAULTS[7] = [0.0 for i in range(len(HNAMES[7]))]
HNAMES[8] = ["sigz(m)", "sigpz", "muxpz", "zscale", "pzscale", "zmu1(m)", "zmu2"]
HTYPES[8] = HTYPES[6]
HDEFAULTS[8] = [0.0 for i in range(len(HNAMES[8]))]

# Line 9
HNAMES[9] = ["Bcurr", "Bkenergy", "Bmass", "Bcharge", "Bfreq", "Tini"]
HTYPES[9] = [float for i in range(len(HNAMES[9]))]
HDEFAULTS[9] = [1.0, 1.0, 510998.946, -1.0, 2856000000.0, 0.0]

# Collect all these
HEADER_NAMES = []
HEADER_TYPES = []
HEADER_DEFAULTS = []
for i in range(1, 10):
    HEADER_NAMES.append(HNAMES[i])
    HEADER_TYPES.append(HTYPES[i])
    HEADER_DEFAULTS.append(HDEFAULTS[i])
# Flattened version
ALL_HEADER_NAMES = [item for sublist in HEADER_NAMES for item in sublist]
ALL_HEADER_TYPES = [item for sublist in HEADER_TYPES for item in sublist]
ALL_HEADER_DEFAULTS = [item for sublist in HEADER_DEFAULTS for item in sublist]
# Make dicts
HEADER_DEFAULT = dict(zip(ALL_HEADER_NAMES, ALL_HEADER_DEFAULTS))
HEADER_TYPE = dict(zip(ALL_HEADER_NAMES, ALL_HEADER_TYPES))


def header_bookkeeper(header, defaults=HEADER_DEFAULT, verbose=True):
    """
    Checks header for missing or bad keys, fills in defaults.
    """
    # Check for bad keys
    for k in header:
        if verbose and k not in defaults:
            print("Warning:", k, "does not belong in header.")
    newheader = header.copy()
    # Fill defaults
    for k in defaults:
        if k not in newheader:
            val = defaults[k]
            newheader[k] = val
            if verbose:
                print("Header bookkeeper: Filling in default for", k, "=", val)

    return newheader


# -----------------
# Some help for keys above
help = {}
# Line 1
help["Npcol"] = (
    "Number of columns of processors, used to decompose domain along Y dimension."
)
help["Nprow"] = (
    "Number of rows of processors, used to decompose domain along Z dimension."
)
# Line 2
help["Dt"] = "Time step size (secs)."
help["Ntstep"] = "Maximum number of time steps."
help["Nbunch"] = (
    "The initial distribution of the bunch can be divided longitudinally into Nbunch slices. See the manual."
)
# Line 3
help["Dim"] = "Random seed integer"
help["Np"] = "Number of macroparticles to track"
help["Flagmap"] = "Type of integrator. Currently must be set to 1."
help["Flagerr"] = (
    "Error study flag. 0 - no misalignment and rotation errors; 1 - misalignment and rotation errors are allowed for Quadrupole, Multipole (Sextupole, Octupole, Decapole) and SolRF elements. This function can also be used to simulate the beam transport through rotated beam line elements such as skew quadrupole etc."
)
help["Flagdiag"] = (
    "Diagnostics flag: 1 - output the information at given time, 2 - output the information at the location of bunch centroid by drifting the particles to that location, 3 or more - no output."
)
help["Flagimg"] = (
    "Image charge flag. If set to 1 then the image charge forces due to the cathode are included. The cathode is always assumed to be at z = 0. To not include the image charge forces set imchgF to 0."
)
help["Zimage"] = (
    "z position beyond which image charge forces are neglected. Set z small to speed up the calculation but large enough so that the results are not affected."
)
# Line 4
help["Nx"] = "Number of mesh points in x"
help["Ny"] = "Number of mesh points in y"
help["Nz"] = "Number of mesh points in z"
help["Flagbc"] = (
    "Field boundary condition flag: Currently must be set to 1 which corresponds to an open boundary condition."
)
help["Xrad"] = "Computational domain size in x"
help["Yrad"] = "Computational domain size in x"
help["Perdlen"] = (
    "Computational domain size in z. Must be greater than the lattice length"
)
# Line 5
help["Flagdist"] = "Type of the initial distribution"
help["Rstartflg"] = (
    "If restartf lag = 1, restart the simulation from the previous check point. If restartf lag = 0, start the simulation from the beginning."
)
help["Flagsbstp"] = "Not used."
help["Nemission"] = (
    "There is a time period where the laser is shining on the cathode and electrons are being emitted. Nemisson gives the number of numerical emission steps. More steps gives more accurate modeling but the computation time varies linearly with the number of steps. If Nemission < 0, there will be no cathode model. The particles are assumed to start in a vacuum."
)
help["Temission"] = (
    "Laser pulse emission time (sec.) Note, this time needs to be somewhat greater than the real emission time in the initial longitudinal distribution so that the time step size is changed after the whole beam is a few time steps out of the cathode."
)

# Line 6-8
help["sigx(m)"] = "Distribution sigma_x in meters"
help["sigpx"] = "Distribution sigma_px, where px is gamma*beta_x"
help["muxpx"] = "Distribution correlation <x px>, where px is gamma*beta_x"
help["xscale"] = "Scale factor for distribution x"
help["pxscale"] = "Scale factor for distribution px"
help["xmu1(m)"] = "Distribution mean for x in meters"
help["xmu2"] = "Distribution mean for px, where px is gamma*beta_x"

help["sigy(m)"] = "Distribution sigma_y in meters"
help["sigpy"] = "Distribution sigma_py, where px is gamma*beta_y"
help["muypy"] = "Distribution correlation <y py>, where py is gamma*beta_y"
help["yscale"] = "Scale factor for distribution y"
help["pyscale"] = "Scale factor for distribution py"
help["ymu1(m)"] = "Distribution mean for y in meters"
help["ymu2"] = "Distribution mean for py, where py is gamma*beta_y"

help["sigz(m)"] = "Distribution sigma_z in meters"
help["sigpz"] = "Distribution sigma_pz, where pz is gamma*beta_z"
help["muzpz"] = "Distribution correlation <z pz>, where pz is gamma*beta_z"
help["zscale"] = "Scale factor for distribution z"
help["pzscale"] = "Scale factor for distribution pz"
help["zmu1(m)"] = "Distribution mean for z in meters"
help["zmu2"] = "Distribution mean for pz, where pz is gamma*beta_z"


# Line 9
help["Bcurr"] = "Beam current in Amps"
help["Bkenergy"] = (
    "Initial beam kinetic energy in eV. WARNING: this one is only used to drift the particle out of the wall. The real initial beam energy needs to be input from xmu6 in the initial distribution or the particle data file for the readin distribution."
)
help["Bmass"] = "Mass of the particles in eV."
help["Bcharge"] = "Particle charge in units of proton charge."
help["Bfreq"] = "Reference frequency in Hz."
help["Tini"] = "Initial reference time in seconds."


def header_str(H):
    """
    Summary information about the header
    """
    qb_pC = H["Bcurr"] / H["Bfreq"] * 1e12
    Nbunch = H["Nbunch"]

    species = identify_species(H["Bmass"], H["Bcharge"])

    if H["Flagimg"]:
        start_condition = (
            "Cathode start at z = 0 m\n   emission time: "
            + str(H["Temission"])
            + " s\n   image charges neglected after z = "
            + str(H["Zimage"])
            + " m"
        )

    else:
        start_condition = "Free space start"

    if H["Rstartflg"] == 0:
        restart_condition = "Simulation starting from the beginning"
    elif ["Rstartflg"] == 1:
        restart_condition = "Restarting simulation from checkpoint."
    else:
        restart_condition = "Bad restart condition: " + str(["Rstartflg"])

    dist_type = distrubution_type(H["Flagdist"])

    summary = f"""================ Impact-T Summary ================
{H["Np"]} particles
{Nbunch} bunch of {species}s
total charge: {qb_pC} pC
Distribution type: {dist_type}
{start_condition}
Processor domain: {H["Nprow"]} x {H["Npcol"]} = {H["Nprow"] * H["Npcol"]} CPUs
Space charge grid: {H["Nx"]} x {H["Ny"]} x {H["Nz"]}
Maximum time steps: {H["Ntstep"]}
Reference Frequency: {H["Bfreq"]} Hz
Initial reference time: {H["Tini"]} s
{restart_condition}
=================================================
"""

    return summary


# -----------------
# Distribution types


def distrubution_type(x):
    """
    Returns a named distribution type from an integer x

    """
    if x in DIST_TYPE:
        return DIST_TYPE[x]

    # Expect Combine
    x = str(x)
    assert len(x) == 3

    dtypes = [DIST_TYPE[int(i)] for i in x]
    return "_".join(dtypes)


DIST_TYPE = {
    1: "uniform",
    2: "gauss3",
    3: "waterbag",
    4: "semigauss",
    5: "kv3d",
    16: "read",
    24: "readParmela",
    25: "readElegant",
    27: "colcoldzsob",
    112: "beercan",  # todo: make general
}
# TODO: ijk distribution

# Inverse
DIST_ITYPE = {}
for k, v in DIST_TYPE.items():
    DIST_ITYPE[v] = k


# -----------------
# Util


def is_commented(line, commentchar="!"):
    sline = line.strip()  # remove spaces
    if len(sline) == 0:
        return True
    return sline[0] == commentchar


# Strip comments, and trailing comments
def remove_comments(lines):
    return [l.split("!")[0] for l in lines if len(l) > 0 and not is_commented(l)]


def parse_line(line, names, types):
    """
    parse line with expected names and types

    Example:
        parse_line(line, ['dt', 'ntstep', 'nbunch'], [float, int,int])

    """
    x = line.split()
    values = [types[i](x[i].lower().replace("d", "e")) for i in range(len(x))]
    return dict(zip(names, values))


def parse_header(lines):
    x = remove_comments(lines)
    d = {}
    for i in range(9):
        d.update(parse_line(x[i], HEADER_NAMES[i], HEADER_TYPES[i]))
    return d


#
def ix_lattice(lines):
    """
    Find index of beginning of lattice, end of header

    """
    slines = remove_comments(lines)
    latline = slines[9]
    for i in range(len(lines)):
        if latline in lines[i]:
            return i

    print("Error: no lattice found in stripped lines:", slines)
    raise


def v_from_line(line):
    """
    Extracts the V values from an element line.

    Returns:
        v: list, with v[0] as the original line, so the indexing starts at 1
    """
    v = line.split("/")[0].split()[3:]  # V data starts with index 4
    v[0] = line  # Preserve original line
    return v


def header_lines(header_dict, annotate=True):
    """
    Re-forms the header dict into lines to be written to the Impact-T input file
    """

    line0 = "! Impact-T input file"
    lines = [line0]
    for i in range(1, 10):
        names = HNAMES[i]

        if annotate:
            lines.append("!" + " ".join(names))

        x = " ".join([str(header_dict[n]) for n in names])
        lines.append(x)
    #' '.join(lines)
    return lines


# -----------------
# Lattice
"""

The Impact-T lattice elements are defined with lines of the form:
Blength, Bnseg, Bmpstp, Btype, V1 ... V23

The V list depends on the Btype.

"""

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Parsers for each type of line

"""
Element type.
"""
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


def parse_type(line):
    """
    Parse a lattice line. This is the fourth item in the line.

    Returns the type as a string.

    """
    if is_commented(line):
        return "comment"
    i = int(line.split()[3])
    if i in ele_type:
        return ele_type[i]
    else:
        # print('Undocumented: ', line)
        return "undocumented"


# Collect a dict of type:list of valid keys
VALID_KEYS = {}
ELE_DEFAULTS = {}


# -----------------------------------------------------------------
def parse_drift(line):
    """
    Drift (type 0)

    V1: zedge
    V2: radius Not used.
    """
    v = v_from_line(line)
    d = {}
    d["zedge"] = parse_float(v[1])
    d["radius"] = parse_float(v[2])
    return d


def drift_v(ele):
    """
    Drift V list from ele dict

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    v = [ele, ele["zedge"], 0.0]
    # optional
    if "radius" in ele:
        v[2] = ele["radius"]
    return v


ELE_DEFAULTS["drift"] = {"zedge": 0, "radius": 0}


# -----------------------------------------------------------------
def parse_misalignments(v):
    """
    Parse misalignment portion of V list
    Common to several elements
    """
    d = {}
    d["x_offset"] = parse_float(v[0])
    d["y_offset"] = parse_float(v[1])
    d["x_rotation"] = parse_float(v[2])
    d["y_rotation"] = parse_float(v[3])
    d["z_rotation"] = parse_float(v[4])
    return d


def misalignment_v(ele):
    """
    V list for misalignments
    """
    v = [0.0, 0.0, 0.0, 0.0, 0.0]
    if "x_offset" in ele:
        v[0] = ele["x_offset"]
    if "y_offset" in ele:
        v[1] = ele["y_offset"]
    if "x_rotation" in ele:
        v[2] = ele["x_rotation"]
    if "y_rotation" in ele:
        v[3] = ele["y_rotation"]
    if "z_rotation" in ele:
        v[4] = ele["z_rotation"]

    return v


ELE_DEFAULTS["misalignment"] = {
    "x_offset": 0,
    "y_offset": 0,
    "x_rotation": 0,
    "y_rotation": 0,
    "z_rotation": 0,
}


# -----------------------------------------------------------------
def parse_point_to_point_spacecharge(line):
    """
    (type -15)
    Switch on the direct point-to-point N-body calculation of the space-charge forces.

    Warning:
        The # of electrons divided by the number of processors should be an integer.

    See:
        Ji Qiang et. al,  Numerical Study of Coulomb Scattering Effects on Electron Beam
        from a Nano-Tip,
        in proceedings of PAC07 conference, June 25-29, Albuquerque, p. 1185, 2007
        https://accelconf.web.cern.ch/p07/PAPERS/TUPMN116.PDF



    V3: cut-off radius of aparticle
    """
    v = v_from_line(line)
    d = {}
    d["cutoff_radius"] = parse_float(v[3])
    return d


def parse_point_to_point_spacecharge_v(ele):
    """
    point_to_point_spacecharge V list from ele dict

    v[0] is the original ele

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    v = [ele, 0, 0, ele["cutoff_radius"]]

    return v


ELE_DEFAULTS["point_to_point_spacecharge"] = {
    "cutoff_radius": 2.8179e-15,  # classical electron radius
}


# -----------------------------------------------------------------
def parse_quadrupole(line):
    """
    Quadrupole (type 1)

    V1: zedge
    V2: quad gradient (T/m)
    V3: file ID
        If > 0, then include fringe field (using Enge function) and
        V3 = effective length of quadrupole.
    V4: radius (m)
    V5: x misalignment error (m)
    V6: y misalignment error (m)
    V7: rotation error x (rad)
    V8: rotation error y (rad)
    V9: rotation error z (rad)

    If V9 != 0, skew quadrupole
    V10: rf quadrupole frequency (Hz)
    V11: rf quadrupole phase (degree)
    """

    v = v_from_line(line)
    d = {}
    d["zedge"] = parse_float(v[1])
    d["b1_gradient"] = parse_float(v[2])
    if parse_float(v[3]) > 0:
        d["L_effective"] = parse_float(v[3])
    else:
        d["file_id"] = int(v[3])
    d["radius"] = parse_float(v[4])
    d2 = parse_misalignments(v[5:10])
    d.update(d2)
    if len(v) > 11:
        d["rf_frequency"] = parse_float(v[10])
        d["rf_phase_deg"] = parse_float(v[11])
    return d


def quadrupole_v(ele):
    """
    Quadrupole V list from ele dict

    V[0] is the original ele

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    v = [ele, ele["zedge"], ele["b1_gradient"], 0, 0.0]
    if "file_id" in ele:
        v[3] = ele["file_id"]
    else:
        v[3] = ele["L_effective"]

    # optional
    if "radius" in ele:
        v[4] = ele["radius"]

    # misalignment list
    v += misalignment_v(ele)

    if "rf_frequency" and "rf_phase_deg" in ele:
        v += [ele["rf_frequency"], ele["rf_phase_deg"]]

    return v


ELE_DEFAULTS["quadrupole"] = {
    "zedge": 0,
    "b1_gradient": 0,
    "L_effective": 0,
    "file_id": 0,
    "radius": 0,
    "rf_frequency": 0,
    "rf_phase_deg": 0,
    "x_offset": 0,
    "y_offset": 0,
    "x_rotation": 0,
    "y_rotation": 0,
    "z_rotation": 0,
}


# -----------------------------------------------------------------
def parse_solenoid(line):
    """

    (type 3)

    V1: zedge
    V2: Bz0 (T)
    V3: file ID
    V4: radius
    V5: x misalignment error Not used.
    V6: y misalignment error Not used.
    V7: rotation error x Not used.
    V8: rotation error y Not used.
    V9: rotation error z Not used.

    The discrete magnetic field data is stored in 1T<V3>.T7 file,
    where <V3> is the file ID integer above

    The read in format of 1T#.T7 is in the manual.
    """

    v = v_from_line(line)
    d = {}
    d["zedge"] = parse_float(v[1])
    d["b_field"] = parse_float(v[2])
    d["filename"] = "1T" + str(int(parse_float(v[3]))) + ".T7"
    d["radius"] = parse_float(v[4])
    # Not used: d2 = parse_misalignments(v[5:10])
    # d.update(d2)

    return d


def solenoid_v(ele):
    """
    Solenoid V list from ele dict

    V[0] is the original ele

    """
    # Let v[0] be the original ele, so the indexing looks the same.

    file_id = int(ele["filename"].split("1T")[1].split(".")[0])

    v = [ele, ele["zedge"], ele["b_field"], file_id, ele["radius"]]

    # misalignment list
    v += misalignment_v(ele)

    return v


ELE_DEFAULTS["solenoid"] = {
    "zedge": 0,
    "b_field": 0,
    "filename": "1T99.T7",
    "radius": 0,
    "x_offset": 0,
    "y_offset": 0,
    "x_rotation": 0,
    "y_rotation": 0,
    "z_rotation": 0,
}


# -----------------------------------------------------------------
def parse_dipole(line):
    """
    Dipole (type 4)

    V1:  zedge
    V2:  x field strength (T)
    V3:  y field strength (T)
    V4:  file ID file ID to contain the geometry information of bend.
    V5:  half of gap width (m).
    V6:  x misalignment error Not used.
    V7:  y misalignment error Not used.
    V8:  rotation error x Not used.
    V9:  rotation error y Not used.
    V10: rotation error z Not used.

    """
    v = v_from_line(line)
    d = {}
    d["zedge"] = parse_float(v[1])
    d["b_field_x"] = parse_float(v[2])
    d["b_field"] = parse_float(v[3])
    d["filename"] = "rfdata" + str(int(parse_float(v[4])))
    d["half_gap"] = parse_float(v[5])
    return d


def dipole_v(ele):
    """
    dipole Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.

    # Get file integer
    f = ele["filename"]
    ii = int(f.split("rfdata")[1])

    v = [ele, ele["zedge"], ele["b_field_x"], ele["b_field"], ii, ele["half_gap"]]

    return v


ELE_DEFAULTS["dipole"] = {
    "zedge": 0,
    "b_field_x": 0,
    "b_field": 0,
    "filename": "rfdata99",
    "half_gap": 0,
}


# -----------------------------------------------------------------
def parse_solrf(line):
    """
    Solrf (type 105)

    V1: zedge, the real used field range in z is [zedge,zedge+Blength].
    V2: scale of RF field
    V3: RF frequency
    V4: theta0  Initial phase in degree.
    V5: file ID
    V6: radius
    V7: x misalignment error
    V8: y misalignment error
    V9: rotation error x
    V10: rotation error y
    V11: rotation error z
    V12: scale of solenoid B field. [Only used with SolRF element.]
    """
    v = v_from_line(line)
    d = {}
    d["zedge"] = parse_float(v[1])
    d["rf_field_scale"] = parse_float(v[2])
    d["rf_frequency"] = parse_float(v[3])
    d["theta0_deg"] = parse_float(v[4])  #
    d["filename"] = "rfdata" + str(int(parse_float(v[5])))
    d["radius"] = parse_float(v[6])
    d2 = parse_misalignments(v[7:12])
    d.update(d2)
    d["solenoid_field_scale"] = parse_float(v[12])
    return d


def solrf_v(ele):
    """
    Solrf V list from ele dict.

    V[0] is the original ele

    """

    # Let v[0] be the original ele, so the indexing looks the same.
    file_id = int(ele["filename"].split("rfdata")[1])
    v = [ele, ele["zedge"], 1.0, 0.0, 0.0, file_id, 0.0]

    # optional
    if "rf_field_scale" in ele:
        v[2] = ele["rf_field_scale"]
    if "rf_frequency" in ele:
        v[3] = ele["rf_frequency"]
    if "theta0_deg" in ele:
        v[4] = ele["theta0_deg"]
    if "radius" in ele:
        v[6] = ele["radius"]

    # misalignment list
    v += misalignment_v(ele)

    if "solenoid_field_scale" in ele:
        v.append(ele["solenoid_field_scale"])
    else:
        v.append(0.0)

    return v


ELE_DEFAULTS["solrf"] = {
    "zedge": 0,
    "rf_field_scale": 0,
    "rf_frequency": 0,
    "theta0_deg": 0,
    "filename": "rfdata99",
    "radius": 0,
    "solenoid_field_scale": 0,
    "x_offset": 0,
    "y_offset": 0,
    "x_rotation": 0,
    "y_rotation": 0,
    "z_rotation": 0,
}


# -----------------------------------------------------------------


def parse_emfield_cartesian(line):
    """
    emfield_cartesian
    111: EMfldCart

    Read in discrete EM field data Ex(MV/m), Ey(MV/m), Ez(MV/m), Bx(MV/m), By(MV/m), Bz(MV/m), as a function of (x,y,z).

    V1: zedge
    V2: rf_field_scale
    V3: RF frequency
    V4: theta0_deg
    V5: file ID
    V6: radius               Not used yet
    V7: x misalignment error Not used yet
    V8: y misalignment error Not used yet
    V9: rotation error x     Not used yet
    V10: rotation error y    Not used yet
    V11: rotation error z    Not used yet

    The discrete field data is stored in 1Tv3.T7 file.
    The read in format of 1Tv3.T7 is in the manual.

    """

    v = v_from_line(line)
    d = {}
    d["zedge"] = parse_float(v[1])
    d["rf_field_scale"] = parse_float(v[2])
    d["rf_frequency"] = parse_float(v[3])
    d["theta0_deg"] = parse_float(v[4])
    d["filename"] = "1T" + str(int(parse_float(v[5]))) + ".T7"
    d["radius"] = parse_float(v[6])
    # Not used: d2 = parse_misalignments(v[7:12])
    # d.update(d2)

    return d


def emfield_cartesian_v(ele):
    """
    emfield_cartesian V list from ele dict

    V[0] is the original ele

    """
    # Let v[0] be the original ele, so the indexing looks the same.

    file_id = int(ele["filename"].split("1T")[1].split(".")[0])

    v = [
        ele,
        ele["zedge"],
        ele["rf_field_scale"],
        ele["rf_frequency"],
        ele["theta0_deg"],
        file_id,
        ele["radius"],
    ]

    # misalignment list (Should be zeros)
    v += misalignment_v(ele)

    return v


ELE_DEFAULTS["emfield_cartesian"] = {
    "zedge": 0,
    "rf_field_scale": 0,
    "rf_frequency": 0,
    "theta0_deg": 0,
    "filename": 0,
    "radius": 0,
    "x_offset": 0,
    "y_offset": 0,
    "x_rotation": 0,
    "y_rotation": 0,
    "z_rotation": 0,
}


def parse_emfield_cylindrical(line):
    """
    emfield_cylindrical
    112: EMfldCyl

    Read in discrete EM field data Ez(MV/m), Er(MV/m), and Hθ(A/m)
    as a function of (r,z) of EM field data (from SUPERFISH output).

    V1: zedge
    V2: rf_field_scale       ! Warning: The manual says 'radius'
    V3: RF frequency
    V4: theta0_deg
    V5: file ID
    V6: radius               Not used yet
    V7: x misalignment error Not used yet
    V8: y misalignment error Not used yet
    V9: rotation error x     Not used yet
    V10: rotation error y    Not used yet
    V11: rotation error z    Not used yet

    The discrete field data is stored in 1Tv3.T7 file.
    The read in format of 1Tv3.T7 is in the manual.

    """

    v = v_from_line(line)
    d = {}
    d["zedge"] = parse_float(v[1])
    d["rf_field_scale"] = parse_float(v[2])
    d["rf_frequency"] = parse_float(v[3])
    d["theta0_deg"] = parse_float(v[4])
    d["filename"] = "1T" + str(int(parse_float(v[5]))) + ".T7"
    d["radius"] = parse_float(v[6])
    # Not used: d2 = parse_misalignments(v[7:12])
    # d.update(d2)

    return d


def emfield_cylindrical_v(ele):
    """
    emfield_cylindrical V list from ele dict

    V[0] is the original ele

    """
    # Let v[0] be the original ele, so the indexing looks the same.

    file_id = int(ele["filename"].split("1T")[1].split(".")[0])

    v = [
        ele,
        ele["zedge"],
        ele["rf_field_scale"],
        ele["rf_frequency"],
        ele["theta0_deg"],
        file_id,
        ele["radius"],
    ]

    # misalignment list (Should be zeros)
    v += misalignment_v(ele)

    return v


ELE_DEFAULTS["emfield_cylindrical"] = {
    "zedge": 0,
    "rf_field_scale": 0,
    "rf_frequency": 0,
    "theta0_deg": 0,
    "filename": 0,
    "radius": 0,
    "x_offset": 0,
    "y_offset": 0,
    "x_rotation": 0,
    "y_rotation": 0,
    "z_rotation": 0,
}


# -----------------------------------------------------------------
def parse_offset_beam(line, warn=False):
    """
    offset_beam (type -1)

    If btype = −1, steer the transverse beam centroid at given location V2(m) to position
    x offset        V3(m)
    Px (γβx) offset V4
    y offset        V5(m)
    Py (γβy) offset V6
    z offset        V7(m)
    Pz (γβz) offset V8

    Assumes zero if these are not present.
    """

    v = v_from_line(line)
    d = {}
    ##print (v, len(v))
    d["s"] = parse_float(v[2])
    olist = ["x_offset", "px_offset", "y_offset", "py_offset", "z_offset", "pz_offset"]
    for i in range(6):
        if i + 3 > len(v) - 1:
            val = 0.0
        else:
            ## print('warning: offset_beam missing numbers. Assuming zeros', line)
            val = parse_float(v[i + 3])
        d[olist[i]] = val
    return d


def offset_beam_v(ele):
    """
    offset_beam Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, ele["s"]] + [
        ele[x]
        for x in [
            "x_offset",
            "px_offset",
            "y_offset",
            "py_offset",
            "z_offset",
            "pz_offset",
        ]
    ]

    return v


ELE_DEFAULTS["offset_beam"] = {
    "s": 0,
    "x_offset": 0,
    "px_offset": 0,
    "y_offset": 0,
    "py_offset": 0,
    "z_offset": 0,
    "pz_offset": 0,
}


# -----------------------------------------------------------------
def parse_write_beam(line):
    """
    Write_beam (type -2)

    If btype = −2, output particle phase-space coordinate information at given location V3(m)
    into filename fort.Bmpstp with particle sample frequency Bnseg. Here, the maximum number
    of phase- space files which can be output is 100. Here, 40 and 50 should be avoided
    since these are used for initial and final phase space output.
    """
    x = line.split()
    v = v_from_line(line)
    d = {}
    d["filename"] = "fort." + x[2]
    d["sample_frequency"] = int(x[1])
    d["s"] = parse_float(v[3])
    if int(x[2]) in [40, 50]:
        print("warning, overwriting file fort." + x[2])
    return d


def write_beam_v(ele):
    """
    write_beam Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, dummy, ele["s"]]
    return v


ELE_DEFAULTS["write_beam"] = {"s": 0, "filename": "fort.99", "sample_frequency": 1}


# -----------------------------------------------------------------
def parse_write_beam_for_restart(line):
    """
    Write_beam_for_restart (type -3)

    If btype = −3, output particle phase-space and prepare restart at given location V3(m)
    into filename fort.(Bmpstp+myid). Here, myid is processor id. On single processor, it is 0.
    If there are multiple restart lines in the input file, only the last line matters.

    """
    x = line.split()
    v = v_from_line(line)
    d = {}
    d["filename"] = "fort." + x[2] + "+myid"
    d["s"] = parse_float(v[3])
    return d


def write_beam_for_restart_v(ele):
    """
    write_beam_for_restart Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, dummy, ele["s"]]
    return v


ELE_DEFAULTS["write_beam_for_restart"] = {"s": 0, "filename": "fort.99+myid"}


# -----------------------------------------------------------------


def parse_change_timestep(line):
    """
    Change_timestep (type -4)

    If btype = −4, change the time step size from the initial Dt (secs)
    into V4 (secs) after location V3(m). The maximum number of time step change is 100.

    """
    v = v_from_line(line)
    d = {}
    d["dt"] = parse_float(v[4])
    d["s"] = parse_float(v[3])

    return d


def change_timestep_v(ele):
    """
    change_timestep Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, dummy, ele["s"], ele["dt"]]
    return v


ELE_DEFAULTS["change_timestep"] = {"s": 0, "dt": 0}


# -----------------------------------------------------------------


def parse_rotationally_symmetric_to_3d(line):
    """
    If btype = −5, switch the simulation from azimuthal symmetry to fully 3d simulation after location V3(m).
    This location should be set as large negative number such as -1000.0 in order to start the 3D simulation immediately after the electron emission.
    If there are multiple such lines in the input file, only the last line matters.

    """
    v = v_from_line(line)
    d = {}
    d["s"] = parse_float(v[3])

    return d


def rotationally_symmetric_to_3d_v(ele):
    """
    rotationally_symmetric_to_3d Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, dummy, ele["s"]]
    return v


ELE_DEFAULTS["rotationally_symmetric_to_3d"] = {"s": 0}


# -----------------------------------------------------------------
def parse_wakefield(line):
    """
    Wakefield (type -6)

    If btype = −6, turn on the wake field effects between location V3(m) and V4(m).
    If Bnseg is greater than 0, the longitudinal and transverse wake function will
    be read in from file “fort.Bmpstp”. If Bnseg ≤ 0, the code will use analytical
    wake function described as follows.

    For analytical wake functions, the wake function parameters:
        (iris radius) a = V5(m)
        (gap) g = V6(m)
        (period) L = V7(m).
    Here, the definition of these parameter can be found from
    SLAC-PUB-9663, “Short-Range Dipole Wakefields in Accelerating Structures for the NLC,” by Karl L.F. Bane.
    This will be updated in the future since the parameters a, g, L might change from cell to cell within a single structure.
    For the backward traveling wave structure, the iris radius “a” has to be set greater
    than 100, gap “g” set to the initialization location of BTW.
        For backward traveling wave structures, the wakes are hardwired inside the code following the report:
    P. Craievich, T. Weiland, I. Zagorodnov, “The short-range wakefields in the BTW accelerating structure of the ELETTRA linac,” ST/M-04/02.

    For −10 < a < 0, it uses the analytical equation from the 1.3 GHz Tesla standing wave structure.

    For a < −10, it assumes the 3.9 GHz structure longitudinal wake function.

    For external supplied wake function, The maximum number data point is 1000.
    The data points are assumed uniformly distributed between 0 and V7(m).
    The V6 has to less than 0.
    Each line of the fort.Bmpstp contains longitudinal wake function (V/m) and transverse wake function (V/m/m).

    """
    x = line.split()
    Bnseg = int(x[1])
    v = v_from_line(line)
    d = {}
    d["s_begin"] = parse_float(v[3])
    d["s"] = parse_float(v[4])

    if Bnseg > 0:
        d["method"] = "from_file"
        d["filename"] = "fort." + str(Bnseg)
    else:
        d["method"] = "analytical"
        a = parse_float(v[5])
        d["iris_radius"] = a
        if a >= 0:
            d["gap"] = parse_float(v[6])
            d["period"] = parse_float(v[7])
        else:
            # Tesla structures, dummy values:
            d["gap"] = -1.0
            d["period"] = -1.0

    return d


def wakefield_v(ele):
    """
    wakefield Impact-T style V list

    """

    # Let v[0] be the original ele, so the indexing looks the same.
    # V1 and V2 are not used.
    dummy = 1
    v = [ele, dummy, dummy, ele["s_begin"], ele["s"]]

    if ele["method"] == "analytical":
        v += [ele["iris_radius"], ele["gap"], ele["period"]]

    return v


ELE_DEFAULTS["wakefield"] = {
    "s_begin": 0,
    "s": 0,
    "method": "analytical",
    "filename": "fort.99",
    "iris_radius": 0,
    "gap": 0,
    "period": 0,
}


# -----------------------------------------------------------------
def parse_stop(line):
    """
    Stop (type -99)

    If bytpe = −99, stop the simulation at given location V3(m).


    """
    v = v_from_line(line)
    d = {"s": parse_float(v[3])}
    return d


def stop_v(ele):
    """
    stop Impact-T style V list

    """

    # Let v[0] be the original ele, so the indexing looks the same.
    # V1 and V2 are not used.
    # Bad documentation? Looks like V1 and V3 are used
    dummy = 0.0
    v = [ele, ele["s"], dummy, ele["s"]]

    return v


ELE_DEFAULTS["stop"] = {"s": 0}


# -----------------------------------------------------------------
def parse_spacecharge(line):
    """
    Spacecharge (type -8)

    if bytpe = −8, switch on/off the space-charge calculation at given location V3(m)
    according to the sign of V2 (> 0 on, otherwise off).

    """
    v = v_from_line(line)
    d = {}
    d["s"] = parse_float(v[3])
    if parse_float(v[2]) > 0:
        d["is_on"] = True
    else:
        d["is_on"] = False

    return d


def spacecharge_v(ele):
    """
    spacecharge Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0

    if ele["is_on"]:
        sign = 1.0
    else:
        sign = -1

    v = [ele, dummy, sign, ele["s"]]

    return v


ELE_DEFAULTS["spacecharge"] = {"s": 0, "is_on": False}


# -----------------------------------------------------------------
def parse_write_slice_info(line):
    """
    If bytpe = −9, output slice-based information at given location V3(m) into file “fort.Bmpstp” using “Bnseg” slices.

    """

    x = line.split()
    Bnseg = int(x[1])
    Bmpstp = int(x[2])
    v = v_from_line(line)

    d = {}
    d["n_slices"] = Bnseg
    d["filename"] = "fort." + str(Bmpstp)
    d["s"] = parse_float(v[3])

    return d


def write_slice_info_v(ele):
    """
    write_slice_info Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, dummy, ele["s"]]
    return v


ELE_DEFAULTS["write_slice_info"] = {"s": 0, "n_slices": 0, "filename": "fort.99"}


# Add comment
ELE_DEFAULTS["comment"] = {}

# Add Blength, Bnseg, Bmpstp, Btype to all

# Form valid keys
VALID_KEYS = {}
for k in ELE_DEFAULTS:
    ELE_DEFAULTS[k].update({"L": 0, "Bnseg": 0, "Bmpstp": 0})
    VALID_KEYS[k] = list(ELE_DEFAULTS[k])

for k in VALID_KEYS:
    VALID_KEYS[k] += ["description", "original", "name", "type"]
    # Add L, s
    if k in itype_of and itype_of[k] >= 0:
        VALID_KEYS[k] += ["L", "s"]

# -----------------------------------------------------------------

# def parse_bpm(line):
#    """
#    BPM ????
#    """
#    return {}
#

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Fieldmaps


fieldmap_parsers = {
    "quadrupole": fieldmaps.read_fieldmap_rfdata,
    "dipole": fieldmaps.read_fieldmap_rfdata,
    "multipole": fieldmaps.read_fieldmap_rfdata,
    "srf_cavity": fieldmaps.read_fieldmap_rfdata,
    "solrf": fieldmaps.read_solrf_fieldmap,
    "solenoid": fieldmaps.read_solenoid_fieldmap,
    "emfield_cylindrical": fieldmaps.read_emfield_cylindrical_fieldmap,  # TODO: better parsing
    "emfield_cartesian": fieldmaps.read_emfield_cartesian_fieldmap,
}


def load_fieldmaps(eles, dir):
    """
    Parses fieldmap data from list of elements.

    """
    fmapdata = {}

    for ele in eles:
        if "filename" not in ele:
            continue
        name = ele["filename"]
        # Skip already parsed files
        if name in fmapdata:
            continue
        type = ele["type"]

        # Pick parser
        if type in fieldmap_parsers:
            file = os.path.join(dir, name)
            # Call the appropriate parser
            fmapdata[name] = fieldmap_parsers[type](file)

    return fmapdata


# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Master element parsing


ele_parsers = {  #'bpm': parse_bpm,
    "drift": parse_drift,
    "quadrupole": parse_quadrupole,
    "solenoid": parse_solenoid,
    "dipole": parse_dipole,
    "solrf": parse_solrf,
    "emfield_cartesian": parse_emfield_cartesian,
    "emfield_cylindrical": parse_emfield_cylindrical,
    "offset_beam": parse_offset_beam,
    "write_beam": parse_write_beam,
    "change_timestep": parse_change_timestep,
    "rotationally_symmetric_to_3d": parse_rotationally_symmetric_to_3d,
    "wakefield": parse_wakefield,
    "spacecharge": parse_spacecharge,
    "point_to_point_spacecharge": parse_point_to_point_spacecharge,
    "write_slice_info": parse_write_slice_info,
    "write_beam_for_restart": parse_write_beam_for_restart,
    "stop": parse_stop,
}


def parse_ele(line):
    """
    Parse an Impact-T lattice line.

    Returns an ele dict.

    Parses everything after / as the 'description'

    """
    if is_commented(line):
        return {"type": "comment", "description": line}

    # Ele
    e = {}

    x = line.split("/")
    # Look for extra info past /
    #
    if len(x) > 1:
        # Remove spaces and comment character !
        e["description"] = x[1].strip().strip("!")

    x = x[0].split()

    e["original"] = line  # Save original line

    # e['itype'] = int(x[3]) #Don't store this
    itype = int(x[3])

    if itype >= 0:
        # Real element. Needs L
        e["L"] = parse_float(x[0])

    if itype in ele_type:
        e["type"] = ele_type[itype]

        if itype >= -99:
            d2 = ele_parsers[e["type"]](line)
            e.update(d2)
    else:
        print("Warning: undocumented type", line)
        e["type"] = "undocumented"

    return e


# -----------------------------------------------------------------


def add_s_position(elelist, s0=0):
    """
    Add 's' to element list according to their length.
    s is at the end of an element.
    Assumes the list is in order.

    TODO: This isn't right
    """
    for ele in elelist:
        if "s" not in ele and "zedge" in ele:
            ele["s"] = ele["zedge"] + ele["L"]
            # Skip these.
            # if e['type'] in ['change_timestep', 'offset_beam', 'spacecharge', 'stop', 'write_beam', 'write_beam_for_restart']:
            continue


def create_names(elelist):
    """
    Invent a name for elements
    """
    counter = {}
    for t in list(ele_type.values()) + ["comment", "undocumented"]:
        counter[t] = 0
    for e in elelist:
        t = e["type"]
        counter[t] = counter[t] + 1
        e["name"] = e["type"] + "_" + str(counter[t])

        # Try from description
        if "description" in e:
            alias_name = tools.find_property(e["description"], "name")
            if alias_name:
                e["name"] = alias_name


def parse_lattice(lines):
    eles = [parse_ele(line) for line in lines]
    add_s_position(eles)
    create_names(eles)
    return eles


def parse_impact_input(filePath, verbose=False):
    """
    Parse and ImpactT.in file into header, lattice, fieldmaps


    """
    # Full path
    path, _ = os.path.split(filePath)

    # Read lines
    try:
        with open(filePath, "r") as f:
            data = f.read()
    except UnicodeDecodeError:
        with open(filePath, "r", encoding="utf8", errors="ignore") as f:
            print(
                "Warning: Non-utf8 characters were detected in the input file and ignored."
            )
            data = f.read()
    except Exception:
        raise ValueError("Unxpected error while reading input file!!!")

    lines = data.split("\n")
    header = parse_header(lines)

    # Check for input particles. Must be named 'partcl.data'.
    if header["Flagdist"] == 16:
        pfile = os.path.join(path, "partcl.data")
        pfile = os.path.abspath(pfile)
        if verbose and not os.path.exists(pfile):
            print("Warning: partcl.data missing in path:", path)
    else:
        pfile = None

    # Find index of the line where the lattice starts
    ix = ix_lattice(lines)

    # Gather lattice lines
    latlines = lines[ix:]

    # This parses all lines.
    eles = parse_lattice(latlines)

    # Get fieldmaps
    fieldmaps = load_fieldmaps(eles, path)

    # Ouput dict
    d = {}
    d["original_input"] = data
    d["input_particle_file"] = pfile
    d["header"] = header
    d["lattice"] = eles
    d["fieldmaps"] = fieldmaps

    return d


# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Particles


def parse_impact_particles(
    filePath,
    names=(
        "x",
        "GBx",
        "y",
        "GBy",
        "z",
        "GBz",
        "charge_over_mass_ratio",
        "charge_per_macroparticle",
        "id",
    ),
    skiprows=0,
):
    """
    Parse Impact-T input and output particle data.
    Typical filenames: 'partcl.data', 'fort.40', 'fort.50'.

    Note that partcl.data has the number of particles in the first line, so
    skiprows=1 should be used.

    Returns a structured numpy array

    Impact-T 3.0+ input/output particles distribions are ASCII files with 9 columns:
    x (m)
    GBy = gamma*beta_x (dimensionless)
    y (m)
    GBy = gamma*beta_y (dimensionless)
    z (m)
    GBz = gamma*beta_z (dimensionless)
    charge_over_mass_ratio
    charge_per_macroparticle
    id
    """

    try:
        return pl.read_csv(
            filePath,
            separator=" ",
            has_header=False,
            skip_rows=skiprows,
            schema=dict.fromkeys(names, pl.Float64),
        ).to_numpy(structured=True)
    except pl.exceptions.ComputeError:
        return np.loadtxt(
            filePath,
            skiprows=skiprows,
            dtype={"names": names, "formats": len(names) * [float]},
            ndmin=1,  # to make sure that 1 particle is parsed the same as many.
        )


# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Parsers for Impact-T fort.X output


def _load_ascii_with_missing_exponent(file_path, **kwargs):
    # Define a regex to find numbers that lack the 'E' before the exponent
    pattern = re.compile(r"([+-]?\d*\.\d+)([+-]\d+)")

    # Create a list to hold the corrected lines
    corrected_lines = []

    # Read the file and correct the lines
    with open(file_path, "r") as file:
        for line in file:
            # Substitute matches with the corrected scientific notation
            fixed_line = pattern.sub(r"\1E\2", line)
            corrected_lines.append(fixed_line)

    # Convert the list of corrected lines to a single string
    corrected_data = "\n".join(corrected_lines)

    # Use numpy's loadtxt with ndmin=2 to ensure at least 2D output
    data = np.loadtxt(corrected_data.splitlines(), **kwargs)

    return data


def load_fortX(filePath, keys):
    data = {}
    # Load the data
    fortdata = _load_ascii_with_missing_exponent(filePath, ndmin=2)
    if len(fortdata) == 0:
        raise ValueError(f"{filePath} is empty")
    for count, key in enumerate(keys):
        data[key] = fortdata[:, count]
    return data


FORT_KEYS = {}
FORT_UNITS = {}

FORT_KEYS[18] = [
    "t",
    "mean_z",
    "mean_gamma",
    "mean_kinetic_energy_MeV",
    "mean_beta",
    "max_r",
    "sigma_gamma",
]
FORT_UNITS[18] = ["s", "m", "1", "MeV", "1", "m", "1"]


def load_fort18(filePath, keys=FORT_KEYS[18]):
    """From impact manual v2:
    1st col: time (secs)
    2nd col: distance (m)
    3rd col: gamma
    4th col: kinetic energy (MeV) 5th col: beta
    6th col: Rmax (m) R is measured from the axis of pipe
    7th col: rms energy deviation normalized by MC^2
    """
    return load_fortX(filePath, keys)


FORT_KEYS[24] = [
    "t",
    "mean_z",
    "mean_x",
    "sigma_x",
    "mean_gammabeta_x",
    "sigma_gammabeta_x",
    "-cov_x__gammabeta_x",
    "norm_emit_x",
]
FORT_UNITS[24] = ["s", "m", "m", "m", "1", "1", "m", "m"]


def load_fort24(filePath, keys=FORT_KEYS[24]):
    """From impact manual:
    fort.24, fort.25: X and Y RMS size information
    1st col: time (secs)
    2nd col: z distance (m)
    3rd col: centroid location (m)
    4th col: RMS size (m)
    5th col: Centroid momentum normalized by MC
    6th col: RMS momentum normalized by MC
    7th col: Twiss parameter: -<x, gamma*beta_x>
    8th col: normalized RMS emittance (m-rad)
    """
    return load_fortX(filePath, keys)


FORT_KEYS[25] = [
    "t",
    "mean_z",
    "mean_y",
    "sigma_y",
    "mean_gammabeta_y",
    "sigma_gammabeta_y",
    "-cov_y__gammabeta_y",
    "norm_emit_y",
]
FORT_UNITS[25] = FORT_UNITS[24]


def load_fort25(filePath, keys=FORT_KEYS[25]):
    """
    Same columns as fort24, Y RMS
    """
    return load_fortX(filePath, keys)


FORT_KEYS[26] = [
    "t",
    "mean_z",
    "sigma_z",
    "mean_gammabeta_z",
    "sigma_gammabeta_z",
    "-cov_z__gammabeta_z",
    "norm_emit_z",
]
FORT_UNITS[26] = FORT_UNITS[25]


def load_fort26(filePath, keys=FORT_KEYS[26]):
    """From impact manual:
    fort.26: Z RMS size information
    1st col: time (secs)
    2nd col: centroid location (m)
    3rd col: RMS size (m)
    4th col: Centroid momentum normalized by MC 5th col: RMS momentum normalized by MC
    6th col: Twiss parameter: -<z, gamma*beta_z>
    7th col: normalized RMS emittance (m-rad)
    """
    return load_fortX(filePath, keys)


FORT_KEYS[27] = [
    "t",
    "mean_z",
    "max_amplitude_x",
    "max_amplitude_gammabeta_x",
    "max_amplitude_y",
    "max_amplitude_gammabeta_y",
    "max_amplitude_z",
    "max_amplitude_gammabeta_z",
]
FORT_UNITS[27] = ["s", "m", "m", "1", "m", "1", "m", "1"]


def load_fort27(filePath, keys=FORT_KEYS[27]):
    """
    fort.27: maximum amplitude information
    1st col: time (secs)
    2nd col: z distance (m)
    3rd col: Max. X (m)
    4th col: Max. Px (MC)
    5th col: Max. Y (m)
    6th col: Max. Py (MC)
    7th col: Max. Z (m) (with respect to centroid)
    8th col: Max. Pz (MC)
    """
    return load_fortX(filePath, keys)


FORT_KEYS[28] = [
    "t",
    "mean_z",
    "loadbalance_min_n_particle",
    "loadbalance_max_n_particle",
    "n_particle",
]
FORT_UNITS[28] = ["s", "m", "1", "1", "1"]


def load_fort28(filePath, keys=FORT_KEYS[28]):
    """From impact manual:
    fort.28: load balance and loss diagnostic
    1st col: time (secs)
    2nd col: z distance (m)
    3rd col: min # of particles on a PE
    4th col: max # of particles on a PE
    5th col: total # of particles in the bunch
    """
    return load_fortX(filePath, keys)


FORT_KEYS[29] = [
    "t",
    "mean_z",
    "moment3_x",
    "moment3_gammabeta_x",
    "moment3_y",
    "moment3_gammabeta_y",
    "moment3_z",
    "moment3_gammabeta_z",
]
FORT_UNITS[29] = ["s", "m", "m", "1", "m", "1", "m", "1"]


def load_fort29(filePath, keys=FORT_KEYS[29]):
    """
    fort.29: cubic root of 3rd moments of the beam distribution
    1st col: time (secs)
    2nd col: z distance (m)
    3rd col: X (m)
    4th col: Px (mc)
    5th col: Y (m)
    6th col: Py (mc)
    7th col: Z (m)
    8th col: Pz (mc)
    """
    return load_fortX(filePath, keys)


FORT_KEYS[30] = [
    "t",
    "mean_z",
    "moment4_x",
    "moment4_gammabeta_x",
    "moment4_y",
    "moment4_gammabeta_y",
    "moment4_z",
    "moment4_gammabeta_z",
]
FORT_UNITS[30] = FORT_UNITS[29]


def load_fort30(filePath, keys=FORT_KEYS[30]):
    """
    fort.30: Fourth root of 4th moments of the beam distribution
    1st col: time (secs) 2nd col: z distance (m) 3rd col: X (m)
    4th col: Px (mc)
    5th col: Y (m)
    6th col: Py (mc)
    7th col: Z (m)
    8th col: Pz (mc)
    """
    return load_fortX(filePath, keys)


# Note: fort.31 is a subset of fort.32, so do not parse

FORT_KEYS[32] = [
    "t",
    "mean_z",
    "raw_norm_cdt",
    "cov_x__x",
    "cov_x__gammabeta_x",
    "cov_x__y",
    "cov_x__gammabeta_y",
    "cov_x__z",
    "cov_x__gammabeta_z",
    "cov_gammabeta_x__gammabeta_x",
    "cov_y__gammabeta_x",
    "cov_gammabeta_x__gammabeta_y",
    "cov_z__gammabeta_x",
    "cov_gammabeta_x__gammabeta_z",
    "cov_y__y",
    "cov_y__gammabeta_y",
    "cov_y__z",
    "cov_y__gammabeta_z",
    "cov_gammabeta_y__gammabeta_y",
    "cov_z__gammabeta_y",
    "cov_gammabeta_y__gammabeta_z",
    "cov_z__z",
    "cov_z__gammabeta_z",
    "cov_gammabeta_z__gammabeta_z",
]
### ???
FORT_UNITS[32] = [
    "s",
    "m",
    "m",
    "m*m",
    "m",
    "m*m",
    "m",
    "m*m",
    "m",
    "1",
    "m",
    "1",
    "m",
    "1",
    "m*m",
    "m",
    "m*m",
    "m",
    "1",
    "m",
    "1",
    "m*m",
    "m",
    "1",
]


def load_fort32(filePath, keys=FORT_KEYS[32]):
    """
    fort.32

    z,z0avg*cdt,cdt,&
    qsum1, xpx,    xy,  xpy,    xz,  xpz, &
         sqsum2,   ypx,  pxpy,  zpx,  pxpz, &
                 sqsum3, ypy,    yz,  ypz, &
                                         sqsum4, zpy,  pypz, &
                                                  sqsum5,  zpz, &
                                                        sqsum6

    Note:
        x, y, z are im m
        px, py, pz are normalized by m*c (=gamma*beta_x, etc.


    """
    return load_fortX(filePath, keys)


# ---------------------------------
# Dipole

FORT_KEYS[34] = FORT_KEYS[24]  # X
FORT_UNITS[34] = FORT_UNITS[24]
FORT_KEYS[35] = FORT_KEYS[25]  # Y
FORT_UNITS[35] = FORT_UNITS[25]
FORT_KEYS[36] = FORT_KEYS[26]  # Z
FORT_UNITS[36] = FORT_UNITS[26]
FORT_KEYS[37] = FORT_KEYS[27]  # Amplitude
FORT_UNITS[37] = FORT_UNITS[27]


def load_fort34(filePath, keys=FORT_KEYS[34]):
    """Same as load_fort24, but in the dipole coordinate system"""
    return load_fortX(filePath, keys)


def load_fort35(filePath, keys=FORT_KEYS[35]):
    """Same as load_fort25, but in the dipole coordinate system"""
    return load_fortX(filePath, keys)


def load_fort36(filePath, keys=FORT_KEYS[36]):
    """Same as load_fort26, but in the dipole coordinate system"""
    return load_fortX(filePath, keys)


def load_fort37(filePath, keys=FORT_KEYS[37]):
    """Same as load_fort27, but in the dipole coordinate system"""
    return load_fortX(filePath, keys)


FORT_KEYS[38] = [
    "t",
    "ref_x",
    "ref_gammabeta_x",
    "ref_y",
    "ref_gammabeta_y",
    "ref_z",
    "ref_gammabeta_z",
]
FORT_UNITS[38] = ["s", "m", "1", "m", "1", "m", "1"]


def load_fort38(filePath, keys=FORT_KEYS[38]):
    """
    fort.38: reference particle information in dipole reference coordinate system (inside dipole ONLY)
    1st col: time (secs)
    2nd col: x distance (m) 3rd col: Px/MC
    4th col: y (m)
    5th col: Py/MC
    6th col: z (m) 7th col: Pz/MC
    """
    return load_fortX(filePath, keys)


# ---------------------------------
# Slice statistics

FORT_KEYS[60] = [
    "slice_z",
    "particles_per_cell",
    "current",
    "norm_emit_x",
    "norm_emit_y",
    "mean_energy",
    "sigma_energy",
]
FORT_UNITS[60] = ["m", "1", "A", "m", "m", "eV", "eV"]


def load_fort60_and_70(filePath, keys=FORT_KEYS[60]):
    """
    1st col: bunch length (m)
    2nd col: number of macroparticles per cell
    3rd col: current (A)
    4th col: x slice emittance (m*rad)
    5th col: y slice emittance (m*rad)
    6th col: mean energy (eV)
    7th col: energy spread (eV)
    """
    return load_fortX(filePath, keys)


# Wrapper functions to provide keyed output


def load_fort40(filePath):
    """
    Returns dict with 'initial_particles'
    """
    data = parse_impact_particles(filePath)
    return {"initial_particles": data}


def load_fort50(filePath):
    """
    Returns dict with 'final_particles'
    """
    data = parse_impact_particles(filePath)
    return {"final_particles": data}


def load_fort60(filePath):
    """
    Returns dict with 'initial_particle_slices'
    """
    data = load_fort60_and_70(filePath)
    return {"initial_particle_slices": data}


def load_fort70(filePath):
    """
    Returns dict with 'final_particle_slices'
    """
    data = load_fort60_and_70(filePath)
    return {"final_particle_slices": data}


def fort_files(path):
    """
    Find fort.X fliles in path
    """
    assert os.path.isdir(path)
    flist = os.listdir(path)
    fortfiles = []
    for f in flist:
        if f.startswith("fort."):
            fortfiles.append(os.path.join(path, f))
    return sorted(fortfiles)


FORT_DESCRIPTION = {
    18: "Time and energy",
    24: "RMS X information",
    25: "RMS Y information",
    26: "RMS Z information",
    27: "Max amplitude information",
    28: "Load balance and loss diagnostics",
    29: "Cube root of third moments of the beam distribution",
    30: "Fourth root of the fourth moments of the beam distribution",
    32: "Covariance matrix of the beam distribution",
    34: "Dipole ONLY: X output information in dipole reference coordinate system",
    35: "Dipole ONLY: Y output information in dipole reference coordinate system",
    36: "Dipole ONLY: Z output information in dipole reference coordinate system ",
    37: "Dipole ONLY: Maximum amplitude information in dipole reference coordinate system",
    38: "Dipole ONLY: Reference particle information in dipole reference coordinate system",
    40: "initial particle distribution at t = 0",
    50: "final particle distribution projected to the centroid location of the bunch",
    60: "Slice information of the initial distribution",
    70: "Slice information of the final distribution",
}


FORT_LOADER = {
    18: load_fort18,
    24: load_fort24,
    25: load_fort25,
    26: load_fort26,
    27: load_fort27,
    28: load_fort28,
    29: load_fort29,
    30: load_fort30,
    32: load_fort32,
    34: load_fort34,
    35: load_fort35,
    36: load_fort36,
    37: load_fort37,
    38: load_fort38,
    40: load_fort40,
    50: load_fort50,
    60: load_fort60,
    70: load_fort70,
}

# Form large unit dict for these types of files
UNITS = {}
for i in [18, 24, 25, 26, 27, 28, 29, 30, 32, 34, 35, 36, 37, 38, 60]:
    for j, k in enumerate(FORT_KEYS[i]):
        UNITS[k] = FORT_UNITS[i][j]


def fort_type(filePath, verbose=False):
    """
    Extract the integer type of a fort.X file, where X is the type.
    """
    fullpath = os.path.abspath(filePath)
    p, f = os.path.split(fullpath)
    s = f.split(".")
    if s[0] != "fort":
        print("Error: not a fort file:", filePath)
    else:
        file_type = int(s[1])
        if file_type not in FORT_DESCRIPTION and verbose:
            print("Warning: unknown fort file_type for:", filePath)
        if file_type not in FORT_LOADER and verbose:
            print("Warning: no fort loader yet for:", filePath)
        return file_type


def load_fort(filePath, type=None, verbose=True):
    """
    Loads a fort file, automatically detecting its type and selecting a loader.

    """
    if not type:
        type = fort_type(filePath)

    if verbose:
        if type in FORT_DESCRIPTION:
            print("Loaded fort", type, ":", FORT_DESCRIPTION[type])
        else:
            print("unknown type:", type)

    # Check for empty file
    if os.stat(filePath).st_size == 0:
        warnings.warn(f"Empty file: {filePath}")
        return None

    if type in FORT_LOADER:
        dat = FORT_LOADER[type](filePath)
    else:
        raise ValueError(f"Need parser for {filePath}")
    return dat


FORT_STAT_TYPES = [18, 24, 25, 26, 27, 28, 29, 30, 32]
FORT_DIPOLE_STAT_TYPES = [34, 35, 36, 37, 38]
FORT_PARTICLE_TYPES = [40, 50]
FORT_SLICE_TYPES = [60, 70]

# All of these
# Not uesed: FORT_OUTPUT_TYPES = FORT_STAT_TYPES + FORT_PARTICLE_TYPES + FORT_SLICE_TYPES


def load_many_fort(path, types=FORT_STAT_TYPES, verbose=False):
    """
    Loads a large dict with data from many fort files.
    Checks that keys do not conflict.

    Default types are for typical statistical information along the simulation path.

    """
    fortfiles = fort_files(path)
    alldat = {}
    for f in fortfiles:
        file_type = fort_type(f, verbose=False)
        if file_type not in types:
            continue

        dat = load_fort(f, type=file_type, verbose=verbose)
        if dat is None:  # empty file
            continue

        for k in dat:
            if k not in alldat:
                alldat[k] = dat[k]

            elif np.allclose(alldat[k], dat[k], atol=1e-20):
                # If the difference between alldat-dat < 1e-20,
                # move on to next key without error.
                # https://numpy.org/devdocs/reference/generated/numpy.isclose.html#numpy.isclose
                pass

            else:
                # Data is not close enough to ignore differences.
                # Check that this data is the same as what's already in there
                assert np.all(alldat[k] == dat[k]), "Conflicting data for key:" + k

    return alldat


def _replace_bare_gammabeta_with_p(key, mc2):
    """
    Return a new key and factor for any key:
    gammabeta_{x}
    where x is 'x', 'y', or 'z'.

    Returns
    -------
    newkey
    factor
    extraunit
    """
    if key.startswith("gammabeta_"):
        factor = mc2
        comp = key[10:]
        assert comp in ("x", "y", "z")
        newkey = f"p{comp}"
        extraunits = unit("eV/c")
    else:
        factor = 1
        newkey = key
        extraunits = unit("1")

    return newkey, factor, extraunits


def _replace_all_gammabeta_with_p(key, mc2):
    """
    Return a new key and factor for any key:
    gammabeta_{x}
    or a covariance key
    where x is 'x', 'y', or 'z'

    Returns
    -------
    newkey: str
    factor: float
    extraunits: pmd_unit
    """
    factor = 1

    if key.startswith("cov_"):
        k1, k2 = key[4:].split("__")
        k1, factor1, extraunits1 = _replace_bare_gammabeta_with_p(k1, mc2)
        k2, factor2, extraunits2 = _replace_bare_gammabeta_with_p(k2, mc2)
        factor = factor1 * factor2
        newkey = f"cov_{k1}__{k2}"
        extraunits = extraunits1 * extraunits2
    else:
        newkey, factor, extraunits = _replace_bare_gammabeta_with_p(key, mc2)

    return newkey, factor, extraunits


def load_stats(path, species="electron", types=FORT_STAT_TYPES, verbose=False):
    """
    Loads all Impact-T statistics output.

    Works with types= FORT_STAT_TYPES and FORT_DIPOLE_STAT_TYPES

    Returns dicts:
        data, units

    Converts gamma_beta_X keys to pX using the mass from the species, in units eV/c.

    converts keys that start with "-" to -1*data

    """

    data = load_many_fort(path, types=types, verbose=verbose)
    units = {}

    mc2 = mass_of(species)

    for k in list(data):
        unit_string = UNITS[k]

        # Replace -
        if k.startswith("-"):
            newkey = k[1:]
            newdata = -1 * data.pop(k)
            if newkey in data:
                if not np.allclose(data[newkey], newdata):
                    raise ValueError(f"Inconsistent duplicate data for {newdata}")
            else:
                data[newkey] = newdata
            k = newkey

        # Special
        if k == "mean_kinetic_energy_MeV":
            newkey = "mean_kinetic_energy"
            data[newkey] = data.pop(k) * 1e6
            unit_string = "eV"
            k = newkey

        u = unit(unit_string)

        # Replace all gammabeta_{k} including cov_{k1}__{k2}
        newkey, factor, extraunits = _replace_all_gammabeta_with_p(k, mc2)
        if k not in data:
            raise ValueError(f"{k} not in data!")

        if newkey != k:
            u = multiply_units(u, extraunits)
            if newkey in data:
                newdata = data[k] * factor
                if not np.allclose(data[newkey], newdata):
                    raise ValueError(f"Inconsistent duplicate data for {newdata}")
            else:
                data[newkey] = data[k] * factor
            k = newkey  # to let the next statement work

        # Add to units
        units[k] = u

    # Remove all gammabeta.
    # Note that this needs to be done separately.
    for key in list(data):
        if "gammabeta" in key:
            data.pop(key)

    # Remove these
    remove_keys = ("raw_norm_cdt",)
    for key in remove_keys:
        if key in data:
            data.pop(key)

    return data, units


def load_slice_info(path, verbose=False):
    """
    Loads slice data. Returns dicts:
        data, units

    data has keys:
        initial_particles
        final_particles
    and each is a dict with the slice statistics arrays

    """
    data = load_many_fort(path, FORT_SLICE_TYPES, verbose=verbose)
    units = {}

    data1 = data[list(data)[0]]
    for k in data1:
        unit_string = UNITS[k]
        units[k] = unit(unit_string)

    return data, units
