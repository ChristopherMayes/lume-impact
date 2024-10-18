# import numpy as np
from . import parsers
from .parsers import itype_of, VALID_KEYS

import numpy as np


# -----------------------------------------------------------------
# Print eles in a MAD style syntax
def ele_str(e):
    line = ""
    if e["type"] == "comment":
        c = e["description"]
        if c == "!":
            return ""
        else:
            # pass
            return c

    line = e["name"] + ": " + e["type"]
    l = len(line)
    for key in e:
        if key in ["name", "type", "original", "itype"]:
            continue
        val = str(e[key])
        s = key + "=" + val
        l += len(s)
        if l > 80:
            append = ",\n      " + s
            l = len(append)
        else:
            append = ", " + s
        line = line + append
    return line


ele_v_function = {
    "dipole": parsers.dipole_v,
    "drift": parsers.drift_v,
    "quadrupole": parsers.quadrupole_v,
    "solenoid": parsers.solenoid_v,
    "solrf": parsers.solrf_v,
    "emfield_cartesian": parsers.emfield_cartesian_v,
    "emfield_cylindrical": parsers.emfield_cylindrical_v,
    "stop": parsers.stop_v,
    "change_timestep": parsers.change_timestep_v,
    "offset_beam": parsers.offset_beam_v,
    "rotationally_symmetric_to_3d": parsers.rotationally_symmetric_to_3d_v,
    "wakefield": parsers.wakefield_v,
    "write_beam": parsers.write_beam_v,
    "write_beam_for_restart": parsers.write_beam_for_restart_v,
    "spacecharge": parsers.spacecharge_v,
    "point_to_point_spacecharge": parsers.parse_point_to_point_spacecharge_v,
    "write_slice_info": parsers.write_slice_info_v,
}


def extract_bmpstp(filename):
    """
    Extracts Bmpstp from filename. Filenames are often of the form:
        fort.Bmpstp
    or:
        fort.(Bmpstp+myid)


    """
    ii = int(filename.split("fort.")[1].split("+myid")[0])  # Extract from filename
    # Not true?
    # if ii >= 100:
    #    print(f'Warning: Bmpstp >= 100 is not supported! Filename = {filename}. Nothing will be written.')
    return ii


def ele_line(ele):
    """
    Write Impact-T stype element line

    All real eles start with the four numbers:
    Length, Bnseg, Bmpstp, itype

    With additional numbers depending on itype.

    """
    type = ele["type"]
    if type == "comment":
        return ele["description"]
    itype = parsers.itype_of[type]
    if itype < 0:
        # Custom usage
        if type == "write_beam":
            Bnseg = ele["sample_frequency"]
            Bmpstp = extract_bmpstp(ele["filename"])
        elif type == "write_slice_info":
            Bnseg = ele["n_slices"]
            Bmpstp = extract_bmpstp(ele["filename"])
        elif type == "write_beam_for_restart":
            Bnseg = 0
            Bmpstp = extract_bmpstp(ele["filename"])
        elif type == "wakefield":
            if ele["method"] == "from_file":
                Bnseg = 1  # Anything > 0
                Bmpstp = extract_bmpstp(ele["filename"])
            else:
                Bnseg = -1  # Anything <= 0
                Bmpstp = 0
        else:
            Bnseg = 0  # ele['nseg']
            Bmpstp = 0  # ele['bmpstp']
    else:
        Bnseg = 0
        Bmpstp = 0

    # Length is only for real elements
    if itype < 0:
        L = 0
    else:
        L = ele["L"]
    dat = [L, Bnseg, Bmpstp, itype]

    if type in ele_v_function:
        v = ele_v_function[type](ele)
        dat += v[1:]
    else:
        print("ERROR: ele_v_function not yet implemented for type: ", type)
        return

    line = str(dat[0])
    for d in dat[1:]:
        line = line + " " + str(d)
    return line + " /" + "!name:" + ele["name"]


def lattice_lines(eles, strict=True):
    line0 = "!=================== LATTICE ==================="
    lines = [line0]
    for e in eles:
        if strict:
            assert_strict_ele(e)

        lines.append(ele_line(e))
    return lines


# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Higher level functions
def ele_dict_from(eles):
    """
    Use names as keys. Names must be unique.

    """
    ele_dict = {}
    for ele in eles:
        if ele["type"] == "comment":
            continue
        name = ele["name"]
        assert name not in ele_dict
        ele_dict[name] = ele
    return ele_dict


# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Layout
# Info for plotting

ELE_HEIGHT = {
    "change_timestep": 1,
    "comment": 1,
    "dipole": 0.5,
    "drift": 0.1,
    "offset_beam": 1,
    "quadrupole": 0.5,
    "solenoid": 0.5,
    "solrf": 1,
    "emfield_cartesian": 0.7,
    "emfield_cylindrical": 0.7,
    "spacecharge": 1,
    "stop": 1,
    "wakefield": 1,
    "write_beam": 1,
    "write_beam_for_restart": 1,
}
ELE_COLOR = {
    "change_timestep": "black",
    "comment": "black",
    "dipole": "red",
    "solenoid": "purple",
    "emfield_cartesian": "darkgreen",
    "emfield_cylindrical": "darkgreen",
    "drift": "black",
    "offset_beam": "black",
    "quadrupole": "blue",
    "solrf": "green",
    "spacecharge": "black",
    "stop": "black",
    "wakefield": "brown",
    "write_beam": "black",
    "write_beam_for_restart": "black",
}


def ele_shape(ele):
    """
    Bounding information for use in layout plotting
    """
    type = ele["type"]
    q_sign = -1  # electron
    # print(type, ele['type'])

    # Look for pure solenoid
    if type == "solrf":
        if ele["rf_field_scale"] == 0 and ele["solenoid_field_scale"] != 0:
            type = "solenoid"

    if type == "quadrupole":
        b1 = q_sign * ele["b1_gradient"] / 5
        if b1 > 0:
            # Focusing
            top = b1
            bottom = 0
        else:
            top = 0
            bottom = b1
    else:
        top = ELE_HEIGHT[type]
        bottom = -top

    # DEBUG
    if "L" not in ele:
        print("ERROR: no L in ele: ", ele)

    d = {}
    d["left"] = ele["s"] - ele["L"]
    d["right"] = ele["s"]
    d["top"] = top
    d["bottom"] = bottom
    # Center points
    d["x"] = ele["s"] - ele["L"] / 2
    d["y"] = 0
    d["color"] = ELE_COLOR[type]
    d["name"] = ele["name"]

    d["all"] = ele_str(ele)  #'\n'.join(str(ele).split(',')) # Con
    # d['description'] = ele['description']

    return d


DUMMY_ELE = {"L": 0, "s": 0, "description": "", "name": "dummy", "type": "drift"}


def ele_shapes(eles):
    """
    Form dataset of al element info

    Only returns shapes for physical elements
    """
    # Automatically get keys
    keys = list(ele_shape(DUMMY_ELE))
    # Prepare lists
    data = {}
    for k in keys:
        data[k] = []
    for e in eles:
        type = e["type"]
        if type in ["comment"]:
            continue
        if parsers.itype_of[type] < 0:
            continue
        d = ele_shape(e)
        for k in keys:
            data[k].append(d[k])
    return data


# -----------------------------------------------------------------
def remove_element_types(lattice, types=["stop", "comment", "write_beam", "wakefield"]):
    """
    Removes particular types of elements from a lattice (list of elements)
    """

    return [ele for ele in lattice if ele["type"] not in types]


def get_stop(lattice):
    """
    Searches for stop elements. Only the last one is supposed to matter.
    """

    s = None
    for ele in lattice:
        if ele["type"] == "stop":
            s = ele["s"]
    return s


def set_stop(lattice, s):
    """
    Sets the stop longitudinal position s by removing any existing stop elements,
    and inserting a new stop element at the beginning.

    Returns a tuple:
        lattice, list of removed_eles
    """
    lat = []
    removed_eles = []
    for ele in lattice:
        if ele["type"] == "stop":
            removed_eles.append(ele)

        else:
            lat.append(ele)

    stop_ele = {"name": "stop_1", "type": "stop", "s": s}
    lat.append(stop_ele)
    return lat, removed_eles


def ele_bounds(eles):
    """
    Get the min, max s postion of a list of eles.

    Only considers elements with 'zedge' in them.
    """
    if not isinstance(eles, list):
        eles = [eles]

    mins = []
    maxs = []
    for ele in eles:
        if ele["type"] == "stop":
            zedge = ele["s"]
            L = 0
        elif "zedge" not in ele:
            continue
        else:
            zedge = ele["zedge"]
            L = ele["L"]
        mins.append(zedge)
        maxs.append(zedge + L)
    return min(mins), max(maxs)


def ele_overlaps_s(ele, smin, smax):
    """
    returns True if any part of an element is within smin, smax
    """
    s = ele["s"]
    if "L" not in ele:
        return (s >= smin) and (s <= smax)
    else:
        s0 = s - ele["L"]
        return (s0 >= smin) and (s0 <= smax) or ((s >= smin) and (s <= smax))


def insert_ele_by_s(ele, eles, verbose=False):
    """
    Inserts ele in a list of eles using 's'
    """
    s = ele["s"]
    for i, ele0 in enumerate(eles):
        if "s" not in ele0:
            continue
        if ele0["s"] > s:
            break

    ii = i
    eles.insert(ii, ele)

    if verbose:
        name = ele["name"]
        name0 = ele0["name"]
        print(
            f"Inserted ele '{name}' before ele '{name0}' at index {ii} out of {len(eles)-1}"
        )


# -----------------------------------------------------------------
# Constructors


def new_write_beam(name=None, s=0, filename=None, sample_frequency=1, ref_eles=[]):
    """
    returns a new write_beam element.

    If a list of ref_eles is given, this considers the naming so there are no conflicts.

    Filenames must be of the form 'fort.i' with i an integer. i <= 70 should be used with caution.
    """

    # Get existing list
    ilist = [70]
    if ref_eles:
        ilist += [
            extract_bmpstp(ele["filename"])
            for ele in ref_eles
            if ele["type"] == "write_beam"
        ]

    assert len(ilist) < 99, "Too many write_beam elements. Only 100 allowed."

    if filename:
        inew = extract_bmpstp(filename)
        assert inew not in ilist, f"Filename is already reserved: {filename}"
    else:
        inew = max(max(ilist), 70) + 1  # Avoid 70 and lower
        filename = f"fort.{inew}"
    if not name:
        name = f"write_beam_{inew}"

    ele = {
        "name": name,
        "s": s,
        "type": "write_beam",
        "filename": filename,
        "sample_frequency": sample_frequency,
    }

    return ele


# -----------------------------------------------------------------
# Helpers


def bad_keys(ele):
    """
    Checks the keys in an element for ones that do not belong.

    """
    type = ele["type"]
    assert type in VALID_KEYS, f"No valid keys for type: {type}"

    valid_keys = VALID_KEYS[type]
    bad = []
    for k in ele:
        if k not in valid_keys:
            bad.append(k)

    return bad


def assert_strict_ele(ele):
    """
    Raises an exception if a key does not belong in an ele.
    """

    klist = bad_keys(ele)
    etype = ele["type"]
    if len(klist) > 0:
        raise ValueError(f"Bad keys for ele type {etype}: {klist}")


def sanity_check_ele(ele):
    """
    Sanity check that writing an element is the same as the original line
    """
    if ele["type"] == "comment":
        return True

    dat1 = ele_line(ele).split("/")[0].split()
    dat2 = ele["original"].split("/")[0].split()

    itype = itype_of[ele["type"]]
    if itype >= 0:
        # These aren't used
        dat2[1] = 0
        dat2[2] = 0
    if itype in [parsers.itype_of["offset_beam"]]:
        # V1 is not used
        dat2[4] = 0
    if itype in [parsers.itype_of["spacecharge"]]:
        # V1 is not used, only v2 sign matters
        dat2[4] = 0
        if float(dat2[5]) > 0:
            dat2[5] = 1.0
        else:
            dat2[5] = -1.0

    if itype in [
        parsers.itype_of["write_beam"],
        parsers.itype_of["stop"],
        parsers.itype_of["write_beam_for_restart"],
    ]:
        # Only V3 is used
        dat2[4] = 0
        dat2[5] = 0
    if itype in [itype_of["change_timestep"]]:
        # Only V3, V4 is used
        dat2[4] = 0
        dat2[5] = 0

    dat1 = np.array([float(x) for x in dat1])
    dat2 = np.array([float(x) for x in dat2])
    if len(dat1) != len(dat2):
        # print(ele)
        # print('bad lengths:')
        # print(dat1)
        # print(dat2)
        return True
    good = np.all(dat2 - dat1 == 0)

    if not good:
        print("------ Not Good ----------")
        print(ele)
        print("This    :", dat1)
        print("original:", dat2)

    return good
