import numpy as np
from numpy import cos, pi
import os
from .tools import safe_loadtxt
from .control import ControlGroup
from subprocess import Popen, PIPE
from tempfile import TemporaryDirectory, NamedTemporaryFile
import warnings

from pmd_beamphysics.interfaces.impact import fourier_field_reconsruction
from pmd_beamphysics import FieldMesh


def write_fieldmap(filePath, fieldmap):
    """
    Master routine for writing a fieldmap
    """

    # Look for symlink
    if "filePath" in fieldmap:
        write_fieldmap_symlink(fieldmap, filePath)
        return

    format = fieldmap["info"]["format"]
    if format == "rfdata":
        write_fieldmap_rfdata(filePath, fieldmap)
    elif format == "solrf":
        write_fieldmap_solrf(filePath, fieldmap)
    elif format == "solenoid_T7":
        warnings.warn("deprecated format: solenoid_T7", DeprecationWarning)
        old_write_solenoid_fieldmap(fieldmap, filePath)
    elif format == "solenoid_fieldmesh":
        write_solenoid_fieldmap(fieldmap, filePath)
    elif format == "emfield_cylindrical_fieldmesh":
        write_emfield_cylindrical_fieldmap(fieldmap, filePath)
    elif format == "emfield_cartesian_fieldmesh":
        write_emfield_cartesian_fieldmap(fieldmap, filePath)
    else:
        print("Missing writer for fieldmap:", fieldmap)
        raise


# Simple routines for symlinking fieldmaps


def read_fieldmap_symlink(filePath):
    return {"filePath": os.path.abspath(filePath)}


def write_fieldmap_symlink(fieldmap, filePath):
    if os.path.exists(filePath):
        pass
    else:
        os.symlink(fieldmap["filePath"], filePath)


# -----------------------
# rfdata fieldmaps
def read_fieldmap_rfdata(filePath):
    """
    Read Impact-T rfdata file, which should be simple two-column ASCII data
    """

    info = {}
    info["format"] = "rfdata"
    info["filePath"] = os.path.abspath(filePath)

    # Read data
    d = {}
    d["info"] = info
    d["data"] = safe_loadtxt(filePath)
    return d


def write_fieldmap_rfdata(filePath, fieldmap):
    """ """
    np.savetxt(filePath, fieldmap["data"])


# -----------------------
# T7 fieldmaps
def old_read_solenoid_fieldmap(filePath):
    """
    Read a T7 style file.

    Format:

    Header:
        zmin, zmax, nz
        rmin, rmax, nr
    Data:
        Br, Bz
        (repeating)

    min, max are in cm
    Er

    """
    d = {}
    # Read header
    with open(filePath) as f:
        line1 = f.readline()
        line2 = f.readline()
    zmin, zmax, nz = line1.split()
    rmin, rmax, nr = line2.split()

    info = {}
    info["format"] = "solenoid_T7"
    info["zmin"] = float(zmin)
    info["zmax"] = float(zmax)
    info["nz"] = int(nz)
    info["rmin"] = float(rmin)
    info["rmax"] = float(rmax)
    info["nr"] = int(nr)

    # Read data
    d["info"] = info
    d["data"] = np.loadtxt(filePath, skiprows=2)

    return d


def old_write_solenoid_fieldmap(fieldmap, filePath):
    """
    Save fieldmap data to file in T7 format.
    fieldmap must have:
    ['info'] = dict with keys: zmin, zmax, nz, rmin, rmax, nr
    ['data'] = array of data
    """
    info = fieldmap["info"]
    line1 = " ".join([str(x) for x in [info["zmin"], info["zmax"], info["nz"]]])
    line2 = " ".join([str(x) for x in [info["rmin"], info["rmax"], info["nr"]]])
    header = line1 + "\n" + line2
    # Save data
    np.savetxt(filePath, fieldmap["data"], header=header, comments="")


def upgrade_old_solenoid_fieldmap(fieldmap):
    """
    Upgrades an old-style solenoid fieldmap
    """
    with NamedTemporaryFile() as tf:
        old_write_solenoid_fieldmap(fieldmap, tf.name)
        new_fieldmap = read_solenoid_fieldmap(tf.name)
    return new_fieldmap


def read_solenoid_fieldmap(filePath):
    """
    Read a T7 style file.
    """

    fm = FieldMesh.from_superfish(filePath, type="magnetic")

    d = {"info": {"format": "solenoid_fieldmesh"}, "field": fm}

    return d


def write_solenoid_fieldmap(fieldmap, filePath):
    """
    Writes a superfish T7 file (Poisson problem, magnetic)
    """
    assert fieldmap["info"]["format"] == "solenoid_fieldmesh"
    fieldmap["field"].write_superfish(filePath)


def read_emfield_cylindrical_fieldmap(filePath):
    """
    Read a T7 style file.
    """

    fm = FieldMesh.from_superfish(filePath)

    d = {"info": {"format": "emfield_cylindrical_fieldmesh"}, "field": fm}

    return d


def read_emfield_cartesian_fieldmap(filePath):
    """
    Read a T7 style file for 111: EMfldCart data
    """

    fm = FieldMesh.from_impact_emfield_cartesian(filePath)

    d = {"info": {"format": "emfield_cartesian_fieldmesh"}, "field": fm}

    return d


def write_emfield_cylindrical_fieldmap(fieldmap, filePath):
    """
    Writes a superfish T7 file (Poisson problem, magnetic)
    """
    assert fieldmap["info"]["format"] == "emfield_cylindrical_fieldmesh"
    fieldmap["field"].write_superfish(filePath)


def write_emfield_cartesian_fieldmap(fieldmap, filePath):
    """
    Writes a T7 style file for 111: EMfldCart data
    """
    assert fieldmap["info"]["format"] == "emfield_cartesian_fieldmesh"
    fieldmap["field"].write_impact_emfield_cartesian(filePath)


# -----------------------
# solrf fieldmaps
def read_solrf_fieldmap(filePath):
    """
    Reads a solrf rfdata file.
    Automatically pareses new and old style fieldmaps into 'fields'
    """

    d = read_fieldmap_rfdata(filePath)
    d["info"]["format"] = "solrf"  # Replace this

    # Remove data
    data = d.pop("data")

    # and add the processed field
    d["field"] = solrf_field_from_data(data)

    return d


def write_fieldmap_solrf(filePath, fieldmap):
    """
    Write solrf rfdata file from fieldmap
    """

    data = data_from_solrf_fieldmap(fieldmap)

    np.savetxt(filePath, data)


def solrf_field_from_data(data):
    """
    Processes array of raw data from a rfdataX file into a dict.

    This automatically detects old- and new-style data.

    Parameters
    ----------
    data: array

    Returns
    -------
    field: dict


    """
    ndim = data.ndim

    if ndim == 1:
        field = process_fieldmap_solrf_fourier(data)
    elif ndim == 2:
        field = process_fieldmap_solrf_derivatives(data)
    else:
        raise ValueError(f"Confusing shape for solrf data: {data.shape}")

    return field


def process_fieldmap_solrf_derivatives(data):
    """
    Process new-style raw solrf data into a dict.
    """
    assert data.ndim == 2

    d = {}
    d["Ez"] = {}
    d["Bz"] = {}

    # Ez
    header = data[0]
    n = int(header[0])
    i1 = 1
    i2 = 1 + n
    d["Ez"]["z0"] = header[1]
    d["Ez"]["z1"] = header[2]
    d["Ez"]["L"] = header[3]
    d["Ez"]["derivative_array"] = data[i1:i2]

    # Bz
    header = data[i2]
    n = int(header[0])
    i1 = i2 + 1
    i2 = i1 + n
    d["Bz"]["z0"] = header[1]
    d["Bz"]["z1"] = header[2]
    d["Bz"]["L"] = header[3]
    d["Bz"]["derivative_array"] = data[i1:i2]

    return d


def process_fieldmap_solrf_fourier(data):
    """
    Processes array of raw data from a rfdataX file into a dict.

    From the documentation:
    Here, the rfdataV5 file contains the Fourier coefficients for both E fields and B fields.
    The first half contains E fields, and the second half contains B fields.
    See manual.

    fourier_coeffients are as Impact-T prefers:
    [0] is
    [1::2] are the cos parts
    [2::2] are the sin parts
    Recontruction of the field at z shown in :
        fieldmap_reconsruction

    """

    d = {}
    d["Ez"] = {}
    d["Bz"] = {}

    # Ez
    n_coef = int(data[0])  # Number of Fourier coefs of on axis
    d["Ez"]["z0"] = data[1]  # distance before the zedge.
    d["Ez"]["z1"] = data[2]  # distance after the zedge.
    d["Ez"]["L"] = data[3]  # length of the Fourier expanded field.
    # Note that (z1-z0)/L = number of periods
    i1 = 4
    i2 = 4 + n_coef
    d["Ez"]["fourier_coefficients"] = data[i1:i2]  # Fourier coefficients on axis

    # Bz
    data2 = data[i2:]
    n_coef = int(data2[0])  # Number of Fourier coefs of on axis
    d["Bz"]["z0"] = data2[1]  # distance before the zedge.
    d["Bz"]["z1"] = data2[2]  # distance after the zedge.
    d["Bz"]["L"] = data2[3]  # length of the Fourier expanded field.
    i1 = 4
    i2 = 4 + n_coef
    d["Bz"]["fourier_coefficients"] = data2[i1:i2]  # Fourier coefficients on axis

    return d


def data_from_solrf_fieldmap(fmap):
    """
    Creates 'rfdataX' array from fieldmap dict.

    This is the inverse of process_fieldmap_solrf


    """
    field = fmap["field"]
    data = []
    if "fourier_coefficients" in field["Ez"]:
        for dat in [field["Ez"], field["Bz"]]:
            coefs = dat["fourier_coefficients"]
            z0 = dat["z0"]
            z1 = dat["z1"]
            L = dat["L"]
            data.append(np.array([len(coefs), z0, z1, L]))
            data.append(coefs)
        data = np.hstack(data)
    elif "derivative_array" in field["Ez"]:
        for dat in [field["Ez"], field["Bz"]]:
            darray = dat["derivative_array"]
            z0 = dat["z0"]
            z1 = dat["z1"]
            L = dat["L"]
            data.append(np.array([[len(darray), z0, z1, L]]))
            data.append(darray)
        data = np.vstack(data)

    return data


# process_fieldmap_solrf(I.input['fieldmaps']['rfdata102'])


def old_fieldmap_reconstruction(fdat, z):
    """
    Transcription of Ji's routine

    Field at z relative to the element's zedge

    """
    z0 = fdat["z0"]
    ## z1 = fdat['z1']  # Not needed here

    zlen = fdat["L"]  # Periodic length

    if zlen == 0:
        return 0

    rawdata = fdat["fourier_coefficients"]

    ncoefreal = (len(rawdata) - 1) // 2

    zmid = zlen / 2

    Fcoef0 = rawdata[0]  # constant factor
    Fcoef1 = rawdata[1::2]  # cos parts
    Fcoef2 = rawdata[2::2]  # sin parts

    kk = 2 * np.pi * (z - zmid - z0) / zlen

    ilist = np.arange(ncoefreal) + 1

    res = (
        Fcoef0 / 2
        + np.sum(Fcoef1 * np.cos(ilist * kk))
        + np.sum(Fcoef2 * np.sin(ilist * kk))
    )

    return res


def fieldmap_reconstruction_solrf(fdat, z, order=0):
    z0 = fdat["z0"]
    # z1 = fdat['z1'] # Not needed here

    L = fdat["L"]  # Periodic length

    if L == 0:
        return 0

    # Handle old and new style fieldmaps
    if "fourier_coefficients" in fdat:
        fz = fourier_field_reconsruction(
            z, fdat["fourier_coefficients"], z0=z0, zlen=L, order=order
        )
    elif "derivative_array" in fdat:
        darray = fdat["derivative_array"]
        zlist = np.linspace(z0, z0 + L, len(darray))
        fz = np.interp(z, zlist, darray[:, order])

    return fz


def riffle(a, b):
    return np.vstack((a, b)).reshape((-1,), order="F")


def run_RFcoef(z, fz, n_coef=20, z0=0, exe="RFcoeflcls"):
    """
    Runs the Fortran executable RFcoeflcls,
    and parses the output as numpy arrays.

    Parameters
    ----------
    z: array
        z-positions

    fz: array
        Field at z

    n_coef: int
        Number of Fourier coefficients to use

    z0: float
        Fieldmap lower bound

    exe: str
        path to the RFcoeflcls executable

    Returns
    -------
    ouput: dict with:
        rfdatax: array
        rfdatax2: array
        rfdata.out: array


    """

    d = TemporaryDirectory()
    workdir = d.name

    nz = len(z)
    if n_coef is None:
        n_coef = nz // 2

    infile = os.path.join(workdir, "rfdata.in")
    np.savetxt(infile, np.array([z, fz]).T)

    exe = os.path.expandvars(exe)
    if not os.path.exists(exe):
        raise ValueError(f"Executable does not exist: {exe}")
    p = Popen([exe], stdin=PIPE, shell=True, cwd=workdir)
    p.communicate(input=f"{n_coef}\n{nz}\n{z0}\n".encode("utf-8"))

    output = {}
    for file in ("rfdatax", "rfdatax2", "rfdata.out"):
        ffile = os.path.join(workdir, file)
        output[file] = np.loadtxt(ffile)

        # Debuging precision
        # output[file+'_raw'] = open(ffile).read()

    return output


FIELD_CALC_ELE_TYPES = ("solrf", "solenoid", "emfield_cylindrical", "emfield_cartesian")


def ele_field(ele, *, x=0, y=0, z=0, t=0, component="Ez", fmaps=None):
    """
    Returns the real value of a field component of an ele
    at position x, y, z at time t.

    Currently only implemented for solrf elements.


    Parameters
    ----------
    ele: dict or ControlGroup
        LUME-Impact element dict or ControlGroup
        If a ControlGroup, the field will be the sum of
        the field in eles in ControlGroup.eles

    x: float
        x-position in meters

    y: float
        y-position in meters

    z: float
        z-position in meters

    y: float
        time in seconds

    component: str
        Field component requested. Currently only:
        'Ez' in V/m
        'Bz' in T
        are avaliable

    fmaps:
        dict with parsed fieldmap data.

    Returns
    -------
    field: float


    """

    if x != 0:
        raise NotImplementedError
    if y != 0:
        raise NotImplementedError
    # if component not in ('Bz', 'Ez'):
    #    raise NotImplementedError

    # Allow ControlGroup
    if isinstance(ele, ControlGroup):
        return sum(
            [
                ele_field(ele1, x=x, y=y, z=z, t=t, component=component, fmaps=fmaps)
                for ele1 in ele.eles
            ]
        )

    # regular ele
    ele_type = ele["type"]

    if ele_type not in FIELD_CALC_ELE_TYPES:
        return 0

    zedge = ele["zedge"]
    L = ele["L"]

    z_local = z - zedge

    if z_local < 0 or z_local > L:
        return 0

    ele_type = ele["type"]
    if ele_type == "solrf":
        if component in ("Bz", "Ez"):
            field = fmaps[ele["filename"]]["field"][component]
            freq = ele["rf_frequency"]
            theta0 = ele["theta0_deg"] * pi / 180

            # if ele['x_offset'] != 0:
            #     raise NotImplementedError
            # if ele['y_offset'] != 0:
            #     raise NotImplementedError
            # if ele['x_rotation'] != 0:
            #     raise NotImplementedError
            # if ele['y_rotation'] != 0:
            #     raise NotImplementedError
            # if ele['z_rotation'] != 0:
            #     raise NotImplementedError

            if component == "Bz":
                scale = ele["solenoid_field_scale"]
            elif component == "Ez":
                scale = ele["rf_field_scale"]
            else:
                raise NotImplementedError(f"component not implemented: {component}")

            fz = fieldmap_reconstruction_solrf(field, z_local)

            # Phase factor
            scale *= cos(2 * pi * freq * t + theta0)
            fz *= scale
        else:
            fz = 0

    elif ele_type == "solenoid":
        if component in ("Bz", "Ez"):
            fm = fmaps[ele["filename"]]["field"]
            fz = np.interp(z_local, fm.coord_vec("z"), np.real(fm[component][0, 0, :]))
        else:
            fz = 0

    elif ele_type == "emfield_cylindrical":
        fm = fmaps[ele["filename"]]["field"]
        theta0 = ele["theta0_deg"] * pi / 180
        freq = ele["rf_frequency"]
        scale = ele["rf_field_scale"]
        fz_complex = np.interp(z_local, fm.coord_vec("z"), fm[component][0, 0, :])

        fz = np.real(np.exp(-1j * (2 * pi * freq * t + theta0)) * fz_complex * scale)

    elif ele_type == "emfield_cartesian":
        fm = fmaps[ele["filename"]]["field"]
        scale = ele["rf_field_scale"]
        point = np.zeros(3)
        point[fm.axis_index("x")] = x
        point[fm.axis_index("y")] = y
        # Note: emfield_cartesian does not honor zmin in the fieldmap.
        point[fm.axis_index("z")] = z_local + fm.mins[fm.axis_index("z")]
        fz = np.real(fm.interpolate(component, point) * scale)

    return fz


# @np.vectorize
def lattice_field(eles, *, z=0, x=0, y=0, t=0, component="Ez", fmaps=None):
    return sum(
        [
            ele_field(ele, z=z, x=x, y=y, t=t, fmaps=fmaps, component=component)
            for ele in eles
        ]
    )
