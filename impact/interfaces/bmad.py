from pmd_beamphysics import FieldMesh
from pmd_beamphysics.fields.analysis import accelerating_voltage_and_phase
import numpy as np
from copy import deepcopy
from collections import Counter


def tao_unique_names(tao):
    """
    Invent a unique name

    Parameters
    ----------
    tao: Pytao.Tao instance

    Returns
    -------
    dict of int:str
        Mapping of ix_ele to a unique name
    """
    # Get all ixs
    ixs = set(tao.lat_list("*", "ele.ix_ele"))
    ixs.update(set(tao.lat_list("*", "ele.ix_ele", flags="-array_out -no_slaves")))
    ixs = list(sorted(ixs))

    names = [tao.ele_head(ix)["name"] for ix in ixs]

    count = Counter(names)
    unique_name = {}
    found = {name: 0 for name in names}
    for ix, name in zip(ixs, names):
        if count[name] > 1:
            new_count = found[name] + 1
            found[name] = new_count
            unique_name[ix] = f"{name}_{new_count}"
        else:
            unique_name[ix] = name
    return unique_name


def ele_info(tao, ele_id):
    """
    Returns a dict of element attributes from ele_head and ele_gen_attribs
    """
    edat = tao.ele_head(ele_id)
    edat.update(tao.ele_gen_attribs(ele_id))
    s = edat["s"]
    L = edat["L"]
    edat["s_begin"] = s - L
    edat["s_center"] = (s + edat["s_begin"]) / 2

    return edat


def tao_create_impact_emfield_cartesian_ele(
    tao, ele_id, *, file_id=666, output_path=None, cache=None, name=None
):
    """
    Create an Impact-T emfield_cartesia element from a running PyTao Tao instance.

    Parameters
    ----------

    tao: Tao object

    ele_id: str:
        element name or index

    file_id: int, default: 666

    output_path: str, default: None
        If given, the 1T{file_id}.T7 file will be written to this path

    cache: dict, default: None
        FieldMesh file cache dict: {filename:FieldMesh(filename)}
        If not none, this will cache fieldmaps and update this dict.


    Returns
    -------
    dict with:
      line: str
          Impact-T style element line

      ele: dict
          LUME-Impact style element

    """

    # Ele info from Tao
    edat = ele_info(tao, ele_id)
    ix_ele = edat["ix_ele"]

    # Keys
    ele_key = edat["key"].upper()
    if ele_key not in ("EM_FIELD",):
        raise NotImplementedError(f"{ele_key}")

    if name is None:
        name = edat["name"]

    # FieldMesh
    grid_params = tao.ele_grid_field(ix_ele, 1, "base", as_dict=False)
    field_file = grid_params["file"].value
    if cache is not None:
        if field_file in cache:
            field_mesh = cache[field_file]
        else:
            # Add to cache
            field_mesh = FieldMesh(field_file)
            cache[field_file] = field_mesh

    else:
        field_mesh = FieldMesh(field_file)

    if not field_mesh.is_static:
        raise NotImplementedError("oscillating fields not yet implemented")

    # Scaling
    master_parameter = grid_params["master_parameter"].value
    if master_parameter == "<None>":
        master_parameter = "FIELD_AUTOSCALE"
    scale = edat[master_parameter]

    # Find zedge
    eleAnchorPt = field_mesh.attrs["eleAnchorPt"]
    if eleAnchorPt == "beginning":
        zedge = edat["s_begin"]
    elif eleAnchorPt == "center":
        # Use full fieldmap!!!
        z0 = field_mesh.mins[field_mesh.axis_index("z")]
        zedge = edat["s_center"] + z0  # Wrong: -L_fm/2
    else:
        raise NotImplementedError(f"{eleAnchorPt} not implemented")

    outdat = {}

    # Add field integrals
    info = outdat["info"] = {}
    for key in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
        z, fz = field_mesh.axis_values("z", key)
        info[f"integral_{key}_dz"] = np.trapezoid(fz, z)

    # Call the fieldmesh method
    dat = field_mesh.to_impact_emfield_cartesian(
        zedge=zedge,
        name=name,
        scale=scale,
        phase=0,
        x_offset=edat["X_OFFSET"],
        y_offset=edat["Y_OFFSET"],
        file_id=file_id,
        output_path=output_path,
    )
    # Add this to output
    outdat.update(dat)

    return outdat


def tao_create_impact_solrf_ele(
    tao,
    ele_id,
    *,
    style="fourier",
    n_coef=30,
    spline_s=1e-6,
    spline_k=5,
    file_id=666,
    output_path=None,
    cache=None,
    name=None,
):
    """
    Create an Impact-T solrf element from a running PyTao Tao instance.

    Parameters
    ----------

    tao: Tao object

    ele_id: str:
        element name or index

    style: str, default: 'fourier'

    zmirror: bool, default: None
        Mirror the field about z=0. This is necessary for non-periodic field such as electron guns.
        If None, will auotmatically try to detect whether this is necessary.

    spline_s: float, default: 0

    spline_k: float, default: 0

    file_id: int, default: 666

    output_path: str, default: None
        If given, the rfdata{file_id} file will be written to this path

    cache: dict, default: None
        FieldMesh file cache dict: {filename:FieldMesh(filename)}
        If not none, this will cache fieldmaps and update this dict.


    Returns
    -------
    dict with:
      line: str
          Impact-T style element line

      ele: dict
          LUME-Impact style element

      fmap: dict with:
            data: ndarray

            info: dict with
                Ez_scale: float

                Bz_scale: float

                Ez_err: float, optional

                Bz_err: float, optional

            field: dict with
                Bz:
                    z0: float
                    z1: float
                    L: float
                    fourier_coefficients: ndarray
                        Only present when style = 'fourier'
                    derivative_array: ndarray
                        Only present when style = 'derivatives'
                Ez:
                    z0: float
                    z1: float
                    L: float
                    fourier_coefficients: ndarray
                        Only present when style = 'fourier'
                    derivative_array: ndarray
                        Only present when style = 'derivatives'


    """

    # Ele info from Tao
    edat = ele_info(tao, ele_id)
    ix_ele = edat["ix_ele"]
    if name is None:
        name = edat["name"]

    # FieldMesh
    grid_params = tao.ele_grid_field(ix_ele, 1, "base", as_dict=False)
    field_file = grid_params["file"].value
    if cache is not None:
        if field_file in cache:
            field_mesh = cache[field_file]
        else:
            # Add to cache
            field_mesh = FieldMesh(field_file)
            cache[field_file] = field_mesh

    else:
        field_mesh = FieldMesh(field_file)

    ele_key = edat["key"].upper()
    freq = edat.get("RF_FREQUENCY", 0)
    assert np.allclose(freq, field_mesh.frequency), f"{freq} != {field_mesh.frequency}"

    # master_parameter = field_mesh.attrs.get('masterParameter', None)
    master_parameter = grid_params["master_parameter"].value
    if master_parameter == "<None>":
        master_parameter = None

    if ele_key == "E_GUN":
        zmirror = True
    else:
        zmirror = False

    z0 = field_mesh.coord_vec("z")

    # Find zedge
    eleAnchorPt = field_mesh.attrs["eleAnchorPt"]
    if eleAnchorPt == "beginning":
        zedge = edat["s_begin"]
    elif eleAnchorPt == "center":
        # Use full fieldmap!!!
        zedge = edat["s_center"] + z0[0]  # Wrong: -L_fm/2
    else:
        raise NotImplementedError(f"{eleAnchorPt} not implemented")

    outdat = {}
    info = outdat["info"] = {}

    # Phase and scale
    if ele_key == "SOLENOID":
        assert master_parameter is not None
        scale = edat[master_parameter]

        bfactor = np.abs(field_mesh.components["magneticField/z"][0, 0, :]).max()
        if not np.isclose(bfactor, 1):
            scale *= bfactor
        phi0_tot = 0
        phi0_oncrest = 0

    elif ele_key in ("E_GUN", "LCAVITY"):
        if master_parameter is None:
            scale = edat["FIELD_AUTOSCALE"]
        else:
            scale = edat[master_parameter]

        Ez0 = field_mesh.components["electricField/z"][0, 0, :]
        efactor = np.abs(Ez0).max()
        if not np.isclose(efactor, 1):
            scale *= efactor

        # Get ref_time_start
        ref_time_start = tao.ele_param(ele_id, "ele.ref_time_start")[
            "ele_ref_time_start"
        ]
        phi0_ref = freq * ref_time_start

        # phi0_fieldmap = field_mesh.attrs['RFphase'] / (2*np.pi) # Bmad doesn't use at this point
        phi0_fieldmap = grid_params["phi0_fieldmap"].value

        # Phase based on absolute time tracking
        phi0_user = sum([edat["PHI0"], edat["PHI0_ERR"]])
        phi0_oncrest = sum([edat["PHI0_AUTOSCALE"], phi0_fieldmap, -phi0_ref])
        phi0_tot = (phi0_oncrest + phi0_user) % 1

        # Useful info for scaling
        acc_v0, acc_phase0 = accelerating_voltage_and_phase(
            z0, Ez0 / np.abs(Ez0).max(), field_mesh.frequency
        )
        # print(f"v=c accelerating voltage per max field {acc_v0} (V/(V/m))")

        # Add phasing info
        info["v=c accelerating voltage per max field"] = acc_v0
        info["phi0_oncrest"] = phi0_oncrest % 1

    else:
        raise NotImplementedError

    # Call the fieldmesh method
    dat = field_mesh.to_impact_solrf(
        zedge=zedge,
        name=name,
        scale=scale,
        phase=phi0_tot * (2 * np.pi),
        style=style,
        n_coef=n_coef,
        spline_s=spline_s,
        spline_k=spline_k,
        x_offset=edat["X_OFFSET"],
        y_offset=edat["Y_OFFSET"],
        zmirror=zmirror,
        file_id=file_id,
        output_path=output_path,
    )
    # Add this to output
    outdat.update(dat)

    return outdat


def tao_create_impact_quadrupole_ele(tao, ele_id, *, default_radius=0.01, name=None):
    """
    Create an Impact-T quadrupole element from a running PyTao Tao instance.

    Parameters
    ----------

    tao: Tao object

    ele_id: str

    Returns
    -------
        dict with:
        line: str
            Impact-T style element line

        ele: dict
            LUME-Impact style element

    """

    edat = ele_info(tao, ele_id)
    if name is None:
        name = edat["name"]

    L_eff = edat["L"]
    L = 2 * L_eff  # Account for some fringe
    radius = edat["X1_LIMIT"]
    if radius == 0:
        radius = default_radius
    assert radius > 0

    zedge = edat["s_center"] - L / 2
    b1_gradient = edat["B1_GRADIENT"]
    x_offset = edat["X_OFFSET"]
    y_offset = edat["Y_OFFSET"]
    tilt = edat["TILT"]

    ele = {
        "L": L,
        "type": "quadrupole",
        "zedge": zedge,
        "b1_gradient": b1_gradient,
        "L_effective": L_eff,
        "radius": radius,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "x_rotation": 0.0,
        "y_rotation": 0.0,
        "z_rotation": tilt,
        "s": edat["s"],
        "name": name,
    }

    line = f"{L} 0 0 1 {zedge} {L_eff} {radius} {x_offset} {y_offset} 0 0 0 "

    return {"ele": ele, "line": line}


def tao_create_impact_lattice_and_fieldmaps(
    tao,
    emfield_cartesian_eles="EM_FIELD::*",
    solrf_eles="E_GUN::*,SOLENOID::*,LCAVITY::*",
    quadrupole_eles="quad::*",
    fieldmap_style="fourier",
    n_coef=30,
):
    """
    Create an Impact-T style lattice and fieldmaps from a running PyTao Tao instance.

    Elements must have associated fieldmaps.

    Parameters
    ----------
    tao: Tao object

    solrf_eles: str, default = 'E_GUN::*,SOLENOID::*,LCAVITY::*'
        Matching string for element names to be converted to Impact-T solrf elements.

    quadrupole_eles: str, default = 'quad::*'
         Matching string for element names to be converted to Impact-T quadrupole elements.

    fieldmap_style: str, default = 'fourier'
        Style of fieldmap to create. One of: ('fourier', 'derivatives').

    n_coef: float

    Returns
    -------
    lattice: list of dict
        List of element dicts that form the lattice
    fieldmaps: dict of

    """

    # Error checking
    if fieldmap_style not in ("fourier", "derivatives"):
        raise ValueError(
            f"fieldmap_style '{fieldmap_style}' not allowed, must be one of: ('fourier', 'derivatives')"
        )

    # Get unique name dict
    unique_name = tao_unique_names(tao)

    # Extract elements to use
    emfield_cartesian_ele_ixs = list(
        tao.lat_list(
            emfield_cartesian_eles, "ele.ix_ele", flags="-array_out -no_slaves"
        )
    )
    solrf_ele_ixs = list(
        tao.lat_list(solrf_eles, "ele.ix_ele", flags="-array_out -no_slaves")
    )

    # Large list
    ele_ixs = emfield_cartesian_ele_ixs + solrf_ele_ixs

    # Make a dict of field_file:file_id
    field_files = {
        ele_ix: tao.ele_grid_field(ele_ix, 1, "base", as_dict=False)["file"].value
        for ele_ix in ele_ixs
    }
    # Make file_id lookup table
    file_id_lookup = {}
    for ix, ix_ele in enumerate(sorted(list(set(field_files.values())))):
        file_id_lookup[ix_ele] = ix + 1  # Start at 1

    # Form lattice and fieldmaps
    lattice = []
    cache = {}
    fieldmaps = {}
    for ix_ele in ele_ixs:
        file_id = file_id_lookup[field_files[ix_ele]]
        # name = unique_name[ix_ele]

        if ix_ele in solrf_ele_ixs:
            res = tao_create_impact_solrf_ele(
                tao,
                ele_id=ix_ele,
                style=fieldmap_style,
                n_coef=n_coef,
                file_id=file_id,
                cache=cache,
                name=None,
            )  # Assume unique. TODO: better logic.
        else:
            res = tao_create_impact_emfield_cartesian_ele(
                tao, ele_id=ix_ele, file_id=file_id, cache=cache, name=None
            )
        ele = res["ele"]
        lattice.append(ele)
        fieldmaps[ele["filename"]] = res["fmap"]

    # Quadrupoles
    quad_ix_eles = tao.lat_list(
        quadrupole_eles, "ele.ix_ele", flags="-array_out -no_slaves"
    )
    for ix_ele in quad_ix_eles:
        name = unique_name[ix_ele]
        ele = tao_create_impact_quadrupole_ele(tao, ix_ele, name=name)["ele"]
        lattice.append(ele)

    # must sort!
    lattice = sorted(lattice, key=lambda d: d["zedge"])

    return lattice, fieldmaps


def impact_from_tao(tao, fieldmap_style="fourier", n_coef=30, cls=None):
    """
    Create a complete Impact object from a running Pytao Tao instance.

    Parameters
    ----------
    tao: Tao object

    fieldmap_style: str, default = 'fourier'
        Style of fieldmap to create. One of: ('fourier', 'derivatives').

    Returns
    -------
    impact_object: Impact
        Converted Impact object
    """

    lattice, fieldmaps = tao_create_impact_lattice_and_fieldmaps(
        tao, fieldmap_style=fieldmap_style, n_coef=n_coef
    )

    # Create blank object
    if cls is None:
        from impact import Impact as cls
    I = cls()
    I.input["fieldmaps"].update(fieldmaps)

    # Remove default eles
    I.lattice.pop(1)  # drift_ele
    stop_ele = I.lattice.pop(-1)
    lattice = I.lattice + deepcopy(lattice) + [stop_ele]

    I.ele["stop_1"]["s"] = tao.lat_list("*", "ele.s").max()
    I.header["Bcurr"] = 0  # Turn off SC
    I.header["Flagerr"] = 1  # Allow offsets

    # Check for cathode start
    if len(tao.lat_list("e_gun::*", "ele.ix_ele")) > 0:
        cathode_start = True
    else:
        cathode_start = False

    # Special settings for cathode start.
    # TODO: pass these in more elegantly.
    if cathode_start:
        I.header["Dt"] = 5e-13
        I.header["Flagimg"] = 1  # Cathode start
        I.header["Zimage"] = 0.12  # Conservative image charge distance
        I.header["Nemission"] = 900  # for cathode start
        timestep_ele = {
            "type": "change_timestep",
            "dt": 1e-12,
            "s": 0.5,
            "name": "change_timestep_1",
        }
        lattice = [timestep_ele] + lattice

    I.input["lattice"] = lattice
    I.ele_bookkeeper()

    return I
