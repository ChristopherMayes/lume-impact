from __future__ import annotations

import typing
import warnings
from collections import Counter
from copy import deepcopy

import numpy as np
from beamphysics import FieldMesh
from beamphysics.fields.analysis import accelerating_voltage_and_phase
from beamphysics.interfaces.impact import create_impact_t_wakefield_data

from ..lattice import new_write_beam

if typing.TYPE_CHECKING:
    from pytao import Tao

# fort.N units for read-in wake tables. IMPACT-T writes its own output to units
# below 100 (per bunch from 40, 50, 60 and 70) and at 188-288, and write_beam
# elements take units from 71 upward.
WAKEFIELD_UNITS = range(300, 400)


def tao_unique_names(tao: Tao, ix_uni: int | str = "", ix_branch: int | str = ""):
    """
    Invent a unique name

    Parameters
    ----------
    tao : pytao.Tao

    Returns
    -------
    dict of int:str
        Mapping of ix_ele to a unique name
    """
    # Get all ixs
    ixs = set(tao.lat_list("*", "ele.ix_ele", ix_uni=ix_uni, ix_branch=ix_branch))
    ixs.update(
        set(
            tao.lat_list(
                "*",
                "ele.ix_ele",
                flags="-array_out -no_slaves",
                ix_uni=ix_uni,
                ix_branch=ix_branch,
            )
        )
    )
    ixs = sorted(ixs)

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


def ele_info(tao: Tao, ele_id, which="model") -> dict[str, float | str | int]:
    """
    Returns a dict of element attributes from ele_head and ele_gen_attribs
    """
    edat = tao.ele_head(ele_id, which=which)
    edat.update(tao.ele_gen_attribs(ele_id, which=which))
    s = edat["s"]
    L = edat["L"]
    edat["s_begin"] = s - L
    edat["s_center"] = (s + edat["s_begin"]) / 2

    return edat


def tao_create_impact_emfield_cartesian_ele(
    tao, ele_id, *, file_id=666, output_path=None, cache=None, name=None
):
    """
    Create an Impact-T emfield_cartesian element from a running PyTao Tao instance.

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
    grid_params = tao.ele_grid_field(ix_ele, 1, "base")
    field_file = grid_params["file"]
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
    master_parameter = grid_params["master_parameter"]
    if master_parameter == "<None>":
        master_parameter = "FIELD_AUTOSCALE"
    scale = edat[master_parameter]

    # check lengths
    z0 = field_mesh.zmin
    z1 = field_mesh.zmax
    L_fm = z1 - z0
    L_ele = edat["L"]
    if not np.isclose(L_fm, L_ele):
        raise NotImplementedError(
            f"Element {ele_id} length {L_ele} currently must be the same as the fieldmap length {L_fm}"
        )

    # Find zedge
    eleAnchorPt = field_mesh.attrs["eleAnchorPt"]
    if eleAnchorPt == "beginning":
        zedge = edat["s_begin"]
    elif eleAnchorPt == "center":
        # Set this so that the anchor is effectively at zedge
        field_mesh.zmin = 0
        zedge = edat["s_center"] - L_fm / 2
    else:
        raise NotImplementedError(f"{eleAnchorPt} not implemented")

    outdat = {}
    # Add field integrals
    info = outdat["info"] = {}
    for key in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
        z, fz = field_mesh.axis_values("z", key)
        info[f"integral_{key}_dz"] = np.trapezoid(fz, z)

    x_offset = edat["X_OFFSET"]
    y_offset = edat["Y_OFFSET"]
    x_rotation = -edat["Y_PITCH"]  # Note: x_rotation is equivalent to a y pitch
    y_rotation = edat["X_PITCH"]

    # TODO: check this
    x_offset = x_offset + np.sin(edat["X_PITCH"]) * L_fm / 2
    y_offset = y_offset - np.sin(edat["Y_PITCH"]) * L_fm / 2

    # TODO: this needs to be corrected because rotations are about the beginning.
    z_offset = 0

    # Call the fieldmesh method
    dat = field_mesh.to_impact_emfield_cartesian(
        zedge=zedge + z_offset,
        name=name,
        scale=scale,
        phase=0,
        x_offset=x_offset,
        y_offset=y_offset,
        x_rotation=x_rotation,
        y_rotation=y_rotation,
        z_rotation=edat["TILT"],
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
    grid_params = tao.ele_grid_field(ix_ele, 1, "base")
    field_file = grid_params["file"]
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
    master_parameter = grid_params["master_parameter"]
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
        phi0_fieldmap = grid_params["phi0_fieldmap"]

        # Phase based on absolute time tracking
        phi0_user = sum([edat["PHI0"], edat.get("PHI0_ERR", 0)])
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


def tao_create_impact_wakefield_eles(tao, zmax=None, n=None, which="model", dz=None):
    """
    Create Impact-T wakefield elements from the short-range wakes in a Tao lattice.

    Each Bmad element carrying a short-range longitudinal wake becomes a
    wakefield (-6) element spanning that element, which reads its wake from a
    fort.N table. Elements with identical wakes share a table.

    Parameters
    ----------
    tao : Tao
        Initialized Tao object.
    zmax : float, optional
        Smallest trailing distance to tabulate [m]. Each table covers at least
        the natural range of its wake model, so this only extends it. IMPACT-T
        extrapolates the table linearly beyond its last row, so set this when
        the bunch is longer than that range.
    n : int, optional
        Number of table rows. Defaults to the number the model needs, clamped to
        the 1000 rows IMPACT-T reads. A tabulated (`z_long`) wake keeps its
        spacing, so for it this only limits the rows.
    which : {"model", "base", "design"}, optional
        Which Tao model to use, by default "model".
    dz : float or dict of str to float, optional
        Spacing to rebin tabulated (`z_long`) wakes to [m], given once or per
        element name: the IMPACT-T mesh width at that element. A bin-averaged
        table is only correct at that width. If omitted, each table keeps its
        Bmad spacing. See `impact.wakefields.rebin_tabular_wakefield`.

    Returns
    -------
    lattice : list of dict
        Wakefield element dicts.
    fieldmaps : dict
        Mapping of fort.N filename to wake table.

    Raises
    ------
    NotImplementedError
        If an element carries a wake that IMPACT-T cannot represent, as described
        in `wakefield_from_tao`, or a lumped wake on a zero-length element.
    ValueError
        If `dz` names an unknown element.

    Notes
    -----
    Impact-T convolves the wake with the bunch on its longitudinal mesh. The
    mesh cell, the bunch length divided by header["Nz"] - 1, must be well below
    the shortest length scale of the wake. IMPACT-T recomputes it every step, so
    a rebinned table matches it only where the bunch length is as assumed.
    """
    from beamphysics.interfaces.impact import IMPACT_T_MAX_WAKEFIELD_ROWS
    from beamphysics.wakefields import TabularWakefield
    from pytao.model import ElementHead

    from ..wakefields import (
        check_wakefield_dz_names,
        rebin_tabular_wakefield,
        wakefield_dz_for,
    )

    # impact.z imports impact.impact, which imports this module.
    from ..z.interfaces.bmad import wakefield_from_tao

    unique_name = tao_unique_names(tao)
    check_wakefield_dz_names(dz, set(unique_name.values()))
    ixs = tao.lat_list("*", "ele.ix_ele", flags="-array_out -no_slaves")

    lattice = []
    fieldmaps = {}
    table_lengths = {}
    units = iter(WAKEFIELD_UNITS)
    for ix_ele in ixs:
        if not ElementHead.model_validate(tao.ele_head(ix_ele, which=which)).has_wake:
            continue
        info = wakefield_from_tao(tao, ix_ele, which=which)
        if info is None:
            continue

        edat = ele_info(tao, ix_ele, which=which)
        name = unique_name[ix_ele]
        if isinstance(info.wakefield, TabularWakefield):
            wakefield = rebin_tabular_wakefield(
                info.wakefield,
                dz=wakefield_dz_for(dz, name),
                zmax=zmax,
                max_rows=IMPACT_T_MAX_WAKEFIELD_ROWS if n is None else n,
            )
            data, length = create_impact_t_wakefield_data(
                wakefield, n=len(wakefield.z_data)
            )
        else:
            wake_zmax = info.wakefield.default_zmax
            if zmax is not None:
                wake_zmax = max(wake_zmax, zmax)
            data, length = create_impact_t_wakefield_data(
                info.wakefield, zmax=wake_zmax, n=n
            )

        # Impact-T applies the table per meter between V3 and V4, so a lumped
        # Bmad wake is spread over the element length.
        if not info.scale_with_length:
            if edat["L"] <= 0:
                raise NotImplementedError(
                    f"Element {name} has a lumped wakefield (scale_with_length = False) "
                    "but no length, so it cannot be expressed as the per-meter wake "
                    "Impact-T applies"
                )
            data = data / edat["L"]

        for filename, fmap in fieldmaps.items():
            if table_lengths[filename] == length and np.array_equal(fmap["data"], data):
                break
        else:
            unit = next(units, None)
            if unit is None:
                raise NotImplementedError(
                    f"More than {len(WAKEFIELD_UNITS)} distinct wakefield tables"
                )
            filename = f"fort.{unit}"
            fieldmaps[filename] = {"info": {"format": "rfdata"}, "data": data}
            table_lengths[filename] = length

        lattice.append(
            {
                "type": "wakefield",
                "name": f"{name}_wakefield",
                "s_begin": edat["s_begin"],
                "s": edat["s"],
                "method": "from_file",
                "filename": filename,
                "table_length": length,
            }
        )

    return lattice, fieldmaps


def tao_create_impact_lattice_and_fieldmaps(
    tao,
    emfield_cartesian_eles="EM_FIELD::*",
    solrf_eles="E_GUN::*,SOLENOID::*,LCAVITY::*",
    quadrupole_eles="quad::*",
    write_beam_eles="monitor::*",
    fieldmap_style="fourier",
    n_coef=30,
    wakefield_zmax=None,
    wakefield_dz=None,
):
    """
    Create an Impact-T style lattice and fieldmaps from a running PyTao Tao instance.

    Elements must have associated fieldmaps.

    Parameters
    ----------
    tao : Tao
        Initialized Tao object

    emfield_cartesian_eles : str or list[str], optional
        Matching string or list of element names to be converted to Impact-T emfield_cartesian elements, by default = "EM_FIELD::*"

    solrf_eles : str or list[str], optional
        Matching string or list of element names to be converted to Impact-T solrf elements, by default = 'E_GUN::*,SOLENOID::*,LCAVITY::*'

    quadrupole_eles : str or list[str], optional
        Matching string or list of element names to be converted to Impact-T quadrupole elements, by default = 'quad::*'

    write_beam_eles : str or list[str], optional
        Matching string or list of element names to be converted to Impact-T write beam elements.
        Note that there is a limit to the number of these that can be created.  By default = 'monitor::*'

    fieldmap_style : str, optional
        Style of fieldmap to create. One of: ('fourier', 'derivatives'), by default = 'fourier'

    n_coef : int, optional
        a number of fourier (?) coefficients, by default 30

    wakefield_zmax : float, optional
        Smallest trailing distance to tabulate short-range wakes over [m]. See
        `tao_create_impact_wakefield_eles`.

    wakefield_dz : float or dict of str to float, optional
        Spacing to rebin tabulated (`z_long`) wakes to [m], once or per element
        name. See `tao_create_impact_wakefield_eles`.

    Returns
    -------
    lattice : list of dict
        List of element dicts that form the lattice
    fieldmaps : dict
        Mapping of element name to fieldmap

    """

    # Error checking
    if fieldmap_style not in ("fourier", "derivatives"):
        raise ValueError(
            f"fieldmap_style '{fieldmap_style}' not allowed, must be one of: ('fourier', 'derivatives')"
        )

    # Get unique name dict
    unique_name = tao_unique_names(tao)

    def get_ixs(match_or_list):
        if match_or_list is None:
            return []
        elif isinstance(match_or_list, str):
            return list(
                tao.lat_list(match_or_list, "ele.ix_ele", flags="-array_out -no_slaves")
            )
        else:
            return [tao.ele_head(ele)["ix_ele"] for ele in match_or_list]

    # Extract elements to use
    emfield_cartesian_ele_ixs = get_ixs(emfield_cartesian_eles)

    solrf_ele_ixs = []
    for ix_ele in get_ixs(solrf_eles):
        head = tao.ele_head(ix_ele)
        ngrid = head["num#grid_field"]
        if ngrid == 0:
            warnings.warn(
                f"Solenoid {head['name']} has no grid. Impact-T cannot use hard-edge solenoids."
            )
        elif ngrid == 1:
            solrf_ele_ixs.append(ix_ele)
        else:
            raise NotImplementedError(
                f"Solenoid {head['name']}has more than one grid: {ngrid}"
            )

    # Large list
    ele_ixs = emfield_cartesian_ele_ixs + solrf_ele_ixs

    # Make a dict of field_file:file_id
    field_files = {}
    for ele_ix in ele_ixs:
        field_files[ele_ix] = tao.ele_grid_field(ele_ix, 1, "base")["file"]

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
    quad_ix_eles = get_ixs(quadrupole_eles)
    for ix_ele in quad_ix_eles:
        name = unique_name[ix_ele]
        ele = tao_create_impact_quadrupole_ele(tao, ix_ele, name=name)["ele"]
        lattice.append(ele)

    # Write beams
    write_beam_ix_eles = get_ixs(write_beam_eles)
    for ix_ele in write_beam_ix_eles:
        head = tao.ele_head(ix_ele)
        ele = new_write_beam(
            name=head["name"],
            s=head["s"],
            ref_eles=lattice,
        )  # ref_eles will ensure that there are no naming conflicts
        lattice.append(ele)

    wake_eles, wake_tables = tao_create_impact_wakefield_eles(
        tao, zmax=wakefield_zmax, dz=wakefield_dz
    )
    lattice += wake_eles
    fieldmaps.update(wake_tables)

    # must sort!
    def get_s(ele):
        if "zedge" in ele:
            return ele["zedge"]
        elif "s" in ele:
            return ele["s"]
        else:
            raise ValueError(f"No s or zedge found in {ele}")

    lattice = sorted(lattice, key=get_s)

    return lattice, fieldmaps


def impact_from_tao(
    tao,
    fieldmap_style="fourier",
    n_coef=30,
    cls=None,
    emfield_cartesian_eles="EM_FIELD::*",
    solrf_eles="E_GUN::*,SOLENOID::*,LCAVITY::*",
    quadrupole_eles="quad::*",
    write_beam_eles="monitor::*",
    wakefield_zmax=None,
    wakefield_dz=None,
):
    """
    Create a complete Impact object from a running Pytao Tao instance.

    Parameters
    ----------
    tao: Tao object

    fieldmap_style: str, default = 'fourier'
        Style of fieldmap to create. One of: ('fourier', 'derivatives').

    emfield_cartesian_eles: str or list, default = "EM_FIELD::*"
        Matching string or list of element names to be converted to Impact-T emfield_cartesian elements.

    solrf_eles: str or list, default = 'E_GUN::*,SOLENOID::*,LCAVITY::*'
        Matching string or list for element names to be converted to Impact-T solrf elements.

    quadrupole_eles: str or list, default = 'quad::*'
         Matching string or list for element names to be converted to Impact-T quadrupole elements.

    write_beam_eles: str or list, default = 'monitor::*'
         Matching string or list for element names to be converted to Impact-T write beam elements.
         Note that there is a limit to the number of these that can be created

    wakefield_zmax: float, optional
         Smallest trailing distance to tabulate short-range wakes over [m].
         Each table covers at least the natural range of its wake model;
         Impact-T extrapolates beyond the table, so set this when the bunch is
         longer than that range.

    wakefield_dz: float or dict of str to float, optional
         Spacing to rebin tabulated (`z_long`) wakes to [m], once or per element
         name: the IMPACT-T mesh width at that element, the bunch length there
         divided by header["Nz"] - 1. If omitted, each table keeps its Bmad
         spacing. See `tao_create_impact_wakefield_eles`.

    Returns
    -------
    impact_object: Impact
        Converted Impact object
    """

    lattice, fieldmaps = tao_create_impact_lattice_and_fieldmaps(
        tao,
        fieldmap_style=fieldmap_style,
        n_coef=n_coef,
        emfield_cartesian_eles=emfield_cartesian_eles,
        solrf_eles=solrf_eles,
        quadrupole_eles=quadrupole_eles,
        write_beam_eles=write_beam_eles,
        wakefield_zmax=wakefield_zmax,
        wakefield_dz=wakefield_dz,
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

    # Impact-T applies wakes only at nonzero current, so space charge is
    # switched off by an element instead.
    if any(ele["type"] == "wakefield" for ele in lattice):
        I.total_charge = float(tao.beam_init(0)["bunch_charge"])
        if I.total_charge == 0:
            warnings.warn(
                "The lattice has wakefields but the Tao bunch_charge is zero, "
                "so they will have no effect"
            )
        lattice = [
            {"type": "spacecharge", "name": "spacecharge_off", "s": 0, "is_on": False}
        ] + lattice

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
