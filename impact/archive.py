# import numpy as np

from pmd_beamphysics.units import write_dataset_and_unit_h5, read_dataset_and_unit_h5
from pmd_beamphysics import ParticleGroup, FieldMesh

from .parsers import header_lines
from .lattice import lattice_lines
from .fieldmaps import (
    solrf_field_from_data,
    data_from_solrf_fieldmap,
    upgrade_old_solenoid_fieldmap,
)
from .tools import fstr, isotime, native_type
from .control import ControlGroup

import warnings
import numpy as np

# ----------------------------
# Basic archive metadata


def impact_init(h5, version=None):
    """
    Set basic information to an open h5 handle

    """

    if not version:
        from impact import __version__

        version = __version__

    d = {
        "dataType": "lume-impact",
        "software": "lume-impact",
        "version": version,
        "date": isotime(),
    }
    for k, v in d.items():
        h5.attrs[k] = fstr(v)


def is_impact_archive(h5, key="dataType", value=np.bytes_("lume-impact")):
    """
    Checks if an h5 handle is a lume-impact archive
    """
    return key in h5.attrs and h5.attrs[key] == value


def find_impact_archives(h5):
    """
    Searches one level for a valid impact archive.
    """
    if is_impact_archive(h5):
        return ["./"]
    else:
        return [g for g in h5 if is_impact_archive(h5[g])]


# ------------------------------------------
# Basic tools


def write_attrs_h5(h5, data, name=None):
    """
    Simple function to write dict data to attribues in a group with name
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    for key in data:
        g.attrs[key] = data[key]
    return g


def read_attrs_h5(h5):
    """
    Simple read attributes from h5 handle
    """
    d = dict(h5.attrs)

    # Convert to native types
    for k, v in d.items():
        d[k] = native_type(v)

    return d


def read_datasets_h5(h5):
    """
    Simple read datasets from h5 handle into numpy arrays
    """
    d = {}
    for k in h5:
        d[k] = h5[k][:]
    return d


def read_list_h5(h5):
    """
    Read list from h5 file.

    A list is a group of groups named with their index, and attributes as the data.

    The format corresponds to that written in write_lattice_h5
    """

    # Convert to ints for sorting
    ixlist = sorted([int(k) for k in h5])
    # Back to strings
    ixs = [str(i) for i in ixlist]
    eles = []
    for ix in ixs:
        e = read_attrs_h5(h5[ix])
        eles.append(e)
    return eles


# ------------------------------------------


def write_control_groups_h5(h5, group_data, name="control_groups"):
    """
    Writes the ControlGroup object data to the attrs in
    an h5 group for archiving.

    See: read_control_groups_h5
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    for name, G in group_data.items():
        g.attrs[name] = fstr(G.dumps())


def read_control_groups_h5(h5, verbose=False):
    """
    Reads ControlGroup object data

    See: write_control_groups_h5
    """
    group_data = {}
    for name in h5.attrs:
        dat = h5.attrs[name]
        G = ControlGroup()
        G.loads(dat)
        group_data[name] = G

        if verbose:
            print("h5 read control_groups:", name, "=", G)

    return group_data


def write_lattice_h5(h5, eles, name="lattice"):
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    for i, ele in enumerate(eles):
        write_attrs_h5(g, ele, name=str(i))


# ------------------------------------------


def write_input_h5(h5, input, name="input", include_fieldmaps=True):
    """
    Write header

    Note that the filename ultimately needs to be ImpactT.in

    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    # ImpactT.in as text
    header = input["header"]
    lattice = input["lattice"]
    lines = header_lines(header) + lattice_lines(lattice)
    g.attrs["ImpactT.in"] = "\n".join(lines)

    # Header
    write_attrs_h5(g, input["header"], name="header")

    # Eles
    write_lattice_h5(g, input["lattice"])

    # Original input
    if "original_input" in input:
        g.attrs["original_input"] = input["original_input"]

    # particle filename
    if "input_particle_file" in input and input["input_particle_file"]:
        g.attrs["input_particle_file"] = input["input_particle_file"]

    # Any fieldmaps
    if "fieldmaps" in input and include_fieldmaps:
        g2 = g.create_group("fieldmaps")

        for name, fieldmap in input["fieldmaps"].items():
            write_fieldmap_h5(g2, fieldmap, name=name)


def read_input_h5(h5, verbose=False):
    """
    Read all Impact-T input from h5 handle.
    """
    d = {}
    d["header"] = read_attrs_h5(h5["header"])
    d["lattice"] = read_list_h5(h5["lattice"])

    if "original_input" in h5.attrs:
        d["original_input"] = h5.attrs["original_input"]

    if "input_particle_file" in h5.attrs:
        d["input_particle_file"] = h5.attrs["input_particle_file"]
        if verbose:
            print("h5 read:", "input_particle_file")
    if "fieldmaps" in h5:
        d["fieldmaps"] = {}
        for k in h5["fieldmaps"]:
            d["fieldmaps"][k] = read_fieldmap_h5(h5["fieldmaps"][k])
        if verbose:
            print("h5 read fieldmaps:", list(d["fieldmaps"]))

        # Legacy fieldmap update
        for ele in d["lattice"]:
            if ele["type"] == "solrf":
                name = ele["filename"]
                fm = d["fieldmaps"][name]
                if fm["info"]["format"] == "rfdata":
                    fm["info"]["format"] = "solrf"
                    fm["field"] = solrf_field_from_data(fm.pop("data"))
                    if verbose:
                        print(f"upgraded solrf fieldmap {name}")

    return d


# ------------------------------------------
# Fieldmaps


def write_fieldmap_h5(h5, fieldmap, name=None):
    """ """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    # Look for symlink fieldmaps
    if "filePath" in fieldmap:
        g.attrs["filePath"] = fieldmap["filePath"]
        return

    # Must be real fieldmap

    # Handle formats
    info = fieldmap["info"]
    format = info["format"]
    if format == "solrf":
        data = data_from_solrf_fieldmap(fieldmap)
    elif format.endswith("_fieldmesh"):
        fieldmap["field"].write(g, name="field")
        data = None
    else:
        data = fieldmap["data"]

    # Info attributes
    write_attrs_h5(g, info, name="info")
    # Data as single dataset
    if data is not None:
        g["data"] = data


def read_fieldmap_h5(h5):
    """ """
    if "filePath" in h5.attrs:
        return {"filePath": h5.attrs["filePath"]}

    # Extract
    info = dict(h5["info"].attrs)
    format = info["format"]

    # Handle solrf
    if format == "solrf":
        data = h5["data"][:]
        fieldmap = {"info": info, "field": solrf_field_from_data(data)}
    elif format.endswith("_fieldmesh"):
        fieldmap = {"info": info, "field": FieldMesh(h5["field"])}
    else:
        data = h5["data"][:]
        fieldmap = {"info": info, "data": data}

    # Now upgrade old fomats
    if format == "solenoid_T7":
        warnings.warn(
            "deprecated format: solenoid_T7. Upgrading to solenoid_fieldmesh",
            DeprecationWarning,
        )
        fieldmap = upgrade_old_solenoid_fieldmap(fieldmap)

    return fieldmap


# ------------------------------------------


def write_output_h5(h5, impact_output, name="output", units=None):
    """
    Writes all of impact_output dict to an h5 handle

    """
    g = h5.create_group(name)

    for stats_label in ["stats", "dipole_stats"]:
        if stats_label in impact_output:
            name2 = stats_label
            g2 = g.create_group(name2)
            for key, data in impact_output[name2].items():
                if units:
                    unit = units[key]
                else:
                    unit = None
                write_dataset_and_unit_h5(g2, key, data, unit)

    if "autophase_info" in impact_output:
        g2 = g.create_group("autophase_info")
        for k, v in impact_output["autophase_info"].items():
            g2.attrs[k] = v

    # TODO: This could be simplified
    if "slice_info" in impact_output:
        name2 = "slice_info"
        g2 = g.create_group(name2)
        for name3, slice_dat in impact_output[name2].items():
            g3 = g2.create_group(name3)
            for key, data in impact_output[name2][name3].items():
                if units:
                    unit = units[key]
                else:
                    unit = None
                write_dataset_and_unit_h5(g3, key, data, unit)

    # Run info
    if "run_info" in impact_output:
        for k, v in impact_output["run_info"].items():
            g.attrs[k] = v

    # Particles
    if "particles" in impact_output:
        write_particles_h5(g, impact_output["particles"], name="particles")


def read_output_h5(h5, expected_units=None, verbose=False):
    """
    Reads a properly archived Impact output and returns a dicts:
        output
        units


    Corresponds exactly to the output of writers.write_output_h5
    """

    o = {}
    o["run_info"] = dict(h5.attrs)

    if "autophase_info" in h5:
        o["autophase_info"] = dict(h5["autophase_info"])

    units = {}
    for stats_label in ["stats", "dipole_stats"]:
        if stats_label in h5:
            name2 = stats_label
            if verbose:
                print(f"reading {name2}")
            g = h5[name2]
            o[name2] = {}
            for key in g:
                if expected_units:
                    expected_unit = expected_units[key]
                else:
                    expected_unit = None
                o[name2][key], units[key] = read_dataset_and_unit_h5(
                    g[key], expected_unit=expected_unit
                )

    # TODO: this could be simplified
    if "slice_info" in h5:
        name2 = "slice_info"
        if verbose:
            print(f"reading {name2}")
        g = h5[name2]
        o[name2] = {}
        for name3 in g:
            g2 = g[name3]
            o[name2][name3] = {}

            for key in g2:
                if expected_units:
                    expected_unit = expected_units[key]
                else:
                    expected_unit = None
                o[name2][name3][key], units[key] = read_dataset_and_unit_h5(
                    g2[key], expected_unit=expected_unit
                )

    if "particles" in h5:
        o["particles"] = read_particles_h5(h5["particles"])

    return o, units


# ----------------------------
# particles


def opmd_init(h5, basePath="/screen/%T/", particlesPath="/"):
    """
    Root attribute initialization.

    h5 should be the root of the file.
    """
    d = {
        "basePath": basePath,
        "dataType": "openPMD",
        "openPMD": "2.0.0",
        "openPMDextension": "BeamPhysics;SpeciesType",
        "particlesPath": particlesPath,
    }
    for k, v in d.items():
        h5.attrs[k] = fstr(v)


def write_particles_h5(h5, particles, name="particles"):
    """
    Write all screens to file, simply named by their index

    See: read_particles_h5
    """
    g = h5.create_group(name)

    # Set base attributes
    opmd_init(h5, basePath="/" + name + "/%T/", particlesPath="/")

    # Loop over particles
    for name, particle_group in particles.items():
        # name = str(i)
        particle_group.write(g, name=name)


def read_particles_h5(h5):
    """
    Reads particles from h5
    """
    dat = {}
    for g in h5:
        dat[g] = ParticleGroup(h5=h5[g])
    return dat


def old_write_impact_particles_h5(
    h5, particle_data, name=None, total_charge=1.0, time=0.0, speciesType="electron"
):
    """
    Old routine. Do not delete, may still be useful.

    """

    # Write particle data at a screen in openPMD BeamPhysics format
    # https://github.com/DavidSagan/openPMD-standard/blob/EXT_BeamPhysics/EXT_BeamPhysics.md

    if name:
        g = h5.create_group(name)
    else:
        g = h5

    n_particle = len(particle_data["x"])
    # -----------
    g.attrs["speciesType"] = fstr(speciesType)
    g.attrs["numParticles"] = n_particle
    g.attrs["chargeLive"] = total_charge
    g.attrs["totalCharge"] = total_charge
    g.attrs["chargeUnitSI"] = 1

    # Position
    g["position/x"] = particle_data["x"]  # in meters
    g["position/y"] = particle_data["y"]
    g["position/z"] = particle_data["z"]
    g["position"].attrs["unitSI"] = 1.0
    for component in [
        "position/x",
        "position/y",
        "position/z",
        "position",
    ]:  # Add units to all components
        g[component].attrs["unitSI"] = 1.0
        g[component].attrs["unitDimension"] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # m

    # momenta
    g["momentum/x"] = particle_data["GBx"]  # gamma*beta_x
    g["momentum/y"] = particle_data["GBy"]  # gamma*beta_y
    g["momentum/z"] = particle_data["GBz"]  # gamma*beta_z
    for component in ["momentum/x", "momentum/y", "momentum/z", "momentum"]:
        g[component].attrs["unitSI"] = 2.73092449e-22  # m_e *c in kg*m / s
        g[component].attrs["unitDimension"] = (
            1.0,
            1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )  # kg*m / s

    # Constant records

    # Weights. All particles should have the same weight (macro charge)
    weight = total_charge / n_particle
    g2 = g.create_group("weight")
    g2.attrs["value"] = weight
    g2.attrs["shape"] = n_particle
    g2.attrs["unitSI"] = 1.0
    g2.attrs["unitDimension"] = (0.0, 0.0, 1, 1.0, 0.0, 0.0, 0.0)  # Amp*s = Coulomb

    # Time
    g2 = g.create_group("time")
    g2.attrs["value"] = 0.0
    g2.attrs["shape"] = n_particle
    g2.attrs["unitSI"] = 1.0
    g2.attrs["unitDimension"] = (0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0)  # s

    # Status
    g2 = g.create_group("status")
    g2.attrs["value"] = 1
    g2.attrs["shape"] = n_particle
    g2.attrs["unitSI"] = 1.0
    g2.attrs["unitDimension"] = (0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)  # dimensionless
