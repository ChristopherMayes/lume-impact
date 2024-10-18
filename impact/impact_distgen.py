from . import Impact, tools
from .evaluate import default_impact_merit
from distgen import Generator

from h5py import File

import os


def run_impact_with_distgen(
    settings=None,
    distgen_input_file=None,
    impact_config=None,
    workdir=None,
    verbose=False,
):
    """
    Creates, runs, and returns an Impact object using distgen input.

    .distgen_input = parsed distgen.Generatator's .input is attached to the object.

    """

    # setup objects
    if isinstance(impact_config, str):
        I = Impact.from_yaml(impact_config)
    else:
        I = Impact(**impact_config)

    if workdir:
        I._workdir = workdir  # TODO: fix in LUME-Base
        I.configure()  # again

    I.verbose = verbose
    G = Generator(distgen_input_file)
    G.verbose = verbose

    if settings:
        for key in settings:
            val = settings[key]
            if key.startswith("distgen:"):
                key = key[len("distgen:") :]
                if verbose:
                    print(f"Setting distgen {key} = {val}")
                G[key] = val
            else:
                # Assume impact
                if verbose:
                    print(f"Setting impact {key} = {val}")
                I[key] = val

    # Get particles
    G.run()
    P = G.particles

    # Attach particles
    I.initial_particles = P

    # Attach distgen input. This is non-standard.
    I.distgen_input = G.input

    I.run()

    return I


def evaluate_impact_with_distgen(
    settings,
    distgen_input_file=None,
    impact_config=None,
    workdir=None,
    archive_path=None,
    merit_f=None,
    verbose=False,
):
    """

    Similar to run_impact_with_distgen, but requires settings a the only positional argument.

    If an archive_path is given, the complete evaluated Impact and Generator objects will be archived
    to a file named using a fingerprint from both objects.

    If merit_f is given, this function will be applied to the evaluated Impact object, and this will be returned.

    Otherwise, a default function will be applied.


    """

    I = run_impact_with_distgen(
        settings=settings,
        distgen_input_file=distgen_input_file,
        impact_config=impact_config,
        workdir=workdir,
        verbose=verbose,
    )

    if merit_f:
        output = merit_f(I)
    else:
        output = default_impact_merit(I)

    if "error" in output and output["error"]:
        raise ValueError("run_impact_with_distgen returned error in output")

    # Recreate Generator object for fingerprint, proper archiving
    # TODO: make this cleaner
    G = Generator(I.distgen_input)

    fingerprint = fingerprint_impact_with_distgen(I, G)
    output["fingerprint"] = fingerprint

    if archive_path:
        path = tools.full_path(archive_path)
        assert os.path.exists(path), f"archive path does not exist: {path}"
        archive_file = os.path.join(path, fingerprint + ".h5")
        output["archive"] = archive_file

        # Call the composite archive method
        archive_impact_with_distgen(I, G, archive_file=archive_file)

    return output


def fingerprint_impact_with_distgen(impact_object, distgen_object):
    """
    Calls fingerprint() of each of these objects
    """
    f1 = impact_object.fingerprint()
    f2 = distgen_object.fingerprint()
    d = {"f1": f1, "f2": f2}
    return tools.fingerprint(d)


def archive_impact_with_distgen(
    impact_object,
    distgen_object,
    archive_file=None,
    impact_group="impact",
    distgen_group="distgen",
):
    """
    Creates a new archive_file (hdf5) with groups for
    impact and distgen.

    Calls .archive method of Impact and Distgen objects, into these groups.
    """

    h5 = File(archive_file, "w")

    # fingerprint = tools.fingerprint(astra_object.input.update(distgen.input))

    g = h5.create_group(distgen_group)
    distgen_object.archive(g)

    g = h5.create_group(impact_group)
    impact_object.archive(g)

    h5.close()
