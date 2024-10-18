from .parsers import header_lines
from .lattice import lattice_lines


def write_input_particles_from_file(src, dest, n_particles, skiprows=1):
    """
    Write a partcl.data file from a source file, setting the number of particles as the first line.

    If the source is another partcl.data file, use skiprows=1
    If the source does not have a header line, use skiprows=0

    Warning: Does not randomize or check the length of the file!
    """
    with open(src, "rt") as fsrc:
        for _ in range(skiprows):
            fsrc.readline()
        with open(dest, "wt") as fdest:
            fdest.write(str(n_particles) + "\n")
            for _ in range(n_particles):
                line = fsrc.readline()
                fdest.write(line)


def write_impact_input(filePath, header, eles):
    """
    Write

    Note that the filename ultimately needs to be ImpactT.in

    """

    lines = header_lines(header) + lattice_lines(eles)
    with open(filePath, "w") as f:
        for line in lines:
            f.write(line + "\n")
