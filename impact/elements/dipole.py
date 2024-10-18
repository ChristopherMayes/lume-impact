import matplotlib.pyplot as plt

import numpy as np


"""
Dipole fieldmap


Here, the input file rfdatav4 contains 22 lines.

The first line contains the switch flag for 1D CSR wakefield. The CSR wakefield will be included for a value greater than 0.
The second line contains the relativistic γ of the beam.
Lines 3 to 10 contain k1, b1(m), k2, b2(m), k3, b3(m), k4, b4(m), the geometric description of the pole faces at the entrance and the exit.
Line 11 is twice the shift z0 (m) at the entrance fringe field region.
Line 12 is twice the shift z0 (m) at the exit fringe field region.
    These two lines are used to determine the shift used in the Enge function fitting.
    The shifts are half of those values.
Lines 13 to 20 contain 8 coefficients in the Enge function.
    Here, we have assumed that the entrance and the exit have the same Enge function coefficients.
Line 21 is the effective starting location along the arc of the bend in meter.
Line 22 is the effective ending location along the arc of the bend in meter.
    Normally, these two numbers are the middle locations of the fringe region.
    However, if transient effects at the exit are important, the total length of the bend may include a section of drift.

"""

dipole_fieldmap_labels = [
    "csr_on",
    "gamma_ref",
    "k1",
    "b1",
    "k2",
    "b2",
    "k3",
    "b3",
    "k4",
    "b4",
    "w1",
    "w2",  # entrance and exit widths?
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "c7",
    "c8",  # Enge. c2 = 1/fint will correspond to Bmad.
    "entrance_s",
    "exit_s",
]


def parse_dipole_fieldmap_data(data):
    return {k: v for k, v in zip(dipole_fieldmap_labels, data)}


def parse_dipole_fieldmap(fname):
    """
    parses the rfdatav4 dipole fieldmap data into a dict

    """
    with open(fname, "r") as f:
        lines = f.readlines()

    data = [float(v.strip()) for v in lines]

    return parse_dipole_fieldmap_data(data)


def add_line(ax, k, b, p=0.1, s0=0, y0=0, **kwargs):
    y = np.linspace(-p, p, 100) + y0
    x = k * y + b + s0

    ax.plot(x, y, **kwargs)


def plot_dipole_fieldmap(data, g=None, L=None, ref_data=None, **kwargs):
    """
    Uses rfdatav4 style data dict to plot the fieldmap edges, etc.


    """

    d = data

    fig, ax = plt.subplots(**kwargs)
    ax.set_aspect("equal")

    ax.set_title("Dipole coordinate system")
    ax.set_xlabel("Z (m)")
    ax.set_ylabel("Y (m)")
    s0 = data["entrance_s"]
    s1 = data["exit_s"]
    if L is None:
        L = s1 - s0

    if g is None:
        g = 0
        L = 1
        # Xend = L
        Yend = 0
    else:
        s = np.linspace(0, L, 100)

        X = np.sin(g * s) / g + s0
        Y = -(1 - np.cos(g * s)) / g
        ax.plot(X, Y, color="black", linestyle="solid")

        # Xend = X[-1]
        Yend = Y[-1]

    add_line(ax, d["k1"], d["b1"], p=0.1, linestyle="--", label="L1")
    add_line(ax, d["k2"], d["b2"], p=0.1, linestyle="--", label="L2")
    add_line(
        ax,
        d["k3"],
        d["b3"],
        y0=Yend,
        p=0.1 * np.cos(g * L),
        linestyle="solid",
        label="L3",
    )
    add_line(
        ax, d["k4"], d["b4"], y0=Yend, p=0.1 * np.cos(g * L), linestyle="--", label="L4"
    )

    if ref_data:
        ax.plot(ref_data["ref_z"], ref_data["ref_x"], color="red", linestyle="dotted")

    ax.legend()


def entrance_edges(e1=0, w1=0.02, s0=0):
    k = np.tan(e1)
    b = w1 / (2 * np.cos(e1))

    return dict(k1=k, k2=k, b1=-b + s0, b2=b + s0)


def exit_edges(e2=0, w2=0.02, g=0, L=1, s0=0):
    theta = g * L

    phi = theta - e2

    # X, Z offset of the end of the arc section
    X = -(1 - np.cos(theta)) / g
    Z = (
        np.sinc(theta / np.pi) * L + s0
    )  # numpy sinc has a pi in it, so need to remove it
    # Z = np.sin(theta)/g + s0  # numpy sinc has a pi in it, so need to remove it
    b0 = Z - X * np.tan(phi)
    b3 = b0 - (w2 / 2) / np.cos(phi)
    b4 = b0 + (w2 / 2) / np.cos(phi)

    k = np.tan(phi)
    return dict(k3=k, k4=k, b3=b3, b4=b4)


def new_dipole_fieldmap_data(
    L=1, g=0, gamma_ref=1, e1=0, e2=0, half_gap=0.01, fint=0.5, csr_on=False
):
    """
    Returns rfdatav4 dict from MAD-style SBEND parameters.

    """

    # CSR and Reference energy
    d = dict(csr_on=int(csr_on), gamma_ref=gamma_ref)

    # edge geometry
    w = half_gap * 3

    # Effective arc start and stop
    s0 = w / 2
    d["entrance_s"] = s0
    d["exit_s"] = L + w / 2 + w  # ??? DEBUG

    # Edge Geometry
    d.update(entrance_edges(e1=e1, w1=w, s0=s0))
    d.update(exit_edges(e2=e2, w2=w, g=g, L=L, s0=s0))

    # Enge coefficients
    d.update(dict(c1=0, c2=1 / fint, c3=0, c4=0, c5=0, c6=0, c7=0, c8=0))

    d["w1"] = w
    d["w2"] = w

    return d


def dipole_fieldmap_lines(data, filename=None):
    text = "\n".join([str(data[key]) for key in dipole_fieldmap_labels])

    if filename:
        with open(filename, "w") as f:
            f.write(text)
    return text
