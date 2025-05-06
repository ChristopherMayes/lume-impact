from pmd_beamphysics.units import nice_array, nice_scale_prefix
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .lattice import ele_shape, remove_element_types, ele_bounds, ele_overlaps_s
from .fieldmaps import lattice_field, FIELD_CALC_ELE_TYPES
import numpy as np

from pmd_beamphysics.labels import mathlabel as _mathlabel


_label_hotfixes = {
    "loadbalance_min_n_particle",
    "loadbalance_max_n_particle",
    "charge_state_n_particle",
    "neg_cov_z__gammabeta_z",
    "neg_cov_y__gammabeta_y",
    "neg_cov_x__gammabeta_x",
    "neg_cov_x__gammabeta_x",
}


def mathlabel(*keys: str, **kwargs):
    fixed_keys = [
        key.replace("_", "-") if key in _label_hotfixes else key for key in keys
    ]
    return _mathlabel(*fixed_keys, **kwargs)


def plot_stat(impact_object, y="sigma_x", x="mean_z", nice=True):
    """
    Plots stat output of key y vs key

    If particles have the same stat key, these will also be plotted.

    If nice, a nice SI prefix and scaling will be used to make the numbers reasonably sized.

    """
    I = impact_object  # convenience
    fig, ax = plt.subplots()

    units1 = str(I.units(x))
    units2 = str(I.units(y))

    X = I.stat(x)
    Y = I.stat(y)

    if nice:
        X, f1, prefix1 = nice_array(X)
        Y, f2, prefix2 = nice_array(Y)
        units1 = prefix1 + units1
        units2 = prefix2 + units2
    else:
        f1 = 1
        f2 = 1
    ax.set_xlabel(x + f" ({units1})")
    ax.set_ylabel(y + f" ({units2})")

    # line plot
    plt.plot(X, Y)

    try:
        ax.scatter(
            [I.particles[name][x] / f1 for name in I.particles],
            [I.particles[name][y] / f2 for name in I.particles],
            color="red",
        )
    except Exception:
        pass

    # return fig


def add_ele_box(ele, ax, xfactor=1, yfactor=1, alpha=0.7):
    """
    Add single element box to axes (matplotlib)
    """
    if "L" not in ele:
        if "s" in ele:
            ax.vlines(ele["s"], -1, 1, linestyles="dashed")
        return ax

    d = ele_shape(ele)
    origin = (d["left"] * xfactor, d["bottom"] * yfactor)
    width = (d["right"] - d["left"]) * xfactor
    height = (d["top"] - d["bottom"]) * yfactor
    rect = patches.Rectangle(origin, width, height, color=d["color"], alpha=alpha)

    # Add the patch to the Axes
    ax.add_patch(rect)


def add_ele_marker(ele, ax, ymin=-1, ymax=1, factor=1, linesyles="dashed"):
    """
    Adds a dashed line according to ele['s']
    """
    if "s" in ele:
        ax.vlines(ele["s"] * factor, ymin, ymax, linestyles=linesyles)


def add_ele_label(ele, axis, bounds=None, factor=1):
    """
    Adds the element name

    If bounds, the position will be clipped to the bounds.
    """
    if "s" not in ele:
        return

    s = ele["s"]

    # Clip for reasonable placement
    if "L" not in ele:
        s0 = s
    else:
        s0 = s - ele["L"]
    s0 = max(bounds[0], s0)
    s = min(bounds[1], s)
    x = (s0 + s) / 2

    axis.text(
        x * factor,
        -1.1,
        ele["name"],
        ha="center",
        va="top",
        # transform=ax.transAxes,
        family="sans-serif",
        size=14,
        rotation=90,
    )


def add_layout_to_axes(
    impact_object,
    axes,
    bounds=None,
    factor=1,
    include_labels=True,
    include_field=False,
    field_t=0,
    include_markers=True,
):
    """
    Adds a layout plot to an axis.

    factor multiplies the horizontal placement, for using nice units.

    """
    lat = remove_element_types(impact_object.lattice, ["drift"])

    # Get element bounds. Don't draw things outside the bounds.
    if not bounds:
        bounds = ele_bounds(lat)

    if include_field:
        add_fieldmaps_to_axes(
            impact_object,
            ax=axes,
            bounds=bounds,
            t=field_t,
            n_pts=1000,
            xfactor=factor,
        )
        ymin, ymax = axes.get_ylim()
        yfactor = (ymax - ymin) / 4
        alpha = 0.1
    else:
        yfactor = 1
        alpha = 0.7

    for ele in lat:
        if "s" not in ele:
            continue

        if "L" not in ele and not include_markers:
            continue

        if not ele_overlaps_s(ele, bounds[0], bounds[1]):
            continue

        if "L" not in ele:
            add_ele_marker(ele, axes, factor=factor)
        else:
            add_ele_box(ele, axes, xfactor=factor, yfactor=yfactor, alpha=alpha)

        if include_labels:
            add_ele_label(ele, axes, bounds=bounds, factor=factor)


def plot_layout(
    impact_object,
    xlim=None,
    include_labels=True,
    include_markers=True,
    include_field=False,
    field_t=0,
    return_figure=False,
    **kwargs,
):
    """
    Simple layout plot
    """

    fig, axes = plt.subplots(**kwargs)

    add_layout_to_axes(
        impact_object,
        axes,
        bounds=xlim,
        include_labels=include_labels,
        include_field=include_field,
        field_t=field_t,
        include_markers=include_markers,
    )

    if return_figure:
        return fig


def plot_stats_with_layout(
    impact_object,
    ykeys=["sigma_x", "sigma_y"],
    ykeys2=["mean_kinetic_energy"],
    xkey="mean_z",
    xlim=None,
    ylim=None,
    ylim2=None,
    nice=True,
    tex=True,
    include_layout=True,
    include_labels=True,
    include_field=False,
    field_t=0,
    include_markers=True,
    include_particles=True,
    include_legend=True,
    return_figure=False,
    ax=None,
    **kwargs,
):
    """
    Plots stat output multiple keys.

    If a list of ykeys2 is given, these will be put on the right hand axis. This can also be given as a single key.

    Logical switches:
        nice: a nice SI prefix and scaling will be used to make the numbers reasonably sized. Default: True

        tex: use mathtext (TeX) for plot labels. Default: True

        include_legend: The plot will include the legend.  Default: True

        include_layout: the layout plot will be displayed at the bottom.  Default: True

        include_labels: the layout will include element labels.  Default: True

        return_figure: return the figure object for further manipulation. Default: False

    """
    I = impact_object  # convenience

    if include_layout:
        fig, all_axis = plt.subplots(2, gridspec_kw={"height_ratios": [4, 1]}, **kwargs)
        ax_layout = all_axis[-1]
        ax_plot = [all_axis[0]]
    elif ax is not None:
        ax_plot = [ax]
        fig = ax.get_figure()
    else:
        fig, all_axis = plt.subplots(**kwargs)
        ax_plot = [all_axis]

    # collect axes
    if isinstance(ykeys, str):
        ykeys = [ykeys]

    if ykeys2:
        if isinstance(ykeys2, str):
            ykeys2 = [ykeys2]
        ax_twinx = ax_plot[0].twinx()
        ax_plot.append(ax_twinx)

    # No need for a legend if there is only one plot
    if len(ykeys) == 1 and not ykeys2:
        include_legend = False

    # assert xkey == 'mean_z', 'TODO: other x keys'

    X = I.stat(xkey)

    # Only get the data we need
    if xlim:
        good = np.logical_and(X >= xlim[0], X <= xlim[1])
        X = X[good]
    else:
        xlim = X.min(), X.max()
        good = slice(None, None, None)  # everything

    # Try particles within these bounds
    Pnames = []
    X_particles = []

    if include_particles:
        try:
            for pname in I.particles:
                xp = I.particles[pname][xkey]
                if xp >= xlim[0] and xp <= xlim[1]:
                    Pnames.append(pname)
                    X_particles.append(xp)
            X_particles = np.array(X_particles)
        except Exception:
            Pnames = []
    else:
        Pnames = []

    # X axis scaling
    units_x = str(I.units(xkey))
    if nice:
        X, factor_x, prefix_x = nice_array(X)
        units_x = prefix_x + units_x
    else:
        factor_x = 1

    # set all but the layout

    # Handle tex labels

    xlabel = mathlabel(xkey, units=units_x, tex=tex)

    for ax in ax_plot:
        ax.set_xlim(xlim[0] / factor_x, xlim[1] / factor_x)
        ax.set_xlabel(xlabel)

    # Draw for Y1 and Y2

    linestyles = ["solid", "dashed"]

    ii = -1  # counter for colors
    for ix, keys in enumerate([ykeys, ykeys2]):
        if not keys:
            continue
        ax = ax_plot[ix]
        linestyle = linestyles[ix]

        # Check that units are compatible
        ulist = [I.units(key) for key in keys]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f"Incompatible units: {ulist[0]} and {u2}"
        # String representation
        unit = str(ulist[0])

        # Data
        data = [I.stat(key)[good] for key in keys]

        if nice:
            factor, prefix = nice_scale_prefix(np.ptp(data))
            unit = prefix + unit
        else:
            factor = 1

        # Make a line and point
        for key, dat in zip(keys, data):
            #
            ii += 1
            color = "C" + str(ii)

            # Handle tex labels
            label = mathlabel(key, units=unit, tex=tex)
            ax.plot(X, dat / factor, label=label, color=color, linestyle=linestyle)

            # Particles
            if Pnames:
                try:
                    Y_particles = np.array([I.particles[name][key] for name in Pnames])
                    ax.scatter(
                        X_particles / factor_x, Y_particles / factor, color=color
                    )
                except Exception:
                    pass

        # Handle tex labels
        ylabel = mathlabel(*keys, units=unit, tex=tex)
        ax.set_ylabel(ylabel)

        # Set limits, considering the scaling.
        if ix == 0 and ylim:
            ymin = ylim[0]
            ymax = ylim[1]
            # Handle None and scaling
            if ymin is not None:
                ymin = ymin / factor
            if ymax is not None:
                ymax = ymax / factor
            new_ylim = (ymin, ymax)
            ax.set_ylim(new_ylim)
        # Set limits, considering the scaling.
        if ix == 1 and ylim2:
            pass
            # TODO
            if ylim2:
                ymin2 = ylim2[0]
                ymax2 = ylim2[1]
                # Handle None and scaling
                if ymin2 is not None:
                    ymin2 = ymin2 / factor
                if ymax2 is not None:
                    ymax2 = ymax2 / factor
                new_ylim2 = (ymin2, ymax2)
                ax_twinx.set_ylim(new_ylim2)
            else:
                pass

    # Collect legend
    if include_legend:
        lines = []
        labels = []
        for ax in ax_plot:
            a, b = ax.get_legend_handles_labels()
            lines += a
            labels += b
        ax_plot[0].legend(lines, labels, loc="best")

    # Layout
    if include_layout:
        # Gives some space to the top plot
        if not include_field:
            ax_layout.set_ylim(-1, 1.5)

        if xkey == "mean_z":
            if not include_field:
                ax_layout.set_axis_off()
            ax_layout.set_xlim(xlim[0], xlim[1])
        else:
            ax_layout.set_xlabel("mean_z")
            xlim = (0, I.stop)
        add_layout_to_axes(
            I,
            ax_layout,
            bounds=xlim,
            include_labels=include_labels,
            include_field=include_field,
            field_t=field_t,
            include_markers=include_markers,
        )

    if return_figure:
        return fig


def add_fieldmaps_to_axes(
    impact_object,
    *,
    ax=None,
    bounds=None,
    t=None,
    n_pts=1000,
    xfactor=1,
    add_legend=False,
):
    """
    Adds fieldmaps to an axes.


    t: float or None
        time to plot the field.
        If None and impact_object has output, the field at the bunch center will be plotted.

    """

    if bounds is None:
        zmin, zmax = 0, impact_object.stop
    else:
        zmin, zmax = bounds
    ax.set_xlim(zmin, zmax)

    # Guard for t=None but no output
    if t is None and not impact_object.output:
        t = 0  # No output, just plot t=0

    if t is None:
        # Must be output
        zlist = impact_object.stat("mean_z")
        tlist = impact_object.stat("t")
    else:
        tlist = np.full(n_pts, t)
        zlist = np.linspace(zmin, zmax, n_pts)

    # pre-filter
    eles = [ele for ele in impact_object.lattice if ele["type"] in FIELD_CALC_ELE_TYPES]

    ax2 = ax.twinx()

    lines = []
    for ax1, component, color, label, units in (
        (ax, "Ez", "green", r"$E_z$", "V/m"),
        (ax2, "Bz", "blue", r"$B_z$", "T"),
    ):
        fz = np.array(
            [
                lattice_field(
                    eles,
                    z=z,
                    x=0,
                    y=0,
                    t=t,
                    component=component,
                    fmaps=impact_object.fieldmaps,
                )
                for z, t in zip(zlist, tlist)
            ]
        )

        y, factor, prefix = nice_array(fz)

        line = ax1.plot(zlist / xfactor, y, color=color, label=label)
        lines += line

        ylabel = f"{label} ({prefix}{units})"
        ax1.set_ylabel(ylabel)

    labels = [line.get_label() for line in lines]
    if add_legend:
        ax.legend(lines, labels)

    # ax.set_xlabel('$z$ (m)')
