from __future__ import annotations

import typing
from collections.abc import Sequence
from typing import Protocol, Union, cast

import matplotlib.axes
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pydantic.dataclasses as dataclasses
from impact.z.constants import MultipoleType
from impact.z.input import AnyInputElement, Multipole, ZElement
from pmd_beamphysics.units import nice_array, nice_scale_prefix
from pydantic import ConfigDict, Field
from typing_extensions import Literal

from ..plot import mathlabel


class ElementToShapeFunction(Protocol):
    """
    A protocol which describes a function that maps an input element to a shape.

    A default implementation is provided in `default_shape_mapper`, but the
    user may override or extend it by following the calling convention defined
    here.
    """

    def __call__(
        self,
        ele: AnyInputElement,
        *,
        s1: float,
        s2: float,
        y1: float,
        y2: float,
        name: str,
    ) -> LayoutShape | None: ...


Point = tuple[float, float]


if typing.TYPE_CHECKING:
    from .input import ImpactZInput
    from .output import ImpactZOutput


def plot_layout(
    by_z: list[ZElement],
    ax: matplotlib.axes.Axes | None = None,
    bounds: tuple[float, float] | None = None,
    include_labels: bool = True,
    include_markers: bool = True,
    include_marker_labels: bool | None = None,
    figsize: tuple[int, int] = (6, 2),
    line_width_scale: float = 1.0,
    shape_mapper: ElementToShapeFunction | None = None,
):
    """
    Make a matplotlib plot of the lattice layout.

    Parameters
    ----------
    by_z : list[ZElement]
        List of element instances coupled with their s (Z) positions.
    ax : matplotlib.axes.Axes, optional
        Axes object on which to draw the layout. If None, a new figure and axes
        are created.
    bounds : (float, float), optional
        Lower and upper bounds for z position. Defaults to None.
    include_labels : bool, optional
        Whether to include element labels in the plot. Defaults to True.
    include_markers : bool, optional
        If True, include zero length markers in the plot. Default is True.
    include_marker_labels : bool, optional
        If True, include labels for markers when `include_markers` is set.
        Default is `include_labels`.
    figsize : tuple of int, optional
        Size of the figure in inches (width, height) when a new figure is created.
        Defaults to (6, 2).

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plotted layout.
    """
    if include_marker_labels is None:
        include_marker_labels = include_labels
    if shape_mapper is None:
        shape_mapper = default_shape_mapper

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    if not by_z:
        return fig, ax

    ax.axhline(y=0, color="Black", linewidth=1)
    ax.yaxis.set_visible(False)

    if bounds is None:
        bounds = by_z[0].z_start, by_z[-1].z_end

    y1 = -1.0
    y2 = 1.0
    for zele in by_z:
        ele = zele.ele
        shape = shape_mapper(
            ele,
            s1=zele.z_start,
            s2=zele.z_end,
            y1=y1,
            y2=y2,
            name=ele.name,
        )
        if shape is not None:
            if np.isclose(ele.length, 0.0):
                if include_markers:
                    plot_marker(
                        ax=ax,
                        s_center=zele.z_start,
                        y1=y1,
                        y2=y2,
                        color=shape.color,
                        line_width_scale=line_width_scale,
                        label=ele.name if include_marker_labels else None,
                    )
                continue

            plot_shape(
                ax=ax,
                shape=shape,
                line_width_scale=line_width_scale,
                label=ele.name if include_labels else None,
            )

    ax.set_xlim(bounds)
    ax.set_ylim(-2, 2)
    return fig, ax


def plot_marker(
    ax: matplotlib.axes.Axes,
    s_center: float,
    y1: float,
    y2: float,
    color: str = "black",
    line_width_scale: float = 1.0,
    label: str | None = None,
):
    ax.vlines(
        s_center,
        y1,
        y2,
        color=color,
        linestyles="dashed",
        linewidth=1.0 * line_width_scale,
    )
    if label:
        ax.annotate(
            xy=(s_center, y1 - 0.1),
            text=label,
            horizontalalignment="center",
            verticalalignment="top",
            clip_on=False,
            color=color,
            rotation=90,
            rotation_mode="default",
            fontsize=8,
            family="sans-serif",
        )


def plot_shape(
    ax: matplotlib.axes.Axes,
    shape: LayoutShape,
    line_width_scale: float = 1.0,
    label: str | None = None,
):
    for curve in shape.to_lines():
        ax.plot(
            curve.xs,
            curve.ys,
            color=shape.color,
            linestyle=curve.linestyle,
            linewidth=curve.linewidth * line_width_scale,
            label=shape.name,
        )
    for patch in shape.to_patches():
        mpl = patch_to_mpl(patch, line_width_scale=line_width_scale)
        ax.add_patch(mpl)

    if label:
        s_center = (shape.s1 + shape.s2) / 2.0
        ax.annotate(
            xy=(s_center, -1.1),
            text=label,
            horizontalalignment="center",
            verticalalignment="top",
            clip_on=False,
            color=shape.color,
            rotation=90,
            rotation_mode="default",
            fontsize=8,
            family="sans-serif",
        )


def plot_stats_with_layout(
    output: ImpactZOutput,
    ykeys: Sequence[str] = ("sigma_x", "sigma_y"),
    ykeys2: Sequence[str] = ("mean_kinetic_energy",),
    xkey: str = "z",
    *,
    input: ImpactZInput | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ylim2: tuple[float, float] | None = None,
    nice: bool = True,
    tex: bool = True,
    include_layout: bool = True,
    include_labels: bool = True,
    include_markers: bool = True,
    include_marker_labels: bool | None = None,
    include_particles: bool = True,
    include_legend: bool = True,
    return_figure: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> object | None:
    """
    Plots stat output multiple keys.

    Parameters
    ----------
    output : object
        The impact object containing the data to plot.
    ykeys : list of str, optional
        Keys to plot on the left y-axis. Default is ("sigma_x", "sigma_y").
    ykeys2 : list of str or str, optional
        Keys to plot on the right y-axis. Default is ("mean_kinetic_energy",).
    xkey : str, optional
        Key to plot on the x-axis. Default is "z".
    input : ImpactZInput or None, optional
        Input object. Required if `include_layout` is True.
    xlim : tuple, optional
        Limits for the x-axis. Default is None.
    ylim : tuple, optional
        Limits for the left y-axis. Default is None.
    ylim2 : tuple, optional
        Limits for the right y-axis. Default is None.
    nice : bool, optional
        If True, a nice SI prefix and scaling will be used to make the numbers reasonably sized. Default is True.
    tex : bool, optional
        If True, use mathtext (TeX) for plot labels. Default is True.
    include_layout : bool, optional
        If True, the layout plot will be displayed at the bottom. Default is True.
    include_labels : bool, optional
        If True, the layout will include element labels. Default is True.
    include_markers : bool, optional
        If True, include zero length markers in the plot. Default is True.
    include_marker_labels : bool, optional
        If True, include labels for markers when `include_markers` is set.
        Default is `include_labels`.
    include_particles : bool, optional
        If True, include particles in the plot. Default is True.
    include_legend : bool, optional
        If True, the plot will include the legend. Default is True.
    return_figure : bool, optional
        If True, return the figure object for further manipulation. Default is False.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto. Default is None.
    **kwargs : dict
        Additional keyword arguments for the plot.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if `return_figure` is True, otherwise None.
    """

    if include_marker_labels is None:
        include_marker_labels = include_labels
    if include_layout:
        fig, all_axis = plt.subplots(2, gridspec_kw={"height_ratios": [4, 1]}, **kwargs)
        ax_plot = [all_axis[0]]
        ax_layout = all_axis[-1]
    elif ax is not None:
        ax_plot = [ax]
        ax_layout = None
        fig = ax.get_figure()
    else:
        fig, all_axis = plt.subplots(**kwargs)
        ax_plot = [all_axis]
        ax_layout = None

    ax_plot = cast(list[matplotlib.axes.Axes], ax_plot)

    # collect axes
    ykeys = [ykeys] if isinstance(ykeys, str) else list(ykeys)

    if ykeys2:
        ykeys2 = [ykeys2] if isinstance(ykeys2, str) else list(ykeys2)
        ax_twinx = ax_plot[0].twinx()
        ax_plot.append(ax_twinx)
    else:
        ax_twinx = None

    # No need for a legend if there is only one plot
    if len(ykeys) == 1 and not ykeys2:
        include_legend = False

    X = output.stat(xkey)

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
            for pname, particles in output.particles.items():
                xp = particles[xkey]
                if xp >= xlim[0] and xp <= xlim[1]:
                    Pnames.append(pname)
                    X_particles.append(xp)
            X_particles = np.array(X_particles)
        except Exception:
            Pnames = []
    else:
        Pnames = []

    # X axis scaling
    units_x = str(output.units(xkey))
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
        ulist = [output.units(key) for key in keys]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f"Incompatible units: {ulist[0]} and {u2}"
        # String representation
        unit = str(ulist[0])

        # Data
        data = [output.stat(key)[good] for key in keys]

        if nice:
            factor, prefix = nice_scale_prefix(np.ptp(data))
            unit = prefix + unit
        else:
            factor = 1

        # Make a line and point
        for key, dat in zip(keys, data):
            ii += 1
            color = f"C{ii}"

            # Handle tex labels
            label = mathlabel(key, units=unit, tex=tex)
            ax.plot(X, dat / factor, label=label, color=color, linestyle=linestyle)

            # Particles
            if Pnames:
                try:
                    Y_particles = np.array(
                        [output.particles[name][key] for name in Pnames]
                    )
                    ax.scatter(
                        X_particles / factor_x,
                        Y_particles / factor,
                        color=color,
                    )
                except Exception:
                    pass

        # Handle tex labels
        ylabel = mathlabel(*keys, units=unit, tex=tex)
        ax.set_ylabel(ylabel)

        # Set limits, considering the scaling.
        if ix == 0 and ylim:
            ymin, ymax = ylim
            # Handle None and scaling
            if ymin is not None:
                ymin = ymin / factor
            if ymax is not None:
                ymax = ymax / factor
            new_ylim = (ymin, ymax)
            ax.set_ylim(new_ylim)
        # Set limits, considering the scaling.
        if ix == 1 and ylim2:
            ymin2, ymax2 = ylim2
            # Handle None and scaling
            if ymin2 is not None:
                ymin2 = ymin2 / factor
            if ymax2 is not None:
                ymax2 = ymax2 / factor
            new_ylim2 = (ymin2, ymax2)

            assert ax_twinx is not None
            ax_twinx.set_ylim(new_ylim2)

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
        if input is None:
            raise ValueError(
                "ImpactZInput object is required to generate a layout plot."
            )

        # Gives some space to the top plot
        assert ax_layout is not None
        ax_layout.set_ylim(-1, 1.5)

        if xkey in ("mean_z", "z"):
            ax_layout.set_axis_off()
            ax_layout.set_xlim(xlim[0], xlim[1])
        else:
            ax_layout.set_xlabel("mean_z")
            xlim = (0, output.stats.z[-1])

        input.plot(
            ax=ax_layout,
            bounds=xlim,
            include_labels=include_labels,
            include_markers=include_markers,
            include_marker_labels=include_marker_labels,
        )

    if return_figure:
        return fig


_dcls_config = ConfigDict()
_point_field = Field(default_factory=lambda: (0.0, 0.0))


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchBase:
    edgecolor: str | None = None
    facecolor: str | None = None
    color: str | None = None
    linewidth: float | None = None
    linestyle: str | None = None
    antialiased: bool | None = None
    hatch: str | None = None
    fill: bool = True
    capstyle: str | None = None
    joinstyle: str | None = None
    alpha: float = 1.0

    @property
    def _patch_args(self):
        return {
            "edgecolor": self.edgecolor,
            "facecolor": self.facecolor,
            "color": self.color or "black",
            "linewidth": self.linewidth,
            "linestyle": self.linestyle,
            "antialiased": self.antialiased,
            "hatch": self.hatch,
            "fill": self.fill,
            "capstyle": self.capstyle,
            "joinstyle": self.joinstyle,
            "alpha": self.alpha,
        }


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchRectangle(PlotPatchBase):
    xy: Point = _point_field
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0
    rotation_point: Literal["xy", "center"] | Point = "xy"

    @property
    def center(self) -> Point:
        return (
            self.xy[0] + self.width / 2,
            self.xy[1] + self.height / 2,
        )


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchCircle(PlotPatchBase):
    xy: Point = _point_field
    radius: float = 0.0


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchPolygon(PlotPatchBase):
    vertices: list[Point] = Field(default_factory=list)


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchEllipse(PlotPatchBase):
    xy: Point = _point_field
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0


PlotPatch = Union[
    PlotPatchRectangle,
    PlotPatchCircle,
    PlotPatchEllipse,
    PlotPatchPolygon,
]


@dataclasses.dataclass(config=_dcls_config)
class PlotCurveLine:
    xs: list[float]
    ys: list[float]
    color: str = "black"
    linestyle: str = "solid"
    linewidth: float = 1.0


@dataclasses.dataclass(config=_dcls_config)
class LayoutShape:
    s1: float = 0.0
    s2: float = 0.0
    y1: float = 0.0
    y2: float = 0.0
    name: str = ""
    color: str = "black"
    line_width: float = 1.0
    fill: bool = False

    @property
    def corner_vertices(self):
        return [
            [self.s1, self.s1, self.s2, self.s2],
            [self.y1, self.y2, self.y2, self.y1],
        ]

    @property
    def dimensions(self):
        return (
            self.s2 - self.s1,
            self.y2 - self.y1,
        )

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.s1 + self.s2) / 2,
            (self.y1 + self.y2) / 2,
        )

    @property
    def lines(self):
        return []

    def to_lines(self) -> list[PlotCurveLine]:
        lines = self.lines
        if not lines:
            return []
        return [
            PlotCurveLine(
                [x for x, _ in line],
                [y for _, y in line],
                linewidth=self.line_width,
                color=self.color,
            )
            for line in self.lines
        ]

    def to_patches(self) -> list[PlotPatch]:
        return []

    @property
    def patch_kwargs(self):
        return {
            "linewidth": self.line_width,
            "color": self.color,
            "fill": self.fill,
        }


def patch_to_mpl(patch: PlotPatch, line_width_scale: float = 1.0):
    patch_args = patch._patch_args
    if patch_args["linewidth"] is not None:
        patch_args["linewidth"] *= line_width_scale

    if isinstance(patch, PlotPatchRectangle):
        return matplotlib.patches.Rectangle(
            xy=patch.xy,
            width=patch.width,
            height=patch.height,
            angle=patch.angle,
            rotation_point=patch.rotation_point,
            **patch_args,
        )
    if isinstance(patch, PlotPatchCircle):
        return matplotlib.patches.Circle(
            xy=patch.xy,
            radius=patch.radius,
            **patch_args,
        )
    if isinstance(patch, PlotPatchPolygon):
        return matplotlib.patches.Polygon(
            xy=patch.vertices,
            **patch_args,
        )

    if isinstance(patch, PlotPatchEllipse):
        return matplotlib.patches.Ellipse(
            xy=patch.xy,
            width=patch.width,
            height=patch.height,
            angle=patch.angle,
            **patch_args,
        )

    raise NotImplementedError(f"Unsupported patch type: {type(patch).__name__}")


@dataclasses.dataclass(config=_dcls_config)
class LayoutBox(LayoutShape):
    def to_patches(self) -> list[PlotPatch]:
        width, height = self.dimensions
        return [
            PlotPatchRectangle(
                xy=(self.s1, self.y1),
                width=width,
                height=height,
                **self.patch_kwargs,
            )
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutXBox(LayoutShape):
    @property
    def lines(self):
        return [
            [(self.s1, self.y1), (self.s2, self.y2)],
            [(self.s1, self.y2), (self.s2, self.y1)],
        ]

    def to_patches(self) -> list[PlotPatch]:
        width, height = self.dimensions
        return [
            PlotPatchRectangle(
                xy=(self.s1, self.y1),
                width=width,
                height=height,
                **self.patch_kwargs,
            )
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutLetterX(LayoutShape):
    @property
    def lines(self):
        return [
            [(self.s1, self.y1), (self.s2, self.y2)],
            [(self.s1, self.y2), (self.s2, self.y1)],
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutBowTie(LayoutShape):
    @property
    def lines(self):
        return [
            [
                (self.s1, self.y1),
                (self.s2, self.y2),
                (self.s2, self.y1),
                (self.s1, self.y2),
                (self.s1, self.y1),
            ]
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutRBowTie(LayoutShape):
    @property
    def lines(self):
        return [
            [
                (self.s1, self.y1),
                (self.s2, self.y2),
                (self.s1, self.y2),
                (self.s2, self.y1),
                (self.s1, self.y1),
            ]
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutDiamond(LayoutShape):
    @property
    def lines(self):
        s_mid, _ = self.center
        return [
            [
                (self.s1, 0),
                (s_mid, self.y1),
                (self.s2, 0),
                (s_mid, self.y2),
                (self.s1, 0),
            ]
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutCircle(LayoutShape):
    def to_patches(self) -> list[PlotPatch]:
        s_mid, _ = self.center
        width, height = self.dimensions
        return [
            PlotPatchEllipse(
                xy=(s_mid, 0),
                width=width,
                height=height,
                **self.patch_kwargs,
            )
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutTriangle(LayoutShape):
    orientation: Literal["u", "d", "l", "r"] = "u"

    @property
    def vertices(self):
        s_mid, y_mid = self.center
        if self.orientation == "u":
            return [(self.s1, self.y2), (self.s2, self.y2), (s_mid, self.y1)]
        if self.orientation == "d":
            return [(self.s1, self.y1), (self.s2, self.y1), (s_mid, self.y2)]
        if self.orientation == "l":
            return [(self.s1, y_mid), (self.s2, self.y2), (self.s2, self.y1)]
        if self.orientation == "r":
            return [(self.s1, self.y1), (self.s1, self.y2), (self.s2, y_mid)]
        raise ValueError(f"Unsupported orientation: {self.orientation}")

    def to_patches(self) -> list[PlotPatch]:
        return [PlotPatchPolygon(vertices=self.vertices, **self.patch_kwargs)]


def default_shape_mapper(
    ele: AnyInputElement,
    s1: float,
    s2: float,
    y1: float,
    y2: float,
    name: str,
):
    from . import input as IZ

    cls_to_shape = {
        # IZ.Drift: (None, "black"),
        IZ.Quadrupole: (LayoutXBox, "blue"),
        IZ.ConstantFocusing: (LayoutBox, "black"),
        IZ.Solenoid: (LayoutXBox, "blue"),
        IZ.Dipole: (LayoutBox, "red"),
        # IZ.Multipole: (LayoutXBox, "black"),
        IZ.DTL: (LayoutXBox, "black"),
        IZ.CCDTL: (LayoutXBox, "black"),
        IZ.CCL: (LayoutXBox, "green"),  # standard tracking, standing/traveling wave
        IZ.SolenoidWithRFCavity: (LayoutBox, "green"),  # RK tracking, standing wave
        IZ.SuperconductingCavity: (LayoutXBox, "green"),
        IZ.TravelingWaveRFCavity: (LayoutBox, "green"),
        IZ.UserDefinedRFCavity: (LayoutXBox, "green"),
        # Control inputs
        IZ.ShiftCentroid: (LayoutDiamond, "blue"),
        IZ.WriteFull: (LayoutTriangle, "red"),
        IZ.DensityProfileInput: (LayoutDiamond, "green"),
        IZ.DensityProfile: (LayoutDiamond, "purple"),
        IZ.Projection2D: (LayoutDiamond, "brown"),
        IZ.Density3D: (LayoutDiamond, "blue"),
        IZ.WritePhaseSpaceInfo: (LayoutBowTie, "red"),
        IZ.WriteSliceInfo: (LayoutRBowTie, "green"),
        IZ.ScaleMismatchParticle6DCoordinates: (LayoutDiamond, "purple"),
        IZ.CollimateBeam: (LayoutDiamond, "brown"),
        IZ.ToggleSpaceCharge: (LayoutCircle, "blue"),
        IZ.RotateBeam: (LayoutDiamond, "red"),
        IZ.BeamShift: (LayoutLetterX, "green"),
        IZ.BeamEnergySpread: (LayoutDiamond, "purple"),
        IZ.ShiftBeamCentroid: (LayoutCircle, "brown"),
        IZ.IntegratorTypeSwitch: (LayoutDiamond, "blue"),
        IZ.BeamKickerByRFNonlinearity: (LayoutDiamond, "red"),
        IZ.RfcavityStructureWakefield: (LayoutDiamond, "green"),
        IZ.EnergyModulation: (LayoutDiamond, "purple"),
        IZ.KickBeamUsingMultipole: (LayoutDiamond, "brown"),
        IZ.HaltExecution: (LayoutDiamond, "blue"),
    }

    if isinstance(ele, Multipole):
        shape_cls = LayoutBox
        color = {
            MultipoleType.sextupole: "magenta",
            MultipoleType.octupole: "green",
            MultipoleType.decapole: "orange",
        }[ele.multipole_type]
    else:
        shape_cls, color = cls_to_shape.get(type(ele), (None, ""))
        if shape_cls is None:
            return None

    return shape_cls(
        color=color,
        s1=s1,
        s2=s2,
        y1=y1,
        y2=y2,
        name=name,
    )
