from bokeh.plotting import figure
from bokeh.models.sources import ColumnDataSource
from bokeh.models import LabelSet, HoverTool
from math import pi

from .lattice import ele_shapes

# For Jupyter
# from bokeh.plotting import output_notebook
# from bokeh.plotting import show
# output_notebook()


def layout_plot(eles, width=1000, height=200):
    """
    Returns a bokeh plot

    Use by:
    from bokeh.plotting import show
    p = layout_plot(eles)
    show(p)

    """
    shapes = ele_shapes(eles)
    ds = ColumnDataSource(shapes)

    TOOLTIPS = [
        ("name", "@name"),
        ("s_begin", "@left"),
        ("s_end", "@right"),
        ("s_center", "@x"),
        ("", "@description"),
        ("", "@all"),
    ]

    # Hover tool behaviour
    hover = HoverTool(
        tooltips=TOOLTIPS,
        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode="vline",
    )

    p = figure(
        width=width,
        height=height,
        tools=[hover, "pan", "wheel_zoom", "box_zoom", "reset"],
    )
    p.quad(
        top="top", bottom="bottom", left="left", right="right", color="color", source=ds
    )

    labels = LabelSet(
        x="x",
        y="y",
        text="name",
        level="glyph",
        angle=pi / 2,
        x_offset=5,
        y_offset=30,
        source=ds,
    )
    p.add_layout(labels)

    return p
