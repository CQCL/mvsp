import colorsys

import matplotlib.colors as mc
from matplotlib.ticker import ScalarFormatter


def adjust_lightness(color, amount=0.5):
    # From https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    try:
        c = mc.cnames[color]
    except:  # noqa: E722
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_colorbar(plot, fig, ax):
    cb_formatter = ScalarFormatter(useMathText=True)
    cb_formatter.set_scientific("%.2e")
    cb_formatter.set_powerlimits((-2, 2))
    cb = fig.colorbar(
        plot, ax=ax, format=cb_formatter, location="right", shrink=0.46, pad=0.03
    )
    cb.ax.yaxis.set_offset_position("left")
    cb.ax.tick_params(labelsize="small")
    cb.ax.yaxis.get_offset_text().set(size="small")
