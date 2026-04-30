PLOT_FONT_FAMILY = "Times New Roman"
PLOT_FONT_SIZE = 25
PLOT_TITLE_SIZE = 16
PLOT_LABEL_SIZE = 25
PLOT_TICK_SIZE = 25
PLOT_LEGEND_SIZE = 25

PLOT_STYLE = {
    "font.family": PLOT_FONT_FAMILY,
    "font.size": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_TITLE_SIZE,
    "axes.labelsize": PLOT_LABEL_SIZE,
    "xtick.labelsize": PLOT_TICK_SIZE,
    "ytick.labelsize": PLOT_TICK_SIZE,
    "legend.fontsize": PLOT_LEGEND_SIZE,
    "figure.titlesize": PLOT_TITLE_SIZE,
}


def apply_plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(PLOT_STYLE)
