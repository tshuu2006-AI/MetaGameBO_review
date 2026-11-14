"""Shared plotting style: palette, fixed method colours and figure helpers.

Colours follow the method, never its position in a plot: every method has one
fixed hue (``METHOD_COLORS``). The MetaGameBO remake
shares the hue of the original and is drawn dashed (``method_style``).

Attributes:
    INK, INK_2, MUTED, GRID, AXIS, SURFACE: Chart chrome colours (light mode).
    CATEGORICAL: The 8 categorical slots, in their fixed (CVD-validated) order.
    SEQUENTIAL: Single-hue blue ramp, light -> dark, for magnitudes / time.
    DIVERGING: Blue <-> red colormap with a neutral grey midpoint.
    METHOD_ORDER: Display order of methods in legends and tables.
    METHOD_COLORS: Fixed colour of each method.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b22d2d"])

METHOD_ORDER = ["ABLR", "Random", "GP", "LFBO", "RGPE", "PFN", "MALIBO",
                "MetaGameBO (original)", "MetaGameBO (remake)"]
# hues of the paper's Figure 3 (MetaBO dropped); the remake shares MetaGameBO's hue, dashed
METHOD_COLORS = {
    "ABLR": "#f28e2b",
    "Random": "#8c8c8c",
    "GP": "#7f3f1f",
    "LFBO": "#17becf",
    "RGPE": "#2ca02c",
    "PFN": "#9467bd",
    "MALIBO": "#1f77ff",
    "MetaGameBO (original)": "#d62728",
    "MetaGameBO (remake)": "#7a0010",
}

LINE_W = 1.6
MARKER = 5


def apply_style() -> None:
    """Set matplotlib rcParams: recessive grid and axes, ink-coloured text."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 9, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "legend.labelcolor": INK_2,
        "lines.linewidth": LINE_W, "font.family": "sans-serif",
        "figure.dpi": 110, "savefig.dpi": 140,
    })


apply_style()


def method_color(name: str) -> str:
    """Fixed colour of a method (secondary ink for unknown names)."""
    return METHOD_COLORS.get(name, INK_2)


def method_style(name: str) -> dict:
    """Line kwargs for a method: fixed colour; dashed for upstream code."""
    remake = name.endswith("(remake)")
    return dict(color=method_color(name), ls="--" if remake else "-", lw=LINE_W)


def ordered(names) -> list:
    """Sort method names by ``METHOD_ORDER`` (unknown names last, alphabetical)."""
    names = list(names)
    known = [m for m in METHOD_ORDER if m in names]
    return known + sorted(n for n in names if n not in METHOD_ORDER)


def subplots(ncols: int = 1, nrows: int = 1, width: float = 4.2, height: float = 3.2, **kw):
    """``plt.subplots`` with a per-panel size; always returns a flat axes list."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows),
                             squeeze=False, **kw)
    return fig, list(axes.ravel())


def save(fig, path: Optional[str]) -> None:
    """Save ``fig`` to ``path`` (creating folders) and close it; no-op if ``path`` is None."""
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def label_line_end(ax, x, y, text: str, color: str = INK_2) -> None:
    """Direct label at the end of a line (text in ink, not series colour)."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=color)
