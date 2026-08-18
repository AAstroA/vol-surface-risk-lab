"""Shared thesis-figure style."""

from __future__ import annotations

import matplotlib as mpl
import seaborn as sns


COLORS = {
    "navy": "#17324D",
    "blue": "#2F6B9A",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#D1495B",
    "gray": "#6B7280",
    "light": "#E8EEF3",
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#B8C2CC",
            "axes.linewidth": 0.8,
            "grid.color": "#DDE4EA",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
