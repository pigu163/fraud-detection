# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# Shared style utilities for all EDA figures; figure-specific drawing lives in separate modules.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline -- COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text


MM_TO_INCH = 1 / 25.4

# Report override: design at the final 89/183 mm width, with readable text.
# Keep the skill baseline above intact; this report needs larger typography.
mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
})


def save_pdf_figure(fig, filename):
    """Save vector PDF with a consistent text floor, including annotations.

    The figure itself must already be sized for its single/double-column slot.
    Explicit small annotation sizes in older helpers are raised here too.
    PDF is the only persistent image format requested by the user.
    """
    for artist in fig.findobj(match=Text):
        if artist.get_text():
            artist.set_fontsize(max(9, artist.get_fontsize()))
    fig.tight_layout(pad=1.2)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return filename.with_suffix(".pdf")
