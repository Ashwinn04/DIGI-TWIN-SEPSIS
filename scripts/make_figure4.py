"""Figure 7 of Paper 514: data flow of the multi-site surveillance layer.

    python scripts/make_figure4.py            # -> outputs/figures/figure4_dataflow.png
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 9})
INK, QUIET, EDGE, ACCENT = "#111111", "#555555", "#444444", "#1f5fa8"
ACCENT_FILL = "#e3ecf7"

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 49); ax.axis("off")
T, D = 8, 6.8   # title and detail sizes (pt)


def box(x, y, w, h, fill="white", edge=EDGE, lw=0.9, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fill, ec=edge, lw=lw, ls=ls))


def text(x, y, s, size=D, weight="normal", color=INK, ha="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va="center",
            fontstyle=style, linespacing=1.15)


def arrow(p0, p1, color=EDGE):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color,
                                 shrinkA=0, shrinkB=1))


# hospitals: container with Digital Twin -> privacy filter
HX, HW, HH = 1, 40, 12.5
rows = [(34, "Hospital 1"), (19.5, "Hospital 2"), (2, "Hospital 12")]
for y, name in rows:
    box(HX, y, HW, HH, edge=QUIET, lw=0.7, ls=(0, (3, 2)))
    text(HX + 1.5, y + HH - 1.8, name, size=T, weight="bold", ha="left")
    box(HX + 1.5, y + 1.0, 16, 7.6)
    text(HX + 9.5, y + 6.7, "Digital Twin", size=T, weight="bold")
    text(HX + 9.5, y + 3.4, "hourly risk scores,\nviral test results", color=QUIET)
    box(HX + 21.5, y + 1.0, 17, 7.6, fill=ACCENT_FILL, edge=ACCENT, lw=1.2)
    text(HX + 30, y + 6.7, "Privacy filter", size=T, weight="bold")
    text(HX + 30, y + 3.4, "suppress counts < k,\nadd Laplace noise (\u03b5)")
    arrow((HX + 17.5, y + 4.8), (HX + 21.5, y + 4.8))
text(HX + HW / 2, 17.0, "\u2026 hospitals 3\u201311 \u2026", color=QUIET, style="italic")
text(HX + HW / 2, 47.9, "Inside each hospital (12 sites in 4 regions)", color=QUIET, style="italic")

AY, AH = 16, 16
AX_, AW = 48, 15
box(AX_, AY, AW, AH)
text(AX_ + AW / 2, AY + 12.0, "Central\naggregator", size=T, weight="bold")
text(AX_ + AW / 2, AY + 5.0, "receives daily\ncounts only;\nsums by region\nand day", color=QUIET)
for y, _ in rows:
    arrow((HX + HW, y + 4.8), (AX_, AY + AH / 2))

DX, DW = 67, 15
box(DX, AY, DW, AH)
text(DX + DW / 2, AY + 12.6, "Detectors", size=T, weight="bold")
text(DX + DW / 2, AY + 6.0, "z-score vs\n60-day baseline;\nEWMA, CUSUM", color=QUIET)
arrow((AX_ + AW, AY + AH / 2), (DX, AY + AH / 2))

LX, LW = 86, 13
box(LX, AY, LW, AH)
text(LX + LW / 2, AY + 12.6, "Alerts", size=T, weight="bold")
text(LX + LW / 2, AY + 6.0, "region and day\nof each alarm;\nheatmap", color=QUIET)
arrow((DX + DW, AY + AH / 2), (LX, AY + AH / 2))

text(73.5, 7.5, "Released per site per day: admissions, high-risk alerts, viral positives.\n"
                "No patient record leaves a hospital.", color=QUIET)

out = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "outputs" / "figures" / "figure4_dataflow.png")
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
print("saved", out)
