"""
Fundamental noise bounds for stochastic reaction networks.
Lestas, Vinnicombe & Paulsson (Nature, 2010).

Figure 1: log-log plot of noise (η² = CV²) vs signal rate S.
  - No feedback (Poisson):     η² = const
  - Engineering feedback:       η² ∝ 1/S
  - Biochemical feedback:       η² ∝ 1/√S   (η ∝ 1/S^{1/4})

Figure 2: intuition schematic (why biochemistry pays more).
"""
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 17, "axes.titlesize": 17,
    "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"
PURPLE="#7E2F8E"; RED="#A2142F"

# ══════════════════════════════════════════════════════════════
# FIGURE 1: noise vs signal rate (log-log)
# ══════════════════════════════════════════════════════════════
S = np.logspace(0, 5, 300)

# Normalise: all meet at S=1, η²=1
eta2_poisson = np.ones_like(S)            # constant (no feedback benefit)
eta2_eng     = 1.0 / S                    # slope -1
eta2_bio     = 1.0 / np.sqrt(S)           # slope -1/2

fig, ax = plt.subplots(figsize=(9, 6.5))
fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.12)

# Shaded regions
ax.fill_between(S, eta2_bio, eta2_eng, alpha=0.12, color=ORANGE,
                label=None)
ax.fill_between(S, eta2_poisson, eta2_bio, alpha=0.08, color=BLUE,
                label=None)

# Lines
ax.loglog(S, eta2_poisson, color="0.5", ls=":", lw=2.0,
          label=r"No feedback (Poisson):  $\eta^2 = \mathrm{const}$")
ax.loglog(S, eta2_bio, color=ORANGE, lw=3.0,
          label=r"Biochemical limit:  $\eta^2 \propto 1/\!\sqrt{S}$"
                r"$\;\;(\eta \propto 1/S^{1/4})$")
ax.loglog(S, eta2_eng, color=BLUE, lw=3.0,
          label=r"Engineering limit:  $\eta^2 \propto 1/S$"
                r"$\;\;(\eta \propto 1/\!\sqrt{S}\,)$")

# Annotations
ax.annotate("Achievable\n(biochemistry)",
            xy=(300, 1/300**0.5 * 1.8), fontsize=11, color=ORANGE,
            ha="center", fontstyle="italic")
ax.annotate("Gap: cost of\nmolecular feedback",
            xy=(3e3, 1/3e3**0.35), fontsize=11, color=RED,
            ha="center", fontweight="bold",
            bbox=dict(fc="white", ec=RED, alpha=0.9, pad=4,
                      boxstyle="round,pad=0.4"))
ax.annotate("",
            xy=(3e3, 1/np.sqrt(3e3)),
            xytext=(3e3, 1/3e3),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=2.0))

ax.set_xlabel(r"Signal rate  $S$  (reactions per unit time)")
ax.set_ylabel(r"Noise  $\eta^2 = \mathrm{Var}(X)/\langle X \rangle^2$")
ax.set_title("Fundamental noise limits: engineering vs. biochemistry",
             fontsize=15, fontweight="bold")
ax.set_xlim(1, 1e5)
ax.set_ylim(5e-6, 3)
ax.legend(loc="lower left", fontsize=11, framealpha=0.95, edgecolor="0.75")
ax.grid(True, which="both", alpha=0.12)

# Slope annotations
ax.text(2e4, 1.3/2e4, r"slope $-1$", fontsize=10, color=BLUE,
        rotation=-32, ha="center")
ax.text(2e4, 2.5/np.sqrt(2e4), r"slope $-\frac{1}{2}$", fontsize=10,
        color=ORANGE, rotation=-16, ha="center")

fig.savefig("noise_bound_loglog.svg", bbox_inches="tight")
fig.savefig("noise_bound_loglog.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved noise_bound_loglog.svg")

# ══════════════════════════════════════════════════════════════
# FIGURE 2: intuition schematic
# ══════════════════════════════════════════════════════════════
fig, (ax_eng, ax_bio) = plt.subplots(1, 2, figsize=(12, 5.5))
fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02,
                    wspace=0.15)
fig.suptitle("Why biochemical feedback is noisier",
             fontsize=15, fontweight="bold")

for ax in (ax_eng, ax_bio):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

def draw_box(ax, xy, w, h, label, color, fontsize=12):
    rect = plt.Rectangle(xy, w, h, linewidth=2, edgecolor=color,
                         facecolor=color+"20", clip_on=False, zorder=2)
    ax.add_patch(rect)
    ax.text(xy[0]+w/2, xy[1]+h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=3)

def draw_arrow(ax, start, end, color="0.3", lw=2, style="->"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# ── Engineering panel ─────────────────────────────────────────
ax_eng.set_title("Engineering feedback", fontsize=14, pad=10)

draw_box(ax_eng, (3.5, 6.5), 3, 1.5, "Plant", BLUE)
draw_box(ax_eng, (3.5, 2.5), 3, 1.5, "Controller", PURPLE)
draw_box(ax_eng, (0.3, 4.8), 2.5, 1.2, "Sensor", GREEN)
draw_box(ax_eng, (7.2, 4.8), 2.5, 1.2, "Actuator", ORANGE)

# Arrows: loop
draw_arrow(ax_eng, (3.5, 5.4), (2.8, 5.4), BLUE)     # plant → sensor
draw_arrow(ax_eng, (1.55, 4.8), (1.55, 4.0), GREEN)   # sensor → down
draw_arrow(ax_eng, (1.55, 3.9), (3.5, 3.25), GREEN)   # → controller
draw_arrow(ax_eng, (6.5, 3.25), (8.45, 3.9), ORANGE)  # controller → up
draw_arrow(ax_eng, (8.45, 4.8), (8.45, 6.0), ORANGE)  # → actuator
draw_arrow(ax_eng, (8.45, 6.1), (6.5, 7.0), ORANGE)   # actuator → plant

# Key label
ax_eng.text(5, 0.6,
            "Sensor reads without\ndisturbing the plant",
            ha="center", va="center", fontsize=11,
            color=GREEN, fontstyle="italic",
            bbox=dict(fc="#77AC3015", ec=GREEN, pad=5,
                      boxstyle="round,pad=0.4"))

# ── Biochemistry panel ────────────────────────────────────────
ax_bio.set_title("Biochemical feedback", fontsize=14, pad=10)

draw_box(ax_bio, (3.5, 6.5), 3, 1.5, "Species $X$", BLUE)
draw_box(ax_bio, (3.5, 2.5), 3, 1.5, "Controller\nmolecules", PURPLE,
         fontsize=11)

# Single feedback loop with back-action
draw_arrow(ax_bio, (3.5, 7.25), (1.5, 7.25), BLUE)
ax_bio.text(0.3, 7.25, r"$X$", fontsize=13, color=BLUE, va="center")
draw_arrow(ax_bio, (1.5, 7.0), (1.5, 3.25), "0.4")
draw_arrow(ax_bio, (1.5, 3.25), (3.5, 3.25), PURPLE)

draw_arrow(ax_bio, (6.5, 3.25), (8.5, 3.25), PURPLE)
draw_arrow(ax_bio, (8.5, 3.5), (8.5, 7.0), "0.4")
draw_arrow(ax_bio, (8.5, 7.0), (6.5, 7.25), ORANGE)

# Back-action annotation
ax_bio.annotate("Sensing\nconsumes/\nproduces $X$",
                xy=(1.5, 5.5), xytext=(0.0, 4.5),
                fontsize=10, color=RED, ha="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

ax_bio.annotate("Actuation\nadds noise\nto $X$",
                xy=(8.5, 5.5), xytext=(10, 4.5),
                fontsize=10, color=RED, ha="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

# Key label
ax_bio.text(5, 0.6,
            "Every measurement and actuation\nevent perturbs the molecule count",
            ha="center", va="center", fontsize=11,
            color=RED, fontstyle="italic",
            bbox=dict(fc="#A2142F10", ec=RED, pad=5,
                      boxstyle="round,pad=0.4"))

fig.savefig("noise_bound_schematic.svg", bbox_inches="tight")
fig.savefig("noise_bound_schematic.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved noise_bound_schematic.svg")
