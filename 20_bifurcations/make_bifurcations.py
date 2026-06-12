"""Three bifurcation diagrams: saddle-node, transcritical, supercritical pitchfork.

Conventions:
- mu on x-axis, x* on y-axis
- stable branch:   solid, MATLAB blue   (#0072BD)
- unstable branch: dashed, MATLAB red   (#D95319)
- bifurcation point: black dot
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# MATLAB default colors
MAT_BLUE = "#0072BD"
MAT_RED  = "#D95319"
LW       = 2.5
FIGSIZE  = (6.2, 4.6)

# Matplotlib rcParams to look like MATLAB
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",  # keep text as text in SVG, editable later
})


def base_axes(ax, title, xlim, ylim):
    ax.axhline(0, color="0.6", linewidth=0.6, zorder=1)
    ax.axvline(0, color="0.6", linewidth=0.6, zorder=1)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$x^{\ast}$")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle=":", alpha=0.4)


def make_legend(ax):
    handles = [
        Line2D([0], [0], color=MAT_BLUE, lw=LW, label="Stable equilibrium"),
        Line2D([0], [0], color=MAT_RED,  lw=LW, ls="--", label="Unstable equilibrium"),
        Line2D([0], [0], marker="o", color="black", lw=0, markersize=7,
               label="Bifurcation point"),
    ]
    ax.legend(handles=handles, loc="best", framealpha=0.95)


# =========================================================================
# 1) Saddle-node bifurcation:  x_dot = mu - x^2
#    equilibria at x = +/- sqrt(mu) for mu > 0, none for mu < 0
# =========================================================================
fig, ax = plt.subplots(figsize=FIGSIZE)
mu_pos = np.linspace(0, 4, 300)

ax.plot(mu_pos,  np.sqrt(mu_pos), color=MAT_BLUE, lw=LW)
ax.plot(mu_pos, -np.sqrt(mu_pos), color=MAT_RED,  lw=LW, ls="--")
ax.plot(0, 0, "o", color="black", markersize=7, zorder=5)

base_axes(ax,
          title=r"Saddle-node bifurcation:  $\dot{x} = \mu - x^{2}$",
          xlim=(-3, 4), ylim=(-2.5, 2.5))
make_legend(ax)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/saddle_node_bifurcation.svg",
            format="svg", bbox_inches="tight")
plt.close(fig)

# =========================================================================
# 2) Transcritical bifurcation:  x_dot = mu*x - x^2 = x(mu - x)
#    equilibria: x = 0 and x = mu
#    stability swaps at mu = 0
# =========================================================================
fig, ax = plt.subplots(figsize=FIGSIZE)
mu_neg = np.linspace(-3, 0, 150)
mu_pos = np.linspace(0,  4, 200)

# Branch x = 0
ax.plot(mu_neg, np.zeros_like(mu_neg), color=MAT_BLUE, lw=LW)            # stable
ax.plot(mu_pos, np.zeros_like(mu_pos), color=MAT_RED,  lw=LW, ls="--")   # unstable

# Branch x = mu
ax.plot(mu_neg, mu_neg, color=MAT_RED,  lw=LW, ls="--")                  # unstable
ax.plot(mu_pos, mu_pos, color=MAT_BLUE, lw=LW)                           # stable

ax.plot(0, 0, "o", color="black", markersize=7, zorder=5)

base_axes(ax,
          title=r"Transcritical bifurcation:  $\dot{x} = \mu x - x^{2}$",
          xlim=(-3, 4), ylim=(-3.2, 4))
make_legend(ax)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/transcritical_bifurcation.svg",
            format="svg", bbox_inches="tight")
plt.close(fig)

# =========================================================================
# 3) Supercritical pitchfork:  x_dot = mu*x - x^3
#    equilibria: x = 0  (stable for mu<0, unstable for mu>0)
#                x = +/- sqrt(mu)  (stable, only for mu>0)
# =========================================================================
fig, ax = plt.subplots(figsize=FIGSIZE)
mu_neg = np.linspace(-3, 0, 150)
mu_pos = np.linspace(0,  4, 200)

# Branch x = 0
ax.plot(mu_neg, np.zeros_like(mu_neg), color=MAT_BLUE, lw=LW)            # stable
ax.plot(mu_pos, np.zeros_like(mu_pos), color=MAT_RED,  lw=LW, ls="--")   # unstable

# Branches x = +/- sqrt(mu), mu > 0, both stable
ax.plot(mu_pos,  np.sqrt(mu_pos), color=MAT_BLUE, lw=LW)
ax.plot(mu_pos, -np.sqrt(mu_pos), color=MAT_BLUE, lw=LW)

ax.plot(0, 0, "o", color="black", markersize=7, zorder=5)

base_axes(ax,
          title=r"Supercritical pitchfork:  $\dot{x} = \mu x - x^{3}$",
          xlim=(-3, 4), ylim=(-2.5, 2.5))
make_legend(ax)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/pitchfork_bifurcation.svg",
            format="svg", bbox_inches="tight")
plt.close(fig)

print("All three SVGs written to /mnt/user-data/outputs/")
