import numpy as np
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"; GOLD="#EDB120"
OUT = "/mnt/user-data/outputs"

x = np.linspace(-4, 4, 600)
relu = np.maximum(0, x)

fig, ax = plt.subplots(figsize=(9, 6))
fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.11)

ax.plot(x, relu, color="k", lw=3.0, zorder=5)

etas = [0.1, 1, 10, 100]
colors = [ORANGE, GOLD, GREEN, BLUE]
for eta, col in zip(etas, colors):
    f = (x + np.sqrt(x**2 + 4/eta)) / 2
    ax.plot(x, f, color=col, lw=2.2)

# Place labels at specific (x, y) positions along each curve
# chosen so they're well-separated and don't overlap
label_specs = [
    (-2.5, 2.6, r"$\eta = 0.1$", ORANGE),
    (-1.5, 0.85, r"$\eta = 1$", GOLD),
    (-0.6, 0.35, r"$\eta = 10$", GREEN),
    (0.3,  0.22, r"$\eta = 100$", BLUE),
    (1.5, -0.22, r"$\max\{0, x\}$", "k"),
]
for xp, yp, lbl, col in label_specs:
    ax.text(xp, yp, lbl, fontsize=12, color=col, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(fc="white", ec="none", pad=1, alpha=0.8))

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f_\eta(x)$")
ax.set_title(
    r"Smooth approximation: $f_\eta(x) = \frac{1}{2}"
    r"\!\left(x + \sqrt{x^2 + 4/\eta}\right)"
    r" \;\to\; \max\{0,x\}$",
    fontsize=14, fontweight="bold")
ax.set_xlim(-4, 4); ax.set_ylim(-0.5, 4.5)
ax.grid(True, alpha=0.15)

fig.savefig(f"{OUT}/smooth_relu.svg", bbox_inches="tight")
fig.savefig("/home/claude/smooth_relu.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved smooth_relu.svg")
