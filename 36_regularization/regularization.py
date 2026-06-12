"""
Regularisation functions for positive integral controllers.
Three figures, each with max(0,x) reference.
"""
import numpy as np
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 16,
    "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"]

alphas = [0.01, 0.1, 1.0, 10.0]

# ══════════════════════════════════════════════════════════════
# FIGURE 1: phi(x) = exp(alpha x)
# ══════════════════════════════════════════════════════════════
x = np.linspace(-5, 5, 1000)
relu = np.maximum(0, x)

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12)

ax.plot(x, relu, "k--", lw=2.0, label=r"$\max(0, x)$", zorder=5)
for alpha, c in zip(alphas, COLORS):
    y = np.exp(alpha * x)
    ax.plot(x, y, color=c, label=rf"$\alpha = {alpha}$")

ax.set_ylim(-0.3, 8.5)
ax.set_xlim(-5, 5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\varphi(x)$")
ax.set_title(r"$\varphi(x) = e^{\alpha x}$", fontsize=17, fontweight="bold")
ax.legend(loc="upper left", framealpha=0.95, edgecolor="0.75", ncol=1)
ax.grid(True, alpha=0.15)
ax.axhline(0, color="0.5", lw=0.5)
ax.axvline(0, color="0.5", lw=0.5)

fig.savefig("regularization_exp.svg", bbox_inches="tight")
fig.savefig("regularization_exp.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved regularization_exp.svg")

# ══════════════════════════════════════════════════════════════
# FIGURE 2: phi(x) = beta * exp(alpha x) / (1 + exp(alpha x))
# ══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12,
                    wspace=0.22)
fig.suptitle(
    r"$\varphi(x) = \beta\, e^{\alpha x}\!/\!\left(1 + e^{\alpha x}\right)$",
    fontsize=17, fontweight="bold")

for ax, beta, panel in [(ax1, 1, "a"), (ax2, 10, "b")]:
    ax.plot(x, relu, "k--", lw=2.0, label=r"$\max(0,x)$")
    for alpha, c in zip(alphas, COLORS):
        y = beta * np.exp(alpha*x) / (1 + np.exp(alpha*x))
        ax.plot(x, y, color=c, label=rf"$\alpha = {alpha}$")

    ax.axhline(beta, color="0.4", ls=":", lw=1.0)
    ax.text(4.6, beta*1.04, rf"$\beta = {beta}$", fontsize=10,
            ha="right", va="bottom", color="0.4")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\varphi(x)$")
    ax.set_title(rf"$\beta = {beta}$", fontsize=14)
    ax.set_xlim(-5, 5)
    ymax = max(beta * 1.25, 5.5) if beta <= 1 else beta * 1.25
    ax.set_ylim(-0.15 * beta, ymax)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="0.75")
    ax.grid(True, alpha=0.15)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axvline(0, color="0.5", lw=0.5)

fig.savefig("regularization_sigmoid.svg", bbox_inches="tight")
fig.savefig("regularization_sigmoid.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved regularization_sigmoid.svg")

# ══════════════════════════════════════════════════════════════
# FIGURE 3: phi(x) = x^2 / (x^2 + h^2)
# ══════════════════════════════════════════════════════════════
hs = [0.01, 0.1, 1.0, 10.0]

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12)

# max(0,x) for reference
ax.plot(x, relu, "k--", lw=2.0,
        label=r"$\max(0,x)$")

for h, c in zip(hs, COLORS):
    y = x**2 / (x**2 + h**2)
    ax.plot(x, y, color=c, label=rf"$h = {h}$")

ax.set_xlim(-5, 5)
ax.set_ylim(-0.05, 1.35)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\varphi(x)$")
ax.set_title(r"$\varphi(x) = x^2\!/\!\left(x^2 + h^2\right)$",
             fontsize=17, fontweight="bold")
ax.legend(loc="center right", framealpha=0.95, edgecolor="0.75")
ax.grid(True, alpha=0.15)
ax.axhline(0, color="0.5", lw=0.5)
ax.axvline(0, color="0.5", lw=0.5)

fig.savefig("regularization_hill.svg", bbox_inches="tight")
fig.savefig("regularization_hill.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved regularization_hill.svg")
