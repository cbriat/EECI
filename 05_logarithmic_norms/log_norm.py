"""
Logarithmic norm illustrations.

Figure 1: ||exp(At)|| vs exp(mu(A)*t) for the 1-norm and 2-norm.
Figure 2: Stability region vs mu_2(A)<0 region in a 2-parameter family.
"""
import numpy as np
from scipy.linalg import expm, norm, eigvalsh
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 16,
    "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"; PURPLE="#7E2F8E"
OUT = "/mnt/user-data/outputs"

# ══════════════════════════════════════════════════════════════
#  FIGURE 1: ||exp(At)||_p  vs  exp(mu_p(A) * t)
# ══════════════════════════════════════════════════════════════
A = np.array([[-1, 5],
              [ 0, -2]])

eigs = np.linalg.eigvals(A)
alpha = eigs.real.max()
print(f"A eigenvalues: {eigs}")
print(f"Spectral abscissa: {alpha:.3f}")

# Logarithmic norms
# mu_2 = lambda_max( (A+A^T)/2 )
mu2 = eigvalsh((A + A.T)/2)[-1]
# mu_1 = max_j ( a_jj + sum_{i!=j} |a_ij| )  (column sums)
mu1 = max(A[0,0] + abs(A[1,0]),  A[1,1] + abs(A[0,1]))

print(f"mu_1(A) = {mu1:.3f}")
print(f"mu_2(A) = {mu2:.3f}")

t = np.linspace(0, 4, 500)

# Compute ||exp(At)||_p for each t
norm1 = np.array([norm(expm(A*ti), 1) for ti in t])
norm2 = np.array([norm(expm(A*ti), 2) for ti in t])

bound1 = np.exp(mu1 * t)
bound2 = np.exp(mu2 * t)

# Proper spectral abscissa bound: beta_p * exp(alpha*t)
# beta_p = max_t ||exp(At)||_p * exp(-alpha*t)
beta1 = max(norm1 * np.exp(-alpha * t))
beta2 = max(norm2 * np.exp(-alpha * t))
spec_bound1 = beta1 * np.exp(alpha * t)
spec_bound2 = beta2 * np.exp(alpha * t)
print(f"beta_1 = {beta1:.3f},  beta_2 = {beta2:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.12,
                    wspace=0.25)
fig.suptitle(r"Logarithmic norm bound:  $\|e^{At}\| \leq e^{\mu(A)\,t}$",
             fontsize=15, fontweight="bold")

# 1-norm panel
ax1.plot(t, norm1, color=BLUE, label=r"$\|e^{At}\|_1$")
ax1.plot(t, bound1, color=ORANGE, ls="--", lw=2.0,
         label=rf"$e^{{\mu_1(A)\,t}}$,  $\mu_1 = {mu1:.1f}$")
ax1.plot(t, spec_bound1, color=GREEN, ls=":", lw=2.0,
         label=rf"$\beta_1 e^{{\alpha t}}$,  $\beta_1 = {beta1:.2f}$, $\alpha = {alpha:.0f}$")
ax1.set_xlabel(r"Time $t$"); ax1.set_ylabel("Norm / bound")
ax1.set_title(r"1-norm", fontsize=14)
ax1.set_xlim(0, 4); ax1.set_ylim(0, 6)
ax1.legend(loc="upper left", framealpha=0.95, edgecolor="0.75")
ax1.grid(True, alpha=0.15)

# 2-norm panel
ax2.plot(t, norm2, color=BLUE, label=r"$\|e^{At}\|_2$")
ax2.plot(t, bound2, color=ORANGE, ls="--", lw=2.0,
         label=rf"$e^{{\mu_2(A)\,t}}$,  $\mu_2 = {mu2:.2f}$")
ax2.plot(t, spec_bound2, color=GREEN, ls=":", lw=2.0,
         label=rf"$\beta_2 e^{{\alpha t}}$,  $\beta_2 = {beta2:.2f}$, $\alpha = {alpha:.0f}$")
ax2.set_xlabel(r"Time $t$"); ax2.set_ylabel("Norm / bound")
ax2.set_title(r"2-norm", fontsize=14)
ax2.set_xlim(0, 4); ax2.set_ylim(0, 6)
ax2.legend(loc="upper left", framealpha=0.95, edgecolor="0.75")
ax2.grid(True, alpha=0.15)

# Annotate the matrix
for ax in (ax1, ax2):
    ax.text(0.97, 0.97,
            r"$A = [[-1,\; 5],\; [0,\; -2]]$"
            "\n" r"$\lambda(A) = \{-1,\, -2\}$  (stable)",
            transform=ax.transAxes, fontsize=10, ha="right", va="top",
            bbox=dict(fc="white", ec="0.7", pad=4,
                      boxstyle="round,pad=0.4"))

fig.savefig(f"{OUT}/log_norm_bounds.svg", bbox_inches="tight")
plt.close()
print("Saved log_norm_bounds.svg")

# ══════════════════════════════════════════════════════════════
#  FIGURE 2: Stability region vs mu_2(A)<0 in parameter space
#
#  Family: A(a,b) = [[a, b], [0, -1]]
#  Eigenvalues: a and -1.  Stable iff a < 0.
#  mu_2(A) < 0  iff  a < -b^2/4.
# ══════════════════════════════════════════════════════════════
b_grid = np.linspace(-6, 6, 500)
a_grid = np.linspace(-10, 2, 500)
B, Aa = np.meshgrid(b_grid, a_grid)

# Regions
stable = Aa < 0                    # Hurwitz stability
mu2_neg = Aa < -B**2 / 4           # mu_2 < 0

# Three zones: mu2<0 (⊂ stable), stable but mu2>=0, unstable
zone = np.zeros_like(Aa)
zone[stable & ~mu2_neg] = 1        # stable, mu2 >= 0
zone[mu2_neg] = 2                  # mu2 < 0 (also stable)
# zone = 0: unstable

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

cmap = ListedColormap(["#FDE8E8", "#D4E6F1", "#A9DFBF"])

fig, ax = plt.subplots(figsize=(8, 6.5))
fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.10)

ax.imshow(zone, origin="lower", aspect="auto",
          extent=[b_grid[0], b_grid[-1], a_grid[0], a_grid[-1]],
          cmap=cmap, vmin=0, vmax=2, interpolation="bilinear")

# Boundaries
ax.axhline(0, color="#A2142F", lw=2.5, label="Stability: $a = 0$")
b_curve = np.linspace(-6, 6, 300)
a_curve = -b_curve**2 / 4
ax.plot(b_curve, a_curve, color=BLUE, lw=2.5,
        label=r"$\mu_2 = 0$:  $a = -b^2/4$")

# Mark a sample point in each region
ax.plot(3, 0.5, "X", color="#A2142F", ms=12, markeredgecolor="k",
        markeredgewidth=0.5, zorder=5)
ax.text(3.4, 0.7, "unstable", fontsize=10, color="#A2142F")

ax.plot(3, -1, "o", color="#0072BD", ms=10, markeredgecolor="k",
        markeredgewidth=0.5, zorder=5)
ax.text(3.4, -0.7, r"stable, $\mu_2 \geq 0$", fontsize=10, color="#0072BD")

ax.plot(0.5, -2, "s", color=GREEN, ms=10, markeredgecolor="k",
        markeredgewidth=0.5, zorder=5)
ax.text(0.9, -1.7, r"$\mu_2 < 0$", fontsize=10, color=GREEN)

ax.set_xlabel(r"Off-diagonal element $b$")
ax.set_ylabel(r"Diagonal element $a$")
ax.set_title(
    r"Family $A(a,b)$ with $A_{11}\!=\!a,\; A_{12}\!=\!b,\;"
    r"A_{21}\!=\!0,\; A_{22}\!=\!{-1}$:"
    r"  stability vs. $\mu_2(A) < 0$",
    fontsize=13, fontweight="bold")
ax.set_xlim(b_grid[0], b_grid[-1])
ax.set_ylim(a_grid[0], a_grid[-1])

handles = [
    Patch(fc="#A9DFBF", ec="k", lw=0.8,
          label=r"$\mu_2(A) < 0$  (contractivity)"),
    Patch(fc="#D4E6F1", ec="k", lw=0.8,
          label=r"Stable but $\mu_2(A) \geq 0$"),
    Patch(fc="#FDE8E8", ec="k", lw=0.8,
          label="Unstable"),
]
ax.legend(handles=handles, loc="lower left", fontsize=11,
          framealpha=0.95, edgecolor="0.75")
ax.grid(True, alpha=0.12)

fig.savefig(f"{OUT}/log_norm_regions.svg", bbox_inches="tight")
plt.close()
print("Saved log_norm_regions.svg")
