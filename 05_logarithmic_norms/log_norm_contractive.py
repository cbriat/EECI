"""
Logarithmic norm bound in the contractive case: A Hurwitz with mu_2(A) < 0.
Both the norm and the bound decay, showing a useful (tight) envelope.
"""
import numpy as np
from scipy.linalg import expm, norm, eigvalsh
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 16,
    "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"; PURPLE="#7E2F8E"
OUT = "/mnt/user-data/outputs"

# Matrix: Hurwitz with mu_2 < 0 but non-normal enough to see a gap
A = np.array([[-1, 3],
              [ 0, -3]])

eigs = np.linalg.eigvals(A)
alpha = eigs.real.max()
mu2 = eigvalsh((A + A.T)/2)[-1]

print(f"A eigenvalues: {eigs}")
print(f"Spectral abscissa alpha(A) = {alpha:.3f}")
print(f"mu_2(A) = {mu2:.3f}")
print(f"Ordering: alpha = {alpha:.3f} <= mu_2 = {mu2:.3f} < 0  (contractive)")

t = np.linspace(0, 8, 600)

norm2 = np.array([norm(expm(A*ti), 2) for ti in t])
bound_mu = np.exp(mu2 * t)

# Proper spectral abscissa bound: beta * exp(alpha*t)
beta2 = max(norm2 * np.exp(-alpha * t))
bound_spec = beta2 * np.exp(alpha * t)
print(f"beta_2 = {beta2:.3f}")

# ── Figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.11)

# Shade the gap between norm and bound
ax.fill_between(t, norm2, bound_mu, alpha=0.12, color=ORANGE,
                label="Conservatism gap")

ax.plot(t, bound_mu, color=ORANGE, ls="--", lw=2.5,
        label=rf"$e^{{\mu_2(A)\,t}}$,   $\mu_2 = {mu2:.3f}$  (upper bound)")
ax.plot(t, norm2, color=BLUE, lw=2.5,
        label=r"$\|e^{At}\|_2$  (actual norm)")
ax.plot(t, bound_spec, color=GREEN, ls=":", lw=2.0,
        label=rf"$\beta e^{{\alpha t}}$,   $\beta = {beta2:.2f}$, $\alpha = {alpha:.0f}$  (spectral)")

ax.axhline(1, color="0.7", lw=0.5)
ax.set_xlabel(r"Time $t$")
ax.set_ylabel(r"$\|e^{At}\|_2$  /  bound")
ax.set_title(
    r"Contractive case: $A$ Hurwitz and $\mu_2(A) < 0$",
    fontsize=15, fontweight="bold")
ax.set_xlim(0, 8)
ax.set_ylim(0, 1.95)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.75",
          fontsize=11)
ax.grid(True, alpha=0.15)

# Matrix annotation
ax.text(0.97, 0.55,
        r"$A = [[-1,\; 3],\; [0,\; -3]]$"
        "\n" r"$\lambda(A) = \{-1,\, -3\}$"
        "\n\n" r"$\|e^{At}\|_2 \leq e^{\mu_2 t} \to 0$"
        "\n" r"$\|e^{At}\|_2 \leq \beta\, e^{\alpha t} \to 0$"
        "\n" r"$\mu_2 < 0 \;\Rightarrow\;$ guaranteed contraction",
        transform=ax.transAxes, fontsize=11, ha="right", va="top",
        bbox=dict(fc="#EBF5E0", ec=GREEN, pad=6,
                  boxstyle="round,pad=0.4"))

fig.savefig(f"{OUT}/log_norm_contractive.svg", bbox_inches="tight")
fig.savefig(f"/home/claude/log_norm_contractive.png", dpi=200,
            bbox_inches="tight")
plt.close()
print("Saved log_norm_contractive.svg")
