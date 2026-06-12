"""
Dual-rail representation of a general linear system dx/dt = Ax + b.

A = [[-1,-1],[1,-1]], b = [1,-1].  Equilibrium: x* = [1, 0].

Dual-rail: x = x+ - x-  where x+, x- >= 0.
  dx+/dt = M x+ + N x- + b+ - eta * x+ o x-
  dx-/dt = M x- + N x+ + b- - eta * x+ o x-
  M = [[-1,0],[1,-1]], N = [[0,1],[0,0]], b+ = [1,0], b- = [0,1].

The eta term (annihilation) pushes x+, x- toward their minimal
representation.  As eta -> inf, x+ -> max(x,0), x- -> max(-x,0).

Three figures: eta = 1, 10, 100.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 13,
    "axes.labelsize": 15, "axes.titlesize": 14,
    "legend.fontsize": 10, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"
PURPLE="#7E2F8E"; GOLD="#EDB120"; GRAY="0.6"
OUT = "/mnt/user-data/outputs"

# Original system
A = np.array([[-1, -1], [1, -1]])
b = np.array([1, -1])

# Dual-rail matrices
M = np.array([[-1, 0], [1, -1]])
N = np.array([[0, 1], [0, 0]])
bp = np.array([1, 0])   # b+
bm = np.array([0, 1])   # b-

# Equilibrium
x_eq = np.linalg.solve(A, -b)
print(f"Equilibrium: x* = {x_eq}")

# Original system RHS
def rhs_orig(t, x):
    return A @ x + b

# Dual-rail RHS
def rhs_dual(t, z, eta):
    xp, xm = z[:2], z[2:]
    annihil = eta * xp * xm
    dxp = M @ xp + N @ xm + bp - annihil
    dxm = M @ xm + N @ xp + bm - annihil
    return np.concatenate([dxp, dxm])

# Initial conditions: x(0) = [-1, 2]
x0 = np.array([-1.0, 2.0])
xp0 = np.maximum(x0, 0)    # [0, 2]
xm0 = np.maximum(-x0, 0)   # [1, 0]

T = 10.0
te = np.linspace(0, T, 1000)

# Solve original
sol_orig = solve_ivp(rhs_orig, [0, T], x0, t_eval=te, rtol=1e-10)

# Solve dual-rail for each eta
for eta in [1, 10, 100]:
    z0 = np.concatenate([xp0, xm0])
    sol = solve_ivp(lambda t, z: rhs_dual(t, z, eta),
                    [0, T], z0, t_eval=te, rtol=1e-10)
    xp1, xp2 = sol.y[0], sol.y[1]
    xm1, xm2 = sol.y[2], sol.y[3]
    diff1 = xp1 - xm1
    diff2 = xp2 - xm2
    x1_orig = sol_orig.y[0]
    x2_orig = sol_orig.y[1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.08,
                        hspace=0.15)
    fig.suptitle(rf"Dual-rail representation ($\eta = {eta}$)",
                 fontsize=15, fontweight="bold")

    # Component 1
    ax1.plot(te, x1_orig, color=GRAY, lw=2.0, ls="--",
             label=r"$x_1(t)$ (original)")
    ax1.plot(te, xp1, color=BLUE, lw=2.2,
             label=r"$x_1^+(t)$")
    ax1.plot(te, xm1, color=ORANGE, lw=2.2,
             label=r"$x_1^-(t)$")
    ax1.plot(te, diff1, color=GREEN, lw=2.0, ls=":",
             label=r"$x_1^+ - x_1^-$")
    ax1.axhline(0, color="0.8", lw=0.5)
    ax1.set_ylabel("Component 1")
    ax1.set_title(r"$x_1$: original, dual-rail positive/negative parts",
                  fontsize=12)
    ax1.legend(loc="upper right", framealpha=0.95, edgecolor="0.75",
               ncol=2)
    ax1.grid(True, alpha=0.15)

    # Component 2
    ax2.plot(te, x2_orig, color=GRAY, lw=2.0, ls="--",
             label=r"$x_2(t)$ (original)")
    ax2.plot(te, xp2, color=BLUE, lw=2.2,
             label=r"$x_2^+(t)$")
    ax2.plot(te, xm2, color=ORANGE, lw=2.2,
             label=r"$x_2^-(t)$")
    ax2.plot(te, diff2, color=GREEN, lw=2.0, ls=":",
             label=r"$x_2^+ - x_2^-$")
    ax2.axhline(0, color="0.8", lw=0.5)
    ax2.set_ylabel("Component 2")
    ax2.set_xlabel("Time $t$")
    ax2.set_title(r"$x_2$: original, dual-rail positive/negative parts",
                  fontsize=12)
    ax2.legend(loc="upper right", framealpha=0.95, edgecolor="0.75",
               ncol=2)
    ax2.grid(True, alpha=0.15)

    # Match ylims across panels for visual consistency
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1]) * 1.05
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0]) * 1.05

    fname = f"dual_rail_eta{eta}"
    fig.savefig(f"{OUT}/{fname}.svg", bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}.svg")

    # Verify accuracy
    err1 = np.max(np.abs(diff1 - x1_orig))
    err2 = np.max(np.abs(diff2 - x2_orig))
    print(f"  eta={eta}: max|x+ - x- - x| = {max(err1,err2):.2e}")
