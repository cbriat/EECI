"""
Two reaction networks: deterministic vs stochastic in phase plane.

Network 1 (corrected): X+Y -> 2Y, Y -> X
  Conservation x+y=c. Positive eq at (k2/k1, c-k2/k1).
  Stochastic: absorbed at (c, 0) when Y hits zero.

Network 2: X -> 0, X+Y -> X, Y -> X+Y, Y -> 2Y
  Positive eq at (k4/k2, k1*k4/(k2*k3)).
  Stochastic: absorption at (0,0).
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 15,
    "legend.fontsize": 10, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"
PURPLE="#7E2F8E"; GOLD="#EDB120"; RED="#A2142F"
OUT = "/mnt/user-data/outputs"

# ══════════════════════════════════════════════════════════════
#  NETWORK 1: X+Y -> 2Y (k1), Y -> X (k2)
#  Conservation x+y = c.
#  Equilibrium: x* = k2/k1, y* = c - k2/k1
#  Absorbing: y=0 (stochastic only).
# ══════════════════════════════════════════════════════════════
k1_n1, k2_n1 = 0.05, 0.5
c_n1 = 25
x_eq1 = k2_n1 / k1_n1
y_eq1 = c_n1 - x_eq1
print(f"Network 1: eq = ({x_eq1:.1f}, {y_eq1:.1f}), c={c_n1}")

def rhs_n1(t, z):
    x, y = z
    return [-k1_n1*x*y + k2_n1*y,
             k1_n1*x*y - k2_n1*y]

def ssa_n1(x0, y0, T, seed):
    rng = np.random.default_rng(seed)
    x, y = int(x0), int(y0)
    t = 0.0
    ts, xs, ys = [0.0], [x], [y]
    while t < T:
        prop1 = k1_n1 * x * y    # X+Y -> 2Y
        prop2 = k2_n1 * y        # Y -> X
        a_tot = prop1 + prop2
        if a_tot <= 0:
            ts.append(T); xs.append(x); ys.append(y)
            break
        dt = rng.exponential(1/a_tot)
        t += dt
        if rng.random() < prop1/a_tot:
            x -= 1; y += 1
        else:
            x += 1; y -= 1
        ts.append(t); xs.append(x); ys.append(y)
    return np.array(ts), np.array(xs), np.array(ys)

ics_n1 = [(20, 5), (3, 22)]
T_n1 = 200.0
te_n1 = np.linspace(0, T_n1, 3000)
colors_ic = [BLUE, ORANGE]

fig, ax = plt.subplots(figsize=(8, 7))
fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.11)
fig.suptitle(r"Network 1:  $X\!+\!Y \to 2Y$,  $Y \to X$"
             r"    ($x+y=c$, absorption at $y\!=\!0$)",
             fontsize=14, fontweight="bold")

for i, (x0, y0) in enumerate(ics_n1):
    sol = solve_ivp(rhs_n1, [0, T_n1], [x0, y0], t_eval=te_n1, rtol=1e-9)
    ax.plot(sol.y[0], sol.y[1], color=colors_ic[i], lw=3.0,
            label=rf"Det. $({x0}, {y0})$", zorder=3)
    ax.plot(x0, y0, "o", color=colors_ic[i], ms=8,
            markeredgecolor="k", markeredgewidth=0.8, zorder=4)
    for s in range(3):
        ts, xs, ys = ssa_n1(x0, y0, T_n1, seed=100*i+s+3)
        ax.plot(xs, ys, color=colors_ic[i], lw=0.7, alpha=0.4)
        if ys[-1] == 0:
            ax.plot(xs[-1], ys[-1], "x", color=colors_ic[i],
                    ms=9, mew=2, zorder=4)

# Equilibrium
ax.plot(x_eq1, y_eq1, "*", color=RED, ms=16,
        markeredgecolor="k", markeredgewidth=0.7, zorder=5)
ax.text(x_eq1-2, y_eq1+2,
        rf"$x^* = ({x_eq1:.0f},\,{y_eq1:.0f})$",
        fontsize=11, color=RED, ha="right",
        bbox=dict(fc="white", ec="0.6", pad=2,
                  boxstyle="round,pad=0.3"))

# Conservation line
xx = np.linspace(0, c_n1, 100)
ax.plot(xx, c_n1 - xx, color="0.75", lw=1.0, ls=":", zorder=1,
        label=rf"$x+y={c_n1}$")

# Mark absorbing boundary
ax.plot([0, c_n1], [0, 0], color=RED, lw=2.5, ls="-", alpha=0.6, zorder=2)
ax.text(c_n1/2, -2.5, r"Absorbing boundary $y\!=\!0$",
        ha="center", fontsize=10, color=RED)

ax.set_xlabel(r"$X$"); ax.set_ylabel(r"$Y$")
ax.set_xlim(-2, c_n1+2); ax.set_ylim(-4, c_n1+2)
ax.set_aspect("equal")
ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.75")
ax.grid(True, alpha=0.15)

ax.text(0.02, 0.45,
        "Det.: converges to positive $x^*$\n"
        r"Stoch.: absorbed at $y\!=\!0$ in finite time",
        transform=ax.transAxes, fontsize=10.5, va="top",
        bbox=dict(fc="#FDE8E8", ec=RED, pad=4,
                  boxstyle="round,pad=0.3"))

fig.savefig(f"{OUT}/absorption_network1.svg", bbox_inches="tight")
plt.close()
print("Saved absorption_network1.svg")

# ══════════════════════════════════════════════════════════════
#  NETWORK 2: X->0, X+Y->X, Y->X+Y, Y->2Y
#  Positive eq: (k4/k2, k1*k4/(k2*k3))
#  Stochastic: absorption at (0,0).
# ══════════════════════════════════════════════════════════════
k1_n2, k2_n2, k3_n2, k4_n2 = 0.2, 0.02, 0.2, 0.2
x_eq2 = k4_n2 / k2_n2
y_eq2 = k1_n2 * k4_n2 / (k2_n2 * k3_n2)
print(f"Network 2: eq = ({x_eq2:.1f}, {y_eq2:.1f})")

def rhs_n2(t, z):
    x, y = max(z[0], 0), max(z[1], 0)
    return [-k1_n2*x + k3_n2*y,
            -k2_n2*x*y + k4_n2*y]

def ssa_n2(x0, y0, T, seed):
    rng = np.random.default_rng(seed)
    x, y = int(x0), int(y0)
    t = 0.0
    ts, xs, ys = [0.0], [x], [y]
    while t < T:
        props = np.array([
            k1_n2 * x,          # X -> 0
            k2_n2 * x * y,      # X+Y -> X  (Δy=-1)
            k3_n2 * y,          # Y -> X+Y  (Δx=+1)
            k4_n2 * y,          # Y -> 2Y   (Δy=+1)
        ])
        a_tot = props.sum()
        if a_tot <= 0:
            ts.append(T); xs.append(0); ys.append(0)
            break
        dt = rng.exponential(1/a_tot)
        t += dt
        r = rng.random() * a_tot
        cum = 0.0
        for j in range(4):
            cum += props[j]
            if r < cum: break
        stoich = [(-1,0), (0,-1), (1,0), (0,1)]
        x += stoich[j][0]; y += stoich[j][1]
        x = max(x, 0); y = max(y, 0)
        ts.append(t); xs.append(x); ys.append(y)
    return np.array(ts), np.array(xs), np.array(ys)

ics_n2 = [(18, 5), (5, 18)]
T_n2 = 200.0
te_n2 = np.linspace(0, T_n2, 3000)

fig, ax = plt.subplots(figsize=(8, 7))
fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.11)
fig.suptitle(r"Network 2:  $X\!\to\!0$,  $X\!+\!Y\!\to\!X$,  "
             r"$Y\!\to\!X\!+\!Y$,  $Y\!\to\!2Y$",
             fontsize=13, fontweight="bold")

for i, (x0, y0) in enumerate(ics_n2):
    sol = solve_ivp(rhs_n2, [0, T_n2], [x0, y0], t_eval=te_n2, rtol=1e-9)
    ax.plot(sol.y[0], sol.y[1], color=colors_ic[i], lw=3.0,
            label=rf"Det. $({x0}, {y0})$", zorder=3)
    ax.plot(x0, y0, "o", color=colors_ic[i], ms=8,
            markeredgecolor="k", markeredgewidth=0.8, zorder=4)
    for s in range(4):
        ts, xs, ys = ssa_n2(x0, y0, T_n2, seed=300*i+s+5)
        ax.plot(xs, ys, color=colors_ic[i], lw=0.7, alpha=0.35)
        if xs[-1] == 0 and ys[-1] == 0:
            ax.plot(0, 0, "x", color=colors_ic[i], ms=9, mew=2,
                    alpha=0.7, zorder=4)

# Equilibrium
ax.plot(x_eq2, y_eq2, "*", color=RED, ms=16,
        markeredgecolor="k", markeredgewidth=0.7, zorder=5)
ax.text(x_eq2+1.5, y_eq2+1.5,
        rf"$x^* = ({x_eq2:.0f},\,{y_eq2:.0f})$",
        fontsize=11, color=RED,
        bbox=dict(fc="white", ec="0.6", pad=2,
                  boxstyle="round,pad=0.3"))

# Mark absorbing state
ax.plot(0, 0, "s", color="k", ms=10, zorder=5)
ax.text(2, -2, "Absorbing $(0,0)$", fontsize=10, color="k")

ax.set_xlabel(r"$X$"); ax.set_ylabel(r"$Y$")
xmax = max(30, int(x_eq2*2.5))
ymax = max(30, int(y_eq2*2.5))
ax.set_xlim(-3, xmax); ax.set_ylim(-3, ymax)
ax.set_aspect("equal")
ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.75")
ax.grid(True, alpha=0.15)

ax.text(0.02, 0.97,
        "Det.: spiral to positive $x^*$\n"
        "Stoch.: absorption at $(0, 0)$",
        transform=ax.transAxes, fontsize=10.5, va="top",
        bbox=dict(fc="#FDE8E8", ec=RED, pad=4,
                  boxstyle="round,pad=0.3"))

fig.savefig(f"{OUT}/absorption_network2.svg", bbox_inches="tight")
plt.close()
print("Saved absorption_network2.svg")
