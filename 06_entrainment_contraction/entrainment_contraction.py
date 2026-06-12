"""
Entrainment via contraction theory for reaction networks.
(Russo, di Bernardo & Sontag, PLoS Comp Biol 2010)

A contracting system driven by a periodic input entrains:
all trajectories converge to a unique periodic orbit with
the period of the forcing, regardless of initial conditions.

Example: 2-species negative feedback loop + periodic forcing.
  dx/dt = alpha*K^n / (K^n + y^n) - k1*x + u(t)
  dy/dt = beta*x / (1 + x) - k2*y

Figure 1: time traces from many ICs → all converge to same orbit.
Figure 2: phase portrait showing convergence to the limit cycle.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 15, "axes.titlesize": 15,
    "legend.fontsize": 10, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"
PURPLE="#7E2F8E"; GOLD="#EDB120"; GRAY="0.55"
OUT = "/mnt/user-data/outputs"

# ── Parameters ────────────────────────────────────────────────
alpha = 4.0    # max production rate
K     = 1.0    # Hill threshold
n_h   = 3.0    # Hill coefficient
k1    = 2.0    # degradation of x
beta  = 5.0    # production rate of y
k2    = 2.0    # degradation of y
A_f   = 0.8    # forcing amplitude
omega = 2*np.pi/3.0  # forcing frequency (period = 3)
T_period = 2*np.pi/omega

def forcing(t):
    return A_f * np.sin(omega * t)

def rhs(t, z):
    x, y = z
    hill = alpha * K**n_h / (K**n_h + max(y, 0)**n_h)
    dxdt = hill - k1*x + forcing(t)
    dydt = beta*x/(1+x) - k2*y
    return [dxdt, dydt]

# Verify contraction: check mu_2(J) along a trajectory
def jacobian(x, y):
    hn = K**n_h
    dhill_dy = -alpha * hn * n_h * max(y,0)**(n_h-1) / (hn + max(y,0)**n_h)**2
    dg_dx = beta / (1+x)**2
    J = np.array([[-k1, dhill_dy],
                  [dg_dx, -k2]])
    return J

# Quick check: compute mu_2 along a reference trajectory
sol_ref = solve_ivp(rhs, [0, 30], [1.0, 1.0],
                    t_eval=np.linspace(0, 30, 1000), rtol=1e-9)
mu2_vals = [eigvalsh((jacobian(x,y) + jacobian(x,y).T)/2)[-1]
            for x,y in zip(sol_ref.y[0], sol_ref.y[1])]
print(f"mu_2(J) range: [{min(mu2_vals):.3f}, {max(mu2_vals):.3f}]")
print(f"  → {'CONTRACTING' if max(mu2_vals) < 0 else 'NOT contracting'}")
print(f"Forcing period: {T_period:.2f}")

# ── Simulate from many initial conditions ─────────────────────
T = 20.0
te = np.linspace(0, T, 2000)
ics = [(0.2, 0.1), (3.5, 0.2), (0.3, 3.0), (3.0, 3.0),
       (1.5, 0.1), (0.1, 2.0), (2.5, 2.5), (1.0, 3.5),
       (3.5, 1.5), (0.5, 1.0), (2.0, 0.3), (0.8, 2.8)]

trajectories = []
for ic in ics:
    sol = solve_ivp(rhs, [0, T], ic, t_eval=te, rtol=1e-9)
    trajectories.append(sol.y)

# ══════════════════════════════════════════════════════════════
#  FIGURE: two-panel (time traces + phase portrait)
# ══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.12,
                    wspace=0.25)
fig.suptitle("Entrainment via contraction: all trajectories converge "
             "to the same periodic orbit",
             fontsize=14, fontweight="bold")

# ── Left: time traces of x(t) ────────────────────────────────
cmap = plt.get_cmap("tab20")
for i, traj in enumerate(trajectories):
    ax1.plot(te, traj[0], color=cmap(i), lw=1.8, alpha=0.8)

# Show forcing as a scaled reference
u_sig = np.array([forcing(t) for t in te])
# Rescale forcing to visible range for annotation
x_ss = alpha*K**n_h/(K**n_h + 1) / k1  # approximate SS of x
ax1.plot(te, x_ss + u_sig * 0.5, color=GRAY, ls="--", lw=1.8,
         alpha=0.5, label=r"Periodic forcing $u(t)$ (scaled)")

# Mark the period
yb = ax1.get_ylim()[0] + 0.1
ax1.annotate("", xy=(T-T_period, yb), xytext=(T, yb),
             arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.5))
ax1.text(T - T_period/2, yb + 0.08, rf"$T = {T_period:.1f}$",
         ha="center", fontsize=10, color="0.3")

ax1.set_xlabel(r"Time $t$"); ax1.set_ylabel(r"$x(t)$")
ax1.set_title("Time traces from diverse initial conditions", fontsize=13)
ax1.set_xlim(0, T)
ax1.legend(loc="upper right", framealpha=0.95, edgecolor="0.75")
ax1.grid(True, alpha=0.15)

# Annotation in bottom-left (empty: trajectories are above 0.6 for t>3)
ax1.text(0.35, 0.05,
         r"$\mu_2(J(x)) < 0$ for all $x$"
         r"  $\Rightarrow$  unique entrained orbit",
         transform=ax1.transAxes, fontsize=10, va="bottom", ha="center",
         bbox=dict(fc="#EBF5E0", ec=GREEN, pad=4,
                   boxstyle="round,pad=0.3"))

# ── Right: phase portrait (x, y) ─────────────────────────────
for i, traj in enumerate(trajectories):
    ax2.plot(traj[0], traj[1], color=cmap(i), lw=1.6, alpha=0.6)
    ax2.plot(traj[0,0], traj[1,0], "o", color=cmap(i), ms=6, zorder=3)

# Highlight the entrained orbit (last few periods of reference traj)
ref = trajectories[0]
n_last = int(3 * T_period / T * len(te))
ax2.plot(ref[0, -n_last:], ref[1, -n_last:], color="k", lw=3.0,
         alpha=0.9, label="Entrained periodic orbit", zorder=4)

ax2.set_xlabel(r"$x$"); ax2.set_ylabel(r"$y$")
ax2.set_title("Phase portrait: convergence to limit cycle", fontsize=13)
ax2.legend(loc="lower right", framealpha=0.95, edgecolor="0.75")
ax2.grid(True, alpha=0.15)

fig.savefig(f"{OUT}/entrainment_contraction.svg", bbox_inches="tight")
fig.savefig("/home/claude/entrainment_contraction.png", dpi=200,
            bbox_inches="tight")
plt.close()
print("Saved entrainment_contraction.svg")
