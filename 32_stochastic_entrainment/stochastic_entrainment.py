"""
Stochastic entrainment: Gupta, Hepp & Khammash demonstration.

Stuart-Landau oscillator (normal form near any Hopf bifurcation)
with periodic forcing + noise:

  dx = [(mu - r^2)x - omega*y + eps*cos(Omega*t)] dt + sigma dW_x
  dy = [(mu - r^2)y + omega*x                   ] dt + sigma dW_y

Figure 1: time series at a demo point outside the deterministic
          Arnold tongue, showing beating (det) vs phase locking (stoch).
Figure 2: Arnold tongue map, three colour-coded regions:
          blue = both entrained, orange = stochastic only, white = neither.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
import matplotlib, matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time as timer

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 16,
    "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE = "#0072BD"; ORANGE = "#D95319"; GRAY = "0.55"

# ── Shared parameters ─────────────────────────────────────────
mu    = 1.0
omega = 1.0
sigma = 0.4
r_star = np.sqrt(mu)

# Demo point: outside det. tongue, inside stoch. tongue
Omega_demo = 1.35          # Dw = 0.35
eps_demo   = 0.30          # < |Dw|*r* = 0.35 → det NOT entrained
Dw_demo    = Omega_demo - omega

print(f"|Dw| = {abs(Dw_demo):.3f},  eps/r* = {eps_demo/r_star:.3f},  "
      f"det entrained: {abs(Dw_demo) < eps_demo/r_star}")

# ══════════════════════════════════════════════════════════════
# FIGURE 1: TIME SERIES
# ══════════════════════════════════════════════════════════════
T_sim = 150.0; dt = 0.005; N = int(T_sim/dt)
tg = np.linspace(0, T_sim, N+1)

# Deterministic
sol = solve_ivp(
    lambda t, s: [(mu-s[0]**2-s[1]**2)*s[0] - omega*s[1]
                    + eps_demo*np.cos(Omega_demo*t),
                  (mu-s[0]**2-s[1]**2)*s[1] + omega*s[0]],
    [0, T_sim], [1.0, 0.0], max_step=0.02, rtol=1e-8, dense_output=True)
xd, yd = sol.sol(tg)

# Stochastic
rng = np.random.default_rng(12)
xs = np.zeros(N+1); ys = np.zeros(N+1)
xs[0], ys[0] = 1.0, 0.0
sqdt = np.sqrt(dt)
for i in range(N):
    x, y = xs[i], ys[i]
    r2 = x*x + y*y
    xs[i+1] = x + ((mu-r2)*x - omega*y + eps_demo*np.cos(Omega_demo*tg[i]))*dt \
              + sigma*rng.standard_normal()*sqdt
    ys[i+1] = y + ((mu-r2)*y + omega*x)*dt \
              + sigma*rng.standard_normal()*sqdt

ref = 1.3*np.cos(Omega_demo*tg)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.10,
                    hspace=0.18)
fig.suptitle(
    "Stochastic entrainment: forcing outside the deterministic Arnold tongue",
    fontsize=14, fontweight="bold")

ax1.plot(tg, xd, color=BLUE, lw=1.8, label=r"Oscillator $x(t)$")
ax1.plot(tg, ref, color=GRAY, lw=1.0, ls="--",
         label=rf"Forcing ($\Omega = {Omega_demo}$)")
ax1.set_ylabel(r"$x(t)$")
ax1.set_title(
    rf"Deterministic: beating  ($|\Delta\omega| = {abs(Dw_demo):.2f}"
    rf" > \varepsilon/r^* = {eps_demo/r_star:.2f}$)", fontsize=13)
ax1.legend(loc="upper right", framealpha=0.95, edgecolor="0.82")
ax1.grid(True, alpha=0.15)

ax2.plot(tg, xs, color=ORANGE, lw=1.2, alpha=0.9,
         label=r"Oscillator $x(t)$")
ax2.plot(tg, ref, color=GRAY, lw=1.0, ls="--",
         label=rf"Forcing ($\Omega = {Omega_demo}$)")
ax2.set_ylabel(r"$x(t)$"); ax2.set_xlabel(r"Time $t$")
ax2.set_title(
    rf"Stochastic ($\sigma = {sigma}$): noise-induced phase locking",
    fontsize=13)
ax2.legend(loc="upper right", framealpha=0.95, edgecolor="0.82")
ax2.grid(True, alpha=0.15)
for ax in (ax1, ax2): ax.set_xlim(0, T_sim)

fig.savefig("entrainment_timeseries.svg", bbox_inches="tight")
fig.savefig("entrainment_timeseries.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved entrainment_timeseries.svg")

# ══════════════════════════════════════════════════════════════
# FIGURE 2: ARNOLD TONGUE
# ══════════════════════════════════════════════════════════════
n_dw, n_eps = 70, 55
dw_arr  = np.linspace(-0.8, 0.8, n_dw)
eps_arr = np.linspace(0.01, 0.9, n_eps)
DW, EPS = np.meshgrid(dw_arr, eps_arr)
Om_flat = (omega + DW).ravel()
ep_flat = EPS.ravel()
M = len(Om_flat)

T_at = 500.0; dt_at = 0.01; N_at = int(T_at/dt_at)
burn_steps = int(120.0 / dt_at)
sqdt_at = np.sqrt(dt_at)

def run_tongue(with_noise, seed=42):
    rng = np.random.default_rng(seed)
    xa = np.ones(M); ya = np.zeros(M)
    sin_a = np.zeros(M); cos_a = np.zeros(M); cnt = 0
    for i in range(N_at):
        ti = i*dt_at
        r2 = xa*xa + ya*ya
        fx = (mu-r2)*xa - omega*ya + ep_flat*np.cos(Om_flat*ti)
        fy = (mu-r2)*ya + omega*xa
        xa += fx*dt_at; ya += fy*dt_at
        if with_noise:
            xa += sigma*rng.standard_normal(M)*sqdt_at
            ya += sigma*rng.standard_normal(M)*sqdt_at
        if i > burn_steps and i % 10 == 0:
            psi = np.arctan2(ya, xa) - Om_flat*ti
            cos_a += np.cos(psi); sin_a += np.sin(psi); cnt += 1
    return (np.sqrt(cos_a**2 + sin_a**2)/cnt).reshape(n_eps, n_dw)

print("Deterministic tongue ...", flush=True)
t0 = timer.time()
R_det = run_tongue(False)
print(f"  done ({timer.time()-t0:.0f} s)")

print("Stochastic tongue (3 seeds) ...", flush=True)
t0 = timer.time()
R_stoc = sum(run_tongue(True, s) for s in [42, 137, 256]) / 3.0
R_stoc = gaussian_filter(R_stoc, sigma=0.8)
print(f"  done ({timer.time()-t0:.0f} s)")

# Three zones
R_th = 0.3
det_on  = R_det  >= R_th
stoc_on = R_stoc >= R_th
zone = np.zeros_like(R_det)
zone[det_on & stoc_on]   = 2   # both
zone[~det_on & stoc_on]  = 1   # stochastic only

from matplotlib.colors import ListedColormap
cmap_zone = ListedColormap(["#F5F5F5", "#FDAE61", "#2166AC"])

fig, ax = plt.subplots(figsize=(9, 6.5))
fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.12)

ax.imshow(zone, origin="lower", aspect="auto",
          extent=[dw_arr[0], dw_arr[-1], eps_arr[0], eps_arr[-1]],
          cmap=cmap_zone, vmin=0, vmax=2, interpolation="nearest")

ax.contour(dw_arr, eps_arr, R_det,  levels=[R_th],
           colors=["#2166AC"], linewidths=2.5)
ax.contour(dw_arr, eps_arr, R_stoc, levels=[R_th],
           colors=["#D95319"], linewidths=2.5)

dw_line = np.linspace(-0.9, 0.9, 300)
ax.plot(dw_line, np.abs(dw_line)*r_star, "k--", lw=1.8)

ax.plot(Dw_demo, eps_demo, "*", color="red", ms=18,
        markeredgecolor="k", markeredgewidth=0.8, zorder=5)
ax.annotate("demo point",
            xy=(Dw_demo, eps_demo), xytext=(0.50, 0.10),
            fontsize=11, ha="center",
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.5),
            bbox=dict(fc="white", ec="0.6", pad=3, boxstyle="round,pad=0.3"))

handles = [
    Patch(fc="#2166AC", ec="k", lw=0.8,
          label="Deterministic + stochastic entrainment"),
    Patch(fc="#FDAE61", ec="k", lw=0.8,
          label=rf"Stochastic entrainment only ($\sigma = {sigma}$)"),
    Patch(fc="#F5F5F5", ec="k", lw=0.8, label="No entrainment"),
    plt.Line2D([],[],color="k",ls="--",lw=1.8,
               label=r"Analytical: $\varepsilon = |\Delta\omega|\, r^*$"),
]
ax.legend(handles=handles, loc="upper left", fontsize=10.5,
          framealpha=0.95, edgecolor="0.75")
ax.set_xlabel(r"Detuning $\Delta\omega = \Omega - \omega$")
ax.set_ylabel(r"Forcing amplitude $\varepsilon$")
ax.set_title("Stochastic widening of the Arnold tongue",
             fontsize=15, fontweight="bold")
ax.set_xlim(dw_arr[0], dw_arr[-1])
ax.set_ylim(eps_arr[0], eps_arr[-1])

fig.savefig("entrainment_arnold_tongue.svg", bbox_inches="tight")
fig.savefig("entrainment_arnold_tongue.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved entrainment_arnold_tongue.svg")
