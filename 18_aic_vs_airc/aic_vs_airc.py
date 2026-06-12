"""
AIC vs AIRC: disturbance on translation rate (gamma: 1 -> 2).
Protein jumps up -> Z2 accumulates -> rein kicks in (AIRC).
eta=100, k_d=50 (strong rein to matter at low z2).
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 14,
    "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"
PURPLE="#7E2F8E"; GRAY="0.55"
OUT = "/mnt/user-data/outputs"

k = 1.0; eta = 100.0; mu = 1.0; theta = 1.0
kd = 50.0
gamma_before, gamma_after = 1.0, 2.0
t_d = 50.0; T = 200.0

def rhs_aic(t, z, gam):
    m, p, z1, z2 = z
    return [k*z1 - m,
            gam*m - p,
            mu - eta*z1*z2,
            theta*p - eta*z1*z2]

def rhs_airc(t, z, gam):
    m, p, z1, z2 = z
    return [k*z1 - m,
            gam*m - p - kd*z2*p,
            mu - eta*z1*z2,
            theta*p - eta*z1*z2]

def solve_sys(rhs_fn, n_pts=1500):
    te1 = np.linspace(0, t_d, n_pts//2)
    te2 = np.linspace(t_d, T, n_pts//2)
    z0 = [0, 0, 0, 0]
    sol1 = solve_ivp(lambda t,z: rhs_fn(t, z, gamma_before),
                     [0, t_d], z0, t_eval=te1, rtol=1e-10)
    sol2 = solve_ivp(lambda t,z: rhs_fn(t, z, gamma_after),
                     [t_d, T], sol1.y[:,-1], t_eval=te2, rtol=1e-10)
    t_all = np.concatenate([sol1.t, sol2.t])
    y_all = np.concatenate([sol1.y, sol2.y], axis=1)
    return t_all, y_all

t_aic, y_aic = solve_sys(rhs_aic)
t_airc, y_airc = solve_sys(rhs_airc)

print(f"AIC  before: p={y_aic[1][len(y_aic[1])//2-1]:.4f}")
print(f"AIRC before: p={y_airc[1][len(y_airc[1])//2-1]:.4f}")
print(f"AIC  after:  p={y_aic[1][-1]:.4f}")
print(f"AIRC after:  p={y_airc[1][-1]:.4f}")
# Peak overshoot after disturbance
idx_d = len(y_aic[1])//2
print(f"AIC  peak after dist: {max(y_aic[1][idx_d:]):.4f}")
print(f"AIRC peak after dist: {max(y_airc[1][idx_d:]):.4f}")

def make_fig(t, y, ctrl_name, subtitle, fname):
    p = y[1]; z1 = y[2]; z2 = y[3]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.09,
                        hspace=0.15)
    fig.suptitle(f"{ctrl_name}: {subtitle}",
                 fontsize=15, fontweight="bold")
    # Top: protein
    ax1.plot(t, p, color=BLUE, lw=2.5, label=r"$p(t)$")
    ax1.axhline(mu/theta, color=GREEN, lw=2.0, ls="--",
                label=r"$\mu/\theta = 1$")
    ax1.axvline(t_d, color=GRAY, lw=1.5, ls=":", alpha=0.5)
    ax1.text(t_d+0.5, mu/theta+0.05, r"$\gamma: 1 \to 2$",
             fontsize=11, va="bottom", color="0.35")
    ax1.set_ylabel(r"Protein $p(t)$")
    ax1.legend(loc="upper right", framealpha=0.95, edgecolor="0.75")
    ax1.grid(True, alpha=0.15)
    # Bottom: controller species
    ax2.plot(t, z1, color=ORANGE, lw=2.5, label=r"$z_1(t)$ (actuator)")
    ax2.plot(t, z2, color=PURPLE, lw=2.5, label=r"$z_2(t)$ (sensor/rein)")
    ax2.axvline(t_d, color=GRAY, lw=1.5, ls=":", alpha=0.5)
    ax2.set_ylabel(r"Controller species")
    ax2.set_xlabel("Time $t$")
    ax2.legend(loc="upper right", framealpha=0.95, edgecolor="0.75")
    ax2.grid(True, alpha=0.15)
    ax2.set_xlim(0, T)
    fig.savefig(f"{OUT}/{fname}.svg", bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}.svg")

make_fig(t_aic, y_aic, "AIC",
         r"antithetic integral controller ($k_d = 0$)",
         "aic_controller")
make_fig(t_airc, y_airc, "AIRC",
         rf"antithetic integral rein controller ($k_d = {kd:.0f}$)",
         "airc_controller")
