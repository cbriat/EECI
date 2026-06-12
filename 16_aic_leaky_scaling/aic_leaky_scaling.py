import numpy as np
from scipy.optimize import fsolve
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 15,
    "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"; PURPLE="#7E2F8E"
OUT = "/mnt/user-data/outputs"

gc = 0.5; g = 1.0  # DC gain of plant (gamma/(delta_m*delta_p))
n_pts = 80
s_vals = np.logspace(0, 3, n_pts)

def find_ss(k, mu, theta, eta):
    def eqs(z):
        m, p, z1, z2 = z
        return [k*z1 - m, m - p,
                mu - eta*z1*z2 - gc*z1,
                theta*p - eta*z1*z2 - gc*z2]
    sol = fsolve(eqs, [max(0.1,k*mu/(theta*g*k+gc)),
                       max(0.1,g*k*mu/(theta*g*k+gc)),
                       max(0.01, mu/(theta*g*k+gc)),
                       max(0.01, mu/(eta*mu/(theta*g*k+gc)))],
                 full_output=True)
    return sol[0][1]

# ── Top: scale eta = mu = theta = s, k=1 ─────────────────────
p_top = np.array([find_ss(k=1, mu=s, theta=s, eta=s) for s in s_vals])
p_top_formula = 1 / (1 + gc/(g*1))  # limit: independent of s

# ── Bottom: scale eta and k = s, mu=theta=1 ──────────────────
# Need eta >> k for the fast-sequestration limit to hold.
# Use eta = s^2 so that eta/k = s >> 1.
p_bot = np.array([find_ss(k=s, mu=1, theta=1, eta=s**2) for s in s_vals])

# Analytical prediction from fast-sequestration limit:
# p* = g*k*mu/(theta*g*k + gc) = g*s/(g*s + gc) = s/(s+0.5)
p_bot_formula = g * s_vals / (g * s_vals + gc)

# Also show the WRONG scaling (eta = k = s) for contrast
p_bot_wrong = np.array([find_ss(k=s, mu=1, theta=1, eta=s) for s in s_vals])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.subplots_adjust(left=0.11, right=0.96, top=0.92, bottom=0.10,
                    hspace=0.22)
fig.suptitle(
    rf"Leaky AIC (deterministic, $\gamma_c = {gc}$, $g = {g:.0f}$): "
    r"two strategies to recover adaptation",
    fontsize=14, fontweight="bold")

# ── Top: scale flux ──────────────────────────────────────────
ax1.semilogx(s_vals, p_top, color=BLUE, lw=2.5,
             label=r"$\eta\!=\!\mu\!=\!\theta\!=\!s$, $k\!=\!1$")
ax1.axhline(1.0, color=GREEN, lw=2.0, ls="--",
            label=r"Set-point $\mu/\theta = 1$")
ax1.set_ylabel(r"$p^*$")
ax1.set_title(
    r"Scale controller flux ($\eta, \mu, \theta$): "
    r"$p^* = \mu/(\theta + \gamma_c/(gk)) \to 1$",
    fontsize=12)
ax1.legend(loc="lower right", framealpha=0.95, edgecolor="0.75")
ax1.grid(True, alpha=0.15)
ax1.set_ylim(0.8, 1.05)

# ── Bottom: scale gain ───────────────────────────────────────
ax2.semilogx(s_vals, p_bot, color=ORANGE, lw=2.5,
             label=r"$k\!=\!s$, $\eta\!=\!s^2$  ($\mu\!=\!\theta\!=\!1$)")
ax2.semilogx(s_vals, p_bot_formula, color=ORANGE, ls=":", lw=1.5,
             alpha=0.6,
             label=r"$p^* = gk\mu/(\theta gk + \gamma_c)$")
ax2.semilogx(s_vals, p_bot_wrong, color=PURPLE, lw=2.0, ls="--",
             label=r"$k\!=\!\eta\!=\!s$ (too slow: $\eta$ not $\gg k$)")
ax2.axhline(1.0, color=GREEN, lw=2.0, ls="--",
            label=r"Set-point $\mu/\theta = 1$")
ax2.set_ylabel(r"$p^*$")
ax2.set_xlabel(r"Scaling factor $s$")
ax2.set_title(
    r"Scale loop gain ($\eta, k$): "
    r"$p^* = \mu/(\theta + \gamma_c/(gk)) \to 1$ "
    r"(requires $\eta \gg k$)",
    fontsize=12)
ax2.legend(loc="center right", framealpha=0.95, edgecolor="0.75")
ax2.grid(True, alpha=0.15)
ax2.set_ylim(0.8, 1.45)

fig.savefig(f"{OUT}/aic_leaky_scaling.svg", bbox_inches="tight")
fig.savefig("/home/claude/aic_leaky_scaling.png", dpi=200,
            bbox_inches="tight")
plt.close()
print("Saved aic_leaky_scaling.svg")
print(f"Top  at s=1000: p*={p_top[-1]:.6f}")
print(f"Bot (eta=s^2) at s=1000: p*={p_bot[-1]:.6f}")
print(f"Bot (eta=s)   at s=1000: p*={p_bot_wrong[-1]:.6f}")
