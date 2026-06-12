import numpy as np
import subprocess, time
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 15,
    "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"
PURPLE="#7E2F8E"; GOLD="#EDB120"; GRAY="0.55"
OUT = "/mnt/user-data/outputs"

k = 1.0
eta = 100.0
ngc = 50
nev = 3_000_000
burn = 500_000

# Sweep gc from 0 to 2 (linear, with 0 included)
gc_vals = np.concatenate([[0], np.linspace(0.01, 2.0, ngc-1)])

inp = " ".join(f"{v:.8f}" for v in gc_vals) + "\n"

print(f"k={k}, eta={eta}, {ngc} gamma_c values, {nev/1e6:.0f}M events each")
t0 = time.time()
proc = subprocess.run(
    ["/home/claude/ssa_leaky", str(k), str(eta),
     str(ngc), str(nev), str(burn)],
    input=inp, capture_output=True, text=True, timeout=120)
print(f"SSA done in {time.time()-t0:.1f}s")

lines = proc.stdout.strip().split("\n")
mean_p = np.zeros(ngc)
mean_z1 = np.zeros(ngc)
mean_z2 = np.zeros(ngc)
for i, line in enumerate(lines):
    vals = [float(x) for x in line.split()]
    mean_p[i], mean_z1[i], mean_z2[i] = vals

# Deterministic steady-state prediction (approximate)
# From dz1/dt = mu - eta*z1*z2 - gc*z1 = 0
# and dz2/dt = theta*p - eta*z1*z2 - gc*z2 = 0
# => mu - theta*p = gc*(z1 - z2)
# For small gc: p ≈ mu/theta - gc*(z1-z2)/theta
# But z1, z2 depend on gc too... just show the SSA result.

setpoint = 1.0  # mu/theta

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.subplots_adjust(left=0.11, right=0.96, top=0.92, bottom=0.10,
                    hspace=0.18)
fig.suptitle(
    r"Leaky AIC: effect of controller dilution $\gamma_c$ on "
    r"steady-state protein  ($k\!=\!1, \eta\!=\!100$)",
    fontsize=14, fontweight="bold")

# Top: E[P] vs gamma_c
ax1.plot(gc_vals, mean_p, "o-", color=BLUE, ms=4, lw=2.0,
         label=r"$E[P]$ (SSA)")
ax1.axhline(setpoint, color=GREEN, lw=2.0, ls="--",
            label=r"Set-point $\mu/\theta = 1$")
ax1.fill_between(gc_vals, setpoint, mean_p, alpha=0.15, color=ORANGE,
                 label="Steady-state error")
ax1.set_ylabel(r"$E[P]$")
ax1.set_title("Mean protein at stationarity", fontsize=13)
ax1.legend(loc="lower left", framealpha=0.95, edgecolor="0.75")
ax1.grid(True, alpha=0.15)

# Bottom: relative error and controller species
ax2.plot(gc_vals, np.abs(mean_p - setpoint) / setpoint * 100,
         "o-", color=ORANGE, ms=4, lw=2.0,
         label=r"$|E[P] - p^*|/p^*$ (\%)")
ax2.set_ylabel("Relative SS error (%)")
ax2.set_xlabel(r"Controller dilution rate $\gamma_c$")
ax2.set_title("Steady-state error grows with controller leakage",
              fontsize=13)
ax2.legend(loc="upper left", framealpha=0.95, edgecolor="0.75")
ax2.grid(True, alpha=0.15)

fig.savefig(f"{OUT}/aic_leaky.svg", bbox_inches="tight")
fig.savefig("/home/claude/aic_leaky.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved aic_leaky.svg")
print(f"E[P] at gc=0: {mean_p[0]:.4f}")
print(f"E[P] at gc=2: {mean_p[-1]:.4f}")
print(f"Relative error at gc=2: {abs(mean_p[-1]-1)*100:.1f}%")
