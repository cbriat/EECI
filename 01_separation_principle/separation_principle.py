"""
Separation principle illustration.

Plant: unstable 2nd-order system.
State-feedback: pole placement for desired closed-loop dynamics.
Observer: Luenberger observer with poles at three speeds:
  - Too slow → sluggish, large transient
  - Borderline → moderate
  - Fast enough → nearly matches ideal state feedback

Figure: output y(t) and estimation error ||e(t)|| for each case,
with the ideal full-state-feedback response as reference.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import place_poles
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 16,
    "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GOLD="#EDB120"
PURPLE="#7E2F8E"; GREEN="#77AC30"

# ── Plant ─────────────────────────────────────────────────────
A = np.array([[0, 1],
              [3, 0.5]])    # unstable: eigs ≈ +1.86, -1.36
B = np.array([[0], [1]])
C = np.array([[1, 0]])
n = 2

eigs_plant = np.linalg.eigvals(A)
print(f"Plant eigenvalues: {eigs_plant}")

# ── State-feedback gain K ─────────────────────────────────────
cl_poles = np.array([-3+2j, -3-2j])
K = place_poles(A, B, cl_poles).gain_matrix
Acl = A - B @ K
print(f"Controller poles: {cl_poles}")
print(f"K = {K}")

# ── Observer gains L for three speeds ─────────────────────────
observer_cases = {
    "Slow":       np.array([-1.0, -1.5]),
    "Borderline": np.array([-3.0, -4.0]),
    "Fast":       np.array([-15.0, -20.0]),
}

Ls = {}
for name, obs_poles in observer_cases.items():
    # L via duality: place_poles(A^T, C^T, poles) → L = result^T
    res = place_poles(A.T, C.T, obs_poles)
    L = res.gain_matrix.T
    Ls[name] = L
    eig_check = np.linalg.eigvals(A - L @ C)
    print(f"Observer '{name}': poles = {obs_poles}, "
          f"check eigs = {eig_check.real.round(3)}")

# ── Simulation ────────────────────────────────────────────────
T = 6.0
x0 = np.array([1.0, 0.0])
xhat0 = np.array([0.0, 0.0])
e0 = x0 - xhat0

tg = np.linspace(0, T, 2000)

# 1) Ideal state feedback: dx/dt = (A-BK)x
def ideal_rhs(t, x):
    return (Acl @ x.reshape(-1,1)).flatten()

sol_ideal = solve_ivp(ideal_rhs, [0, T], x0, t_eval=tg, rtol=1e-10)
y_ideal = (C @ sol_ideal.y).flatten()

# 2) Output feedback for each observer speed
# Augmented state: [x1, x2, e1, e2]
# dx/dt = (A-BK)x + BK e
# de/dt = (A-LC) e
results = {}
for name, L in Ls.items():
    Aobs = A - L @ C
    def aug_rhs(t, z, Acl=Acl, B=B, K=K, Aobs=Aobs):
        x = z[:2]; e = z[2:]
        dx = Acl @ x + (B @ K @ e.reshape(-1,1)).flatten()
        de = Aobs @ e
        return np.concatenate([dx, de])

    z0 = np.concatenate([x0, e0])
    sol = solve_ivp(aug_rhs, [0, T], z0, t_eval=tg, rtol=1e-10)
    y_out = (C @ sol.y[:2]).flatten()
    e_norm = np.sqrt(sol.y[2]**2 + sol.y[3]**2)
    results[name] = (y_out, e_norm)

# ── Figure ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.09,
                    hspace=0.15)
fig.suptitle("Separation principle: observer speed vs. closed-loop performance",
             fontsize=14, fontweight="bold")

colors = {"Slow": ORANGE, "Borderline": GOLD, "Fast": GREEN}
obs_poles_str = {
    "Slow": r"$-1,\;-1.5$",
    "Borderline": r"$-3,\;-4$",
    "Fast": r"$-15,\;-20$",
}

# Top: output y(t)
ax1.plot(tg, y_ideal, color=BLUE, lw=2.5, ls="--",
         label=r"Ideal state feedback")
for name in ["Slow", "Borderline", "Fast"]:
    y_out, _ = results[name]
    ax1.plot(tg, y_out, color=colors[name], lw=2.0,
             label=f"Observer: {name.lower()} ({obs_poles_str[name]})")

ax1.axhline(0, color="0.5", lw=0.5)
ax1.set_ylabel(r"Output  $y(t) = C x(t)$")
ax1.set_title(
    r"Output response  (controller poles at $-3 \pm 2j$)",
    fontsize=13)
ax1.legend(loc="upper right", framealpha=0.95, edgecolor="0.75",
           fontsize=10)
ax1.grid(True, alpha=0.15)

# Bottom: estimation error
for name in ["Slow", "Borderline", "Fast"]:
    _, e_norm = results[name]
    ax1_twin = ax2
    ax2.plot(tg, e_norm, color=colors[name], lw=2.0,
             label=f"{name} ({obs_poles_str[name]})")

ax2.axhline(0, color="0.5", lw=0.5)
ax2.set_ylabel(r"$\|e(t)\| = \|x(t) - \hat{x}(t)\|$")
ax2.set_xlabel(r"Time $t$")
ax2.set_title("Observer estimation error", fontsize=13)
ax2.legend(loc="upper right", framealpha=0.95, edgecolor="0.75",
           fontsize=10)
ax2.grid(True, alpha=0.15)
ax2.set_xlim(0, T)
emax = max(results[n][1].max() for n in results)
ax2.set_ylim(-0.05 * emax, emax * 1.1)

fig.savefig("separation_principle.svg", bbox_inches="tight")
fig.savefig("separation_principle.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved separation_principle.svg")
