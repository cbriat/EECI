"""
AIC controlling a gene-expression cascade (mRNA → protein).

Plant:
  ∅ → M   rate θ Z₁        (actuation)
  M → ∅   rate δ_M M        (mRNA degradation)
  M → M+X rate β M           (translation)
  X → ∅   rate γ X           (protein degradation)

Controller:
  ∅ → Z₁  rate μ             (reference)
  ∅ → Z₂  rate k X           (measurement)
  Z₁+Z₂→∅ rate η Z₁ Z₂      (annihilation)

At η = 3.5 the equilibrium is an unstable spiral (past Hopf at η ≈ 3.0).
Deterministic oscillations grow; stochastic moments converge.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib, matplotlib.pyplot as plt
import time as timer

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 16,
    "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"
PURPLE="#7E2F8E"; GOLD="#EDB120"

# ── Parameters ────────────────────────────────────────────────
theta  = 5.0     # actuation gain
delta_M= 1.0     # mRNA degradation
beta   = 2.0     # translation rate
gamma  = 0.5     # protein degradation
mu     = 10.0    # reference
k_m    = 1.0     # measurement gain
eta    = 3.5     # annihilation rate (past Hopf ≈ 3.0 → unstable)

x_ss  = mu / k_m
m_ss  = gamma * x_ss / beta
z1_ss = delta_M * m_ss / theta
z2_ss = mu / (eta * z1_ss)
print(f"Setpoint: x* = {x_ss:.0f},  m* = {m_ss:.1f}")
print(f"Controller: z1* = {z1_ss:.2f},  z2* = {z2_ss:.1f}")

# Eigenvalues
p = eta * z2_ss; q = eta * z1_ss
J = np.array([[-delta_M, 0, theta, 0],
              [beta, -gamma, 0, 0],
              [0, 0, -p, -q],
              [0, k_m, -p, -q]])
eigs = np.linalg.eigvals(J)
cpx = sorted([e for e in eigs if abs(e.imag) > 0.01],
             key=lambda e: abs(e.real))
print(f"Oscillatory eigenvalues: {cpx[0].real:+.4f} ± {abs(cpx[0].imag):.3f}i")

T_sim = 150.0

# ── Deterministic ODE ─────────────────────────────────────────
def ode(t, y):
    M, X, Z1, Z2 = y
    return [theta*Z1 - delta_M*M,
            beta*M - gamma*X,
            mu - eta*Z1*Z2,
            k_m*X - eta*Z1*Z2]

sol = solve_ivp(ode, [0, T_sim], [0, 0, 0, 0],
                max_step=0.02, rtol=1e-9, dense_output=True)
tg = np.linspace(0, T_sim, 8000)
Md, Xd, Z1d, Z2d = sol.sol(tg)

# ── Stochastic SSA ────────────────────────────────────────────
def ssa_aic(T, seed):
    rng = np.random.default_rng(seed)
    dt_rec = 0.1
    t_rec = np.arange(0, T + dt_rec, dt_rec)
    n_rec = len(t_rec)
    rec = np.zeros((n_rec, 4))
    M, X, Z1, Z2 = 0, 0, 0, 0
    t, idx, n_ev = 0.0, 0, 0
    while t < T:
        while idx < n_rec and t_rec[idx] <= t:
            rec[idx] = [M, X, Z1, Z2]; idx += 1
        a = np.array([theta*Z1,       # ∅→M
                       delta_M*M,      # M→∅
                       beta*M,         # M→M+X
                       gamma*X,        # X→∅
                       mu,             # ∅→Z₁
                       k_m*X,          # ∅→Z₂
                       eta*Z1*Z2],     # Z₁+Z₂→∅
                      dtype=float)
        a_tot = a.sum()
        if a_tot <= 0: t += 0.1; continue
        t += rng.exponential(1.0 / a_tot)
        r = rng.random() * a_tot
        cum = 0.0
        for j in range(7):
            cum += a[j]
            if r < cum:
                if   j == 0: M  += 1
                elif j == 1: M  -= 1
                elif j == 2: X  += 1
                elif j == 3: X  -= 1
                elif j == 4: Z1 += 1
                elif j == 5: Z2 += 1
                elif j == 6: Z1 -= 1; Z2 -= 1
                break
        n_ev += 1
    while idx < n_rec: rec[idx] = [M, X, Z1, Z2]; idx += 1
    return t_rec, rec, n_ev

print("Running SSA (3 paths) ...", flush=True)
t0 = timer.time()
paths = []
for s in [42, 137, 256]:
    t_s, rec_s, nev = ssa_aic(T_sim, s)
    paths.append((t_s, rec_s))
    print(f"  seed {s}: {nev:,} events")
print(f"  done ({timer.time()-t0:.1f} s)")

# ── Figure ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.10,
                    hspace=0.18)
fig.suptitle(
    r"Antithetic Integral Controller:  $x^* = \mu/k = "
    f"{x_ss:.0f}$  (perfect adaptation)",
    fontsize=14, fontweight="bold")

# Top: deterministic
ax1.plot(tg, Xd, color=BLUE, lw=2.0, label=r"$X(t)$")
ax1.axhline(x_ss, color="k", ls="--", lw=1.2, alpha=0.5,
            label=rf"Setpoint $\mu/k = {x_ss:.0f}$")
ax1.set_ylabel("Protein count")
ax1.set_title(
    r"Deterministic: growing oscillations (unstable spiral, "
    rf"$\mathrm{{Re}}\,\lambda = +{cpx[0].real:.3f}$)",
    fontsize=12)
ax1.legend(loc="upper left", framealpha=0.95, edgecolor="0.82")
ax1.grid(True, alpha=0.15)

# Bottom: stochastic
colors_s = [ORANGE, GREEN, PURPLE]
for i, (t_s, rec_s) in enumerate(paths):
    lbl = r"$X(t)$ SSA" if i == 0 else None
    ax2.plot(t_s, rec_s[:, 1], color=colors_s[i], lw=0.9, alpha=0.8,
             label=lbl)
ax2.axhline(x_ss, color="k", ls="--", lw=1.2, alpha=0.5,
            label=rf"Setpoint $\mu/k = {x_ss:.0f}$")
ax2.set_ylabel("Protein count")
ax2.set_xlabel("Time")
ax2.set_title(
    r"Stochastic: bounded fluctuations, moments converge",
    fontsize=12)
ax2.legend(loc="upper left", framealpha=0.95, edgecolor="0.82")
ax2.grid(True, alpha=0.15)

for ax in (ax1, ax2):
    ax.set_xlim(0, T_sim)
    ax.set_ylim(bottom=-0.5)

fig.savefig("aic_birth_death.svg", bbox_inches="tight")
fig.savefig("aic_birth_death.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved aic_birth_death.svg")
