"""
Vilar et al. (2002) circadian clock model.
Deterministic ODE vs stochastic SSA comparison.

Species: DA, DR (gene states, binary), MA, MR (mRNA), A, R (proteins), C (complex)
16 reactions modelling activator/repressor gene regulation with negative feedback.

Reference: Vilar et al., PNAS 99(9):5988-5992, 2002.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt
import time as timer

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 17,
    "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

# MATLAB colours
BLUE   = "#0072BD"
ORANGE = "#D95319"
GOLD   = "#EDB120"
PURPLE = "#7E2F8E"
GREEN  = "#77AC30"

# ── Parameters (Vilar et al. 2002, Table 1) ──────────────────
alphaA  = 50.0     # activated transcription, activator gene
alphaAp = 500.0    # basal transcription, activator gene
alphaR  = 0.01     # basal transcription, repressor gene
alphaRp = 50.0     # activated transcription, repressor gene
betaA   = 50.0     # translation rate, activator
betaR   = 5.0      # translation rate, repressor
deltaMA = 10.0     # mRNA degradation, activator
deltaMR = 0.5      # mRNA degradation, repressor
deltaA  = 1.0      # protein degradation, activator
deltaR  = 0.2      # protein degradation, repressor
gammaA  = 1.0      # A binding to activator promoter
thetaA  = 50.0     # A unbinding from activator promoter
gammaR  = 1.0      # A binding to repressor promoter
thetaR  = 100.0    # A unbinding from repressor promoter
gammaC  = 2.0      # complex formation A + R -> C

T_sim = 150.0      # total simulation time (hours)

# ── State vector: [DA, DR, MA, MR, A, R, C] ──────────────────
#    DA, DR in {0,1}; rest >= 0
#    DA' = 1-DA, DR' = 1-DR  (conservation)

# ── Deterministic ODE ─────────────────────────────────────────
def ode_rhs(t, y):
    DA, DR, MA, MR, A, R, C = y
    DAp = 1.0 - DA
    DRp = 1.0 - DR
    dDA = gammaA * A * DAp - thetaA * DA
    dDR = gammaR * A * DRp - thetaR * DR
    dMA = alphaA * DAp + alphaAp * DA - deltaMA * MA
    dMR = alphaR * DRp + alphaRp * DR - deltaMR * MR
    dA  = (betaA * MA - deltaA * A
           - gammaA * A * DAp + thetaA * DA
           - gammaR * A * DRp + thetaR * DR
           - gammaC * A * R)
    dR  = betaR * MR - deltaR * R - gammaC * A * R + deltaA * C
    dC  = gammaC * A * R - deltaA * C
    return [dDA, dDR, dMA, dMR, dA, dR, dC]

# Initial condition: genes inactive, small amount of activator
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

print("Solving deterministic ODE ...", flush=True)
t0 = timer.time()
sol = solve_ivp(ode_rhs, [0, T_sim], y0, method="LSODA",
                max_step=0.05, rtol=1e-8, atol=1e-10,
                dense_output=True)
tg = np.linspace(0, T_sim, 6000)
yg = sol.sol(tg)
DA_d, DR_d, MA_d, MR_d, A_d, R_d, C_d = yg
print(f"  done ({timer.time()-t0:.1f} s)")

# ── Stochastic SSA (Gillespie) ────────────────────────────────
# Stoichiometry: columns = reactions, rows = [DA,DR,MA,MR,A,R,C]
S = np.array([
 # R1  R2  R3  R4  R5  R6  R7  R8  R9  R10 R11 R12 R13 R14 R15 R16
 [+1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # DA
 [ 0,  0, +1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # DR
 [ 0,  0,  0,  0, +1, +1,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0], # MA
 [ 0,  0,  0,  0,  0,  0, +1, +1,  0,  0,  0, -1,  0,  0,  0,  0], # MR
 [-1, +1, -1, +1,  0,  0,  0,  0, +1,  0,  0,  0, -1,  0, -1,  0], # A
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0, +1,  0,  0,  0, -1, -1, +1], # R
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, +1, -1], # C
], dtype=int)

def gillespie_vilar(T, seed):
    rng = np.random.default_rng(seed)
    # Record at fixed intervals
    dt_rec = 0.05
    t_rec = np.arange(0, T + dt_rec, dt_rec)
    n_rec = len(t_rec)
    rec = np.zeros((n_rec, 7))

    state = np.array([0, 0, 0, 0, 0, 0, 0], dtype=int)
    t = 0.0
    idx = 0  # next recording index
    n_events = 0

    while t < T:
        # Record state at all passed recording times
        while idx < n_rec and t_rec[idx] <= t:
            rec[idx] = state
            idx += 1

        DA, DR, MA, MR, A, R, C = state
        DAp = 1 - DA
        DRp = 1 - DR

        # 16 propensities
        a = np.array([
            gammaA * A * DAp,     # R1
            thetaA * DA,          # R2
            gammaR * A * DRp,     # R3
            thetaR * DR,          # R4
            alphaA * DAp,         # R5
            alphaAp * DA,         # R6
            alphaR * DRp,         # R7
            alphaRp * DR,         # R8
            betaA * MA,           # R9
            betaR * MR,           # R10
            deltaMA * MA,         # R11
            deltaMR * MR,         # R12
            deltaA * A,           # R13
            deltaR * R,           # R14
            gammaC * A * R,       # R15
            deltaA * C,           # R16
        ], dtype=float)

        a_tot = a.sum()
        if a_tot <= 0:
            break

        tau = rng.exponential(1.0 / a_tot)
        t += tau

        # Select reaction
        r = rng.random() * a_tot
        cum = 0.0
        for j in range(16):
            cum += a[j]
            if r < cum:
                state = state + S[:, j]
                break
        n_events += 1

    # Fill remaining recording slots
    while idx < n_rec:
        rec[idx] = state
        idx += 1

    return t_rec, rec, n_events

print("Running SSA ...", flush=True)
t0 = timer.time()
t_ssa, rec_ssa, n_ev = gillespie_vilar(T_sim, seed=42)
DA_s, DR_s = rec_ssa[:,0], rec_ssa[:,1]
MA_s, MR_s = rec_ssa[:,2], rec_ssa[:,3]
A_s,  R_s  = rec_ssa[:,4], rec_ssa[:,5]
C_s        = rec_ssa[:,6]
print(f"  done ({timer.time()-t0:.1f} s, {n_ev:,} events)")

# ── Figure 1: Deterministic ──────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.10,
                    hspace=0.12)
fig.suptitle("Vilar et al. circadian clock — deterministic dynamics (RRE)",
             fontsize=14, fontweight="bold")

axes[0].plot(tg, A_d, color=BLUE,   label=r"Activator $A$")
axes[0].plot(tg, R_d, color=ORANGE, label=r"Repressor $R$")
axes[0].plot(tg, C_d, color=GREEN,  label=r"Complex $C$")
axes[0].set_ylabel("Protein count")
axes[0].legend(loc="upper right", ncol=3, framealpha=0.95, edgecolor="0.82")
axes[0].grid(True, alpha=0.2)

axes[1].plot(tg, MA_d, color=PURPLE, label=r"$M_A$  (activator mRNA)")
axes[1].plot(tg, MR_d, color=GOLD,   label=r"$M_R$  (repressor mRNA)")
axes[1].set_ylabel("mRNA count")
axes[1].set_xlabel("Time [h]")
axes[1].legend(loc="upper right", ncol=2, framealpha=0.95, edgecolor="0.82")
axes[1].grid(True, alpha=0.2)

for ax in axes:
    ax.set_xlim(0, T_sim)

fig.savefig("vilar_deterministic.svg", bbox_inches="tight")
fig.savefig("vilar_deterministic.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved vilar_deterministic.svg")

# ── Figure 2: Stochastic ─────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.10,
                    hspace=0.12)
fig.suptitle("Vilar et al. circadian clock — stochastic dynamics (SSA)",
             fontsize=14, fontweight="bold")

axes[0].plot(t_ssa, A_s, color=BLUE,   lw=1.2, alpha=0.85,
             label=r"Activator $A$")
axes[0].plot(t_ssa, R_s, color=ORANGE, lw=1.2, alpha=0.85,
             label=r"Repressor $R$")
axes[0].plot(t_ssa, C_s, color=GREEN,  lw=1.2, alpha=0.85,
             label=r"Complex $C$")
axes[0].set_ylabel("Protein count")
axes[0].legend(loc="upper right", ncol=3, framealpha=0.95, edgecolor="0.82")
axes[0].grid(True, alpha=0.2)

axes[1].plot(t_ssa, MA_s, color=PURPLE, lw=1.2, alpha=0.85,
             label=r"$M_A$  (activator mRNA)")
axes[1].plot(t_ssa, MR_s, color=GOLD,   lw=1.2, alpha=0.85,
             label=r"$M_R$  (repressor mRNA)")
axes[1].set_ylabel("mRNA count")
axes[1].set_xlabel("Time [h]")
axes[1].legend(loc="upper right", ncol=2, framealpha=0.95, edgecolor="0.82")
axes[1].grid(True, alpha=0.2)

for ax in axes:
    ax.set_xlim(0, T_sim)

fig.savefig("vilar_stochastic.svg", bbox_inches="tight")
fig.savefig("vilar_stochastic.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved vilar_stochastic.svg")

# ══════════════════════════════════════════════════════════════
#  NOISE-INDUCED OSCILLATIONS
#  alphaR raised from 0.01 -> 1.1 (past the Hopf bifurcation)
#  Deterministic: converges to a stable focus
#  Stochastic:    sustained noisy oscillations
# ══════════════════════════════════════════════════════════════
alphaR_new = 1.1

def ode_rhs_new(t, y):
    DA,DR,MA,MR,A,R,C = y
    return [
        gammaA*A*(1-DA) - thetaA*DA,
        gammaR*A*(1-DR) - thetaR*DR,
        alphaA*(1-DA) + alphaAp*DA - deltaMA*MA,
        alphaR_new*(1-DR) + alphaRp*DR - deltaMR*MR,
        betaA*MA - deltaA*A - gammaA*A*(1-DA) + thetaA*DA
          - gammaR*A*(1-DR) + thetaR*DR - gammaC*A*R,
        betaR*MR - deltaR*R - gammaC*A*R + deltaA*C,
        gammaC*A*R - deltaA*C,
    ]

T_long = 150.0

print("\n--- Noise-induced oscillations (alphaR = 1.1) ---")
print("Solving deterministic ODE ...", flush=True)
t0 = timer.time()
sol2 = solve_ivp(ode_rhs_new, [0, T_long], [0,0,0,0,0,0,0],
                 method="LSODA", max_step=0.05, rtol=1e-8, atol=1e-10,
                 dense_output=True)
tg2 = np.linspace(0, T_long, 8000)
yg2 = sol2.sol(tg2)
print(f"  done ({timer.time()-t0:.1f} s)")
print(f"  steady state: A={yg2[4,-1]:.1f}, R={yg2[5,-1]:.0f}, C={yg2[6,-1]:.0f}")

print("Running SSA ...", flush=True)
t0 = timer.time()

# SSA with modified alphaR
def gillespie_vilar2(T, seed):
    rng = np.random.default_rng(seed)
    dt_rec = 0.05
    t_rec = np.arange(0, T + dt_rec, dt_rec)
    n_rec = len(t_rec)
    rec = np.zeros((n_rec, 7))
    state = np.array([0,0,0,0,0,0,0], dtype=int)
    t, idx, n_events = 0.0, 0, 0
    while t < T:
        while idx < n_rec and t_rec[idx] <= t:
            rec[idx] = state; idx += 1
        DA,DR,MA,MR,A,R,C = state
        a = np.array([
            gammaA*A*(1-DA), thetaA*DA,
            gammaR*A*(1-DR), thetaR*DR,
            alphaA*(1-DA), alphaAp*DA,
            alphaR_new*(1-DR), alphaRp*DR,
            betaA*MA, betaR*MR,
            deltaMA*MA, deltaMR*MR,
            deltaA*A, deltaR*R,
            gammaC*A*R, deltaA*C,
        ], dtype=float)
        a_tot = a.sum()
        if a_tot <= 0: break
        t += rng.exponential(1.0/a_tot)
        r = rng.random()*a_tot
        cum = 0.0
        for j in range(16):
            cum += a[j]
            if r < cum: state = state + S[:,j]; break
        n_events += 1
    while idx < n_rec: rec[idx] = state; idx += 1
    return t_rec, rec, n_events

t_s2, rec2, nev2 = gillespie_vilar2(T_long, seed=42)
print(f"  done ({timer.time()-t0:.1f} s, {nev2:,} events)")

# ── Figure 3: Deterministic (converging) ──────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.10,
                    hspace=0.12)
fig.suptitle(
    r"Vilar clock, $\alpha_R = 1.1$ (past Hopf bifurcation)"
    r" — deterministic: convergence",
    fontsize=13, fontweight="bold")

axes[0].plot(tg2, yg2[4], color=BLUE,   label=r"Activator $A$")
axes[0].plot(tg2, yg2[5], color=ORANGE, label=r"Repressor $R$")
axes[0].plot(tg2, yg2[6], color=GREEN,  label=r"Complex $C$")
axes[0].set_ylabel("Protein count")
axes[0].legend(loc="upper right", ncol=3, framealpha=0.95, edgecolor="0.82")
axes[0].grid(True, alpha=0.2)

axes[1].plot(tg2, yg2[2], color=PURPLE, label=r"$M_A$  (activator mRNA)")
axes[1].plot(tg2, yg2[3], color=GOLD,   label=r"$M_R$  (repressor mRNA)")
axes[1].set_ylabel("mRNA count")
axes[1].set_xlabel("Time [h]")
axes[1].legend(loc="upper right", ncol=2, framealpha=0.95, edgecolor="0.82")
axes[1].grid(True, alpha=0.2)
for ax in axes: ax.set_xlim(0, T_long)

fig.savefig("vilar_deterministic_convergent.svg", bbox_inches="tight")
fig.savefig("vilar_deterministic_convergent.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved vilar_deterministic_convergent.svg")

# ── Figure 4: Stochastic (still oscillating!) ─────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.10,
                    hspace=0.12)
fig.suptitle(
    r"Vilar clock, $\alpha_R = 1.1$ (past Hopf bifurcation)"
    r" — stochastic: noise-induced oscillations",
    fontsize=13, fontweight="bold")

axes[0].plot(t_s2, rec2[:,4], color=BLUE,   lw=1.2, alpha=0.85,
             label=r"Activator $A$")
axes[0].plot(t_s2, rec2[:,5], color=ORANGE, lw=1.2, alpha=0.85,
             label=r"Repressor $R$")
axes[0].plot(t_s2, rec2[:,6], color=GREEN,  lw=1.2, alpha=0.85,
             label=r"Complex $C$")
axes[0].set_ylabel("Protein count")
axes[0].legend(loc="upper right", ncol=3, framealpha=0.95, edgecolor="0.82")
axes[0].grid(True, alpha=0.2)

axes[1].plot(t_s2, rec2[:,2], color=PURPLE, lw=1.2, alpha=0.85,
             label=r"$M_A$  (activator mRNA)")
axes[1].plot(t_s2, rec2[:,3], color=GOLD,   lw=1.2, alpha=0.85,
             label=r"$M_R$  (repressor mRNA)")
axes[1].set_ylabel("mRNA count")
axes[1].set_xlabel("Time [h]")
axes[1].legend(loc="upper right", ncol=2, framealpha=0.95, edgecolor="0.82")
axes[1].grid(True, alpha=0.2)
for ax in axes: ax.set_xlim(0, T_long)

fig.savefig("vilar_stochastic_oscillating.svg", bbox_inches="tight")
fig.savefig("vilar_stochastic_oscillating.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved vilar_stochastic_oscillating.svg")
