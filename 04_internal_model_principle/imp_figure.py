"""
Internal Model Principle — two consistent figures.
Plant G(s) = 5/((s+1)(s+3)) throughout.

Figure 1: REFERENCE TRACKING (dashed = reference to follow)
  step, ramp, sinusoid — with/without matching internal model.

Figure 2: DISTURBANCE REJECTION (dashed = disturbance to reject)
  step, ramp, sinusoid — with/without matching internal model.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 13,
    "axes.labelsize": 14, "axes.titlesize": 14,
    "legend.fontsize": 9.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "lines.linewidth": 2.2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GRAY="0.55"
OUT = "/mnt/user-data/outputs"

# Plant: x1'=x2, x2'=-3x1-4x2+5u, y=x1
ap1, ap2, bp = -3.0, -4.0, 5.0
T = 15.0; t0_sig = 0.5
te = np.linspace(0, T, 3000)

# Signal generators
def step(t):  return 1.0 if t >= t0_sig else 0.0
def ramp(t):  return max(0, t - t0_sig)
w_d = 2.0
def sine(t):  return np.sin(w_d*t) if t >= t0_sig else 0.0

# Controller gains
Kp, Ki = 3.0, 2.0
Kp2, Ki2, Kii2 = 4.0, 4.0, 1.0
Kr = 5.0

# ── Simulation helpers ────────────────────────────────────────
def sim(rhs, n_states):
    sol = solve_ivp(rhs, [0,T], [0]*n_states, t_eval=te, rtol=1e-9)
    return sol.y[0]  # output y = x1

# ══════════════════════════════════════════════════════════════
#  FIGURE 1: REFERENCE TRACKING
# ══════════════════════════════════════════════════════════════

# Step: P vs PI
y_P_step = sim(lambda t,z: [z[1], ap1*z[0]+ap2*z[1]+bp*Kp*(step(t)-z[0])], 2)
y_PI_step = sim(lambda t,z: [z[1], ap1*z[0]+ap2*z[1]+bp*(Kp*(step(t)-z[0])+Ki*z[2]),
                              step(t)-z[0]], 3)

# Ramp: PI vs PII
y_PI_ramp = sim(lambda t,z: [z[1], ap1*z[0]+ap2*z[1]+bp*(Kp*(ramp(t)-z[0])+Ki*z[2]),
                              ramp(t)-z[0]], 3)
y_PII_ramp = sim(lambda t,z: [z[1],
    ap1*z[0]+ap2*z[1]+bp*(Kp2*(ramp(t)-z[0])+Ki2*z[2]+Kii2*z[3]),
    ramp(t)-z[0], z[2]], 4)

# Sinusoid: PI vs PI+resonant
y_PI_sine = sim(lambda t,z: [z[1], ap1*z[0]+ap2*z[1]+bp*(Kp*(sine(t)-z[0])+Ki*z[2]),
                              sine(t)-z[0]], 3)
y_PIR_sine = sim(lambda t,z: [z[1],
    ap1*z[0]+ap2*z[1]+bp*(Kp*(sine(t)-z[0])+Ki*z[2]+z[3]),
    sine(t)-z[0], w_d*z[4]+Kr*(sine(t)-z[0]), -w_d*z[3]], 5)

ref_step = np.array([step(t) for t in te])
ref_ramp = np.array([ramp(t) for t in te])
ref_sine = np.array([sine(t) for t in te])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.subplots_adjust(left=0.05, right=0.98, top=0.78, bottom=0.13, wspace=0.28)
fig.suptitle("Internal Model Principle: reference tracking",
             fontsize=15, fontweight="bold")

panels_track = [
    (ref_step, y_P_step, y_PI_step,
     "Step tracking",
     r"Exosystem: $\dot{w}\!=\!0$  (pole at $s\!=\!0$)",
     r"P only", r"PI: $1/s$ in loop", (-0.1, 1.4)),
    (ref_ramp, y_PI_ramp, y_PII_ramp,
     "Ramp tracking",
     r"Exosystem: $\ddot{w}\!=\!0$  (two poles at $s\!=\!0$)",
     r"PI: $1/s$", r"PII: $1/s^2$ in loop", (-0.5, ref_ramp[-1]*1.1)),
    (ref_sine, y_PI_sine, y_PIR_sine,
     "Sinusoidal tracking",
     r"Exosystem: $\ddot{w}\!+\!\omega^2 w\!=\!0$  (poles at $\pm j\omega$)",
     r"PI only", r"PI + resonant $s/(s^2\!+\!\omega^2)$", (-1.5, 1.5)),
]

for ax, (ref, y_bad, y_good, title, sub, lbl_bad, lbl_good, ylims) in \
        zip(axes, panels_track):
    ax.plot(te, ref, color=GRAY, ls="--", lw=1.5, label="Reference $r(t)$")
    ax.plot(te, y_bad, color=ORANGE, label=lbl_bad + " (SS error)")
    ax.plot(te, y_good, color=BLUE, label=lbl_good)
    ax.set_title(f"{title}\n{sub}", fontsize=11.5)
    ax.set_xlabel("Time"); ax.set_ylabel(r"$y(t)$")
    ax.set_xlim(0, T); ax.set_ylim(*ylims)
    ax.legend(loc="best", framealpha=0.95, edgecolor="0.75", fontsize=9)
    ax.grid(True, alpha=0.15)

fig.savefig(f"{OUT}/imp_tracking.svg", bbox_inches="tight")
plt.close()
print("Saved imp_tracking.svg")

# ══════════════════════════════════════════════════════════════
#  FIGURE 2: DISTURBANCE REJECTION
# ══════════════════════════════════════════════════════════════

# Step disturbance at plant input, regulate y→0
y_P_dstep = sim(lambda t,z: [z[1],
    ap1*z[0]+ap2*z[1]+bp*(Kp*(-z[0])+step(t))], 2)
y_PI_dstep = sim(lambda t,z: [z[1],
    ap1*z[0]+ap2*z[1]+bp*(Kp*(-z[0])+Ki*z[2]+step(t)),
    -z[0]], 3)

# Ramp disturbance
y_PI_dramp = sim(lambda t,z: [z[1],
    ap1*z[0]+ap2*z[1]+bp*(Kp*(-z[0])+Ki*z[2]+ramp(t)),
    -z[0]], 3)
y_PII_dramp = sim(lambda t,z: [z[1],
    ap1*z[0]+ap2*z[1]+bp*(Kp2*(-z[0])+Ki2*z[2]+Kii2*z[3]+ramp(t)),
    -z[0], z[2]], 4)

# Sinusoidal disturbance
y_PI_dsine = sim(lambda t,z: [z[1],
    ap1*z[0]+ap2*z[1]+bp*(Kp*(-z[0])+Ki*z[2]+0.5*sine(t)),
    -z[0]], 3)
y_PIR_dsine = sim(lambda t,z: [z[1],
    ap1*z[0]+ap2*z[1]+bp*(Kp*(-z[0])+Ki*z[2]+z[3]+0.5*sine(t)),
    -z[0], w_d*z[4]+Kr*(-z[0]), -w_d*z[3]], 5)

d_step = np.array([step(t) for t in te])
d_ramp = np.array([ramp(t) for t in te])
d_sine = np.array([0.5*sine(t) for t in te])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.subplots_adjust(left=0.05, right=0.98, top=0.78, bottom=0.13, wspace=0.28)
fig.suptitle("Internal Model Principle: disturbance rejection",
             fontsize=15, fontweight="bold")

panels_reject = [
    (d_step, y_P_dstep, y_PI_dstep,
     "Step disturbance",
     r"Exosystem: $\dot{d}\!=\!0$  (pole at $s\!=\!0$)",
     r"P only", r"PI: $1/s$ in loop"),
    (d_ramp, y_PI_dramp, y_PII_dramp,
     "Ramp disturbance",
     r"Exosystem: $\ddot{d}\!=\!0$  (two poles at $s\!=\!0$)",
     r"PI: $1/s$", r"PII: $1/s^2$ in loop"),
    (d_sine, y_PI_dsine, y_PIR_dsine,
     "Sinusoidal disturbance",
     r"Exosystem: $\ddot{d}\!+\!\omega^2 d\!=\!0$  (poles at $\pm j\omega$)",
     r"PI only", r"PI + resonant $s/(s^2\!+\!\omega^2)$"),
]

for ax, (d_sig, y_bad, y_good, title, sub, lbl_bad, lbl_good) in \
        zip(axes, panels_reject):
    ax.plot(te, d_sig, color=GRAY, ls="--", lw=1.5, alpha=0.7,
            label="Disturbance $d(t)$")
    ax.plot(te, y_bad, color=ORANGE, label=lbl_bad + " (persistent)")
    ax.plot(te, y_good, color=BLUE, label=lbl_good)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_title(f"{title}\n{sub}", fontsize=11.5)
    ax.set_xlabel("Time"); ax.set_ylabel(r"$y(t)$")
    ax.set_xlim(0, T)
    ax.legend(loc="best", framealpha=0.95, edgecolor="0.75", fontsize=9)
    ax.grid(True, alpha=0.15)

fig.savefig(f"{OUT}/imp_rejection.svg", bbox_inches="tight")
plt.close()
print("Saved imp_rejection.svg")
