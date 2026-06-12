"""Three figures for the slide 'Three reasons to use feedback':

    1. Stabilization of an unstable plant
    2. Exact disturbance rejection (P vs PI)
    3. Robustness to plant-gain uncertainty (feed-forward vs PI)

All figures use the MATLAB default color palette and are exported as SVG.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# MATLAB default colors + helpers
# ------------------------------------------------------------------
BLUE      = "#0072BD"
RED       = "#D95319"
YELLOW    = "#EDB120"
PURPLE    = "#7E2F8E"
GREEN     = "#77AC30"
GRAY      = "#808080"
DARK_GRAY = "#404040"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13.5,
    "legend.fontsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",       # keep text as text, editable in Inkscape
    "mathtext.fontset": "cm",     # Computer Modern math: dot accents sit on top
})

OUT = "/mnt/user-data/outputs"

# =============================================================
# Figure 1.  Stabilization of an unstable scalar plant
#   dot x = a x + u,   a > 0  (open-loop unstable)
#   Open-loop:  u = 0      -> x(t) = x0 exp(a t)        (diverges)
#   Closed-loop: u = -k x  -> x(t) = x0 exp((a-k) t)    (decays for k > a)
# =============================================================
fig, ax = plt.subplots(figsize=(6.6, 4.4))
a, k, x0 = 0.5, 1.5, 1.0

t   = np.linspace(0, 4, 500)
xol = x0 * np.exp(a       * t)
xcl = x0 * np.exp((a - k) * t)

ax.plot(t, xol, color=RED,  lw=2.5,
        label=r"Open-loop  $u=0\;\Rightarrow\;\dot x=a\,x$")
ax.plot(t, xcl, color=BLUE, lw=2.5,
        label=r"Closed-loop  $u=-k\,x\;\Rightarrow\;\dot x=(a-k)\,x$")

ax.axhline(0, color=DARK_GRAY, lw=0.7)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x(t)$")
ax.set_title(rf"Stabilization of an unstable plant   "
             rf"($a={a},\;k={k},\;x_0={x0}$)")
ax.set_xlim(0, 4)
ax.set_ylim(-0.5, 8)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper left", framealpha=0.95)
plt.tight_layout()
plt.savefig(f"{OUT}/feedback_reason_1_stabilization.svg",
            format="svg", bbox_inches="tight")
plt.close(fig)

# =============================================================
# Figure 2.  Exact disturbance rejection
#   Plant:  dot y = u + d            (integrator with load disturbance)
#   Reference  r = 1 (step at t = 0)
#   Disturbance d = 0.5 step at t = t_d
#   Compare P  (offset remains)  vs  PI  (offset driven to zero)
# =============================================================
fig, ax = plt.subplots(figsize=(7.2, 4.4))
r_val, d_val, t_d = 1.0, 0.5, 6.0
t_end, N = 15.0, 1500
t  = np.linspace(0, t_end, N)
dt = t[1] - t[0]


def simulate_P(Kp):
    y = np.zeros(N)
    for i in range(1, N):
        d_now = d_val if t[i - 1] >= t_d else 0.0
        u     = Kp * (r_val - y[i - 1])
        y[i]  = y[i - 1] + dt * (u + d_now)
    return y


def simulate_PI(Kp, Ki):
    y, xi = np.zeros(N), 0.0
    for i in range(1, N):
        d_now = d_val if t[i - 1] >= t_d else 0.0
        e     = r_val - y[i - 1]
        xi   += dt * e
        u     = Kp * e + Ki * xi
        y[i]  = y[i - 1] + dt * (u + d_now)
    return y


y_p  = simulate_P(Kp=1.0)
y_pi = simulate_PI(Kp=1.0, Ki=1.0)

ax.axhline(r_val, color=DARK_GRAY, lw=1.3, ls="--",
           label=fr"Reference $r={r_val:g}$")
ax.plot(t, y_p,  color=RED,  lw=2.5, label=r"P  ($K_p=1$)")
ax.plot(t, y_pi, color=BLUE, lw=2.5, label=r"PI ($K_p=K_i=1$)")

ax.axvline(t_d, color=GRAY, lw=0.8, ls=":")
ax.annotate(rf"step disturbance $d={d_val:g}$ applied",
            xy=(t_d, 1.68), xytext=(t_d - 0.2, 1.68),
            fontsize=10, va="center", ha="right")

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$y(t)$")
ax.set_title(r"Exact disturbance rejection: P vs PI   "
             r"($\dot{y} = u + d$)")
ax.set_xlim(0, t_end)
ax.set_ylim(0, 1.75)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="lower right", framealpha=0.95)
plt.tight_layout()
plt.savefig(f"{OUT}/feedback_reason_2_disturbance_rejection.svg",
            format="svg", bbox_inches="tight")
plt.close(fig)

# =============================================================
# Figure 3.  Robustness to plant-gain uncertainty
#   Plant:  dot y = -y + b u,   actual gain  b  unknown.
#   Nominal value b_nom = 1.
#   Left  panel:  open-loop feed-forward  u = r / b_nom
#   Right panel:  PI closed-loop
#   Same b values, same reference, same time axis.
# =============================================================
fig, (ax_ol, ax_cl) = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)

b_nom, r_val = 1.0, 1.0
b_vals       = [0.5, 0.75, 1.0, 1.5, 2.0]
colors       = [BLUE, GREEN, DARK_GRAY, YELLOW, RED]  # nominal in dark gray

Kp, Ki   = 2.0, 2.0
t_end, N = 10.0, 1000
t  = np.linspace(0, t_end, N)
dt = t[1] - t[0]

for b, c in zip(b_vals, colors):
    # Feed-forward (no feedback)
    y_ol = np.zeros(N)
    u_ff = r_val / b_nom
    for i in range(1, N):
        y_ol[i] = y_ol[i - 1] + dt * (-y_ol[i - 1] + b * u_ff)
    ax_ol.plot(t, y_ol, color=c, lw=2.0, label=rf"$b={b:g}$")

    # PI closed-loop
    y_cl, xi = np.zeros(N), 0.0
    for i in range(1, N):
        e        = r_val - y_cl[i - 1]
        xi      += dt * e
        u        = Kp * e + Ki * xi
        y_cl[i]  = y_cl[i - 1] + dt * (-y_cl[i - 1] + b * u)
    ax_cl.plot(t, y_cl, color=c, lw=2.0, label=rf"$b={b:g}$")

for ax_ in (ax_ol, ax_cl):
    ax_.axhline(r_val, color=DARK_GRAY, lw=1.2, ls="--")
    ax_.set_xlabel(r"$t$")
    ax_.set_xlim(0, t_end)
    ax_.set_ylim(0, 2.35)
    ax_.grid(True, linestyle=":", alpha=0.5)

ax_ol.set_ylabel(r"$y(t)$")
ax_ol.set_title(r"Open-loop feed-forward:  $u = r/b_{\rm nom}$")
ax_cl.set_title(r"Closed-loop PI:  $K_p = K_i = 2$")
ax_ol.legend(loc="upper left",  framealpha=0.95, title=r"plant gain $b$")
ax_cl.legend(loc="lower right", framealpha=0.95, title=r"plant gain $b$")

fig.suptitle(
    r"Robustness to plant-gain uncertainty   "
    r"($\dot y = -y + b\,u$,  $b_{\rm nom}=1$,  $r=1$)",
    fontsize=13, y=1.00,
)
plt.tight_layout()
plt.savefig(f"{OUT}/feedback_reason_3_robustness.svg",
            format="svg", bbox_inches="tight")
plt.close(fig)

print("All three SVGs written to", OUT)
