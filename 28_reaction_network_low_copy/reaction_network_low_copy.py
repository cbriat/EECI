"""
Failure of Langevin and LNA at low molecule counts.

Same dimerisation network, parameters chosen for steady-state counts of
~2–4 molecules, where the stochastic propensity  k1 X(X-1)/2  differs
strongly from the macroscopic rate  k1 X^2/2 .

Three approximation failures are visible:
  1. The RRE (and LNA centred on it) predict X*=2.0, while the true
     stochastic mean is ~2.5  (macroscopic propensity overcounts
     dimerisation at low X).
  2. The CLE, even with reflecting boundary at 0, settles to a mean of
     ~1.7, lower than both the RRE and the SSA.
  3. The LNA envelope, being Gaussian and centred on the RRE, extends
     into the unphysical negative region.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 13, "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 1.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GOLD="#EDB120"; PURPLE="#7E2F8E"; BLACK="black"

# ── Parameters ────────────────────────────────────────────────
k0, k1, k2, k3 = 1.5, 0.3, 0.15, 0.2
T, seed = 80.0, 42
n0 = [0.0, 0.0]

# ── (a) Deterministic ODE ─────────────────────────────────────
def rre(t, n):
    X, Y = n
    return [k0 - k1*X**2 - k2*X,  0.5*k1*X**2 - k3*Y]

sol = solve_ivp(rre, [0,T], n0, dense_output=True, max_step=0.1, rtol=1e-8)
tg = np.linspace(0, T, 1200)
det_X, det_Y = sol.sol(tg)
Xss = det_X[-1]; Yss = det_Y[-1]
print(f"RRE steady state:  X*={Xss:.2f},  Y*={Yss:.2f}")

# ── (b) SSA (two realisations) ────────────────────────────────
def gillespie(T, sd):
    rng = np.random.default_rng(sd)
    ts, Xs, Ys = [0.0], [0], [0]
    t, nX, nY = 0.0, 0, 0
    while t < T:
        a0=k0; a1=k1*nX*(nX-1)/2 if nX>=2 else 0.0; a2=k2*nX; a3=k3*nY
        at = a0+a1+a2+a3
        if at <= 0: break
        t += rng.exponential(1/at)
        if t > T: break
        r = rng.random()*at
        if   r < a0:        nX += 1
        elif r < a0+a1:     nX -= 2; nY += 1
        elif r < a0+a1+a2:  nX -= 1
        else:               nY -= 1
        ts.append(t); Xs.append(nX); Ys.append(nY)
    ts.append(T); Xs.append(nX); Ys.append(nY)
    return np.asarray(ts), np.asarray(Xs), np.asarray(Ys)

ssa = [gillespie(T, seed+i*17) for i in range(2)]

# ── (c) CLE with reflecting boundary at 0 ────────────────────
def cle(T, dt, sd):
    rng = np.random.default_rng(sd)
    N = int(T/dt); t = np.linspace(0,T,N+1)
    X, Y = np.zeros(N+1), np.zeros(N+1)
    sqdt = np.sqrt(dt)
    for i in range(N):
        nX, nY = X[i], Y[i]
        a0 = k0
        a1 = 0.5*k1*nX**2 if nX > 0 else 0.0   # macroscopic propensity
        a2 = k2*nX         if nX > 0 else 0.0
        a3 = k3*nY         if nY > 0 else 0.0
        dW = rng.standard_normal(4)*sqdt
        X[i+1] = max(0.0, nX + (a0-2*a1-a2)*dt
                    + np.sqrt(a0)*dW[0] - 2*np.sqrt(a1)*dW[1] - np.sqrt(a2)*dW[2])
        Y[i+1] = max(0.0, nY + (a1-a3)*dt
                    + np.sqrt(a1)*dW[1] - np.sqrt(a3)*dW[3])
    return t, X, Y

t_cle, X_cle, Y_cle = cle(T, 0.002, seed)

# ── (d) LNA ───────────────────────────────────────────────────
def lna_rhs(t, s):
    X, Y, Sxx, Sxy, Syy = s
    dX = k0 - k1*X**2 - k2*X
    dY = 0.5*k1*X**2 - k3*Y
    Jxx = -2*k1*X - k2;  Jxy = 0.0;  Jyx = k1*X;  Jyy = -k3
    a0=k0; a1=0.5*k1*X**2; a2=k2*X; a3=k3*Y
    Dxx = a0+4*a1+a2;  Dxy = -2*a1;  Dyy = a1+a3
    return [dX, dY,
            2*(Jxx*Sxx+Jxy*Sxy)+Dxx,
            Jxx*Sxy+Jxy*Syy+Jyx*Sxx+Jyy*Sxy+Dxy,
            2*(Jyx*Sxy+Jyy*Syy)+Dyy]

sl = solve_ivp(lna_rhs, [0,T], [0,0,0,0,0], dense_output=True,
               max_step=0.1, rtol=1e-8)
la = sl.sol(tg)
lna_X, lna_Y = la[0], la[1]
stdX = np.sqrt(np.maximum(la[2],0)); stdY = np.sqrt(np.maximum(la[4],0))
print(f"LNA stationary std: σ_X={stdX[-1]:.2f}, σ_Y={stdY[-1]:.2f}")

# ── Long SSA for empirical mean ───────────────────────────────
rng_l = np.random.default_rng(999)
nX, nY, tl = 0, 0, 0.0
x_samples = []
while tl < 3000:
    a0=k0; a1=k1*nX*(nX-1)/2 if nX>=2 else 0; a2=k2*nX; a3=k3*nY
    at=a0+a1+a2+a3
    if at<=0: break
    tl += rng_l.exponential(1/at)
    r=rng_l.random()*at
    if r<a0: nX+=1
    elif r<a0+a1: nX-=2; nY+=1
    elif r<a0+a1+a2: nX-=1
    else: nY-=1
    if tl > 200: x_samples.append(nX)
ssa_mean = np.mean(x_samples)
print(f"SSA empirical mean X ≈ {ssa_mean:.2f}")

# Long CLE (clamped) for empirical mean
rng_c = np.random.default_rng(999)
dtc=0.002; nXc,nYc=0.0,0.0; xcle_samp=[]
for kk in range(int(3000/dtc)):
    a0=k0
    a1=0.5*k1*nXc**2 if nXc>0 else 0
    a2=k2*nXc if nXc>0 else 0
    a3=k3*nYc if nYc>0 else 0
    dW=rng_c.standard_normal(4)*np.sqrt(dtc)
    nXc=max(0,nXc+(a0-2*a1-a2)*dtc+np.sqrt(a0)*dW[0]-2*np.sqrt(a1)*dW[1]-np.sqrt(a2)*dW[2])
    nYc=max(0,nYc+(a1-a3)*dtc+np.sqrt(a1)*dW[1]-np.sqrt(a3)*dW[3])
    if kk*dtc>200: xcle_samp.append(nXc)
cle_mean = np.mean(xcle_samp)
print(f"CLE (clamped) mean X ≈ {cle_mean:.2f}")

# ── Figure ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
fig.subplots_adjust(left=0.10, right=0.96, top=0.89, bottom=0.08, hspace=0.12)
fig.suptitle(
    "Dimerisation network at low copy numbers\n"
    r"Failure of the CLE and LNA approximations",
    fontsize=13, fontweight="bold")

# ---- X panel ----
ax1.fill_between(tg, lna_X-2*stdX, lna_X+2*stdX,
                 color=GOLD, alpha=0.35, lw=0,
                 label=r"LNA $\pm 2\sigma$ envelope")
ax1.plot(tg, det_X, BLACK, ls="--", lw=2, label="Deterministic ODE (RRE)")
for i,(ts,xs,_) in enumerate(ssa):
    ax1.step(ts, xs, where="post", color=[BLUE,PURPLE][i], lw=1.0, alpha=0.85,
             label="SSA sample paths" if i==0 else None)
ax1.plot(t_cle, X_cle, color=ORANGE, lw=1.0, alpha=0.80,
         label="Langevin SDE (CLE, clamped at 0)")

# unphysical zone
ax1.axhspan(-2, 0, color="red", alpha=0.06, zorder=0)
ax1.axhline(0, color="red", lw=0.6, alpha=0.4)

ax1.set_ylabel(r"Molecules of $X$")
ax1.legend(loc="upper right", framealpha=0.95, edgecolor="0.82", ncol=2)
ax1.grid(True, alpha=0.2)
ax1.set_title(
    r"$\varnothing\!\to\!X,\;"
    r"2X\!\to\!Y,\;"
    r"X\!\to\!\varnothing,\;"
    r"Y\!\to\!\varnothing$"
    r"$\qquad(k_0\!=\!1.5,\;k_1\!=\!0.3,\;k_2\!=\!0.15,\;k_3\!=\!0.2)$",
    fontsize=10, pad=4)

# Mean annotations (right margin)
yoff = 0.15
ax1.annotate(f"SSA mean $\\approx {ssa_mean:.1f}$",
             xy=(T, ssa_mean), xytext=(T+1, ssa_mean+yoff),
             fontsize=8, color=BLUE, va="bottom",
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5))
ax1.annotate(f"RRE $X^*\\!= {Xss:.1f}$",
             xy=(T, Xss), xytext=(T+1, Xss-yoff),
             fontsize=8, color="0.3", va="top",
             arrowprops=dict(arrowstyle="-", color="0.3", lw=0.5))
ax1.annotate(f"CLE mean $\\approx {cle_mean:.1f}$",
             xy=(T, cle_mean), xytext=(T+1, cle_mean-0.6),
             fontsize=8, color=ORANGE, va="top",
             arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5))

# ---- Y panel ----
ax2.fill_between(tg, lna_Y-2*stdY, lna_Y+2*stdY,
                 color=GOLD, alpha=0.35, lw=0)
ax2.plot(tg, det_Y, BLACK, ls="--", lw=2)
for i,(ts,_,ys) in enumerate(ssa):
    ax2.step(ts, ys, where="post", color=[BLUE,PURPLE][i], lw=1.0, alpha=0.85)
ax2.plot(t_cle, Y_cle, color=ORANGE, lw=1.0, alpha=0.80)

ax2.axhspan(-2, 0, color="red", alpha=0.06, zorder=0)
ax2.axhline(0, color="red", lw=0.6, alpha=0.4)
ax2.set_ylabel(r"Molecules of $Y$")
ax2.set_xlabel(r"Time  $t$")
ax2.grid(True, alpha=0.2)

# unphysical label
for ax in (ax1, ax2):
    ax.set_ylim(bottom=min(-1.5, ax.get_ylim()[0]))
    yl = ax.get_ylim()
    if yl[0] < 0:
        ax.text(T*0.005, -0.15, "unphysical ($<0$)",
                fontsize=7.5, color="red", alpha=0.6, va="top")
for ax in (ax1, ax2):
    ax.set_xlim(0, T)

fig.savefig("reaction_network_low_copy.svg", bbox_inches="tight")
fig.savefig("reaction_network_low_copy.png", dpi=200, bbox_inches="tight")
print("\nFigure saved.")
plt.close(fig)
