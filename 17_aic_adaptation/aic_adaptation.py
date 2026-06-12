"""
Four figures: perfect/leaky × deterministic/stochastic.
k=0.5, eta=200. Disturbance: delta_p 1->2 at t_d=40.
Leaky: gamma_c=0.15 (visible error).
Stochastic plots include deterministic reference curve.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib, matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 15,
    "legend.fontsize": 10.5, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"; GRAY="0.55"
OUT = "/mnt/user-data/outputs"

k = 0.5; eta = 200.0; mu = 1.0; theta = 1.0
dp_before, dp_after = 1.0, 2.0
t_d = 40.0; T = 80.0

def rhs(t, z, dp, gc):
    m, p, z1, z2 = z
    return [k*z1 - m, m - dp*p,
            mu - eta*z1*z2 - gc*z1,
            theta*p - eta*z1*z2 - gc*z2]

def solve_det(gc, n_pts=1200):
    te1 = np.linspace(0, t_d, n_pts//2)
    te2 = np.linspace(t_d, T, n_pts//2)
    z0 = [0, 0, 0, 0]
    sol1 = solve_ivp(lambda t,z: rhs(t,z,dp_before,gc),
                     [0, t_d], z0, t_eval=te1, rtol=1e-10)
    sol2 = solve_ivp(lambda t,z: rhs(t,z,dp_after,gc),
                     [t_d, T], sol1.y[:,-1], t_eval=te2, rtol=1e-10)
    return (np.concatenate([sol1.t, sol2.t]),
            np.concatenate([sol1.y[1], sol2.y[1]]))

def ssa_trajectory(gc, seed, t_grid):
    rng = np.random.default_rng(seed)
    m, p, z1, z2 = 0, 0, 0, 0
    t = 0.0; dp = dp_before
    times = [0.0]; prots = [0]
    while t < T:
        a1=k*z1; a2=float(m); a3=float(m); a4=dp*p
        a5=mu; a6=theta*p; a7=eta*z1*z2; a8=gc*z1; a9=gc*z2
        at=a1+a2+a3+a4+a5+a6+a7+a8+a9
        if at<=0: at=mu
        dt=rng.exponential(1.0/at)
        if t < t_d <= t+dt: dp = dp_after
        t += dt
        r=rng.random()*at; c=0
        if r<(c:=c+a1): m+=1
        elif r<(c:=c+a2): p+=1
        elif r<(c:=c+a3): m-=1
        elif r<(c:=c+a4): p-=1
        elif r<(c:=c+a5): z1+=1
        elif r<(c:=c+a6): z2+=1
        elif r<(c:=c+a7): z1-=1; z2-=1
        elif r<(c:=c+a8): z1-=1
        else: z2-=1
        times.append(t); prots.append(p)
    times=np.array(times); prots=np.array(prots)
    idx=np.clip(np.searchsorted(times,t_grid,side="right")-1,0,len(prots)-1)
    return prots[idx]

def make_det_fig(gc, title, fname):
    t, p = solve_det(gc)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.86, bottom=0.13)
    ax.plot(t, p, color=BLUE, lw=2.5, label=r"$p(t)$")
    ax.axhline(mu/theta, color=GREEN, lw=2.0, ls="--",
               label=r"$\mu/\theta = 1$")
    ax.axvline(t_d, color=GRAY, lw=1.5, ls=":", alpha=0.5)
    ax.text(t_d+1, mu/theta+0.06, r"$\delta_p: 1 \to 2$",
            fontsize=12, va="bottom", color="0.35")
    ax.set_xlabel("Time $t$"); ax.set_ylabel(r"Protein $p(t)$")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, T)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.75")
    ax.grid(True, alpha=0.15)
    fig.savefig(f"{OUT}/{fname}.svg", bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}.svg")

def make_stoch_fig(gc, title, fname, n_runs=1000):
    t_grid = np.linspace(0, T, 800)
    # Deterministic reference
    t_det, p_det = solve_det(gc)
    # Sample paths
    sp1 = ssa_trajectory(gc, seed=7, t_grid=t_grid)
    sp2 = ssa_trajectory(gc, seed=42, t_grid=t_grid)
    # Mean
    p_runs = np.zeros((n_runs, len(t_grid)))
    for r in range(n_runs):
        p_runs[r] = ssa_trajectory(gc, seed=1000+r, t_grid=t_grid)
        if (r+1) % 250 == 0:
            print(f"    {r+1}/{n_runs}", flush=True)
    p_mean = p_runs.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.86, bottom=0.13)
    ax.plot(t_grid, sp1, color=BLUE, lw=0.5, alpha=0.2,
            label="Sample paths")
    ax.plot(t_grid, sp2, color=ORANGE, lw=0.5, alpha=0.2)
    ax.plot(t_det, p_det, color=ORANGE, lw=1.8, ls="--", alpha=0.7,
            label="Deterministic")
    ax.plot(t_grid, p_mean, color=BLUE, lw=2.5,
            label=rf"$E[P(t)]$ ({n_runs} runs)")
    ax.axhline(mu/theta, color=GREEN, lw=2.0, ls="--",
               label=r"$\mu/\theta = 1$")
    ax.axvline(t_d, color=GRAY, lw=1.5, ls=":", alpha=0.5)
    ax.text(t_d+1, mu/theta+0.06, r"$\delta_p: 1 \to 2$",
            fontsize=12, va="bottom", color="0.35")
    ax.set_xlabel("Time $t$"); ax.set_ylabel(r"Protein $P(t)$")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, T)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.75")
    ax.grid(True, alpha=0.15)
    fig.savefig(f"{OUT}/{fname}.svg", bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}.svg")

GC_LEAKY = 0.15

make_det_fig(gc=0,
    title=r"Perfect adaptation (deterministic, $\gamma_c = 0$)",
    fname="aic_perfect_det")
make_det_fig(gc=GC_LEAKY,
    title=rf"Near-perfect adaptation (deterministic, $\gamma_c = {GC_LEAKY}$)",
    fname="aic_leaky_det")

print("Stochastic perfect AIC:")
make_stoch_fig(gc=0, n_runs=1000,
    title=r"Perfect adaptation (stochastic, $\gamma_c = 0$)",
    fname="aic_perfect_stoch")
print("Stochastic leaky AIC:")
make_stoch_fig(gc=GC_LEAKY, n_runs=1000,
    title=rf"Near-perfect adaptation (stochastic, $\gamma_c = {GC_LEAKY}$)",
    fname="aic_leaky_stoch")
