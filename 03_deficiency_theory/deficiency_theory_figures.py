"""
Deficiency theory figures for EECI course.
Two weakly reversible reaction networks:
  Example 1 (δ=0):  A ⇌ B ⇌ A+B ⇌ A          [complex-balanced]
  Example 2 (δ=1):  ∅ ⇌ A+B ⇌ 2A  and  A ⇌ B  [not complex-balanced]

Generates 8 SVG figures:
  1. def0_graph.svg          — network graph
  2. def0_phase.svg          — phase portrait
  3. def0_stationary.svg     — stationary distribution + Poisson fit
  4. cb_check_def0.svg       — complex-balance flow check
  5. def1_graph.svg          — network graph
  6. def1_phase.svg          — phase portrait
  7. def1_stationary.svg     — stationary distribution + Poisson fit
  8. cb_check_def1.svg       — complex-balance flow check
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.stats import poisson
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib, matplotlib.pyplot as plt
import time as timer

OUT = "/mnt/user-data/outputs"

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 16,
    "legend.fontsize": 11, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.0, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#0072BD"; ORANGE="#D95319"; GREEN="#77AC30"
PURPLE="#7E2F8E"; GOLD="#EDB120"; RED="#A2142F"

# ══════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════
def draw_node(ax, xy, label, color, r=0.38, fs=16):
    circ = plt.Circle(xy, r, fc="white", ec=color, lw=2.5, zorder=3)
    ax.add_patch(circ)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color, zorder=4)

def draw_edge(ax, p1, p2, color, offset=0.06, rad=0.12, r_node=0.35):
    x1,y1 = p1; x2,y2 = p2
    dx,dy = x2-x1, y2-y1
    L = np.hypot(dx,dy)
    nx,ny = -dy/L, dx/L
    sx = x1 + r_node*dx/L + offset*nx
    sy = y1 + r_node*dy/L + offset*ny
    ex = x2 - r_node*dx/L + offset*nx
    ey = y2 - r_node*dy/L + offset*ny
    ax.annotate("", xy=(ex,ey), xytext=(sx,sy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)

def draw_flow_arrow(ax, p1, p2, rate, max_rate, color,
                    offset=0.06, rad=0.12, r_node=0.42):
    x1,y1 = p1; x2,y2 = p2
    dx,dy = x2-x1, y2-y1
    L = np.hypot(dx,dy)
    nx,ny = -dy/L, dx/L
    sx = x1 + r_node*dx/L + offset*nx
    sy = y1 + r_node*dy/L + offset*ny
    ex = x2 - r_node*dx/L + offset*nx
    ey = y2 - r_node*dy/L + offset*ny
    lw = 1.0 + 4.0 * rate / max_rate
    ax.annotate("", xy=(ex,ey), xytext=(sx,sy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=10+8*rate/max_rate,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)
    mx = (sx+ex)/2 + 0.22*nx + rad*0.5*(ey-sy)/L
    my = (sy+ey)/2 + 0.22*ny - rad*0.5*(ex-sx)/L
    ax.text(mx, my, f"{rate:.2f}", fontsize=8, ha="center", va="center",
            color=color, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))

def balance_badge(ax, xy, inflow, outflow, r_node=0.42):
    diff = abs(inflow - outflow)
    col = GREEN if diff < 0.01 else RED
    bx, by = xy[0], xy[1] - r_node - 0.28
    ax.text(bx, by, f"in={inflow:.2f}\nout={outflow:.2f}",
            fontsize=7.5, ha="center", va="top", color=col,
            fontfamily="monospace",
            bbox=dict(fc="white", ec=col, lw=1.5, pad=3,
                      boxstyle="round,pad=0.3", alpha=0.95))

def phase_portrait(ax, rhs_fn, eq, ics, T=12, xmax=3.8):
    teval = np.linspace(0, T, 600)
    xg = np.linspace(0.1, xmax, 16)
    yg = np.linspace(0.1, xmax, 16)
    Xg, Yg = np.meshgrid(xg, yg)
    UV = np.array([[rhs_fn(0,[x,y]) for x in xg] for y in yg])
    Ug = UV[:,:,0]; Vg = UV[:,:,1]
    sp = np.sqrt(Ug**2+Vg**2)
    ax.quiver(Xg, Yg, Ug/sp, Vg/sp, sp, cmap="Greys", alpha=0.3,
              scale=28, width=0.004, zorder=1)
    cmap = plt.get_cmap("tab10")
    for i, ic in enumerate(ics):
        sol = solve_ivp(rhs_fn, [0,T], ic, t_eval=teval, rtol=1e-9)
        c = cmap(i%10)
        ax.plot(sol.y[0], sol.y[1], color=c, lw=1.5, alpha=0.7, zorder=2)
        ax.plot(sol.y[0,0], sol.y[1,0], "o", color=c, ms=5, zorder=3)
    ax.plot(eq[0], eq[1], "*", color="red", ms=18,
            markeredgecolor="k", markeredgewidth=0.7, zorder=5)
    ax.annotate(rf"$x^*\!=\!({eq[0]:.2f},\,{eq[1]:.2f})$",
                xy=eq, xytext=(eq[0]+0.55, eq[1]+0.55), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2),
                bbox=dict(fc="white", ec="0.5", pad=3,
                          boxstyle="round,pad=0.3"))
    ax.set_xlabel(r"$[A]$"); ax.set_ylabel(r"$[B]$")
    ax.set_xlim(0, xmax); ax.set_ylim(0, xmax)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)

def plot_stationary(P, mA, mB, eq, title, fname, na_range, nb_range):
    fig, ax_main = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(left=0.10, right=0.85, top=0.88, bottom=0.10)
    Psub = P[na_range[0]:na_range[1]+1, nb_range[0]:nb_range[1]+1].T
    extent = [na_range[0]-0.5, na_range[1]+0.5,
              nb_range[0]-0.5, nb_range[1]+0.5]
    im = ax_main.imshow(Psub, origin="lower", aspect="auto",
                         extent=extent, cmap="viridis",
                         interpolation="bilinear")
    ax_main.plot(eq[0], eq[1], "+", color="red", ms=14, mew=2.5, zorder=5)
    ax_main.set_xlabel(r"$n_A$"); ax_main.set_ylabel(r"$n_B$")

    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", size="22%", pad=0.25, sharex=ax_main)
    ax_right = divider.append_axes("right", size="22%", pad=0.25,
                                    sharey=ax_main)
    cax = divider.append_axes("right", size="4%", pad=0.55)

    na_grid = np.arange(na_range[0], na_range[1]+1)
    nb_grid = np.arange(nb_range[0], nb_range[1]+1)
    mA_sub = mA[na_range[0]:na_range[1]+1]
    mB_sub = mB[nb_range[0]:nb_range[1]+1]
    mean_A = np.sum(na_grid * mA_sub) / mA_sub.sum()
    mean_B = np.sum(nb_grid * mB_sub) / mB_sub.sum()
    pois_A = poisson.pmf(na_grid, mean_A)
    pois_B = poisson.pmf(nb_grid, mean_B)

    ax_top.bar(na_grid, mA_sub, width=0.9, color="#0072BD",
               edgecolor="none", alpha=0.7)
    ax_top.plot(na_grid, pois_A, "k-", lw=1.8)
    ax_top.set_ylabel(r"$P(n_A)$", fontsize=11)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.text(0.97, 0.92, rf"Poisson($\lambda\!=\!{mean_A:.1f}$)",
                transform=ax_top.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(fc="white", ec="0.7", pad=2,
                          boxstyle="round,pad=0.3", alpha=0.9))
    for s in ("top","right"): ax_top.spines[s].set_visible(False)

    ax_right.barh(nb_grid, mB_sub, height=0.9, color="#D95319",
                  edgecolor="none", alpha=0.7)
    ax_right.plot(pois_B, nb_grid, "k-", lw=1.8)
    ax_right.set_xlabel(r"$P(n_B)$", fontsize=11)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.text(0.95, 0.03, rf"Poisson($\lambda\!=\!{mean_B:.1f}$)",
                  transform=ax_right.transAxes, fontsize=9, ha="right",
                  va="bottom", bbox=dict(fc="white", ec="0.7", pad=2,
                          boxstyle="round,pad=0.3", alpha=0.9))
    for s in ("top","right"): ax_right.spines[s].set_visible(False)

    ax_main.set_xlim(na_range[0]-0.5, na_range[1]+0.5)
    ax_main.set_ylim(nb_range[0]-0.5, nb_range[1]+0.5)
    fig.colorbar(im, cax=cax, label=r"$P(n_A, n_B)$")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(f"{OUT}/{fname}.svg", bbox_inches="tight")
    plt.close()
    print(f"  {fname}.svg")

def savefig(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight")
    plt.close()
    print(f"  {name}.svg")

# ══════════════════════════════════════════════════════════════
#  PARAMETERS
# ══════════════════════════════════════════════════════════════
# Example 1:  A ⇌ B ⇌ A+B ⇌ A  (δ=0)
k1 = [1.0, 0.8, 0.6, 0.5, 0.4, 0.7]  # k1..k6

def rhs1(t, z):
    a, b = z
    return [-k1[0]*a + (k1[1]+k1[2])*b - k1[3]*a*b,
            (k1[0]+k1[5])*a - k1[1]*b - k1[4]*a*b]

eq1 = fsolve(lambda z: rhs1(0,z), [1.5, 1.5])

# Example 2:  ∅ ⇌ A+B ⇌ 2A, A ⇌ B  (δ=1)
k2 = [2.0, 0.3, 0.5, 0.2, 0.8, 0.6]

def rhs2(t, z):
    a, b = z
    return [k2[0] + (k2[2]-k2[1])*a*b - k2[3]*a**2
              - k2[4]*a + k2[5]*b,
            k2[0] - (k2[1]+k2[2])*a*b + k2[3]*a**2
              + k2[4]*a - k2[5]*b]

eq2 = fsolve(lambda z: rhs2(0,z), [2.0, 2.0])

print(f"Equilibria: Ex1=({eq1[0]:.3f},{eq1[1]:.3f}), "
      f"Ex2=({eq2[0]:.3f},{eq2[1]:.3f})")

Omega = 10.0
ics_all = [(0.15,0.15),(3.5,0.2),(0.2,3.5),(3.5,3.5),(2.8,0.5),
           (0.3,2.8),(3.0,2.0),(0.8,3.2),(0.4,1.8),(2.2,0.3)]

# ══════════════════════════════════════════════════════════════
#  1 & 5.  NETWORK GRAPHS
# ══════════════════════════════════════════════════════════════
print("Network graphs:")

# --- δ=0 graph ---
fig, ax = plt.subplots(figsize=(6, 5.5))
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02)
ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.1, 1.9); ax.set_aspect("equal")
ax.axis("off")
fig.suptitle(r"Example 1: deficiency $\delta = 0$", fontsize=15,
             fontweight="bold")
ax.set_title("Weakly reversible, single linkage class", fontsize=12)
n1 = {"AB": (0, 1.3), "A": (-1.2, -0.4), "B": (1.2, -0.4)}
draw_node(ax, n1["AB"], r"$A\!+\!B$", PURPLE, r=0.42, fs=15)
draw_node(ax, n1["A"],  r"$A$", BLUE)
draw_node(ax, n1["B"],  r"$B$", ORANGE)
for p1,p2,col in [(n1["A"],n1["B"],GREEN),(n1["B"],n1["A"],GREEN),
                   (n1["B"],n1["AB"],GOLD),(n1["AB"],n1["B"],GOLD),
                   (n1["AB"],n1["A"],RED),(n1["A"],n1["AB"],RED)]:
    draw_edge(ax, p1, p2, col)
ax.text(0, -1.0, r"$n=3,\;\ell=1,\;s=2$" "\n" r"$\delta = 3-1-2 = 0$"
        "\n\nDeficiency Zero Thm:\nunique complex-balanced\n"
        "equilibrium (globally stable)",
        ha="center", va="top", fontsize=10.5,
        bbox=dict(fc="#EBF5E0", ec=GREEN, pad=6,
                  boxstyle="round,pad=0.4"))
savefig(fig, "def0_graph")

# --- δ=1 graph ---
fig, ax = plt.subplots(figsize=(7, 5.5))
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02)
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.3, 2.0); ax.set_aspect("equal")
ax.axis("off")
fig.suptitle(r"Example 2: deficiency $\delta = 1$", fontsize=15,
             fontweight="bold")
ax.set_title("Weakly reversible, two linkage classes", fontsize=12)
n2 = {"0":(-1.5,1.2),"AB":(0,1.2),"2A":(1.5,1.2),
      "A":(-0.8,-0.4),"B":(0.8,-0.4)}
draw_node(ax, n2["0"],  r"$\varnothing$", "0.45", r=0.32, fs=17)
draw_node(ax, n2["AB"], r"$A\!+\!B$", PURPLE, r=0.42, fs=14)
draw_node(ax, n2["2A"], r"$2A$", BLUE, r=0.32, fs=15)
draw_node(ax, n2["A"],  r"$A$", GREEN, r=0.30, fs=16)
draw_node(ax, n2["B"],  r"$B$", ORANGE, r=0.30, fs=16)
for p1,p2,col in [(n2["0"],n2["AB"],RED),(n2["AB"],n2["0"],RED),
                   (n2["AB"],n2["2A"],GOLD),(n2["2A"],n2["AB"],GOLD)]:
    draw_edge(ax, p1, p2, col, offset=0.05, rad=0.10, r_node=0.38)
draw_edge(ax,n2["A"],n2["B"],PURPLE,offset=0.05,rad=0.12,r_node=0.32)
draw_edge(ax,n2["B"],n2["A"],PURPLE,offset=0.05,rad=0.12,r_node=0.32)
ax.plot([-1.8,1.8],[0.55,0.55],color="0.8",lw=1,ls=":",zorder=0)
ax.text(0, 1.85, "Linkage class 1", fontsize=10, ha="center",
        color="0.5", fontstyle="italic")
ax.text(0, 0.35, "Linkage class 2", fontsize=10, ha="center",
        color="0.5", fontstyle="italic")
ax.text(0, -1.2, r"$n=5,\;\ell=2,\;s=2$" "\n" r"$\delta = 5-2-2 = 1$"
        "\n\nDeficiency One Thm:\n" r"$\leq 1$ equilibrium per class,"
        "\nstable if it exists,\nnot complex-balanced",
        ha="center", va="top", fontsize=10.5,
        bbox=dict(fc="#FFF5E6", ec=GOLD, pad=6,
                  boxstyle="round,pad=0.4"))
savefig(fig, "def1_graph")

# ══════════════════════════════════════════════════════════════
#  2 & 6.  PHASE PORTRAITS
# ══════════════════════════════════════════════════════════════
print("Phase portraits:")

fig, ax = plt.subplots(figsize=(6.5, 6))
fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.10)
fig.suptitle(r"Deficiency $\delta=0$: global convergence",
             fontsize=15, fontweight="bold")
phase_portrait(ax, rhs1, eq1, ics_all)
savefig(fig, "def0_phase")

ics2 = [(0.2,0.2),(4.0,0.3),(0.3,4.0),(4.0,4.0),(3.5,1.0),
        (0.5,3.5),(3.5,2.5),(1.0,3.8),(0.5,2.0),(2.5,0.4)]
fig, ax = plt.subplots(figsize=(6.5, 6))
fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.10)
fig.suptitle(r"Deficiency $\delta=1$: convergence (not complex-balanced)",
             fontsize=14, fontweight="bold")
phase_portrait(ax, rhs2, eq2, ics2, T=10, xmax=4.2)
savefig(fig, "def1_phase")

# ══════════════════════════════════════════════════════════════
#  3 & 7.  STATIONARY DISTRIBUTIONS (SSA)
# ══════════════════════════════════════════════════════════════
print("Stationary distributions (SSA) ...")

def ssa_ex1(T_events, seed):
    rng = np.random.default_rng(seed)
    nA, nB = 13, 17
    burn = T_events // 5
    hist = {}
    for ev in range(T_events):
        props = np.array([k1[0]*nA, k1[1]*nB, k1[2]*nB,
                          k1[3]*nA*nB/Omega, k1[4]*nA*nB/Omega,
                          k1[5]*nA])
        a_tot = props.sum()
        if a_tot <= 0: break
        r = rng.random()*a_tot; cum=0.0
        for j in range(6):
            cum += props[j]
            if r < cum: break
        stoich = [(-1,1),(1,-1),(1,0),(-1,0),(0,-1),(0,1)]
        nA += stoich[j][0]; nB += stoich[j][1]
        if ev > burn:
            hist[(nA,nB)] = hist.get((nA,nB), 0) + 1
    return hist

def ssa_ex2(T_events, seed):
    rng = np.random.default_rng(seed)
    nA, nB = 32, 21
    burn = T_events // 5
    hist = {}
    for ev in range(T_events):
        props = np.array([
            k2[0]*Omega, k2[1]*nA*nB/Omega, k2[2]*nA*nB/Omega,
            k2[3]*nA*(nA-1)/(2*Omega) if nA>=2 else 0,
            k2[4]*nA, k2[5]*nB])
        a_tot = props.sum()
        if a_tot <= 0: break
        r = rng.random()*a_tot; cum=0.0
        for j in range(6):
            cum += props[j]
            if r < cum: break
        stoich = [(1,1),(-1,-1),(1,-1),(-1,1),(-1,1),(1,-1)]
        nA += stoich[j][0]; nB += stoich[j][1]
        nA = max(nA,0); nB = max(nB,0)
        if ev > burn:
            hist[(nA,nB)] = hist.get((nA,nB), 0) + 1
    return hist

def hist_to_array(hist, pad_a=0, pad_b=0):
    max_a = max(max(k[0] for k in hist), pad_a)
    max_b = max(max(k[1] for k in hist), pad_b)
    arr = np.zeros((max_a+1, max_b+1))
    for (a,b), cnt in hist.items():
        if 0<=a<=max_a and 0<=b<=max_b: arr[a,b] = cnt
    arr /= arr.sum()
    return arr

t0 = timer.time()
h1 = ssa_ex1(5_000_000, 42)
h2 = ssa_ex2(5_000_000, 42)
print(f"  SSA done ({timer.time()-t0:.0f}s)")

P1 = hist_to_array(h1, 40, 45)
P2 = hist_to_array(h2, 65, 50)
mA1=P1.sum(1); mB1=P1.sum(0)
mA2=P2.sum(1); mB2=P2.sum(0)

plot_stationary(P1, mA1, mB1, (eq1[0]*Omega, eq1[1]*Omega),
    r"Stationary distribution — deficiency $\delta\!=\!0$"
    r"  (complex-balanced, $\Omega\!=\!10$)",
    "def0_stationary", (0, 35), (0, 40))

plot_stationary(P2, mA2, mB2, (eq2[0]*Omega, eq2[1]*Omega),
    r"Stationary distribution — deficiency $\delta\!=\!1$"
    r"  (not complex-balanced, $\Omega\!=\!10$)",
    "def1_stationary", (10, 60), (5, 45))

# ══════════════════════════════════════════════════════════════
#  4 & 8.  COMPLEX-BALANCE FLOW CHECKS
# ══════════════════════════════════════════════════════════════
print("Complex-balance checks:")
a_s, b_s = eq1
rates_e1 = {"A→B": k1[0]*a_s, "B→A": k1[1]*b_s, "B→AB": k1[2]*b_s,
            "AB→B": k1[3]*a_s*b_s, "AB→A": k1[4]*a_s*b_s,
            "A→AB": k1[5]*a_s}
max_r1 = max(rates_e1.values())
flows_e1 = {
    "A":  {"in": rates_e1["B→A"]+rates_e1["AB→A"],
           "out": rates_e1["A→B"]+rates_e1["A→AB"]},
    "B":  {"in": rates_e1["A→B"]+rates_e1["AB→B"],
           "out": rates_e1["B→A"]+rates_e1["B→AB"]},
    "AB": {"in": rates_e1["B→AB"]+rates_e1["A→AB"],
           "out": rates_e1["AB→B"]+rates_e1["AB→A"]},
}

# --- δ=0 flow check ---
fig, ax = plt.subplots(figsize=(7, 7))
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)
ax.set_xlim(-2.0,2.0); ax.set_ylim(-1.8,2.0); ax.set_aspect("equal")
ax.axis("off")
fig.suptitle(r"Deficiency $\delta\!=\!0$: complex-balanced equilibrium",
             fontsize=15, fontweight="bold")
ax.set_title(r"Arrow thickness $\propto$ reaction rate at $x^*$",
             fontsize=11, color="0.4")
nn1 = {"AB":(0,1.4), "A":(-1.3,-0.3), "B":(1.3,-0.3)}
draw_node(ax, nn1["AB"], r"$A\!+\!B$", PURPLE, r=0.45, fs=15)
draw_node(ax, nn1["A"],  r"$A$", BLUE, r=0.38)
draw_node(ax, nn1["B"],  r"$B$", ORANGE, r=0.38)
for p1,p2,r,c in [
    (nn1["A"],nn1["B"],rates_e1["A→B"],GREEN),
    (nn1["B"],nn1["A"],rates_e1["B→A"],GREEN),
    (nn1["B"],nn1["AB"],rates_e1["B→AB"],GOLD),
    (nn1["AB"],nn1["B"],rates_e1["AB→B"],GOLD),
    (nn1["AB"],nn1["A"],rates_e1["AB→A"],RED),
    (nn1["A"],nn1["AB"],rates_e1["A→AB"],RED)]:
    draw_flow_arrow(ax, p1, p2, r, max_r1, c, r_node=0.42)
for nm,pos in [("A",nn1["A"]),("B",nn1["B"]),("AB",nn1["AB"])]:
    balance_badge(ax, pos, flows_e1[nm]["in"], flows_e1[nm]["out"], 0.42)
ax.text(0, -1.65, r"$\sum_{\mathrm{in}} = \sum_{\mathrm{out}}$ at every"
        r" complex  $\Rightarrow$  complex-balanced",
        ha="center", fontsize=13, color=GREEN, fontweight="bold")
savefig(fig, "cb_check_def0")

# --- δ=1 flow check ---
a_s2, b_s2 = eq2
rates_e2 = {"0→AB": k2[0], "AB→0": k2[1]*a_s2*b_s2,
            "AB→2A": k2[2]*a_s2*b_s2, "2A→AB": k2[3]*a_s2**2,
            "A→B": k2[4]*a_s2, "B→A": k2[5]*b_s2}
max_r2 = max(rates_e2.values())
flows_e2 = {
    "0":  {"in": rates_e2["AB→0"], "out": rates_e2["0→AB"]},
    "AB": {"in": rates_e2["0→AB"]+rates_e2["2A→AB"],
           "out": rates_e2["AB→0"]+rates_e2["AB→2A"]},
    "2A": {"in": rates_e2["AB→2A"], "out": rates_e2["2A→AB"]},
    "A":  {"in": rates_e2["B→A"], "out": rates_e2["A→B"]},
    "B":  {"in": rates_e2["A→B"], "out": rates_e2["B→A"]},
}

fig, ax = plt.subplots(figsize=(8, 7))
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)
ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.0,2.2); ax.set_aspect("equal")
ax.axis("off")
fig.suptitle(r"Deficiency $\delta\!=\!1$: equilibrium is NOT complex-balanced",
             fontsize=15, fontweight="bold")
ax.set_title(r"Arrow thickness $\propto$ reaction rate at $x^*$",
             fontsize=11, color="0.4")
nn2 = {"0":(-1.6,1.2),"AB":(0,1.2),"2A":(1.6,1.2),
       "A":(-0.8,-0.5),"B":(0.8,-0.5)}
draw_node(ax, nn2["0"],  r"$\varnothing$", "0.45", r=0.35, fs=17)
draw_node(ax, nn2["AB"], r"$A\!+\!B$", PURPLE, r=0.45, fs=14)
draw_node(ax, nn2["2A"], r"$2A$", BLUE, r=0.35, fs=15)
draw_node(ax, nn2["A"],  r"$A$", GREEN, r=0.32, fs=16)
draw_node(ax, nn2["B"],  r"$B$", ORANGE, r=0.32, fs=16)
for p1,p2,r,c in [
    (nn2["0"],nn2["AB"],rates_e2["0→AB"],RED),
    (nn2["AB"],nn2["0"],rates_e2["AB→0"],RED),
    (nn2["AB"],nn2["2A"],rates_e2["AB→2A"],GOLD),
    (nn2["2A"],nn2["AB"],rates_e2["2A→AB"],GOLD),
    (nn2["A"],nn2["B"],rates_e2["A→B"],PURPLE),
    (nn2["B"],nn2["A"],rates_e2["B→A"],PURPLE)]:
    rn = 0.40 if p1 in [nn2["A"],nn2["B"]] or \
                 p2 in [nn2["A"],nn2["B"]] else 0.42
    draw_flow_arrow(ax, p1, p2, r, max_r2, c, r_node=rn)
for nm,pos in [("0",nn2["0"]),("AB",nn2["AB"]),("2A",nn2["2A"]),
               ("A",nn2["A"]),("B",nn2["B"])]:
    balance_badge(ax, pos, flows_e2[nm]["in"], flows_e2[nm]["out"], 0.38)
ax.text(0, -1.75, r"$\sum_{\mathrm{in}} \neq \sum_{\mathrm{out}}$"
        r" at $A\!+\!B$, $2A$, $A$, $B$"
        r"  $\Rightarrow$  not complex-balanced",
        ha="center", fontsize=13, color=RED, fontweight="bold")
savefig(fig, "cb_check_def1")

# ══════════════════════════════════════════════════════════════
#  EXAMPLE 3: EDELSTEIN NETWORK (δ=1, NOT weakly reversible)
#  A ⇌ 2A,  A+B ⇌ C,  C → B      (bistability)
#  Conservation: B + C = B_tot → reduce to 2D (A, B)
# ══════════════════════════════════════════════════════════════
print("Edelstein network (bistability):")

ke = dict(k1=3.0, k2=0.5, k3=2.0, k4=0.5, k5=1.0, Bt=5.0)

def rhs_edel(t, z):
    a, b = z
    c = ke['Bt'] - b
    return [ke['k1']*a - ke['k2']*a**2 - ke['k3']*a*b + ke['k4']*c,
            -ke['k3']*a*b + (ke['k4']+ke['k5'])*c]

# Equilibria
from scipy.optimize import brentq
def f_edel(a):
    return (ke['k1'] - ke['k2']*a
            - ke['k3']*ke['k5']*ke['Bt']/(ke['k3']*a+ke['k4']+ke['k5']))
roots_a = []
ag = np.linspace(0.001, 20, 5000)
fg = [f_edel(a) for a in ag]
for i in range(len(fg)-1):
    if fg[i]*fg[i+1] < 0:
        roots_a.append(brentq(f_edel, ag[i], ag[i+1]))

eq_edel = []
for a in [0.0] + roots_a:
    b = (ke['k4']+ke['k5'])*ke['Bt']/(ke['k3']*a+ke['k4']+ke['k5'])
    eq_edel.append((a, b))
    fp = -ke['k2'] + ke['k3']**2*ke['k5']*ke['Bt']/(ke['k3']*a+ke['k4']+ke['k5'])**2 if a>0 else -1
    stab = "stable" if fp < 0 else "UNSTABLE"
    print(f"  eq: a={a:.3f}, b={b:.3f}  ({stab})")

# ── Network graph ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02)
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.5, 2.0); ax.set_aspect("equal")
ax.axis("off")
fig.suptitle(r"Example 3 (Edelstein): $\delta = 1$, not weakly reversible",
             fontsize=14, fontweight="bold")
ax.set_title("Bistability: two stable equilibria", fontsize=12)

ne = {"A":(-1.8,0.8),"2A":(-0.2,0.8),"AB":(1.8,0.8),
      "C":(1.8,-0.3),"B":(0.2,-0.3)}
draw_node(ax, ne["A"],  r"$A$", BLUE, r=0.32, fs=16)
draw_node(ax, ne["2A"], r"$2A$", BLUE, r=0.32, fs=15)
draw_node(ax, ne["AB"], r"$A\!+\!B$", PURPLE, r=0.42, fs=14)
draw_node(ax, ne["C"],  r"$C$", GOLD, r=0.32, fs=16)
draw_node(ax, ne["B"],  r"$B$", ORANGE, r=0.32, fs=16)

# LC1: A ⇌ 2A (weakly reversible)
draw_edge(ax, ne["A"], ne["2A"], GREEN, offset=0.05, rad=0.15, r_node=0.35)
draw_edge(ax, ne["2A"], ne["A"], GREEN, offset=0.05, rad=0.15, r_node=0.35)
# LC2: A+B ⇌ C → B (NOT weakly reversible)
draw_edge(ax, ne["AB"], ne["C"], RED, offset=0.05, rad=0.12, r_node=0.38)
draw_edge(ax, ne["C"], ne["AB"], RED, offset=0.05, rad=0.12, r_node=0.38)
draw_edge(ax, ne["C"], ne["B"], GOLD, offset=0.0, rad=0.0, r_node=0.35)

# LC labels
ax.text(-1.0, 1.5, "LC 1 (w.r.)", fontsize=10, ha="center",
        color=GREEN, fontstyle="italic")
ax.text(1.0, 1.5, "LC 2 (not w.r.)", fontsize=10, ha="center",
        color=RED, fontstyle="italic")
# Arrow to highlight irreversibility
ax.text(1.0, -0.6, r"$C \to B$ irreversible",
        fontsize=9, ha="center", color=GOLD, fontstyle="italic")

ax.text(0, -1.4, r"$n=5,\;\ell=2,\;s=2,\;\delta=1$" "\n"
        "Not weakly reversible\n"
        r"$\Rightarrow$ Def. One Thm uniqueness does not apply"
        "\n" r"$\Rightarrow$ multiple equilibria possible",
        ha="center", va="top", fontsize=10.5,
        bbox=dict(fc="#FDE8E8", ec=RED, pad=6,
                  boxstyle="round,pad=0.4"))
savefig(fig, "edelstein_graph")

# ── Phase portrait (2D reduced: A vs B) ──────────────────────
fig, ax = plt.subplots(figsize=(7, 6.5))
fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.10)
fig.suptitle("Edelstein network: bistability ($A$ vs $B$, $C = B_{tot} - B$)",
             fontsize=14, fontweight="bold")

# Vector field
xmax_e, ymax_e = 8.0, ke['Bt']+0.2
ag2 = np.linspace(0.01, xmax_e, 20)
bg2 = np.linspace(0.01, ymax_e, 20)
Ag, Bg = np.meshgrid(ag2, bg2)
Cg = np.clip(ke['Bt'] - Bg, 0, None)
Ug = ke['k1']*Ag - ke['k2']*Ag**2 - ke['k3']*Ag*Bg + ke['k4']*Cg
Vg = -ke['k3']*Ag*Bg + (ke['k4']+ke['k5'])*Cg
sp = np.sqrt(Ug**2+Vg**2)
ax.quiver(Ag, Bg, Ug/sp, Vg/sp, sp, cmap="Greys", alpha=0.25,
          scale=30, width=0.004, zorder=1)

# Trajectories from many initial conditions
ics_e = [(0.1, 4.5), (0.5, 4.0), (1.0, 3.5), (1.2, 2.0),
         (0.3, 1.0), (0.8, 0.5), (2.0, 4.5), (3.0, 3.0),
         (5.0, 4.0), (6.0, 3.0), (7.0, 2.0), (4.0, 0.5),
         (6.0, 0.5), (1.5, 1.5), (2.5, 2.5), (4.5, 2.0)]
cmap_t = plt.get_cmap("tab20")
for i, ic in enumerate(ics_e):
    if ic[1] > ke['Bt']: continue
    sol = solve_ivp(rhs_edel, [0, 25], ic, t_eval=np.linspace(0,25,800),
                    rtol=1e-9)
    c = cmap_t(i % 20)
    ax.plot(sol.y[0], sol.y[1], color=c, lw=1.3, alpha=0.7, zorder=2)
    ax.plot(sol.y[0,0], sol.y[1,0], "o", color=c, ms=4, zorder=3)

# Mark equilibria
for j, (ae, be) in enumerate(eq_edel):
    if j == 0:
        mk, col, lbl = "s", GREEN, "Stable (trivial)"
    elif j == 1:
        mk, col, lbl = "D", RED, "Unstable (saddle)"
    else:
        mk, col, lbl = "*", GREEN, "Stable (active)"
    ax.plot(ae, be, mk, color=col, ms=14 if mk=="*" else 10,
            markeredgecolor="k", markeredgewidth=0.7, zorder=5,
            label=lbl)

ax.set_xlabel(r"$[A]$"); ax.set_ylabel(r"$[B]$")
ax.set_xlim(0, xmax_e); ax.set_ylim(0, ymax_e)
ax.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="0.75")
ax.grid(True, alpha=0.15)

savefig(fig, "edelstein_phase")

print("Done — 10 figures saved.")
