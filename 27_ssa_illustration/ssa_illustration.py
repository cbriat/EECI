"""
Pedagogical SSA illustration — slide-sized, focused on the annotated events.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 17,
    "lines.linewidth": 2.2, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
})

BLUE = "#0072BD"; RED = "#D95319"; GOLD = "#C8A000"; BLACK = "#222222"
LGRAY = "#D0D0D0"

# ── Parameters ────────────────────────────────────────────────
k0 = 2.5;  k1 = 0.5;  x0 = 5
T = 2.0

# ── SSA ───────────────────────────────────────────────────────
rng = np.random.default_rng(15)
times, states, taus, rxn_ids, prop_list = [0.0], [x0], [], [], []
nX, t = x0, 0.0
while t < T:
    a0 = k0; a1 = k1 * nX; at = a0 + a1
    prop_list.append((a0, a1, at))
    tau = rng.exponential(1.0 / at)
    t += tau
    if t > T: break
    taus.append(tau)
    if rng.random() * at < a0:
        nX += 1; rxn_ids.append(0)
    else:
        nX -= 1; rxn_ids.append(1)
    times.append(t); states.append(nX)
times.append(T); states.append(nX)
n_ev = len(taus)

# ── Figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6.5))
fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.09)

n_annot = min(5, n_ev)

# ---- Step-function trajectory ----
ax.step(times, states, where="post", color=BLACK, lw=2.4, zorder=4)

# ---- Markers ----
for i in range(n_ev):
    t_ev = times[i + 1]; x_aft = states[i + 1]
    col = BLUE if rxn_ids[i] == 0 else RED
    mk  = "^"  if rxn_ids[i] == 0 else "v"
    ax.plot(t_ev, x_aft, mk, color=col, ms=12, zorder=6,
            markeredgewidth=0)

# ---- Vertical guides at annotated events ----
for i in range(n_annot + 1):
    ax.axvline(times[i], color=LGRAY, ls=":", lw=0.8, zorder=2)

# ---- Tau brackets ----
y_tau = -1.2
for i in range(n_annot):
    t_s, t_e = times[i], times[i + 1]
    t_m = 0.5 * (t_s + t_e)
    ax.annotate("", xy=(t_e, y_tau), xytext=(t_s, y_tau),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.5,
                                shrinkA=0, shrinkB=0))
    ax.text(t_m, y_tau - 0.4,
            rf"$\tau_{{{i+1}}}\!=\!{taus[i]:.3f}$",
            ha="center", va="top", fontsize=11, color=GOLD)

# ---- Reaction annotations ----
for i in range(n_annot):
    t_ev  = times[i + 1]
    x_aft = states[i + 1]
    col   = BLUE if rxn_ids[i] == 0 else RED
    a0, a1, at = prop_list[i]

    if rxn_ids[i] == 0:
        rxn_lbl = r"$\varnothing \!\to\! X$"
        p_lbl = rf"$p = {a0:.1f}/{at:.1f}$"
    else:
        rxn_lbl = r"$X \!\to\! \varnothing$"
        p_lbl = rf"$p = {a1:.1f}/{at:.1f}$"

    y_base = max(states[:i+2]) + 2.0 + 1.2 * (i % 2)

    ax.annotate(
        f"{rxn_lbl}\n{p_lbl}",
        xy=(t_ev, x_aft + 0.15),
        xytext=(t_ev, y_base),
        fontsize=11, color=col, ha="center", va="bottom",
        linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                        alpha=0.5, shrinkB=4),
    )

# ---- Initial-state box ----
a0_0, a1_0, at_0 = prop_list[0]
ax.text(0.06, x0 - 0.5,
        rf"$X = {x0}$" "\n"
        rf"$a_0 = k_0 = {a0_0:.1f}$" "\n"
        rf"$a_1 = k_1 X = {a1_0:.1f}$" "\n"
        rf"$a_{{\mathrm{{tot}}}} = {at_0:.1f}$",
        fontsize=12, color="0.3", va="top", linespacing=1.5,
        bbox=dict(fc="white", ec="0.75", pad=4, alpha=0.95,
                  boxstyle="round,pad=0.4"))

# ---- Algorithm box ----
algo = (
    "SSA iteration:\n"
    r"1.  Draw $\tau \sim \mathrm{Exp}\!\left(a_{\mathrm{tot}}\right)$" "\n"
    r"2.  Select reaction $j$ with prob.\ $a_j\! /\, a_{\mathrm{tot}}$" "\n"
    r"3.  $t \leftarrow t + \tau$,  $X \leftarrow X + \zeta_j$"
)
ax.text(0.99, 0.97, algo, transform=ax.transAxes,
        fontsize=12, va="top", ha="right", linespacing=1.7,
        bbox=dict(boxstyle="round,pad=0.5", fc="white",
                  ec="0.65", alpha=0.95))

# ---- Legend ----
ax.plot([], [], "^", color=BLUE, ms=11, markeredgewidth=0,
        label=r"Production $\varnothing \to X$")
ax.plot([], [], "v", color=RED,  ms=11, markeredgewidth=0,
        label=r"Degradation $X \to \varnothing$")
ax.plot([], [], color=BLACK, lw=2.4, label=r"State $X(t)$")
ax.legend(loc="upper left", framealpha=0.95, edgecolor="0.82",
          fontsize=12, handletextpad=0.4)

# ---- Axes ----
ax.set_xlabel(r"Time $t$")
ax.set_ylabel(r"Molecule count $X(t)$", labelpad=8)
ax.set_title(
    "Stochastic simulation algorithm (SSA)"
    r" — anatomy of a sample path",
    fontweight="bold", pad=10)
ax.set_xlim(-0.12, T)
ax.set_ylim(-3.0, max(states) + 6)
ax.grid(True, alpha=0.12)

fig.savefig("ssa_illustration.svg", bbox_inches="tight")
fig.savefig("ssa_illustration.png", dpi=200, bbox_inches="tight")
print("Figure saved.")
plt.close(fig)
