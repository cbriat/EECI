"""
Memorylessness of the exponential distribution.
Three figures: density overlay, Poisson process timeline, histogram.
"""
import numpy as np
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
PURPLE="#7E2F8E"; GOLD="#EDB120"; RED="#A2142F"
OUT = "/mnt/user-data/outputs"

lam = 1.0

# ══════════════════════════════════════════════════════════════
#  FIGURE 1: Conditional density overlay
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.12)

t = np.linspace(0, 7, 500)
f_exp = lam * np.exp(-lam * t)

ax.plot(t, f_exp, color=BLUE, lw=3.0,
        label=r"$f_T(t) = \lambda e^{-\lambda t}$")

s_vals = [1.0, 2.0, 3.0]
colors_s = [ORANGE, GREEN, PURPLE]
for s, col in zip(s_vals, colors_s):
    t_shifted = t + s
    f_cond = lam * np.exp(-lam * t)
    ax.plot(t_shifted, f_cond, color=col, lw=2.2, ls="--",
            label=rf"$f_{{T-{s:.0f} \mid T>{s:.0f}}}(t)$")
    ax.axvline(s, color=col, lw=0.8, ls=":", alpha=0.4)

ax.set_xlabel(r"Time $t$")
ax.set_ylabel("Density")
ax.set_title(
    r"Memorylessness: $P(T > s+t \mid T > s) = P(T > t)$",
    fontsize=14, fontweight="bold")
ax.set_xlim(0, 7); ax.set_ylim(0, 1.15)
ax.legend(loc="center right", framealpha=0.95, edgecolor="0.75")
ax.grid(True, alpha=0.15)

# Text in upper right (above the legend, where curves are flat)
ax.text(0.97, 0.97,
        "All conditional densities are\n"
        "identical exponentials, shifted",
        transform=ax.transAxes, fontsize=11, ha="right", va="top",
        fontstyle="italic", color="0.35")

fig.savefig(f"{OUT}/memoryless_density.svg", bbox_inches="tight")
plt.close()
print("Saved memoryless_density.svg")

# ══════════════════════════════════════════════════════════════
#  FIGURE 2: Poisson process timeline
# ══════════════════════════════════════════════════════════════
# Use hand-picked events for clean spacing
events = np.array([0.8, 2.0, 3.4, 4.8, 7.2, 8.5, 9.5, 10.8, 11.8])

fig, ax = plt.subplots(figsize=(12, 5.0))
fig.subplots_adjust(left=0.04, right=0.97, top=0.82, bottom=0.12)

# Timeline
ax.axhline(0, color="0.3", lw=2.5)
ax.plot(events, np.zeros_like(events), "o", color=BLUE, ms=12, zorder=4,
        markeredgecolor="white", markeredgewidth=1.5)

# Label some events
for i, ev in enumerate(events[:7]):
    ax.text(ev, -0.25, rf"$t_{i+1}$", ha="center", fontsize=10,
            color=BLUE)

# Inter-arrival brackets (below the line)
starts = np.concatenate([[0], events[:-1]])
for i in range(6):
    t1, t2 = starts[i], events[i]
    mid = (t1+t2)/2
    ax.annotate("", xy=(t2, -0.55), xytext=(t1, -0.55),
                arrowprops=dict(arrowstyle="<->", color="0.5", lw=1.2))
    ax.text(mid, -0.72, rf"$\tau_{i+1}$", ha="center", fontsize=10,
            color="0.45")

# "now" line between event 4 and 5
s = events[3] + 0.45 * (events[4] - events[3])
ax.axvline(s, color=RED, lw=3, ls="-", zorder=3)
ax.text(s+0.1, 1.25, r'"now" ($s$)', ha="left", fontsize=13,
        color=RED, fontweight="bold")

# Already waited (from last event to s)
t_last = events[3]
ax.annotate("", xy=(s-0.05, 0.50), xytext=(t_last, 0.50),
            arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2.5))
ax.text(s-0.2, 0.65, "already waited", ha="right",
        fontsize=11, color=ORANGE, fontweight="bold")

# Remaining wait (from s to next event) — offset higher
t_next = events[4]
ax.annotate("", xy=(t_next, 0.85), xytext=(s+0.05, 0.85),
            arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2.5))
ax.text(s+0.2, 1.0, "remaining wait", ha="left",
        fontsize=11, color=GREEN, fontweight="bold")

ax.set_xlabel(r"Time $t$")
ax.set_xlim(-0.3, 12.2)
ax.set_ylim(-1.0, 1.6)
ax.set_yticks([])
ax.set_title(
    "Poisson process: the remaining wait has the same distribution "
    "as a fresh inter-arrival time",
    fontsize=13, fontweight="bold")

# Annotation box in the right side
ax.text(0.97, -0.60,
        r"$\tau_i \sim \mathrm{Exp}(\lambda)$ i.i.d."
        "    "
        r"Remaining wait $\sim \mathrm{Exp}(\lambda)$, "
        "regardless of time already waited",
        transform=ax.transAxes, fontsize=11, ha="right", va="top",
        bbox=dict(fc="#EBF5E0", ec=GREEN, pad=5,
                  boxstyle="round,pad=0.4"))

fig.savefig(f"{OUT}/memoryless_timeline.svg", bbox_inches="tight")
plt.close()
print("Saved memoryless_timeline.svg")

# ══════════════════════════════════════════════════════════════
#  FIGURE 3: Histogram comparison
# ══════════════════════════════════════════════════════════════
N = 200_000
rng = np.random.default_rng(42)
samples = rng.exponential(1/lam, size=N)

s = 2.0
mask = samples > s
residuals = samples[mask] - s

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.12)

bins = np.linspace(0, 8, 60)
ax.hist(samples, bins=bins, density=True, alpha=0.5, color=BLUE,
        edgecolor="white", lw=0.5,
        label=rf"All samples $T \sim \mathrm{{Exp}}({lam:.0f})$")
ax.hist(residuals, bins=bins, density=True, alpha=0.5, color=ORANGE,
        edgecolor="white", lw=0.5,
        label=rf"Residuals $T - {s:.0f} \mid T > {s:.0f}$")

t_th = np.linspace(0, 8, 300)
ax.plot(t_th, lam * np.exp(-lam * t_th), "k-", lw=2.5, alpha=0.7,
        label=r"$\mathrm{Exp}(\lambda)$ density")

ax.set_xlabel(r"$t$")
ax.set_ylabel("Density")
ax.set_title(
    rf"Empirical test: $T - s \mid T > s$ has the same distribution "
    rf"as $T$  ($s = {s:.0f}$,  $N = {N//1000}$k samples)",
    fontsize=13, fontweight="bold")
ax.set_xlim(0, 8); ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.75",
          fontsize=11)
ax.grid(True, alpha=0.15)

fig.savefig(f"{OUT}/memoryless_histogram.svg", bbox_inches="tight")
plt.close()
print("Saved memoryless_histogram.svg")
