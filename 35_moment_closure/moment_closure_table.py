import matplotlib
import matplotlib.pyplot as plt
import textwrap

matplotlib.rcParams.update({
    "svg.fonttype": "path", "mathtext.fontset": "cm",
    "font.family": "serif", "font.size": 12,
})

methods = [
    ("Normal\n(Gaussian)",
     "Cumulants of order ≥ 3 are zero",
     "Negative concentrations;\nfails at low copy numbers"),
    ("Log-normal",
     "Species follow a\nlog-normal distribution",
     "Poor near zero;\nno multimodality"),
    ("Poisson",
     "Var(X) = E[X],\nspecies independent",
     "Fano = 1 assumed;\nviolated by feedback"),
    ("Beta / binomial",
     r"X ∈ {0,…,N};  X² = X" "\n" "for binary species",
     "Only for bounded\n(e.g. gene-state) species"),
    ("Zero cumulant\n(order n)",
     "Cumulants of order > n\nset to zero",
     "Can diverge;\ncombinatorial growth"),
    ("Derivative\nmatching",
     "Match time derivatives of\nexact & approx. moments",
     "Algebraically heavy;\npossible neg. variances"),
    ("Maximum\nentropy",
     "Max Shannon entropy\ngiven known moments",
     "Optimisation at each step;\nimplicit ODEs"),
    ("Conditional\nmoment",
     "Condition on discrete\nspecies (e.g. gene state)",
     "Needs timescale separation;\nfew discrete states"),
    ("Linear mapping\n(LMA)",
     "Linearise propensities\naround operating point",
     "Degrades far from\nsteady state"),
]

n = len(methods)
col_colors = ["#0072BD", "#2E8B57", "#A2142F"]
header_bg  = "#2C3E50"
row_even   = "#F7F9FB"
row_odd    = "#FFFFFF"

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.axis("off")
fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.01)

fig.suptitle("Moment closure methods for stochastic reaction networks",
             fontsize=16, fontweight="bold", y=0.98)

col_widths = [0.19, 0.38, 0.38]
col_labels = ["Method", "Key assumption", "Main limitation"]

table_data = [[m, a, l] for m, a, l in methods]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    colWidths=col_widths,
    loc="center",
    cellLoc="left",
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.4)

# Style header
for j in range(3):
    cell = table[0, j]
    cell.set_facecolor(header_bg)
    cell.set_text_props(color="white", fontweight="bold", fontsize=12.5)
    cell.set_edgecolor("white")
    cell.set_linewidth(1.5)

# Style body
for i in range(1, n + 1):
    bg = row_even if i % 2 == 0 else row_odd
    for j in range(3):
        cell = table[i, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.8)
        txt = cell.get_text()
        if j == 0:
            txt.set_fontweight("bold")
            txt.set_color(col_colors[0])
            txt.set_fontsize(11)
        elif j == 1:
            txt.set_color("#2E4A2E")
            txt.set_fontsize(10.5)
        else:
            txt.set_color(col_colors[2])
            txt.set_fontsize(10.5)
            txt.set_fontstyle("italic")

fig.savefig("moment_closure_table.svg", bbox_inches="tight")
fig.savefig("moment_closure_table.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved moment_closure_table.svg")
