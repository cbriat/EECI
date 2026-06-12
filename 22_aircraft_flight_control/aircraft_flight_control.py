"""Schematic of an airliner (top view) with control systems labelled,
plus a side panel summarizing the control systems involved.

Output: SVG figure for use as a slide on aerospace applications of control.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle, FancyBboxPatch
import numpy as np

# ------------------------------------------------------------------
# Colors (MATLAB palette + neutrals)
# ------------------------------------------------------------------
MAT_BLUE   = "#0072BD"
MAT_RED    = "#D95319"
MAT_YELLOW = "#EDB120"
MAT_PURPLE = "#7E2F8E"
MAT_GREEN  = "#77AC30"
MAT_CYAN   = "#4DBEEE"
PLANE_FILL = "#E8E8E8"
PLANE_EDGE = "#303030"
ENGINE_FILL = "#5A5A5A"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
})

# ------------------------------------------------------------------
# Figure layout: plane on the left, summary panel on the right
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 7.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1], wspace=0.02)
ax = fig.add_subplot(gs[0, 0])
ax_text = fig.add_subplot(gs[0, 1])
ax.set_aspect("equal")
ax.set_xlim(-13, 14)
ax.set_ylim(-9.5, 9.5)
ax.axis("off")
ax_text.axis("off")

# ------------------------------------------------------------------
# Wing / stab geometry parameters
# ------------------------------------------------------------------
NOSE_X = 10
TAIL_X = -8

# Main wing
W_ROOT_LE = 2.5
W_ROOT_TE = -2.0
W_TIP_LE = -2.0
W_TIP_TE = -5.0
W_SPAN = 7.5

# Horizontal stab
HS_ROOT_LE = -5.3
HS_ROOT_TE = -7.4
HS_TIP_LE  = -6.8
HS_TIP_TE  = -7.8
HS_SPAN = 3.2

# ------------------------------------------------------------------
# Fuselage outline
# ------------------------------------------------------------------
fus_pts = [
    (TAIL_X - 0.4, 0.0),
    (TAIL_X - 0.1, -0.55),
    (TAIL_X + 0.4, -0.75),
    (NOSE_X - 1.5, -0.75),
    (NOSE_X - 0.4, -0.50),
    (NOSE_X, 0.0),
    (NOSE_X - 0.4, 0.50),
    (NOSE_X - 1.5, 0.75),
    (TAIL_X + 0.4, 0.75),
    (TAIL_X - 0.1, 0.55),
]
ax.add_patch(Polygon(fus_pts, closed=True,
                     facecolor=PLANE_FILL, edgecolor=PLANE_EDGE, lw=1.5, zorder=2))

# ------------------------------------------------------------------
# Wings (swept back)
# ------------------------------------------------------------------
def make_lifting_surface(root_le, root_te, tip_le, tip_te, span, y_root=0.75):
    return [
        (root_le,  y_root),
        (tip_le,   span),
        (tip_te,   span),
        (root_te,  y_root),
    ]

right_wing = make_lifting_surface(W_ROOT_LE, W_ROOT_TE, W_TIP_LE, W_TIP_TE, W_SPAN)
left_wing  = [(x, -y) for (x, y) in right_wing]

for pts in (right_wing, left_wing):
    ax.add_patch(Polygon(pts, closed=True,
                         facecolor=PLANE_FILL, edgecolor=PLANE_EDGE, lw=1.5, zorder=1))

# Horizontal stab
right_hstab = make_lifting_surface(HS_ROOT_LE, HS_ROOT_TE, HS_TIP_LE, HS_TIP_TE, HS_SPAN)
left_hstab  = [(x, -y) for (x, y) in right_hstab]

for pts in (right_hstab, left_hstab):
    ax.add_patch(Polygon(pts, closed=True,
                         facecolor=PLANE_FILL, edgecolor=PLANE_EDGE, lw=1.5, zorder=1))

# ------------------------------------------------------------------
# Vertical fin (seen from above: thin diamond along centerline)
# ------------------------------------------------------------------
vfin = [(-5.3, 0.0), (-7.0, 0.45), (-8.1, 0.0), (-7.0, -0.45)]
ax.add_patch(Polygon(vfin, closed=True,
                     facecolor="#C4C4C4", edgecolor=PLANE_EDGE, lw=1.5, zorder=3))

# ------------------------------------------------------------------
# Helpers for control-surface strips
# ------------------------------------------------------------------
def chord_at(span_frac, root_le, root_te, tip_le, tip_te):
    le = root_le + span_frac * (tip_le - root_le)
    te = root_te + span_frac * (tip_te - root_te)
    return le, te

def trailing_strip(y_in, y_out, chord_frac,
                   root_le, root_te, tip_le, tip_te, total_span, y_root=0.75):
    """Trailing-edge strip occupying chord_frac of local chord, from y_in to y_out."""
    s_in  = (y_in  - y_root) / (total_span - y_root)
    s_out = (y_out - y_root) / (total_span - y_root)
    le_in,  te_in  = chord_at(s_in,  root_le, root_te, tip_le, tip_te)
    le_out, te_out = chord_at(s_out, root_le, root_te, tip_le, tip_te)
    chord_in  = le_in  - te_in
    chord_out = le_out - te_out
    fwd_in  = te_in  + chord_frac * chord_in
    fwd_out = te_out + chord_frac * chord_out
    return [(fwd_in, y_in), (fwd_out, y_out), (te_out, y_out), (te_in, y_in)]

def midchord_strip(y_in, y_out, x_le_frac, x_te_frac,
                   root_le, root_te, tip_le, tip_te, total_span, y_root=0.75):
    s_in  = (y_in  - y_root) / (total_span - y_root)
    s_out = (y_out - y_root) / (total_span - y_root)
    le_in,  te_in  = chord_at(s_in,  root_le, root_te, tip_le, tip_te)
    le_out, te_out = chord_at(s_out, root_le, root_te, tip_le, tip_te)
    chord_in  = le_in  - te_in
    chord_out = le_out - te_out
    # x_le_frac, x_te_frac are measured forward from TE as fraction of chord
    aft_in   = te_in  + x_le_frac * chord_in
    fwd_in   = te_in  + x_te_frac * chord_in
    aft_out  = te_out + x_le_frac * chord_out
    fwd_out  = te_out + x_te_frac * chord_out
    return [(fwd_in, y_in), (fwd_out, y_out), (aft_out, y_out), (aft_in, y_in)]

# ------------------------------------------------------------------
# Control surfaces (color-coded)
# ------------------------------------------------------------------
# Ailerons: outboard trailing edge
ail_r = trailing_strip(4.7, 7.4, 0.28,
                       W_ROOT_LE, W_ROOT_TE, W_TIP_LE, W_TIP_TE, W_SPAN)
ail_l = [(x, -y) for (x, y) in ail_r]

# Flaps: inboard trailing edge
flap_r = trailing_strip(1.2, 4.4, 0.28,
                        W_ROOT_LE, W_ROOT_TE, W_TIP_LE, W_TIP_TE, W_SPAN)
flap_l = [(x, -y) for (x, y) in flap_r]

# Spoilers: mid-chord top of wing (panels)
spoil_r = midchord_strip(2.0, 4.2, 0.30, 0.55,
                         W_ROOT_LE, W_ROOT_TE, W_TIP_LE, W_TIP_TE, W_SPAN)
spoil_l = [(x, -y) for (x, y) in spoil_r]

# Elevators: trailing edge of horizontal stab
elev_r = trailing_strip(1.0, 3.0, 0.45,
                        HS_ROOT_LE, HS_ROOT_TE, HS_TIP_LE, HS_TIP_TE, HS_SPAN)
elev_l = [(x, -y) for (x, y) in elev_r]

# Rudder: aft portion of vertical fin (in top view, the back half of the diamond)
rudder = [(-7.0, 0.45), (-8.1, 0.0), (-7.0, -0.45), (-6.6, 0.0)]

# Draw all control surfaces
surfaces = [
    (ail_r,   MAT_RED,    "Aileron"),
    (ail_l,   MAT_RED,    None),
    (flap_r,  MAT_YELLOW, "Flap"),
    (flap_l,  MAT_YELLOW, None),
    (spoil_r, MAT_PURPLE, "Spoiler"),
    (spoil_l, MAT_PURPLE, None),
    (elev_r,  MAT_GREEN,  "Elevator"),
    (elev_l,  MAT_GREEN,  None),
    (rudder,  MAT_CYAN,   "Rudder"),
]
for pts, color, _ in surfaces:
    ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                         edgecolor=PLANE_EDGE, lw=0.8, zorder=4))

# ------------------------------------------------------------------
# Engines (ovals under wings)
# ------------------------------------------------------------------
ENG_X = -0.2
ENG_Y = 3.4
for sign in (+1, -1):
    e = Ellipse((ENG_X, sign * ENG_Y), width=2.6, height=1.0,
                facecolor=ENGINE_FILL, edgecolor=PLANE_EDGE, lw=1, zorder=3)
    ax.add_patch(e)

# ------------------------------------------------------------------
# Body-frame axes drawn at the CG
# ------------------------------------------------------------------
CG = (-0.8, 0.0)
AX_LEN = 2.2  # arrow length

# x forward (roll)
ax.annotate("", xy=(CG[0] + AX_LEN, CG[1]), xytext=CG,
            arrowprops=dict(arrowstyle="->", lw=2.0, color="black"), zorder=10)
ax.text(CG[0] + AX_LEN + 0.15, CG[1] - 0.05, r"$x$  (roll $\phi$)",
        fontsize=10, va="center", ha="left", zorder=10)

# y right wing (pitch). NOTE: with nose pointing right in the figure, the
# pilot's right wing is at the bottom of the figure, so y points down here.
ax.annotate("", xy=(CG[0], CG[1] - AX_LEN), xytext=CG,
            arrowprops=dict(arrowstyle="->", lw=2.0, color="black"), zorder=10)
ax.text(CG[0] + 0.15, CG[1] - AX_LEN - 0.10, r"$y$  (pitch $\theta$)",
        fontsize=10, va="top", ha="left", zorder=10)

# z into the page (yaw): circle with X
ax.add_patch(Circle(CG, 0.30, facecolor="white", edgecolor="black",
                    lw=1.5, zorder=11))
ax.plot([CG[0] - 0.21, CG[0] + 0.21],
        [CG[1] - 0.21, CG[1] + 0.21], "k", lw=1.5, zorder=12)
ax.plot([CG[0] - 0.21, CG[0] + 0.21],
        [CG[1] + 0.21, CG[1] - 0.21], "k", lw=1.5, zorder=12)
ax.text(CG[0] - 0.5, CG[1] + 0.05, r"$z$  (yaw $\psi$)",
        fontsize=10, va="center", ha="right", zorder=10)

# ------------------------------------------------------------------
# Callout labels with leader lines
# ------------------------------------------------------------------
def callout(text, xy, xytext, color="black"):
    ax.annotate(text, xy=xy, xytext=xytext,
                fontsize=11, color="black",
                arrowprops=dict(arrowstyle="-", lw=0.8, color=color,
                                shrinkA=0, shrinkB=2),
                ha="center", va="center", zorder=15,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=color, lw=1.2))

# Ailerons
callout("Aileron\n(roll $\\phi$)", xy=(-3.8, 6.2), xytext=(2.0, 8.6),
        color=MAT_RED)
callout("Aileron\n(roll $\\phi$)", xy=(-3.8, -6.2), xytext=(2.0, -8.6),
        color=MAT_RED)

# Flaps
callout("Flap\n(high lift)", xy=(-2.7, 2.6), xytext=(-9.8, 5.6),
        color=MAT_YELLOW)
callout("Flap\n(high lift)", xy=(-2.7, -2.6), xytext=(-9.8, -5.6),
        color=MAT_YELLOW)

# Spoilers
callout("Spoiler\n(lift dump, on upper wing)", xy=(-1.5, 3.4), xytext=(7.5, 6.5),
        color=MAT_PURPLE)

# Elevators
callout("Elevator\n(pitch $\\theta$)", xy=(-7.4, 2.0), xytext=(-11.5, 3.3),
        color=MAT_GREEN)

# Rudder
callout("Rudder\n(yaw $\\psi$, on vertical fin)",
        xy=(-7.5, 0.2), xytext=(-11.6, -1.0), color=MAT_CYAN)

# Engines
callout("Engine\n(thrust, FADEC)", xy=(0.6, 3.4), xytext=(8.0, 3.4),
        color="#404040")

# ------------------------------------------------------------------
# Title (over the plane axes)
# ------------------------------------------------------------------
ax.set_title("Aircraft flight control: control surfaces and body-frame axes",
             fontsize=13, pad=6)

# ------------------------------------------------------------------
# Side panel: summary of control systems
# ------------------------------------------------------------------
ax_text.set_xlim(0, 1)
ax_text.set_ylim(0, 1)

# Background box
ax_text.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#FAFAFA",
                                 edgecolor="#999999", lw=1.0,
                                 transform=ax_text.transAxes))

header = "CONTROL SYSTEMS INVOLVED"
ax_text.text(0.5, 0.955, header,
             fontsize=13, fontweight="bold",
             ha="center", va="top",
             transform=ax_text.transAxes)

sections = [
    ("Primary flight controls (3-axis attitude)",
     [r"Ailerons  $\rightarrow$  roll  ($\phi$)",
      r"Elevators $\rightarrow$  pitch ($\theta$)",
      r"Rudder    $\rightarrow$  yaw  ($\psi$)"]),
    ("Secondary flight controls (lift / drag)",
     ["Flaps, slats: high-lift devices",
      "Spoilers: lift dumpers, speed brakes",
      "Trim tabs: hands-off neutral"]),
    ("Propulsion control",
     ["Throttle, autothrottle",
      "FADEC (engine control unit)"]),
    ("Automatic flight control",
     ["Autopilot (attitude, alt, hdg, speed holds)",
      "Flight Management System (FMS)",
      "Fly-by-wire control laws",
      "Yaw damper, stability augmentation",
      "Flight envelope protection"]),
    ("Sensing and state estimation",
     ["Inertial Reference Unit (IRU)",
      "GPS / GNSS",
      "Air data computer (pitot, static)",
      "Radio altimeter, ILS / GBAS"]),
]

y = 0.895
line_h = 0.030
section_gap = 0.014
for title, items in sections:
    ax_text.text(0.05, y, title,
                 fontsize=10.5, fontweight="bold",
                 ha="left", va="top",
                 color=MAT_BLUE,
                 transform=ax_text.transAxes)
    y -= line_h
    for item in items:
        ax_text.text(0.08, y, "\u2022  " + item,
                     fontsize=10, ha="left", va="top",
                     transform=ax_text.transAxes)
        y -= line_h
    y -= section_gap

plt.savefig("/mnt/user-data/outputs/aircraft_flight_control.svg",
            format="svg", bbox_inches="tight")
plt.close(fig)
print("Wrote /mnt/user-data/outputs/aircraft_flight_control.svg")
