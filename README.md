# EECI Course Figures — Control Theory & Reaction Networks

Lecture figures for Corentin Briat's EECI course covering dynamical systems,
control theory, stochastic reaction networks, and the antithetic integral
controller (AIC).

## Prerequisites

```
pip install numpy scipy matplotlib
```

For SSA-based figures (folders 11–18), a C compiler is needed:
```
gcc -O3 -o ssa_aic ssa_aic.c -lm
```

For the H∞ augmented plant (folder 24): `pdflatex` + `pdf2svg`.

## Conventions

- **Font**: serif, Computer Modern math (`mathtext.fontset: "cm"`)
- **SVG output**: text as paths (`svg.fonttype: "path"`)
- **MATLAB colors**: `#0072BD` (blue), `#D95319` (orange), `#EDB120` (gold),
  `#7E2F8E` (purple), `#77AC30` (green)
- **Line width**: ≥ 2.2 for main curves
- Legends positioned to avoid covering plot data

Each folder contains a Python script, generated SVG figures, a README,
and a companion Jupyter notebook.

---

## Part 1 — Dynamical Systems & Control

| # | Topic | Figures |
|---|-------|---------|
| 20 | Bifurcation diagrams | 3 |
| 21 | Three reasons for feedback | 3 |
| 22 | Aircraft flight control schematic | 1 |
| 23 | LQR weight sweep | 1 |
| 24 | Augmented plant H∞ (TikZ) | 1 |
| 25 | LQG vs robust controller | 1 |
| 26 | MPC comparison | 1 |
| 01 | Separation principle | 1 |
| 02 | LQG fragility | 1 |
| 04 | Internal model principle | 2 |
| 05 | Logarithmic norms | 3 |
| 06 | Entrainment via contraction | 1 |

## Part 2 — Reaction Network Theory

| # | Topic | Figures |
|---|-------|---------|
| 03 | Deficiency theory | 10 |
| 08 | Absorption at zero | 2 |
| 09 | Dual-rail (linear system) | 3 |
| 10 | Smooth ReLU approximation | 1 |
| 19 | Dual-rail (nonlinear, non-uniqueness) | 1 |
| 36 | Regularization functions | 4 |

## Part 3 — Stochastic Systems

| # | Topic | Figures |
|---|-------|---------|
| 27 | SSA (Gillespie) illustration | 1 |
| 28 | Reaction network at low copy numbers | 1 |
| 29 | Kurtz convergence theorem | 1 |
| 30 | Finite State Projection (FSP) | 5 |
| 31 | Vilar circadian clock | 4 |
| 32 | Stochastic entrainment (Arnold tongue) | 2 |
| 07 | Memorylessness (exponential) | 3 |
| 34 | Noise bounds | 2 |
| 35 | Moment closure methods | 1 |

## Part 4 — Antithetic Integral Controller

| # | Topic | Figures |
|---|-------|---------|
| 33 | AIC on birth-death process | 1 |
| 11 | AIC variance heatmap (SSA) | 1 |
| 12 | AIC det. vs stoch. mean | 2 |
| 13 | AIC ergodicity | 1 |
| 14 | AIC Z₁–Z₂ covariance | 1 |
| 15 | Leaky AIC (dilution effect) | 1 |
| 16 | Leaky AIC scaling strategies | 1 |
| 17 | AIC adaptation (4 scenarios) | 4 |
| 18 | AIC vs AIRC (rein controller) | 2 |
