# EECI Course Figures — Control Theory & Reaction Networks

Lecture figures for Corentin Briat's EECI course covering dynamical systems,
control theory, stochastic reaction networks, and the antithetic integral
controller (AIC).

## Structure

Each folder contains a self-contained Python script (or C + Python for
SSA-based figures) that generates the corresponding SVG figures.

### Prerequisites

```
pip install numpy scipy matplotlib
```

For SSA-based figures (folders 11–18), a C compiler is needed:
```
gcc -O3 -o ssa_aic ssa_aic.c -lm
```

### Conventions

All figures use:
- **Font**: serif, Computer Modern math (`mathtext.fontset: "cm"`)
- **SVG output**: text as paths (`svg.fonttype: "path"`)
- **MATLAB colors**: `#0072BD` (blue), `#D95319` (orange), `#EDB120` (gold),
  `#7E2F8E` (purple), `#77AC30` (green)
- **Line width**: ≥ 2.2 for main curves
- Legends positioned to avoid covering plot data

---

| # | Topic | Figures |
|---|-------|---------|
| 01 | Separation principle | 1 |
| 02 | LQG fragility | 1 |
| 03 | Deficiency theory | 10 |
| 04 | Internal model principle | 2 |
| 05 | Logarithmic norms | 3 |
| 06 | Entrainment via contraction | 1 |
| 07 | Memorylessness (exponential) | 3 |
| 08 | Absorption at zero | 2 |
| 09 | Dual-rail (linear system) | 3 |
| 10 | Smooth ReLU approximation | 1 |
| 11 | AIC variance heatmap (SSA) | 1 |
| 12 | AIC det. vs stoch. mean | 2 |
| 13 | AIC ergodicity | 1 |
| 14 | AIC Z₁–Z₂ covariance | 1 |
| 15 | Leaky AIC (dilution effect) | 1 |
| 16 | Leaky AIC scaling strategies | 1 |
| 17 | AIC adaptation (4 scenarios) | 4 |
| 18 | AIC vs AIRC (rein controller) | 2 |
| 19 | Dual-rail (nonlinear, non-uniqueness) | 1 |
