# AIC Z₁–Z₂ Covariance

Stationary Cov(Z₁,Z₂) vs η (k=1). Compiled C SSA, 3M events/point.
Cov ≈ -2.7 (always negative from sequestration). Correlation ρ ≈ -0.4.
Var(Z₂) decreases with η.

**Build & Run:**
```
gcc -O3 -o ssa_cov ssa_cov.c -lm
python aic_z1z2_covariance.py
```
