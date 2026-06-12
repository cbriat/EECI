# Leaky AIC — Dilution Effect

Z₁→∅ and Z₂→∅ at rate γ_c. Sweep γ_c ∈ [0,2] at k=1, η=100.
E[P] drops from 1.000 (γ_c=0) to 0.41 (γ_c=2, 59% error).

**Build & Run:**
```
gcc -O3 -o ssa_leaky ssa_leaky.c -lm
python aic_leaky.py
```
