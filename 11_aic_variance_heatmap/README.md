# AIC Variance Heatmap (SSA)

Gene expression + antithetic integral controller (Briat et al., 2016).
All rates = 1, μ = θ = 1. Heatmap of Var[P]_CL / Var[P]_OL over (k, η).
Computed via compiled C SSA kernel (2M events/point, 20×20 grid).

The stochastic system is ergodic for all k, η > 0 (no unstable region).
Ratio always ≥ 1.3: AIC amplifies noise for this network.

**Build & Run:**
```
gcc -O3 -o ssa_aic ssa_aic.c -lm
python run_ssa.py    # runs C binary, saves data
python plot_aic.py   # generates heatmap
```
