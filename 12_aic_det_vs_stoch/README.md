# AIC: Deterministic vs Stochastic Mean

Two figures comparing p(t) from ODE vs E[P(t)] from 1000 SSA runs:
- **Stable (k=1, η=100):** both converge to p*=1.
- **Limit cycle (k=5, η=100):** ODE oscillates permanently, stochastic
  mean converges (phase randomization across runs).

**Run:** `python aic_det_vs_stoch.py`
