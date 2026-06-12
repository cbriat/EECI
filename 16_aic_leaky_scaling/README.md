# Leaky AIC — Scaling Strategies (Deterministic)

Two strategies to recover adaptation under leakage (γ_c = 0.5):
- **Top:** Scale η = μ = θ = s (controller flux). Error ∝ (γ_c/s)² → 0.
- **Bottom:** Scale η, k = s (loop gain). Requires η ≫ k; shown with
  η = s² (works) vs η = s (saturates at 37% error).

**Run:** `python aic_leaky_scaling.py`
