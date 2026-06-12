# Dual-Rail Representation (Linear System)

dx/dt = Ax + b with A = [[-1,-1],[1,-1]], b = [1,-1]. Equilibrium x* = [1,0].
Dual-rail decomposition: x = x⁺ - x⁻ with annihilation η·x⁺∘x⁻.

Three figures for η = 1, 10, 100. Larger η gives tighter decomposition
(less slack). x⁺ - x⁻ = x to ~10⁻⁷ for all η.

**Run:** `python dual_rail.py`
