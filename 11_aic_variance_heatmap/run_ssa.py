import numpy as np
import subprocess, time

nk, neta = 20, 20
nevents = 2_000_000
burn = 400_000

k_vals = np.logspace(-0.5, 2, nk)
eta_vals = np.logspace(-0.5, 3, neta)

# Prepare input
inp = " ".join(f"{v:.8f}" for v in k_vals) + "\n"
inp += " ".join(f"{v:.8f}" for v in eta_vals) + "\n"

print(f"Grid: {nk}x{neta}, {nevents/1e6:.1f}M events/point, "
      f"burn={burn/1e6:.1f}M")
t0 = time.time()

proc = subprocess.run(
    ["/home/claude/ssa_aic", str(nk), str(neta), str(nevents), str(burn)],
    input=inp, capture_output=True, text=True, timeout=240)

elapsed = time.time() - t0
print(f"SSA done in {elapsed:.1f}s")

# Parse output
lines = proc.stdout.strip().split("\n")
ratio = np.zeros((neta, nk))
Var_OL = 1.5
for i, line in enumerate(lines):
    vals = [float(x) for x in line.split()]
    for j, v in enumerate(vals):
        ratio[i, j] = v / Var_OL

np.savez("/home/claude/aic_ssa_data.npz",
         k_vals=k_vals, eta_vals=eta_vals, ratio=ratio)
print(f"Ratio range: [{ratio.min():.3f}, {ratio.max():.3f}]")
