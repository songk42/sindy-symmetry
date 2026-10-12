"""Run the frequency/length comparison sweep (see symmetric_sindy_v1.ipynb, "Comparing
sample trajectory frequency & length") from the command line, on a GPU node, using cupy
for the POD SVD (fast -- the CPU `svds` fallback in sindy_pipeline.compute_pod is too
slow for a ~40000-snapshot file like the current NEWEST_DATA).

Usage: submitted via run_window_sweep.slurm.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sindy_pipeline import compute_pod, build_symmetry_constraints, run_slice, print_row

NEWEST_DATA = "/gscratch/stf/songk42/ehd/sindy_data_fine.npz"
# Restrict the POD basis itself to the trimmed, transient-free window -- not just the
# amplitude series afterward. t=800 is where the growing instability settles into
# sustained switching (see the "Reproducing the paper's figures" trajectory plots);
# excluding t<800 keeps the transient/steady-state portion from influencing which
# spatial modes come out as PC1-3 at all.
POD_T_START = 800.0
# Cache the (GPU-computed) POD so a rerun -- e.g. after changing what run_slice reports,
# not the data or window -- doesn't need another GPU job just to redo the same SVD.
# Deleted automatically if NEWEST_DATA/r/t_start change (compute_pod checks its meta).
POD_CACHE = "/gscratch/stf/songk42/ehd/pod_cache_trimmed_t800.npz"

import subprocess
subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"])

Phi, sigma, Vt, t_full = compute_pod(NEWEST_DATA, r=6, method="cupy", t_start=POD_T_START,
                                      cache_path=POD_CACHE)
print(f"POD of {NEWEST_DATA} (t>={POD_T_START} only): sigma[:6] = {np.round(sigma[:6], 1)}")
print(f"  n_t={len(t_full)}, t in [{t_full[0]:.4g}, {t_full[-1]:.4g}]")
constraints = build_symmetry_constraints(r=3)

print()
print("=== FREQUENCY SWEEP: fixed window t=[1000,2000], varying dt via stride ===")
for stride in (1, 2, 4, 8):
    row = run_slice(sigma, Vt, t_full, r=3, t_start=1000, t_end=2000, stride=stride,
                     label=f"stride={stride} (dt~{0.05 * stride:g})",
                     constraints=constraints, check_sim=True)
    print_row(row)
    print()

print("=== LENGTH SWEEP: fixed dt=0.05 (stride=1), window end fixed at t=2000, varying start ===")
for t_start in (1900, 1800, 1600, 1400, 1200, 1000, 800):
    row = run_slice(sigma, Vt, t_full, r=3, t_start=t_start, t_end=2000, stride=1,
                     label=f"t_start={t_start} (span={2000 - t_start})",
                     constraints=constraints, check_sim=True)
    print_row(row)
    print()
