"""Multi-sample version of run_window_sweep.py: instead of one window per (dt,
duration) cell, draw several non-overlapping windows and report mean +/- std. See
sindy_pipeline.run_multi_slice / multi_window_starts for why -- a single window per
cell can't separate a real dt/duration trend from "which switching-cycle phase this
one window happened to start at" (see job 40061710's length sweep, whose ratio bounced
-189 to -244 with no visible trend as t_start varied).

Uses the t=[0,4000] extension (job 40135125, continuing job 40044144's t=[0,2000]
which continued the original bootstrap run) -- POD_T_START=800 gives 3200 time units
of switching-regime data to draw independent windows from, vs. 1200 before.

Usage: submitted via run_multi_window_sweep.slurm.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sindy_pipeline import compute_pod, build_symmetry_constraints, run_multi_slice, print_multi_summary

NEWEST_DATA = "/gscratch/stf/songk42/ehd/sindy_data_fine.npz"
POD_T_START = 800.0
POD_CACHE = "/gscratch/stf/songk42/ehd/pod_cache_trimmed_t800.npz"

import subprocess
subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"])

# cupy_rsvd (randomized SVD), not the full dense "cupy" SVD: at n_t~64000
# (t=[800,4000]) the dense SVD's workspace risks OOM even on an 80GB A100 (it was
# already ~15GB of workspace at n_t~24000 on an 11GB card). r=6 is tiny relative to
# n_t, so the randomized method is both cheaper and sufficiently accurate here.
Phi, sigma, Vt, t_full = compute_pod(NEWEST_DATA, r=6, method="cupy_rsvd",
                                      t_start=POD_T_START, cache_path=POD_CACHE)
print(f"POD of {NEWEST_DATA} (t>={POD_T_START} only): sigma[:6] = {np.round(sigma[:6], 1)}")
print(f"  n_t={len(t_full)}, t in [{t_full[0]:.4g}, {t_full[-1]:.4g}]"
      f"  (span={t_full[-1] - t_full[0]:.4g})")
constraints = build_symmetry_constraints(r=3)

print()
print("=== FREQUENCY SWEEP: duration=800, varying dt via stride, 4 samples each ===")
for stride in (1, 2, 4, 8):
    result = run_multi_slice(sigma, Vt, t_full, r=3, duration=800.0, stride=stride,
                              n_samples=4, label=f"stride={stride} (dt~{0.05 * stride:g})",
                              constraints=constraints)
    print_multi_summary(result)
    print()

print("=== DURATION SWEEP: dt=0.05 (stride=1), varying duration, up to 8 samples each ===")
for duration, n_samples in [(200.0, 8), (400.0, 8), (800.0, 4), (1200.0, 2)]:
    result = run_multi_slice(sigma, Vt, t_full, r=3, duration=duration, stride=1,
                              n_samples=n_samples, label=f"duration={duration:g}",
                              constraints=constraints)
    print_multi_summary(result)
    print()
