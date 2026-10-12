"""Time compute_pod's SVD methods (CPU `svds` vs GPU `cupy`/`cupy_rsvd`) on real
sindy data files, so the CPU-only vs GPU tradeoff documented in compute_pod's
docstring has actual numbers behind it instead of just the OOM anecdote.

Usage: python -u benchmark_svd.py <npz_path> <method> [r]
Prints one line: "<method> <npz_path> n_t=<n_t> r=<r> seconds=<elapsed>"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sindy_pipeline import compute_pod

npz_path = sys.argv[1]
method = sys.argv[2]
r = int(sys.argv[3]) if len(sys.argv) > 3 else 6

t0 = time.perf_counter()
Phi, sigma, Vt, t = compute_pod(npz_path, r=r, method=method)
elapsed = time.perf_counter() - t0

print(f"{method} {npz_path} n_t={len(t)} r={r} seconds={elapsed:.3f} sigma={sigma}")
