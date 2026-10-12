"""How different is the POD *spatial basis* itself across different snapshot windows?

Every sweep so far (window_sweep, multi_window_sweep, r_fidelity_sweep) computes POD
ONCE on the whole t>=800 range and then only slices/strides the resulting AMPLITUDE
series per window -- `compute_pod`'s own docstring flags that a POD fit on a different
ensemble of snapshots can give a different spatial basis, but this was never actually
tested. If the basis itself drifts between windows -- especially plausible since PC2/
PC3 are nearly degenerate (sigma ratio ~1.006, so an SVD is free to rotate within that
2D subspace) -- then "a2"/"a3" wouldn't even mean the same physical thing window to
window, which would explain coefficient instability at a level below anything the
SINDy fit itself could fix.

For each of several non-overlapping windows, computes a FRESH POD (fresh SVD, not
sliced from the shared r=6 cache) and compares the resulting Phi (spatial modes) and
sigma (singular values) across windows via:
  - per-mode cosine similarity |Phi_i(window a) . Phi_i(window b)| (sign-fixed against
    the global t>=800 basis), for each of the 6 modes independently
  - principal angles between the full r=6 column spaces of Phi(window a) and
    Phi(window b) (rotation/mixing-invariant -- catches within-subspace mode swapping
    that per-mode cosine similarity would miss)
  - principal angle between just the {mode2, mode3} 2D subspace across windows (the
    near-degenerate pair specifically)
  - sigma2/sigma3 ratio per window (how close to degenerate that pair is locally)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.linalg import subspace_angles

from sindy_pipeline import compute_pod

NPZ_PATH = "/gscratch/stf/songk42/ehd/sindy_data_fine.npz"
GLOBAL_CACHE = "/gscratch/stf/songk42/ehd/pod_cache_trimmed_t800.npz"
R = 6

windows = [("w1 [800,1600)", 800.0, 1600.0),
           ("w2 [1600,2400)", 1600.0, 2400.0),
           ("w3 [2400,3200)", 2400.0, 3200.0),
           ("w4 [3200,4000]", 3200.0, 4000.0)]

print("Loading global reference basis (cached, t=[800,4000])...", flush=True)
Phi_global, sigma_global, _, _ = compute_pod(NPZ_PATH, r=R, t_start=800.0, method="svds",
                                              cache_path=GLOBAL_CACHE)
print(f"  global sigma = {np.round(sigma_global, 1)}", flush=True)

bases = {"global": (Phi_global, sigma_global)}
for label, t0, t1 in windows:
    t_start_time = time.time()
    Phi, sigma, _, t = compute_pod(NPZ_PATH, r=R, t_start=t0, t_end=t1, method="svds")
    print(f"  {label}: n_t={len(t)} sigma={np.round(sigma, 1)}  ({time.time() - t_start_time:.1f}s)",
          flush=True)
    bases[label] = (Phi, sigma)

# Sign-fix every basis's columns against the global reference (SVD sign is arbitrary
# per mode; without this, cosine similarities would be meaningless).
Phi_ref = bases["global"][0]
for label in bases:
    Phi, sigma = bases[label]
    signs = np.sign(np.sum(Phi * Phi_ref, axis=0))
    signs[signs == 0] = 1.0
    bases[label] = (Phi * signs[None, :], sigma)

labels = ["global"] + [w[0] for w in windows]

print("\n=== Per-mode cosine similarity vs. global basis (sign-fixed) ===")
header = "mode  " + "  ".join(f"{l:>16s}" for l in labels)
print(header)
for i in range(R):
    row = [f"{np.dot(bases[l][0][:, i], Phi_ref[:, i]):16.5f}" for l in labels]
    print(f"a{i+1}    " + "  ".join(row))

print("\n=== Pairwise full-subspace (r=6) principal angles (degrees, max over the 6) ===")
print("      " + "  ".join(f"{l:>16s}" for l in labels))
for la in labels:
    row = []
    for lb in labels:
        ang = np.degrees(subspace_angles(bases[la][0], bases[lb][0])).max()
        row.append(f"{ang:16.3f}")
    print(f"{la:6s}" + "  ".join(row))

print("\n=== Pairwise {mode2, mode3} 2D-subspace principal angle (degrees, max) ===")
print("      " + "  ".join(f"{l:>16s}" for l in labels))
for la in labels:
    row = []
    for lb in labels:
        ang = np.degrees(subspace_angles(bases[la][0][:, 1:3], bases[lb][0][:, 1:3])).max()
        row.append(f"{ang:16.3f}")
    print(f"{la:6s}" + "  ".join(row))

print("\n=== sigma2/sigma3 ratio per window (how close to degenerate locally) ===")
for l in labels:
    sigma = bases[l][1]
    print(f"  {l}: sigma2/sigma3 = {sigma[1] / sigma[2]:.5f}  (sigma={np.round(sigma, 1)})")

# Save the raw bases for any follow-up plotting/analysis.
out = {}
for l in labels:
    key = l.split()[0]
    out[f"Phi_{key}"] = bases[l][0]
    out[f"sigma_{key}"] = bases[l][1]
out_path = Path(__file__).resolve().parent.parent / "logs" / "pod_modes_per_window.npz"
np.savez(out_path, **out)
print(f"\nSaved {out_path}", flush=True)
