"""claude_artifacts review, item 6 (Section 6): does a degree-2 (quadratic-only)
library with the same symmetry+energy constraints PLUS TrappingSR3's global-stability
promotion fix the switching-vanishes failure, or was the cubic library just absorbing
closure error as the review's Section 6 asked?

TrappingSR3 requires _n_tgts/_include_bias and a strictly quadratic PolynomialLibrary;
it accepts extra constraint_lhs/constraint_rhs and concatenates them with its own
auto-derived trapping/enstrophy constraints (confirmed by reading
pysindy/optimizers/trapping_sr3.py directly -- kwargs.pop("constraint_lhs") is
concatenated with the trapping constraints it builds internally). Reuses the same
symmetry+energy constraint machinery via build_symmetry_constraints(degree=2).

Same representative window as every other single-window test this session
(t_start=800, duration=800), for direct comparability with logs/representative_traj_summary.md.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pysindy as ps

from sindy_pipeline import compute_pod, build_symmetry_constraints, slice_amplitudes, G, A_SYMB

POD_NPZ = "/gscratch/stf/songk42/ehd/sindy_data_fine.npz"
POD_CACHE = "/gscratch/stf/songk42/ehd/pod_cache_trimmed_t800.npz"
T_START, DURATION = 800.0, 800.0
r = 3

print("Loading POD cache...", flush=True)
Phi, sigma, Vt, t_full = compute_pod(POD_NPZ, r=6, t_start=800.0, method="svds",
                                      cache_path=POD_CACHE)
X_raw, t = slice_amplitudes(sigma, Vt, t_full, r=r, t_start=T_START, t_end=T_START + DURATION, stride=1)

constraints = build_symmetry_constraints(r=r, degree=2, group=G, a_symb=A_SYMB)
print(f"degree=2 library: {len(constraints['full_library'])} features, "
      f"{constraints['constraints_lhs'].shape[0]} symmetry+energy constraint rows", flush=True)

X_scale = np.max(np.abs(X_raw))
X = X_raw / X_scale

for eta, beta, lam in [(1e-2, 1e-2, 1e-1), (1.0, 1e-2, 0.0), (0.1, 1e-3, 1e-2)]:
    print(f"\n--- TrappingSR3(eta={eta}, beta={beta}, reg_weight_lam={lam}) ---", flush=True)
    try:
        opt = ps.TrappingSR3(
            _n_tgts=r, _include_bias=True,
            constraint_lhs=constraints["constraints_lhs"],
            constraint_rhs=constraints["constraints_rhs"],
            constraint_order="target",
            eta=eta, beta=beta, reg_weight_lam=lam,
            regularizer="l1" if lam > 0 else "l2",
            max_iter=500, verbose=False,
        )
        model = ps.SINDy(feature_library=constraints["Theta"], optimizer=opt)
        t0 = time.time()
        model.fit(X, t=t, feature_names=[s.name for s in A_SYMB])
        print(f"  fit ok in {time.time() - t0:.1f}s", flush=True)
        model.print()
        Xi = model.coefficients().T
        Xi_raw = np.array(Xi)
        from sindy_pipeline import _exponent_tuple
        for j, m in enumerate(constraints["full_library"]):
            e = np.array(_exponent_tuple(m, A_SYMB), float)
            Xi_raw[j, :] = Xi[j, :] * X_scale / np.prod(X_scale ** e)
        feat_names = constraints["Theta"].get_feature_names([s.name for s in A_SYMB])
        i_a1a2 = feat_names.index("a1 a2")
        xi2 = Xi_raw[i_a1a2, 1]
        print(f"  xi2 (a1*a2 in a2') = {xi2:.4g}  (no cubic xi6 to compare at degree=2)", flush=True)

        t0 = time.time()
        x0 = X[0]
        t_eval = np.linspace(0, 800.0, 2000)
        Y = model.simulate(x0, t_eval, integrator_kws=dict(max_step=0.2)) * X_scale
        sim_time = time.time() - t0
        real_rate = int(np.sum(np.diff(np.sign(X_raw[:, 0])) != 0)) / DURATION * 1000.0
        if np.isfinite(Y).all():
            sim_rate = int(np.sum(np.diff(np.sign(Y[:, 0])) != 0)) / DURATION * 1000.0
            print(f"  sim ok ({sim_time:.1f}s): a1 range={np.ptp(Y[:, 0]):.4g}  "
                  f"real_rate={real_rate:.1f}/1000  sim_rate={sim_rate:.1f}/1000  "
                  f"fidelity={sim_rate / real_rate:.3g}", flush=True)
        else:
            print(f"  sim diverged/non-finite ({sim_time:.1f}s, {Y.shape[0]}/{len(t_eval)} steps)",
                  flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
