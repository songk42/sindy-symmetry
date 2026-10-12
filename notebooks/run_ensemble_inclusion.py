"""claude_artifacts review, item 4 (Section 5): E-SINDy-style inclusion probabilities
across the same window ensemble used throughout this session (8-cell matrix: freq
sweep duration=800 x stride 1/2/4/8 x4 windows each, + duration sweep 200/400/800/1200
x stride=1, up to 8 windows each -- same cells as run_r_fidelity_sweep.py), r=3,
standard constrained SINDy (fast, fit-only, no simulation).

Converts "2 of 11 slices break the a2<->a3 symmetry" style anecdotes into per-
coefficient inclusion probabilities, medians and IQRs -- a term whose inclusion
probability isn't near 0 or 1 is a coin flip the point estimate was hiding.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "claude_artifacts"))

import numpy as np

from sindy_pipeline import (compute_pod, build_symmetry_constraints, run_slice,
                             multi_window_starts, G as G3, A_SYMB)
import robustness_diagnostics as rd

POD_NPZ = "/gscratch/stf/songk42/ehd/sindy_data_fine.npz"
POD_CACHE = "/gscratch/stf/songk42/ehd/pod_cache_trimmed_t800.npz"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
r = 3

print("Loading POD cache...", flush=True)
Phi, sigma, Vt, t_full = compute_pod(POD_NPZ, r=6, t_start=800.0, method="svds",
                                      cache_path=POD_CACHE)
t_min, t_max = 800.0, 4000.0

cells = [dict(duration=800.0, stride=s, n_samples=4) for s in (1, 2, 4, 8)]
cells += [dict(duration=d, stride=1, n_samples=n)
          for d, n in [(200.0, 8), (400.0, 8), (800.0, 4), (1200.0, 2)]]

windows = []
for cell in cells:
    starts = multi_window_starts(t_min, t_max, cell["duration"], cell["n_samples"])
    for t_start in starts:
        windows.append(dict(t_start=t_start, t_end=t_start + cell["duration"],
                             stride=cell["stride"]))
print(f"{len(windows)} windows total across {len(cells)} cells", flush=True)

constraints = build_symmetry_constraints(r=r, group=G3, a_symb=A_SYMB)


def fit_fn(w):
    row = run_slice(sigma, Vt, t_full, r=r, t_start=w["t_start"], t_end=w["t_end"],
                     stride=w["stride"], constraints=constraints, augment=False,
                     group=G3, a_symb=A_SYMB, weak=False, check_sim=False,
                     check_fit_quality=False, max_iter=500)
    if row.get("fit") is None:
        raise RuntimeError(row.get("error", "fit failed"))
    return row["fit"]["Xi_raw"]


t0 = time.time()
res = rd.inclusion_probabilities(fit_fn, windows)
print(f"n_fits={res['n_fits']} n_failed={res['n_failed']}  ({time.time() - t0:.1f}s)", flush=True)

feat_names = constraints["full_library"]
feat_names = [str(f) for f in feat_names]
a_names = [s.name for s in A_SYMB]

print("\n=== Inclusion probability, median, IQR per (feature, target) -- only terms with", flush=True)
print("0 < inclusion_prob < 1 (i.e. NOT unanimous across the 38 windows) shown ===", flush=True)
print(f"{'feature':<14}{'target':<8}{'incl_prob':>10}{'median':>14}{'iqr':>14}{'sign_consist':>14}",
      flush=True)
lines_md = ["| feature | target | inclusion_prob | median | IQR | sign_consistency |", "|---|---|---|---|---|---|"]
n_coinflip = 0
for j, fname in enumerate(feat_names):
    for k, aname in enumerate(a_names):
        p = res["inclusion_prob"][j, k]
        if 0.0 < p < 1.0:
            n_coinflip += 1
            med, iqr, sc = res["median"][j, k], res["iqr"][j, k], res["sign_consistency"][j, k]
            print(f"{fname:<14}{aname:<8}{p:>10.2f}{med:>14.4g}{iqr:>14.4g}{sc:>14.2f}", flush=True)
            lines_md.append(f"| {fname} | {aname} | {p:.2f} | {med:.4g} | {iqr:.4g} | {sc:.2f} |")

print(f"\n{n_coinflip} of {len(feat_names) * len(a_names)} (feature,target) pairs are non-unanimous "
      f"across {res['n_fits']} windows.", flush=True)

print("\n=== For reference: terms with inclusion_prob == 1.0 (always included) ===", flush=True)
always = [(feat_names[j], a_names[k]) for j in range(len(feat_names)) for k in range(len(a_names))
          if res["inclusion_prob"][j, k] == 1.0]
print(always, flush=True)

LOGS_DIR.mkdir(exist_ok=True)
out_lines = [
    "# E-SINDy inclusion probabilities, r=3 constrained, 38-window ensemble",
    "",
    f"n_fits={res['n_fits']}, n_failed={res['n_failed']}.",
    "",
    f"{n_coinflip} of {len(feat_names) * len(a_names)} (feature, target) coefficient slots are "
    "non-unanimous (0 < inclusion_prob < 1) across the 38 windows:",
    "",
] + lines_md
(LOGS_DIR / "ensemble_inclusion_summary.md").write_text("\n".join(out_lines) + "\n")
print(f"\nSaved {LOGS_DIR / 'ensemble_inclusion_summary.md'}", flush=True)
np.savez(LOGS_DIR / "ensemble_inclusion_coeffs.npz", coefficients=res["coefficients"],
          inclusion_prob=res["inclusion_prob"], median=res["median"], iqr=res["iqr"],
          sign_consistency=res["sign_consistency"])
