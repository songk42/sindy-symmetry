"""claude_artifacts review, item 4 (Section 4): MIOSR isn't installed in this pysindy
build (needs a MIP solver) -- pysindy's own reg_weight_lam sweep on ConstrainedSR3 is
the fallback the review itself names ("Fix it with sparsity rather than scaling:
reg_weight_lam > 0, or MIOSR"). Sweeps reg_weight_lam across the SAME 38-window
ensemble as run_ensemble_inclusion.py and reports how the collinear a1/a1^3 pair (the
worst-conditioned pair identified by library_conditioning) and the xi2/xi6 saddle
ratio's cross-window consistency respond.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from sindy_pipeline import (compute_pod, build_symmetry_constraints, run_slice,
                             multi_window_starts, xi_ratio, G as G3, A_SYMB)

POD_NPZ = "/gscratch/stf/songk42/ehd/sindy_data_fine.npz"
POD_CACHE = "/gscratch/stf/songk42/ehd/pod_cache_trimmed_t800.npz"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
r = 3
LAMBDAS = [0.0, 0.001, 0.01, 0.05, 0.1]

print("Loading POD cache...", flush=True)
Phi, sigma, Vt, t_full = compute_pod(POD_NPZ, r=6, t_start=800.0, method="svds",
                                      cache_path=POD_CACHE)
t_min, t_max = 800.0, 4000.0

cells = [dict(duration=800.0, stride=s, n_samples=4) for s in (1, 2, 4, 8)]
cells += [dict(duration=d, stride=1, n_samples=n)
          for d, n in [(200.0, 8), (400.0, 8), (800.0, 4), (1200.0, 2)]]
windows = []
for cell in cells:
    for t_start in multi_window_starts(t_min, t_max, cell["duration"], cell["n_samples"]):
        windows.append(dict(t_start=t_start, t_end=t_start + cell["duration"], stride=cell["stride"]))
print(f"{len(windows)} windows", flush=True)

constraints = build_symmetry_constraints(r=r, group=G3, a_symb=A_SYMB)
feat_names = [str(f) for f in constraints["full_library"]]
i_a1 = feat_names.index("a1")
i_a1cub = feat_names.index("a1**3")

lines_md = ["| reg_weight_lam | mean nnz | a1-coef (mean+-std) | a1^3-coef (mean+-std) | "
            "xi2/xi6 (mean+-std) | frac saddle (|ratio|>10) |", "|---|---|---|---|---|---|"]

for lam in LAMBDAS:
    t0 = time.time()
    a1_coefs, a1cub_coefs, ratios, nnzs, n_ok = [], [], [], [], 0
    for w in windows:
        row = run_slice(sigma, Vt, t_full, r=r, t_start=w["t_start"], t_end=w["t_end"],
                         stride=w["stride"], constraints=constraints, augment=False,
                         group=G3, a_symb=A_SYMB, weak=False, reg_weight_lam=lam,
                         check_sim=False, check_fit_quality=False, max_iter=500)
        if row.get("fit") is None:
            continue
        n_ok += 1
        Xi_raw = row["fit"]["Xi_raw"]
        a1_coefs.append(Xi_raw[i_a1, 0])
        a1cub_coefs.append(Xi_raw[i_a1cub, 0])
        ratios.append(row["ratio"])
        nnzs.append(int(np.sum(np.abs(row["fit"]["Xi"]) > 1e-10)))
    a1_coefs, a1cub_coefs, ratios = map(np.array, (a1_coefs, a1cub_coefs, ratios))
    frac_saddle = float(np.mean(np.abs(ratios) > 10))
    print(f"lam={lam:<7g} n_ok={n_ok}/{len(windows)}  nnz={np.mean(nnzs):.1f}  "
          f"a1={a1_coefs.mean():.4g}+-{a1_coefs.std():.2g}  "
          f"a1^3={a1cub_coefs.mean():.4g}+-{a1cub_coefs.std():.2g}  "
          f"xi2/xi6={ratios.mean():.4g}+-{ratios.std():.2g}  frac_saddle={frac_saddle:.2f}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    lines_md.append(f"| {lam:g} | {np.mean(nnzs):.1f} | {a1_coefs.mean():.4g}+-{a1_coefs.std():.2g} | "
                     f"{a1cub_coefs.mean():.4g}+-{a1cub_coefs.std():.2g} | "
                     f"{ratios.mean():.4g}+-{ratios.std():.2g} | {frac_saddle:.2f} |")

LOGS_DIR.mkdir(exist_ok=True)
out_lines = ["# reg_weight_lam sweep (ConstrainedSR3), r=3, same 38-window ensemble", "",
             "MIOSR unavailable in this pysindy build (no MIP solver); this is the review's own",
             "named fallback for the same collinearity concern.", ""] + lines_md
(LOGS_DIR / "regularization_sweep_summary.md").write_text("\n".join(out_lines) + "\n")
print(f"\nSaved {LOGS_DIR / 'regularization_sweep_summary.md'}", flush=True)
