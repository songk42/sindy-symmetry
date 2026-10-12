"""Act on claude_artifacts/sindy_noise_robustness_review.md's suggested order of work,
items 1-3 (cheap, decisive, no pysindy-simulation cost):

  1. Kramers-Moyal residual analysis (closure_noise) -- the decisive test for whether
     switching is a noise-activated escape (Langevin closure territory) or a missing
     deterministic term.
  2. Lyapunov time of the amplitude series (predictability_time) -- resolves whether a
     50-time-unit forecast failure is a fit defect or just past the predictability
     horizon.
  3. Conditioning of the constrained regression (library_conditioning) -- quantifies
     the a1*a_i / a1^2*a_i collinearity flagged in the review.

Fits r=3 constrained SINDy on the FULL available range (t=800..4000, not just one
800-unit window) -- deliberately maximizes averaging of any fluctuating unresolved
forcing into the deterministic drift, giving the best available f_model for the
closure-noise test (least-squares should average genuine noise toward zero, per the
review's own "more data -> more damped" prediction).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "claude_artifacts"))

import numpy as np

from sindy_pipeline import (compute_pod, slice_amplitudes, build_symmetry_constraints,
                             fit_symmetric_sindy, G as G3, A_SYMB)
import robustness_diagnostics as rd

POD_NPZ = "/gscratch/stf/songk42/ehd/sindy_data_fine.npz"
POD_CACHE = "/gscratch/stf/songk42/ehd/pod_cache_trimmed_t800.npz"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

print("Loading POD cache...", flush=True)
Phi, sigma, Vt, t_full = compute_pod(POD_NPZ, r=6, t_start=800.0, method="svds",
                                      cache_path=POD_CACHE)

r = 3
X_raw, t = slice_amplitudes(sigma, Vt, t_full, r=r, t_start=800.0, t_end=4000.0, stride=1)
dt = float(np.mean(np.diff(t)))
print(f"Fitting r={r} constrained SINDy on FULL range: n={len(t)}, dt={dt:.4g}, "
      f"t=[{t[0]:.4g},{t[-1]:.4g}]", flush=True)

constraints = build_symmetry_constraints(r=r, group=G3, a_symb=A_SYMB)
t0 = time.time()
fit = fit_symmetric_sindy(X_raw, t, r=r, constraints=constraints, augment=False,
                           group=G3, a_symb=A_SYMB, weak=False, max_iter=500)
print(f"  fit done in {time.time() - t0:.1f}s, nnz={int(np.sum(np.abs(fit['Xi']) > 1e-10))}",
      flush=True)

# --------------------------------------------------------------------------- 1 & 2
print("\n=== predictability_time (Rosenstein largest-LE) ===", flush=True)
t0 = time.time()
ly = rd.predictability_time(X_raw, dt=dt)
print(f"  lambda_max={ly['lambda_max']:.4g}  lyapunov_time={ly['lyapunov_time']:.4g}  "
      f"fit_r2={ly['fit_r2']:.3f}  theiler_steps={ly['theiler_steps']}  ({time.time() - t0:.1f}s)",
      flush=True)

print("\n=== switching_stats (real data, a1, full range) ===", flush=True)
sw = rd.switching_stats(t, X_raw[:, 0])
print(f"  n_switches={sw['n_switches']}  rate_per_1000={sw['rate_per_1000']:.2f}  "
      f"mean_dwell={sw['mean_dwell']:.3g}  cv_dwell={sw['cv_dwell']:.3g}", flush=True)

# --------------------------------------------------------------------------- 3 (decisive)
print("\n=== closure_noise (Kramers-Moyal residual) -- decisive test ===", flush=True)
X_scale = fit["X_scale"]


def f_model_raw(Xr):
    return fit["model"].predict(Xr / X_scale) * X_scale


Xdot = np.gradient(X_raw, dt, axis=0)
t0 = time.time()
cn = rd.closure_noise(t, X_raw, f_model=f_model_raw, Xdot=Xdot, timescale=ly["lyapunov_time"])
print(f"  dt={cn['dt']:.4g}  timescale={cn['timescale']:.4g} (= Lyapunov time)  "
      f"({time.time() - t0:.1f}s)", flush=True)
for v in cn["per_variable"]:
    print(f"  a{v['var'] + 1}: unresolved_frac={v.get('unresolved_frac', float('nan')):.3f}  "
          f"corr_time/dt={v['corr_time_over_dt']:.3f}  sigma_eff={v['sigma_eff']:.4g}  "
          f"excursion_ratio={v['excursion_ratio']:.4g}  state_dependent={v['state_dependent']:.3f}  "
          f"skew={v['resid_skew']:.3f}  kurt={v['resid_excess_kurt']:.3f}", flush=True)

# --------------------------------------------------------------------------- 4
print("\n=== library_conditioning ===", flush=True)
Theta = fit["constraints"]["Theta"].transform(fit["X"])
cond = rd.library_conditioning(Theta, constraint_lhs=constraints["constraints_lhs"], n_targets=r)
print(f"  n_free_params={cond['n_free_params']} of {cond['n_params_unconstrained']} "
      f"({cond['n_constraints_indep']} indep. constraints)", flush=True)
print(f"  cond_Theta={cond['cond_Theta']:.3g}  cond_reduced={cond['cond_reduced']:.3g}  "
      f"eff_rank_reduced={cond['eff_rank_reduced']}", flush=True)
print(f"  max_abs_corr={cond['max_abs_corr']:.4g} (pair {cond['worst_pair']})  "
      f"max_VIF={cond['max_VIF']:.4g}  mutual_coherence={cond['mutual_coherence']:.4g}", flush=True)
feat_names = fit["feat_names"]
i, j = cond["worst_pair"]
print(f"  worst-pair features: '{feat_names[i]}' vs '{feat_names[j]}'", flush=True)

# --------------------------------------------------------------------------- save + write-up
LOGS_DIR.mkdir(exist_ok=True)
np.savez(LOGS_DIR / "robustness_diagnostics_r3_fullrange.npz",
          tau=ly["tau"], log_divergence=ly["log_divergence"],
          switch_times=sw["switch_times"], Xi_raw=fit["Xi_raw"])

lines = [
    "# Robustness diagnostics (claude_artifacts review, items 1-3), r=3, full range t=[800,4000]",
    "",
    "Model: symmetry+energy-constrained SINDy (ConstrainedSR3, reg_weight_lam=0), fit on the",
    "FULL t=[800,4000] range (n=64000) rather than one window, to maximize averaging of any",
    "fluctuating unresolved forcing into the deterministic drift before testing for it.",
    "",
    f"**Lyapunov time**: lambda_max={ly['lambda_max']:.4g} /tu, "
    f"lyapunov_time={ly['lyapunov_time']:.4g} tu (fit R^2={ly['fit_r2']:.3f}).",
    "",
    f"**Real switching (full range)**: {sw['n_switches']} switches, "
    f"rate={sw['rate_per_1000']:.2f}/1000tu, mean dwell={sw['mean_dwell']:.3g} tu, "
    f"CV(dwell)={sw['cv_dwell']:.3g}.",
    "",
    "## Kramers-Moyal closure-noise test (decisive)",
    "",
    "| var | unresolved_frac | corr_time/dt | sigma_eff | excursion_ratio | state_dep | skew | kurt |",
    "|---|---|---|---|---|---|---|---|",
]
for v in cn["per_variable"]:
    lines.append(f"| a{v['var']+1} | {v.get('unresolved_frac', float('nan')):.3f} | "
                 f"{v['corr_time_over_dt']:.3f} | {v['sigma_eff']:.4g} | "
                 f"{v['excursion_ratio']:.4g} | {v['state_dependent']:.3f} | "
                 f"{v['resid_skew']:.3f} | {v['resid_excess_kurt']:.3f} |")
lines += [
    "",
    "## Conditioning",
    "",
    f"n_free_params={cond['n_free_params']}/{cond['n_params_unconstrained']}, "
    f"cond_reduced={cond['cond_reduced']:.3g}, max_abs_corr={cond['max_abs_corr']:.4g} "
    f"(worst pair: `{feat_names[i]}` vs `{feat_names[j]}`), max_VIF={cond['max_VIF']:.4g}, "
    f"mutual_coherence={cond['mutual_coherence']:.4g}.",
]
(LOGS_DIR / "robustness_diagnostics_summary.md").write_text("\n".join(lines) + "\n")
print(f"\nSaved {LOGS_DIR / 'robustness_diagnostics_summary.md'}", flush=True)
