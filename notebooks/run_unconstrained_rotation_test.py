"""Test: does freeing the linear a2<->a3 coupling term (forbidden by the enforced
discrete symmetry g1: a2'must be odd-in-a2, so any a2'-term lacking a factor of a2 --
including a bare a3, or a1*a3, a1^2*a3, etc. -- is forced to exactly zero, confirmed
directly against the printed constrained-fit equations) recovers the missing phase
(theta) dynamics of the near-degenerate (a2,a3) pair?

Uses the existing `fit_plain_sindy` (unconstrained STLSQ, already in sindy_pipeline.py)
on the SAME full range (t=800..4000, r=3) as the constrained fit, since an
unconstrained fit is free to put weight on the linear a2/a3 cross terms the
constrained fit cannot touch. Then runs the identical polar (rho, theta) R^2 /
Kramers-Moyal decomposition as run_phase_amplitude_closure_test.py on THIS fit's
vector field, for direct comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "claude_artifacts"))

import numpy as np

from sindy_pipeline import (compute_pod, slice_amplitudes, build_symmetry_constraints,
                             fit_symmetric_sindy, fit_plain_sindy,
                             xi_ratio_generic, G as G3, A_SYMB)


def equation_strings_generic(fit, a_symb, raw=True, precision=4):
    """Same formatting as sindy_pipeline.equation_strings, but works for fits with no
    'constraints' key (e.g. fit_plain_sindy)."""
    Xi = fit["Xi_raw"] if raw else fit["Xi"]
    names = [s.name for s in a_symb]
    lines = []
    for i, nm in enumerate(names):
        terms = [f"{Xi[j, i]:+.{precision}g} {fit['feat_names'][j]}"
                 for j in range(len(fit["feat_names"])) if abs(Xi[j, i]) > 1e-8]
        lines.append(f"({nm})' = " + " ".join(terms))
    return lines
import robustness_diagnostics as rd

POD_NPZ = "/gscratch/stf/songk42/ehd/sindy_data_fine.npz"
POD_CACHE = "/gscratch/stf/songk42/ehd/pod_cache_trimmed_t800.npz"

print("Loading POD cache...", flush=True)
Phi, sigma, Vt, t_full = compute_pod(POD_NPZ, r=6, t_start=800.0, method="svds",
                                      cache_path=POD_CACHE)
r = 3
X_raw, t = slice_amplitudes(sigma, Vt, t_full, r=r, t_start=800.0, t_end=4000.0, stride=1)
dt = float(np.mean(np.diff(t)))

print(f"Fitting r=3 UNCONSTRAINED (plain STLSQ) on full range: n={len(t)}", flush=True)
fit_plain = fit_plain_sindy(X_raw, t, a_symb=A_SYMB)
print("\nUnconstrained equations:")
for line in equation_strings_generic(fit_plain, A_SYMB, raw=True):
    print(" ", line)

print(f"\nFitting r=3 CONSTRAINED (symmetric) on full range, for comparison...", flush=True)
constraints = build_symmetry_constraints(r=r, group=G3, a_symb=A_SYMB)
fit_constrained = fit_symmetric_sindy(X_raw, t, r=r, constraints=constraints, augment=False,
                                       group=G3, a_symb=A_SYMB, weak=False, max_iter=500)
print("Constrained equations:")
for line in equation_strings_generic(fit_constrained, A_SYMB, raw=True):
    print(" ", line)

a1, a2, a3 = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]
rho = np.sqrt(a2 ** 2 + a3 ** 2)
theta = np.unwrap(np.arctan2(a3, a2))
rho_safe = np.where(rho > 1e-9, rho, 1e-9)
drho_dt_data = np.gradient(rho, dt)
dtheta_dt_data = np.gradient(theta, dt)


def r2(actual, model):
    return 1.0 - np.sum((actual - model) ** 2) / np.sum((actual - actual.mean()) ** 2)


def polar_report(name, fit):
    X_scale = fit["X_scale"]
    F = fit["model"].predict(X_raw / X_scale) * X_scale
    Fa1, Fa2, Fa3 = F[:, 0], F[:, 1], F[:, 2]
    drho_dt_model = (a2 * Fa2 + a3 * Fa3) / rho_safe
    dtheta_dt_model = (a2 * Fa3 - a3 * Fa2) / rho_safe ** 2
    print(f"\n[{name}] R^2(a1')={r2(np.gradient(a1, dt), Fa1):.4f}  "
          f"R^2(a2')={r2(np.gradient(a2, dt), Fa2):.4f}  R^2(a3')={r2(np.gradient(a3, dt), Fa3):.4f}")
    print(f"[{name}] R^2(rho')={r2(drho_dt_data, drho_dt_model):.4f}   "
          f"R^2(theta')={r2(dtheta_dt_data, dtheta_dt_model):.4f}")
    ratio = xi_ratio_generic(fit["Xi_raw"], fit["feat_names"], A_SYMB)
    print(f"[{name}] xi2/xi6 (saddle ratio) = {ratio['ratio']:.4g}")
    return dtheta_dt_model


print("\n" + "=" * 70)
dtheta_model_constrained = polar_report("constrained", fit_constrained)
dtheta_model_plain = polar_report("unconstrained", fit_plain)

print("\n(reference from previous script: constrained R^2(theta') = -0.0068)")
