"""Follow-up to run_robustness_diagnostics.py: the Kramers-Moyal residual on Cartesian
(a2, a3) came back COLORED (corr_time/dt ~8.5-15, not ~1), which per the review's own
guide means "missing deterministic term", not white noise -- the opposite of a clean
Langevin signature. But a2/a3 are exactly the near-degenerate POD mode pair (sigma2/
sigma3 ~1.006) whose *individual identity* was already shown to be unstable across
windows (per-window POD comparison, an earlier analysis). Within THIS fit's single
fixed global basis a2/a3 are well-defined, but the pair's natural symmetry is
rotational/dihedral (the enforced group acts on it via sign flips + swap) -- so if the
"missing dynamics" the closure-noise test found is really just phase wander of the
(a2,a3) pair (its angle around the origin) rather than genuine amplitude dynamics, that
points to a near-neutral (Goldstone-like) mode, not missing coupling to modes 4-6 and
not stochastic forcing either -- a third, more benign explanation.

Tests this directly, with NO refit needed: reuse the existing fitted vector field F(X)
in raw Cartesian units, transform to polar (rho, theta) for the (a2, a3) pair via the
exact chain rule (drho/dt = (a2 Fa2 + a3 Fa3)/rho, dtheta/dt = (a2 Fa3 - a3 Fa2)/rho^2),
and run the same Kramers-Moyal-style residual analysis in (rho, theta) instead of
(a2, a3). If theta's residual is much larger/more diffusive/whiter than rho's, phase
wander is the dominant "unresolved" component.
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

print("Loading POD cache + refitting r=3 on full range (same as previous script)...", flush=True)
Phi, sigma, Vt, t_full = compute_pod(POD_NPZ, r=6, t_start=800.0, method="svds",
                                      cache_path=POD_CACHE)
r = 3
X_raw, t = slice_amplitudes(sigma, Vt, t_full, r=r, t_start=800.0, t_end=4000.0, stride=1)
dt = float(np.mean(np.diff(t)))
constraints = build_symmetry_constraints(r=r, group=G3, a_symb=A_SYMB)
fit = fit_symmetric_sindy(X_raw, t, r=r, constraints=constraints, augment=False,
                           group=G3, a_symb=A_SYMB, weak=False, max_iter=500)
X_scale = fit["X_scale"]


def f_model_raw(Xr):
    return fit["model"].predict(Xr / X_scale) * X_scale


F = f_model_raw(X_raw)  # (n, 3) raw-unit model tendencies, evaluated once on the whole series
a1, a2, a3 = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]
Fa1, Fa2, Fa3 = F[:, 0], F[:, 1], F[:, 2]

rho = np.sqrt(a2 ** 2 + a3 ** 2)
theta = np.unwrap(np.arctan2(a3, a2))
print(f"rho: min={rho.min():.3g} max={rho.max():.3g} mean={rho.mean():.3g} std={rho.std():.3g}")
print(f"theta (unwrapped): range={theta.max() - theta.min():.3g} rad "
      f"({(theta.max() - theta.min()) / (2 * np.pi):.2f} full turns), std={theta.std():.3g} rad",
      flush=True)

# Model-implied polar tendencies via the exact chain rule from the fitted Cartesian field.
rho_safe = np.where(rho > 1e-9, rho, 1e-9)
drho_dt_model = (a2 * Fa2 + a3 * Fa3) / rho_safe
dtheta_dt_model = (a2 * Fa3 - a3 * Fa2) / rho_safe ** 2

# Data-side derivatives (finite difference; unwrapped theta so the angle derivative is
# not corrupted by the -pi/pi branch cut).
drho_dt_data = np.gradient(rho, dt)
dtheta_dt_data = np.gradient(theta, dt)


def r2(actual, model):
    return 1.0 - np.sum((actual - model) ** 2) / np.sum((actual - actual.mean()) ** 2)


print("\n=== R^2 of the EXISTING Cartesian-fit model, viewed in polar coordinates ===", flush=True)
print(f"  R^2(rho')   = {r2(drho_dt_data, drho_dt_model):.4f}   (cf. R^2(a2')={r2(np.gradient(a2, dt), Fa2):.4f}, "
      f"R^2(a3')={r2(np.gradient(a3, dt), Fa3):.4f})")
print(f"  R^2(theta') = {r2(dtheta_dt_data, dtheta_dt_model):.4f}", flush=True)

# Kramers-Moyal-style one-step residuals in (rho, theta), same construction as
# robustness_diagnostics.closure_noise but done by hand since rho/theta aren't part of
# the fitted state.
taper = 2
sl = slice(taper, len(t) - taper - 1)
inc_rho = (rho[1:] - rho[:-1])[sl] - drho_dt_model[:-1][sl] * dt
dtheta_step = (theta[1:] - theta[:-1])[sl]  # already unwrapped, no branch-cut issue
inc_theta = dtheta_step - dtheta_dt_model[:-1][sl] * dt


def stats_for(name, inc, scale_for_excursion, timescale):
    r_ = inc - inc.mean()
    ac = rd._acorr(r_)
    z = np.flatnonzero(ac <= 0)
    n0 = z[0] if z.size else len(ac)
    corr_time = float(dt * (0.5 + ac[1:n0].sum())) if n0 > 1 else float(dt)
    D = float(np.var(r_) / (2 * dt))
    sigma_eff = float(np.sqrt(2 * D))
    exc = sigma_eff * np.sqrt(timescale) / scale_for_excursion
    print(f"  {name}: corr_time/dt={corr_time / dt:.3f}  sigma_eff={sigma_eff:.4g}  "
          f"excursion_ratio={exc:.4g}  skew={float(__import__('scipy.stats', fromlist=['skew']).skew(r_)):.3f}",
          flush=True)
    return dict(corr_time_over_dt=corr_time / dt, sigma_eff=sigma_eff, excursion_ratio=exc)


print("\n=== Kramers-Moyal residual, polar (rho, theta) vs. Cartesian (a2, a3) reference ===",
      flush=True)
ly = rd.predictability_time(X_raw, dt=dt)
timescale = ly["lyapunov_time"]
print(f"(timescale = Lyapunov time = {timescale:.4g} tu)")
stats_for("rho  ", inc_rho, rho.std(), timescale)
stats_for("theta", inc_theta, max(theta.std(), 1e-9), timescale)

print("\n(reference, Cartesian a2/a3 from previous script: corr_time/dt~8.5-8.6, "
      "excursion_ratio~0.44 each)", flush=True)

# Fraction of total (a2,a3) Kramers-Moyal residual variance that is radial vs tangential,
# decomposed pointwise (equivalent construction, cross-checks the rho/theta numbers above).
inc_a2 = (a2[1:] - a2[:-1])[sl] - Fa2[:-1][sl] * dt
inc_a3 = (a3[1:] - a3[:-1])[sl] - Fa3[:-1][sl] * dt
rho_at = rho[:-1][sl]
rho_hat_x, rho_hat_y = a2[:-1][sl] / np.where(rho_at > 1e-9, rho_at, 1e-9), a3[:-1][sl] / np.where(rho_at > 1e-9, rho_at, 1e-9)
tang_hat_x, tang_hat_y = -rho_hat_y, rho_hat_x
r_radial = inc_a2 * rho_hat_x + inc_a3 * rho_hat_y
r_tangential = inc_a2 * tang_hat_x + inc_a3 * tang_hat_y
var_radial, var_tangential = np.var(r_radial), np.var(r_tangential)
print(f"\nRadial vs tangential share of Cartesian (a2,a3) residual variance: "
      f"radial={var_radial:.4g} ({var_radial / (var_radial + var_tangential):.1%}), "
      f"tangential={var_tangential:.4g} ({var_tangential / (var_radial + var_tangential):.1%})",
      flush=True)

lines = [
    "# Amplitude/phase decomposition of the a2/a3 closure-noise residual (follow-up)",
    "",
    f"rho range=[{rho.min():.3g},{rho.max():.3g}], theta unwrapped range="
    f"{(theta.max() - theta.min()):.3g} rad ({(theta.max() - theta.min()) / (2*np.pi):.2f} turns).",
    "",
    f"R^2(rho')={r2(drho_dt_data, drho_dt_model):.4f}, R^2(theta')={r2(dtheta_dt_data, dtheta_dt_model):.4f} "
    f"(cf. R^2(a2')={r2(np.gradient(a2, dt), Fa2):.4f}, R^2(a3')={r2(np.gradient(a3, dt), Fa3):.4f}).",
    "",
    f"Radial vs tangential share of (a2,a3) Kramers-Moyal residual variance: "
    f"radial {var_radial / (var_radial + var_tangential):.1%}, "
    f"tangential {var_tangential / (var_radial + var_tangential):.1%}.",
]
(LOGS_DIR / "phase_amplitude_closure_summary.md").write_text("\n".join(lines) + "\n")
print(f"\nSaved {LOGS_DIR / 'phase_amplitude_closure_summary.md'}", flush=True)
