"""Correctness metrics for a fitted polynomial model Xi (h, n) against the true Xi_true (same library).

  coef_rel_err        ||Xi - Xi_true||_F / ||Xi_true||_F
  support_f1, exact_support            support at an absolute 1e-10 cutoff (strict: any leftover term is a miss)
  support_f1_rel, exact_support_rel    support at |xi| > REL_TOL * max|xi| of the same model, so negligible
                                       leftover terms don't count (true support uses the same rule)
  on_support_err      ||(Xi - Xi_true)[true support]|| / ||Xi_true||: error in the terms that should be there,
                      ignoring spurious ones (pair with exact_support_rel for the "successful runs only" RMSE of
                      Yang et al. 2024)
  vf_rel_err          ||Theta(X) Xi - Theta(X) Xi_true||_F / ||Theta(X) Xi_true||_F on the clean attractor states:
                      how wrong the learned vector field is where the dynamics live
  vpt_lyap            valid prediction time in Lyapunov times: forecast with the learned model and with the true
                      model (both RK4 in scaled coordinates) from N_IC states on the clean attractor, until the error
                      exceeds VPT_EPS times the attractor's RMS spread; mean over initial conditions, capped at
                      min(VPT_HORIZON Lyapunov times, MAX_SAMPLES samples) = vpt_horizon. Divergent models stop
                      where they blow up.
  traj_nrmse_clean    trajectory RMSE against the data: simulate the learned model (RK4) from N_TRAJ_IC states of the
  traj_nrmse_noisy    data for TRAJ_HORIZON Lyapunov times (at most MAX_SAMPLES samples) and compare with the data
                      samples that follow, RMS over samples, initial conditions and coordinates, divided by the
                      attractor's RMS spread. "clean": start from and compare with the clean trajectory; "noisy": start
                      from and compare with the noisy data passed to set_noisy (= clean if none). inf if any forecast
                      blows up. traj_nrmse_{clean,noisy}_true are the same numbers for Xi_true (the floor: integration
                      and dt-fit error, plus the noise itself for "noisy").
"""
import numpy as np

REL_TOL = 1e-3
VPT_EPS = 0.3
VPT_HORIZON = 10.0  # Lyapunov times
MAX_SAMPLES = 2000  # ...but never longer than this many data samples (~20 periods); vpt_horizon records the cap
N_IC = 8
SUBSTEPS = 4  # RK4 substeps per data sample
TRAJ_HORIZON = 1.0  # Lyapunov times
N_TRAJ_IC = 32


def poly_eval(X, powers):
    """Theta(X) for monomials with exponent rows `powers` (h, n); X (..., n) -> (..., h)."""
    return np.prod(X[..., None, :] ** powers, axis=-1)


def _support(Xi, rel):
    scale = np.max(np.abs(Xi))
    if scale == 0:
        return np.zeros_like(Xi, bool)
    return np.abs(Xi) > (rel * scale if rel else 1e-10)


def _f1(est, true):
    tp = np.sum(est & true)
    prec = tp / max(est.sum(), 1)
    rec = tp / max(true.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-12), bool(np.array_equal(est, true))


class Evaluator:
    """Precomputes everything that depends only on the system, then scores many fits cheaply."""

    def __init__(self, X_clean, Xi_true, powers, dt, lyap):
        self.Xi_true = Xi_true
        self.powers = np.asarray(powers)
        self.Theta = poly_eval(X_clean, self.powers)
        self.F_true = self.Theta @ Xi_true
        self.true_rel = _support(Xi_true, REL_TOL)
        self.true_abs = _support(Xi_true, 0)
        # forecast setup: evenly spaced initial conditions from the second half of the trajectory
        idx = np.linspace(len(X_clean) // 2, len(X_clean) - 1, N_IC + 2).astype(int)[1:-1]
        self.x0 = X_clean[idx]
        self.spread = np.sqrt(np.mean(np.var(X_clean, axis=0)))
        self.bound = 10 * np.max(np.abs(X_clean))
        self.lyap = lyap if lyap and lyap > 0 else None
        t_lyap = 1.0 / self.lyap if self.lyap else 100 * dt  # no exponent: 100 samples per "Lyapunov time"
        self.t_lyap = t_lyap
        self.h = dt / SUBSTEPS
        self.n_steps = min(int(np.ceil(VPT_HORIZON * t_lyap / self.h)), MAX_SAMPLES * SUBSTEPS)
        self.horizon = self.n_steps * self.h / t_lyap
        self.ref = self._integrate(Xi_true)
        # trajectory-RMSE setup: N_TRAJ_IC starts spread over the whole trajectory, n_traj samples ahead of each
        self.X_clean = X_clean
        self.n_traj = max(1, min(int(np.ceil(TRAJ_HORIZON * t_lyap / dt)), MAX_SAMPLES, len(X_clean) - 1))
        self.traj_idx = np.linspace(0, len(X_clean) - 1 - self.n_traj, N_TRAJ_IC).astype(int)
        self.traj_true = {}
        self.set_noisy(X_clean)

    def set_noisy(self, X_noisy):
        """Noisy data (same length as X_clean) that traj_nrmse_noisy compares against."""
        self.X_noisy = X_noisy
        self.traj_true = dict(zip(("clean", "noisy"), self.traj_nrmse(self.Xi_true)))

    def traj_nrmse(self, Xi):
        """(clean, noisy) trajectory NRMSE; both forecasts are integrated together."""
        k = self.traj_idx[:, None] + np.arange(1, self.n_traj + 1)[None]  # (N_TRAJ_IC, n_traj) sample indices
        x0 = np.vstack([self.X_clean[self.traj_idx], self.X_noisy[self.traj_idx]])
        traj = self._integrate(Xi, x0, self.n_traj * SUBSTEPS)[SUBSTEPS::SUBSTEPS]  # (n_traj, 2 N_TRAJ_IC, n)
        out = []
        for j, X_ref in enumerate((self.X_clean, self.X_noisy)):
            err = traj[:, j * N_TRAJ_IC:(j + 1) * N_TRAJ_IC] - X_ref[k].transpose(1, 0, 2)
            out.append(np.inf if np.isnan(err).any() else float(np.sqrt(np.mean(err ** 2))) / self.spread)
        return out

    def _integrate(self, Xi, x0=None, n_steps=None):
        """RK4 trajectories from all x0 at once (default: the VPT starts and horizon); returns (n_steps + 1, n_x0,
        n), NaN after blow-up."""
        f = lambda x: poly_eval(x, self.powers) @ Xi
        x = (self.x0 if x0 is None else x0).copy()
        n_steps = self.n_steps if n_steps is None else n_steps
        out = np.full((n_steps + 1,) + x.shape, np.nan)
        out[0] = x
        alive = np.ones(len(x), bool)
        h = self.h
        with np.errstate(all="ignore"):
            for k in range(n_steps):
                k1 = f(x)
                k2 = f(x + 0.5 * h * k1)
                k3 = f(x + 0.5 * h * k2)
                k4 = f(x + h * k3)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                alive &= np.all(np.isfinite(x), axis=1) & (np.max(np.abs(x), axis=1) < self.bound)
                if not alive.any():
                    break
                x = np.where(alive[:, None], x, np.nan)
                out[k + 1] = x
        return out

    def vpt(self, Xi):
        traj = self._integrate(Xi)
        err = np.linalg.norm(traj - self.ref, axis=-1) / self.spread  # NaN (blown up) compares as invalid
        valid = err < VPT_EPS
        # first invalid step per initial condition (n_steps + 1 if always valid)
        first_bad = np.where(~valid.all(axis=0), np.argmin(valid, axis=0), self.n_steps + 1)
        return float(np.mean((first_bad - 1).clip(min=0) * self.h / self.t_lyap))

    def __call__(self, Xi, forecast=True, traj=True):
        Xi_true = self.Xi_true
        norm_true = np.linalg.norm(Xi_true)
        f1, exact = _f1(_support(Xi, 0), self.true_abs)
        f1_rel, exact_rel = _f1(_support(Xi, REL_TOL), self.true_rel)
        with np.errstate(all="ignore"):
            vf = np.linalg.norm(self.Theta @ Xi - self.F_true) / np.linalg.norm(self.F_true)
        clean, noisy = self.traj_nrmse(Xi) if traj else (np.nan, np.nan)
        return dict(
            coef_rel_err=np.linalg.norm(Xi - Xi_true) / norm_true,
            support_f1=f1, exact_support=exact,
            support_f1_rel=f1_rel, exact_support_rel=exact_rel,
            on_support_err=np.linalg.norm((Xi - Xi_true)[self.true_abs]) / norm_true,
            vf_rel_err=vf,
            vpt_lyap=self.vpt(Xi) if forecast else np.nan,
            vpt_horizon=self.horizon,
            traj_nrmse_clean=clean, traj_nrmse_noisy=noisy,
            traj_nrmse_clean_true=self.traj_true["clean"], traj_nrmse_noisy_true=self.traj_true["noisy"],
            traj_horizon_samples=self.n_traj,
            n_terms=int(np.sum(np.abs(Xi) > 1e-10)),
            n_true_terms=int(np.sum(Xi_true != 0)),
        )
