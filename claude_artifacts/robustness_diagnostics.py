"""
Diagnostics for deciding *which* robustness problem a SINDy fit actually has.

Operates only on the low-dimensional state series (e.g. POD amplitudes a_i(t)) and
a fitted model -- never the raw field -- so it runs in seconds on a laptop.

Four questions, four functions:

  1. library_conditioning   -- is the (constrained) regression ill-posed?
                               [Feng/Mangan arXiv:2603.11330, Raut arXiv:2603.05201]
  2. predictability_time    -- is a failed 50-time-unit forecast a fit defect or a
                               Lyapunov-time effect? (Rosenstein largest-LE estimate)
  3. switching_stats        -- what IS the data's switching statistic, with error bars?
  4. closure_noise          -- how much of the tendency is unresolved, is that residual
                               white, and is it big enough to be the thing that drives
                               switching? [Langevin regression, Callaham arXiv:2009.01006]

No pysindy dependency; `closure_noise` takes any callable f(X) -> Xdot.
"""
import numpy as np
from scipy import linalg, signal, stats


# ---------------------------------------------------------------- 1. conditioning
def library_conditioning(Theta, constraint_lhs=None, n_targets=None, tol=1e-10):
    """Conditioning of the regression actually being solved.

    Theta : (n_samples, n_features) library matrix (model.feature_library.transform(X)).
    constraint_lhs : (n_constraints, n_targets*n_features) equality-constraint matrix
        C with C vec(Xi) = 0 (pysindy 'target' order => vec is row-major over targets).
        If given, conditioning is reported for the REDUCED problem on the constraint
        null space -- the parameterisation the optimizer really sees -- which is what
        governs noise amplification, not Theta alone.
    """
    Theta = np.asarray(Theta, float)
    out = {}
    s = linalg.svdvals(Theta)
    out["n_samples"], out["n_features"] = Theta.shape
    out["cond_Theta"] = float(s[0] / max(s[-1], 1e-300))
    out["eff_rank_Theta"] = int((s > tol * s[0]).sum())

    # pairwise collinearity between library columns (mean-removed, unit-norm)
    Z = Theta - Theta.mean(0)
    nrm = np.linalg.norm(Z, axis=0)
    nz = nrm > 0
    Zn = Z[:, nz] / nrm[nz]
    Corr = Zn.T @ Zn
    np.fill_diagonal(Corr, 0.0)
    out["max_abs_corr"] = float(np.abs(Corr).max())
    iw, jw = np.unravel_index(np.abs(Corr).argmax(), Corr.shape)
    idx = np.flatnonzero(nz)
    out["worst_pair"] = (int(idx[iw]), int(idx[jw]))
    # mutual coherence on raw (un-centred) unit columns = the compressed-sensing quantity
    U = Theta / np.maximum(np.linalg.norm(Theta, axis=0), 1e-300)
    Gm = np.abs(U.T @ U)
    np.fill_diagonal(Gm, 0.0)
    out["mutual_coherence"] = float(Gm.max())
    # VIF per column
    with np.errstate(divide="ignore"):
        R = Zn.T @ Zn
        try:
            out["max_VIF"] = float(np.max(np.diag(linalg.pinv(R))))
        except linalg.LinAlgError:
            out["max_VIF"] = np.inf

    if constraint_lhs is not None:
        C = np.asarray(constraint_lhs, float)
        n_feat = Theta.shape[1]
        n_tgt = n_targets if n_targets is not None else C.shape[1] // n_feat
        assert C.shape[1] == n_tgt * n_feat, "constraint width != n_targets*n_features"
        N = linalg.null_space(C)                       # (n_tgt*n_feat, p_free)
        out["n_constraints_indep"] = int(np.linalg.matrix_rank(C))
        out["n_free_params"] = int(N.shape[1])
        out["n_params_unconstrained"] = int(n_tgt * n_feat)
        # Effective design of the reduced LS problem:
        # vec(Xi) = N z  =>  residual = vec(Xdot) - (I_tgt kron Theta) N z   (target order)
        # Build A = blkdiag-free via reshape: apply Theta to each target block of N.
        blocks = [Theta @ N[k * n_feat:(k + 1) * n_feat, :] for k in range(n_tgt)]
        A = np.vstack(blocks)                          # (n_tgt*n_samples, p_free)
        sa = linalg.svdvals(A)
        out["cond_reduced"] = float(sa[0] / max(sa[-1], 1e-300))
        out["eff_rank_reduced"] = int((sa > tol * sa[0]).sum())
    return out


# ------------------------------------------------------- 2. predictability horizon
def predictability_time(X, dt, n_ref=500, theiler=None, max_steps=None,
                        fit_frac=None, sat_frac=0.8, seed=0):
    """Largest Lyapunov exponent by Rosenstein's mean-log-divergence, on the full
    state (no delay embedding needed when all state variables are available).

    Returns lambda_max (1/time), lyapunov_time = 1/lambda, and the divergence curve
    so the linear-scaling region can be eyeballed. A forecast NRMSE ~ 1 at horizons
    >> lyapunov_time is expected for a chaotic system and is NOT evidence of a bad fit.
    """
    X = np.atleast_2d(np.asarray(X, float))
    n, d = X.shape
    if max_steps is None:
        max_steps = min(n // 4, 2000)
    if theiler is None:                                 # exclude temporally-close points
        theiler = _dominant_period_steps(X[:, 0], dt)
    rng = np.random.default_rng(seed)
    cand = np.arange(n - max_steps)
    refs = rng.choice(cand, size=min(n_ref, len(cand)), replace=False)

    div = np.zeros(max_steps + 1)
    cnt = np.zeros(max_steps + 1)
    for i in refs:
        d2 = ((X[:n - max_steps] - X[i]) ** 2).sum(1)
        d2[max(0, i - theiler):i + theiler + 1] = np.inf
        j = int(np.argmin(d2))
        if not np.isfinite(d2[j]) or d2[j] <= 0:
            continue
        k = np.arange(max_steps + 1)
        sep = np.linalg.norm(X[i + k] - X[j + k], axis=1)
        good = sep > 0
        div[k[good]] += np.log(sep[good])
        cnt[k[good]] += 1
    ok = cnt > 0
    curve = np.full(max_steps + 1, np.nan)
    curve[ok] = div[ok] / cnt[ok]
    tau = np.arange(max_steps + 1) * dt

    # Fit the linear-growth region. By default locate it automatically: growth is
    # linear only before the divergence curve saturates at the attractor diameter,
    # so cut at the first point reaching `sat_frac` of the total rise.
    fin = np.isfinite(curve)
    if fit_frac is not None:
        lo, hi = int(fit_frac[0] * max_steps), int(fit_frac[1] * max_steps)
    else:
        c0, c1 = np.nanmin(curve), np.nanmax(curve)
        reach = np.flatnonzero(fin & (curve >= c0 + sat_frac * (c1 - c0)))
        hi = int(reach[0]) if reach.size else max_steps
        lo = max(1, int(0.1 * hi))
    seg = slice(max(lo, 1), max(hi, lo + 3))
    m = np.isfinite(curve[seg])
    slope, r2 = np.nan, np.nan
    if m.sum() > 2:
        xs, ys = tau[seg][m], curve[seg][m]
        slope, icpt = np.polyfit(xs, ys, 1)
        ss = np.sum((ys - (slope * xs + icpt)) ** 2)
        r2 = 1.0 - ss / max(np.sum((ys - ys.mean()) ** 2), 1e-300)
    return {"lambda_max": float(slope),
            "lyapunov_time": float(1.0 / slope) if slope and slope > 0 else np.inf,
            "fit_r2": float(r2), "theiler_steps": int(theiler),
            "tau": tau, "log_divergence": curve, "fit_slice": (seg.start, seg.stop)}


def _dominant_period_steps(x, dt, default=10):
    x = np.asarray(x, float) - np.mean(x)
    if len(x) < 16:
        return default
    f, P = signal.welch(x, fs=1.0 / dt, nperseg=min(len(x), 4096))
    f, P = f[1:], P[1:]
    if len(f) == 0 or P.max() <= 0:
        return default
    return max(default, int(round((1.0 / f[int(np.argmax(P))]) / dt)))


# ------------------------------------------------------------- 3. switching statistic
def switching_stats(t, a, min_dwell=0.0, hysteresis=0.0):
    """Zero-crossing / sign-switching statistics of a scalar series, with a dwell-time
    filter and optional amplitude hysteresis so fast jitter about 0 isn't counted.

    Report rate WITH its across-window spread; a model's switching rate is only
    meaningfully 'wrong' relative to this spread.
    """
    t = np.asarray(t, float); a = np.asarray(a, float)
    thr = hysteresis * np.std(a)
    state = np.sign(a[0]) or 1.0
    times = []
    for k in range(1, len(a)):
        if state > 0 and a[k] < -thr:
            state = -1.0; times.append(t[k])
        elif state < 0 and a[k] > thr:
            state = 1.0; times.append(t[k])
    times = np.asarray(times)
    if min_dwell > 0 and times.size:
        keep = [times[0]]
        for x in times[1:]:
            if x - keep[-1] >= min_dwell:
                keep.append(x)
        times = np.asarray(keep)
    span = t[-1] - t[0]
    iv = np.diff(times)
    return {"n_switches": int(times.size),
            "rate_per_1000": float(1000.0 * times.size / span) if span > 0 else np.nan,
            "mean_dwell": float(iv.mean()) if iv.size else np.nan,
            "cv_dwell": float(iv.std() / iv.mean()) if iv.size and iv.mean() > 0 else np.nan,
            "switch_times": times}


# ------------------------------------------------------------- 4. closure/residual noise
def closure_noise(t, X, f_model, Xdot=None, timescale=None, taper=2):
    """Is the unresolved part of the dynamics big enough -- and white enough -- to be
    modelled as stochastic forcing (Langevin), rather than as missing library terms?

    t, X    : time vector and state series (n_samples, n_vars) the model was fit on.
    f_model : callable X -> Xdot_pred (e.g. `model.predict`).
    Xdot    : optional measured/estimated derivative, used only for `unresolved_frac`.
    timescale : dynamical time to measure a diffusive excursion over (default: dominant
                spectral period of variable 0; pass the Lyapunov time to compare against
                predictability instead).

    Diffusion is estimated from one-step increments (Kramers-Moyal), not from a
    differentiated series, so it is unbiased for white forcing:
        r_k = X_{k+1} - X_k - f(X_k) dt ,  D = var(r)/(2 dt) ,  sigma_eff = sqrt(2 D).

    Readouts per state variable:
      unresolved_frac  1 - R^2 of the deterministic tendency fit (needs Xdot).
      corr_time        increment-residual autocorrelation time. ~dt => white, so a
                       diffusion/Langevin closure is the right object; >> dt => a
                       missing *deterministic* term (or a coloured closure).
      sigma_eff        forcing amplitude, state units per sqrt(time).
      excursion_ratio  sigma_eff*sqrt(timescale) / std(x): size of a noise-driven
                       excursion over one dynamical time, relative to the attractor.
                       O(0.1-1) => noise-activated transitions are plausible;
                       << 0.1 => they are not, and switching must be deterministic.
      state_dependent  |corr(|r|, |x|)|: multiplicative (state-dependent) forcing.
    """
    X = np.atleast_2d(np.asarray(X, float))
    t = np.asarray(t, float)
    dt = float(np.mean(np.diff(t)))
    F = np.asarray(f_model(X), float)
    sl = slice(taper, len(t) - taper - 1) if taper else slice(0, len(t) - 1)
    inc = (X[1:] - X[:-1])[sl] - F[:-1][sl] * dt        # Kramers-Moyal residual
    Xs = X[:-1][sl]
    if timescale is None:
        timescale = _dominant_period_steps(X[:, 0], dt) * dt
    out = {"dt": dt, "timescale": float(timescale), "per_variable": []}
    for k in range(X.shape[1]):
        r = inc[:, k] - inc[:, k].mean()
        ac = _acorr(r)
        z = np.flatnonzero(ac <= 0)
        n0 = z[0] if z.size else len(ac)
        corr_time = float(dt * (0.5 + ac[1:n0].sum())) if n0 > 1 else float(dt)
        D = float(np.var(r) / (2 * dt))
        sigma_eff = float(np.sqrt(2 * D))
        rec = {"var": k,
               "corr_time": corr_time,
               "corr_time_over_dt": corr_time / dt,
               "D": D,
               "sigma_eff": sigma_eff,
               "excursion_ratio": float(sigma_eff * np.sqrt(timescale) / np.std(Xs[:, k])),
               "resid_skew": float(stats.skew(r)),
               "resid_excess_kurt": float(stats.kurtosis(r)),
               "state_dependent": float(abs(np.corrcoef(np.abs(r), np.abs(Xs[:, k]))[0, 1]))}
        if Xdot is not None:
            Xd = np.atleast_2d(np.asarray(Xdot, float))
            res = (Xd - F)[:, k]
            denom = np.var(Xd[:, k])
            rec["unresolved_frac"] = float(np.var(res) / denom) if denom > 0 else np.nan
        out["per_variable"].append(rec)
    return out


def _acorr(x, nmax=None):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    nmax = nmax or min(n // 2, 5000)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:nmax].real
    return ac / ac[0] if ac[0] > 0 else ac


# ------------------------------------------------- 5. constraints, built numerically
def monomial_powers(n_vars, degree, include_bias=True):
    """Exponent array in pysindy/sklearn `PolynomialLibrary` column order."""
    from itertools import combinations_with_replacement as cwr
    rows = []
    for d in range(0 if include_bias else 1, degree + 1):
        for c in cwr(range(n_vars), d):
            p = [0] * n_vars
            for i in c:
                p[i] += 1
            rows.append(tuple(p))
    return np.array(rows, int)


def poly_features(X, powers):
    X = np.atleast_2d(np.asarray(X, float))
    return np.prod(X[:, None, :] ** powers[None, :, :], axis=2)


def symmetry_constraint_matrix(group, n_vars, degree, energy_pairs=None,
                               include_bias=True, n_probe=None, seed=0, dedup=True):
    """Equality constraints C vec(Xi)=0 enforcing Theta(Xg^T) Xi = Theta(X) Xi g^T.

    M_k (with Theta(gx) = Theta(x) M_k) is obtained by solving Theta(Z) M = Theta(Zg^T)
    at random probe points instead of symbolically -- exact, since both sides are linear
    in the monomial basis, and it generalises to any r and any linear group action
    without re-deriving substitution rules. Verify with `residual` in the returned dict.

    energy_pairs : list of ((target_i, powers_i), (target_j, powers_j), ...) groups whose
        coefficients must sum to zero (the energy-preserving quadratic condition of
        Loiseau & Brunton, arXiv:1611.03271). `powers_*` are exponent tuples.

    vec order is 'target' (row-major over targets), matching pysindy
    `constraint_order="target"`.
    """
    group = np.asarray(group, float)
    powers = monomial_powers(n_vars, degree, include_bias)
    nf = len(powers)
    idx = {tuple(p): i for i, p in enumerate(map(tuple, powers))}
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_probe or max(20 * nf, 200), n_vars))
    TZ = poly_features(Z, powers)
    rows, resid = [], 0.0
    for g in group:
        M, *_ = np.linalg.lstsq(TZ, poly_features(Z @ g.T, powers), rcond=None)
        resid = max(resid, float(np.abs(TZ @ M - poly_features(Z @ g.T, powers)).max()))
        rows.append(np.kron(np.eye(n_vars), M) - np.kron(g, np.eye(nf)))
    C = np.vstack(rows) if rows else np.zeros((0, n_vars * nf))
    if energy_pairs:
        for grp in energy_pairs:
            r = np.zeros(n_vars * nf)
            for tgt, pw in grp:
                r[tgt * nf + idx[tuple(pw)]] += 1.0
            C = np.vstack([C, r[None, :]])
    rank = int(np.linalg.matrix_rank(C)) if C.size else 0
    if dedup and C.size and rank < C.shape[0]:
        _, _, piv = linalg.qr(C.T, pivoting=True)
        C = C[np.sort(piv[:rank]), :]
    return {"C": C, "rank": rank, "n_params": n_vars * nf,
            "n_free_params": int(linalg.null_space(C).shape[1]) if C.size else n_vars * nf,
            "powers": powers, "residual": resid}


# ----------------------------------------------------------- 6. ensemble / inclusion
def inclusion_probabilities(fit_fn, windows, thresh=None, rel_thresh=1e-3):
    """Bootstrap-aggregated coefficient statistics (E-SINDy style, arXiv:2111.10992).

    fit_fn  : callable(window) -> Xi array (n_features, n_targets), e.g. a closure around
              an existing pipeline's fit for one (t_start, duration, stride) slice.
    windows : iterable of whatever fit_fn accepts (data windows, or (window, seed) pairs).

    Returns per-coefficient inclusion probability, median and IQR. A term whose
    inclusion probability is not near 0 or 1 is not a discovery -- it is a coin flip
    the point estimate hides, which is exactly the failure mode that shows up as
    window-dependent fixed-point structure.
    """
    Xis = []
    for w in windows:
        try:
            Xi = np.asarray(fit_fn(w), float)
        except Exception as e:                            # keep the sweep alive
            Xis.append(None)
            continue
        Xis.append(Xi)
    good = [x for x in Xis if x is not None]
    if not good:
        raise RuntimeError("every fit failed")
    A = np.stack(good)                                    # (n_fits, n_feat, n_tgt)
    scale = np.abs(A).max()
    tol = thresh if thresh is not None else rel_thresh * scale
    return {"n_fits": len(good), "n_failed": len(Xis) - len(good),
            "inclusion_prob": (np.abs(A) > tol).mean(0),
            "median": np.median(A, 0),
            "iqr": np.percentile(A, 75, axis=0) - np.percentile(A, 25, axis=0),
            "sign_consistency": np.abs(np.mean(np.sign(A), 0)),
            "coefficients": A}
