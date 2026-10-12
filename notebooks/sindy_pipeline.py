"""Reusable symmetry-constrained SINDy pipeline for the EHD POD amplitudes.

Extracted from symmetric_sindy_v1.ipynb / symmetric_sindy_v1.py so the same, verified
pipeline (single global scale, symmetry-augmented trajectories, energy constraint,
QR-deduplicated equality constraints, ConstrainedSR3) can be run over many *slices*
of one data file -- different dt (via `stride`) and different time windows (via
`t_start`/`t_end`) -- for a fair, code-identical comparison instead of hand-copying
notebook cells per slice.

POD (the SVD of the charge field) is computed ONCE per raw data file with
`compute_pod`, then `slice_amplitudes` cheaply windows/subsamples the resulting
amplitude time series -- no need to recompute the SVD (or touch a GPU) per slice,
since we're testing the SINDy step's sensitivity to dt/window, not the POD step's.

Usage sketch
------------
    from sindy_pipeline import compute_pod, run_slice

    Phi, sigma, Vt, t_full = compute_pod("/path/to/sindy_data_fine_v2.npz", r=6)
    row = run_slice(sigma, Vt, t_full, r=3, t_start=850, t_end=1000, stride=1,
                     label="t=850-1000, dt=0.05")
    print(row["diag"], row["xi_ratio"], row["bounded"])
"""
from __future__ import annotations

from itertools import combinations_with_replacement as cwr
from pathlib import Path

import numpy as np
import pysindy as ps
import scipy.linalg
import sympy as sp
from scipy.signal import savgol_filter
from scipy.sparse.linalg import svds

# ---------------------------------------------------------------------------
# Symmetry group (Guan et al. 2021, eq. 3.1) and its sympy substitution rules
# ---------------------------------------------------------------------------
G = np.array([
    [[1, 0, 0], [0, -1, 0], [0, 0, 1]],    # [a1,a2,a3] -> [a1,-a2,a3]
    [[1, 0, 0], [0, 1, 0], [0, 0, -1]],    # [a1,a2,a3] -> [a1,a2,-a3]
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]],   # [a1,a2,a3] -> [a1,-a2,-a3]
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],    # [a1,a2,a3] -> [-a1,a3,a2]
    [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],  # [a1,a2,a3] -> [-a1,-a3,-a2]
])

A1, A2, A3 = sp.symbols("a1 a2 a3")
A_SYMB = [A1, A2, A3]


def symmetry_subs_rules(group=G, a_symb=A_SYMB):
    return [{a: sum(g[i, j] * a_symb[j] for j in range(3)) for i, a in enumerate(a_symb)}
            for g in group]


# ---------------------------------------------------------------------------
# POD: computed once per raw data file, then sliced/strided cheaply
# ---------------------------------------------------------------------------
def compute_pod(npz_path, r=6, field="charge", method="svds", t_start=None, t_end=None,
                 cache_path=None, force_recompute=False):
    """SVD of the (time-mean-subtracted) `field` in `npz_path`.

    Returns (Phi, sigma, Vt, t): Phi (n_space, r), sigma (r,) descending, Vt (r, n_t),
    t (n_t,) -- t is the (possibly restricted) time vector matching Vt's columns.
    `method="svds"` (default) is a truncated Lanczos SVD -- CPU-only, fast for r <<
    n_t, no GPU/cupy needed. `method="cupy"` uses a full dense GPU SVD instead
    (matches what symmetric_sindy_v1.py does); its U/Vt/workspace scale with n_t and
    can OOM once n_t gets into the tens of thousands (confirmed at n_t~24000 on an
    11GB GPU, though it fit on an 80GB A100). `method="cupy_rsvd"` is a randomized SVD
    on GPU -- memory stays ~O(n_space * n_t) for the data itself, not O(n_t^2)-ish
    workspace, so it's the one to use for large n_t; accuracy is approximate but fine
    for r this small relative to n_t (2 power iterations included).

    `t_start`/`t_end`: restrict which snapshots go into the SVD itself, not just the
    amplitude series afterward (that's what `slice_amplitudes` does on an *existing*
    POD). This matters: a POD fit on the full trajectory (transient included) can
    give a different spatial basis -- and thus different amplitude dynamics for the
    "same" later window -- than a POD fit only on the window of interest, since POD
    modes are just whatever captures the most variance in whatever ensemble you feed
    it. Pass these to exclude an initial transient (or any other regime you don't
    want influencing the basis) from the POD itself.

    `cache_path`: if given, load (Phi, sigma, Vt, t) from there instead of recomputing
    when the file exists and its recorded (npz_path, r, field, t_start, t_end, mtime,
    size) match this call; otherwise compute normally and save there for next time.
    mtime/size (of npz_path, not cache_path) are included specifically so a rebuild of
    npz_path at the SAME path with different content (e.g. a simulation continuation
    extending it to a later t) invalidates the cache instead of silently returning a
    stale, shorter-trajectory POD -- npz_path alone isn't enough to detect that. The
    GPU-heavy `method="cupy"` case is exactly what this is for -- e.g.
    `run_window_sweep.py` previously recomputed the same full-trajectory SVD from
    scratch on every GPU job, even reruns of an identical (file, r, window) triple.
    Pass `force_recompute=True` to ignore an existing cache unconditionally.
    """
    def _cache_key():
        st = Path(npz_path).stat()
        return dict(npz_path=str(npz_path), r=r, field=field, t_start=t_start, t_end=t_end,
                    mtime=st.st_mtime, size=st.st_size)

    if cache_path is not None and not force_recompute and Path(cache_path).exists():
        cached = np.load(cache_path, allow_pickle=True)
        meta = cached["meta"].item()
        if meta == _cache_key():
            return cached["Phi"], cached["sigma"], cached["Vt"], cached["t"]
        # else: cache exists but is for a different (npz_path, r, field, t_start, t_end)
        # or npz_path's content has changed since the cache was written (mtime/size)
        # -- fall through and recompute, then overwrite it below.

    data = np.load(npz_path)
    t = data["t"]
    idx = list(data["feature_names"]).index(field)
    u = data["u"][..., idx]  # (n_x, n_y, n_t)
    if t_start is not None or t_end is not None:
        mask = np.ones(len(t), dtype=bool)
        if t_start is not None:
            mask &= t >= t_start
        if t_end is not None:
            mask &= t <= t_end
        u, t = u[..., mask], t[mask]
    n_x, n_y, n_t = u.shape
    snapshots = u.reshape(n_x * n_y, n_t).astype(np.float64)
    snapshots -= snapshots.mean(axis=1, keepdims=True)

    if method == "svds":
        Phi, sigma, Vt = svds(snapshots, k=r)
        order = np.argsort(sigma)[::-1]
        Phi, sigma, Vt = Phi[:, order], sigma[order], Vt[order, :]
    elif method == "cupy":
        import cupy
        s_cu = cupy.asarray(snapshots)
        Phi_cu, sigma_cu, Vt_cu = cupy.linalg.svd(s_cu, full_matrices=False)
        Phi = cupy.asnumpy(Phi_cu[:, :r])
        sigma = cupy.asnumpy(sigma_cu[:r])
        Vt = cupy.asnumpy(Vt_cu[:r, :])
    elif method == "cupy_rsvd":
        # Randomized SVD (Halko/Martinsson/Tropp): a full dense cupy.linalg.svd's
        # U/Vt/workspace scale with n_t and OOM'd at n_t~24000 even on an 80GB A100
        # once we needed n_t~64000 for the t=[0,4000] extension. Since r is tiny
        # relative to n_t, project onto a random (r+oversample)-dim subspace first --
        # memory becomes O(n_space * n_t) for `snapshots` itself (a few GB) plus
        # O(n_space/n_t * (r+oversample)), not O(n_t^2)-ish workspace.
        import cupy
        rng = cupy.random.default_rng(0)
        p = min(10, snapshots.shape[1] - r)  # oversampling
        A = cupy.asarray(snapshots)
        Omega = rng.standard_normal((A.shape[1], r + p))
        Y = A @ Omega
        for _ in range(2):  # power iterations: sharpens separation of close sigma's
            Y = A @ (A.T @ Y)
        Q, _ = cupy.linalg.qr(Y)
        B = Q.T @ A  # (r+p, n_t) -- SVD of this is cheap regardless of n_t
        Ub, sigma_cu, Vt_cu = cupy.linalg.svd(B, full_matrices=False)
        Phi_cu = Q @ Ub
        Phi = cupy.asnumpy(Phi_cu[:, :r])
        sigma = cupy.asnumpy(sigma_cu[:r])
        Vt = cupy.asnumpy(Vt_cu[:r, :])
    else:
        raise ValueError(f"unknown method {method!r}")

    if cache_path is not None:
        np.savez(cache_path, Phi=Phi, sigma=sigma, Vt=Vt, t=t, meta=_cache_key())

    return Phi, sigma, Vt, t


def slice_amplitudes(sigma, Vt, t, r=3, t_start=None, t_end=None, stride=1):
    """Amplitude time series a_i = sigma_i * v_i (i<r), restricted to [t_start, t_end]
    and subsampled every `stride`-th sample -- a cheap proxy for a coarser dt without
    re-extracting from the raw simulation dumps at a different NDMD.
    """
    X_full = (sigma[:r, None] * Vt[:r, :]).T  # (n_t, r)
    mask = np.ones(len(t), dtype=bool)
    if t_start is not None:
        mask &= t >= t_start
    if t_end is not None:
        mask &= t <= t_end
    X = X_full[mask][::stride]
    t_slice = t[mask][::stride]
    return X, t_slice


# ---------------------------------------------------------------------------
# Data-independent part of the fit: library, M_k's, symmetry + energy constraints
# ---------------------------------------------------------------------------
def _exponent_tuple(monom, vars_):
    poly = sp.Poly(monom, *vars_)
    return tuple([0] * len(vars_)) if poly.is_zero else poly.monoms()[0]


def _build_Mk(library, subs_rule, vars_, monomial_index):
    """Theta(g x) = Theta(x) @ M_k, i.e. theta_j(g x) = sum_i Theta_x[i] * M[i, j]."""
    n = len(library)
    M = sp.zeros(n, n)
    for j, theta_j in enumerate(library):
        transformed = sp.expand(theta_j.subs(subs_rule, simultaneous=True))
        for exps, coeff in sp.Poly(transformed, *vars_).terms():
            M[monomial_index[exps], j] += coeff
    return M


def build_symmetry_constraints(r=3, degree=3, include_bias=True, group=G, a_symb=A_SYMB):
    """Everything that depends only on (r, degree, group), not on the data: the
    library, the M_k matrices, and the symmetry + energy constraint rows (already
    deduplicated via QR -- concatenating every C_k leaves many linearly dependent
    rows, which makes ConstrainedSR3's internal C @ inv1 @ C.T singular otherwise).
    """
    Theta = ps.PolynomialLibrary(degree=degree, include_interaction=True, include_bias=include_bias)
    Theta.fit(np.zeros((2, r)))

    full_library = Theta.get_feature_names([s.name for s in a_symb])
    full_library = [sp.sympify(" * ".join(f.split(" "))) for f in full_library]
    monomial_index = {_exponent_tuple(m, a_symb): idx for idx, m in enumerate(full_library)}

    subs_rules = symmetry_subs_rules(group, a_symb)
    M_list = [_build_Mk(full_library, sr, a_symb, monomial_index) for sr in subs_rules]

    n_lib = len(full_library)
    C_list = []
    for g, Mk in zip(group, M_list):
        Mk_num = np.array(Mk, dtype=float)
        C_list.append(np.kron(np.eye(g.shape[0]), Mk_num) - np.kron(g, np.eye(n_lib)))

    # energy-preserving quadratic constraint (Loiseau & Brunton / Schlegel-Noack):
    # a2^2 in a1' <-> a1*a2 in a2' (the rest of the family is pinned by the symmetry
    # constraints above already).
    row = np.zeros(len(a_symb) * n_lib)
    row[0 * n_lib + monomial_index[_exponent_tuple(a_symb[1] ** 2, a_symb)]] = 1.0
    row[1 * n_lib + monomial_index[_exponent_tuple(a_symb[0] * a_symb[1], a_symb)]] = 1.0
    energy_lhs = row[None, :]

    constraints_lhs = np.vstack(C_list + [energy_lhs])
    constraints_rhs = np.zeros(constraints_lhs.shape[0])
    rank = np.linalg.matrix_rank(constraints_lhs)
    if rank == 0:
        constraints_lhs = np.empty((0, constraints_lhs.shape[1]))
        constraints_rhs = np.empty((0,))
    else:
        _, _, piv = scipy.linalg.qr(constraints_lhs.T, pivoting=True)
        idx = piv[:rank]
        constraints_lhs, constraints_rhs = constraints_lhs[idx], constraints_rhs[idx]
    keep = ~np.isclose(constraints_lhs, 0).all(axis=1)
    constraints_lhs, constraints_rhs = constraints_lhs[keep], constraints_rhs[keep]

    return dict(Theta=Theta, full_library=full_library, monomial_index=monomial_index,
                M_list=M_list, C_list=C_list, energy_lhs=energy_lhs,
                constraints_lhs=constraints_lhs, constraints_rhs=constraints_rhs,
                group=group, a_symb=a_symb)


# ---------------------------------------------------------------------------
# The data-dependent fit
# ---------------------------------------------------------------------------
def fit_symmetric_sindy(X_raw, t, r=3, degree=3, include_bias=True, reg_weight_lam=0.0,
                         group=G, a_symb=A_SYMB, constraints=None, augment=True,
                         weak=False, n_subdomains=100, weak_seed=0, max_iter=500):
    """Single global scale -> (optionally) symmetry-augmented trajectories -> ConstrainedSR3 fit.

    Returns a dict: model (fitted ps.SINDy or ps.WeakSINDy), X (scaled amplitudes),
    X_scale, Xi (normalised-coordinate coefficients, n_lib x n_targets), Xi_raw (same,
    converted back to the raw sigma*v amplitude units), constraints (the bundle used,
    so it can be reused across many fits with the same r/degree/group), feat_names,
    augment, weak.

    `augment=True` (default): fit on the data plus all 5 symmetry-transformed copies
    of it (paper eq. 4.8), each as its own pysindy trajectory (not concatenated along
    time, so the derivative estimator never differentiates across the seam between
    copies). `augment=False`: fit on the single window only. NOTE: given the symmetry
    equality constraints already force exact equivariance (M_g Xi = Xi g^T) and every
    g in `group` is orthogonal (signed permutation matrices), each augmented copy's
    loss term equals the original window's EXACTLY (‖(X_dot - Theta(X)Xi) g^T‖ =
    ‖X_dot - Theta(X)Xi‖ for orthogonal g) -- augmentation only multiplies the total
    objective by a constant and provably cannot move the fitted Xi -- for a
    DETERMINISTIC derivative estimator. Confirmed empirically for the plain-derivative
    (non-weak) form: differs by ~1e-7 relative (solver tolerance noise). The same
    integrate-by-parts argument holds for the weak functional in principle (it's also
    linear in X and Theta(X)), but WeakSINDy's own subdomain placement is randomized
    (see `weak_seed` below) and consumes a different-length draw from the RNG stream
    depending on how many trajectories are in the fit list -- so `augment` DOES change
    Xi when `weak=True`, purely as a side effect of that RNG mechanics (observed ~6%
    relative difference on one EHD window, same weak_seed both ways), not because the
    equivariance argument fails. Use `augment=False` with `weak=True` to sidestep this
    confound entirely, since augmentation was already established to add nothing.

    `weak=False` (default): ps.SINDy with FiniteDifference(order=2) -- estimates X_dot
    pointwise, sensitive to per-sample noise. `weak=True`: ps.WeakSINDy instead --
    the weak/integral formulation (Messenger & Bortz), which integrates the ODE
    against `n_subdomains` compactly-supported test functions instead of estimating a
    pointwise derivative, trading pointwise-noise sensitivity for sensitivity to how
    well `n_subdomains` test functions resolve the trajectory's actual timescales.
    Uses `st_grids=t_traj` in place of `t=t_traj`; WeakSINDy accepts a plain 1D time
    grid for pure-ODE (no spatial dimension) data, verified against known Lorenz
    coefficients before use here.

    IMPORTANT: pysindy's WeakSINDy places its `n_subdomains` test-function subdomains
    via `np.random.uniform` against the GLOBAL numpy RNG, with no seed parameter of its
    own and no exposed way to pass one in. Two fits on identical data can therefore
    return different Xi. `weak_seed` (only used when `weak=True`) calls
    `np.random.seed(weak_seed)` immediately before `model.fit(...)` so a given
    (data, weak_seed) pair is reproducible; it does NOT isolate this call from global
    numpy RNG state elsewhere in the process (harmless here since nothing else in this
    module draws random numbers, but worth knowing before using this alongside code
    that does). Pass different `weak_seed` values across an otherwise-identical sweep
    to separately characterize subdomain-placement variance vs. genuine window-to-
    window variance.

    `max_iter` is passed to `ConstrainedSR3` -- pysindy's own default (30) was too low
    for the weak form on real EHD windows here (hit "did not converge" at 30 for
    several slices; not observed for the plain-derivative form, but raising it for both
    keeps the comparison on equal footing).
    """
    if constraints is None:
        constraints = build_symmetry_constraints(r=r, degree=degree, include_bias=include_bias,
                                                   group=group, a_symb=a_symb)

    X_scale = np.max(np.abs(X_raw))
    X = X_raw / X_scale
    if augment:
        X_traj = [X] + [X @ g.T for g in group]
        t_traj = [t for _ in X_traj]
    else:
        X_traj = [X]
        t_traj = [t]

    Theta = ps.PolynomialLibrary(degree=degree, include_interaction=True, include_bias=include_bias)
    Theta.fit(X)

    optimizer = ps.ConstrainedSR3(
        constraint_lhs=constraints["constraints_lhs"],
        constraint_rhs=constraints["constraints_rhs"],
        constraint_order="target",
        equality_constraints=True,
        reg_weight_lam=reg_weight_lam,
        max_iter=max_iter,
    )
    if weak:
        np.random.seed(weak_seed)  # see docstring: WeakSINDy's subdomain placement is unseeded otherwise
        model = ps.WeakSINDy(feature_library=Theta, optimizer=optimizer, n_subdomains=n_subdomains)
        model.fit(X_traj, st_grids=t_traj, feature_names=[s.name for s in a_symb])
    else:
        model = ps.SINDy(differentiation_method=ps.FiniteDifference(order=2),
                          feature_library=Theta, optimizer=optimizer)
        model.fit(X_traj, t=t_traj, feature_names=[s.name for s in a_symb])

    Xi = model.coefficients().T  # (n_lib, n_targets), normalised coords
    full_library = constraints["full_library"]
    Xi_raw = np.array(Xi)
    for j, m in enumerate(full_library):
        e = np.array(_exponent_tuple(m, a_symb), float)
        Xi_raw[j, :] = Xi[j, :] * X_scale / np.prod(X_scale ** e)

    return dict(model=model, X=X, X_scale=X_scale, Xi=Xi, Xi_raw=Xi_raw,
                constraints=constraints, augment=augment, weak=weak,
                feat_names=Theta.get_feature_names([s.name for s in a_symb]))


def fit_plain_sindy(X_raw, t, degree=3, threshold=0.1, alpha=0.05, max_iter=20, a_symb=None):
    """Baseline: textbook SINDy (STLSQ, Brunton/Proctor/Kutz 2016) with NO symmetry or
    energy constraints and no augmentation -- just fit the polynomial library to this
    window's amplitudes directly. Same global-scale preprocessing as
    `fit_symmetric_sindy` (X = X_raw / max|X_raw|) so coefficient magnitudes /
    threshold behavior are comparable, but nothing about the known symmetry group is
    used. For comparing against the constrained fits: does exact symmetry enforcement
    help or hurt fit quality / switching fidelity on this data?

    Returns a dict shaped like `fit_symmetric_sindy`'s (model, X, X_scale, Xi, Xi_raw,
    feat_names, a_symb) so it drops into the same `simulate_from`/plotting code; has no
    `constraints` key (nothing to check symmetry residual against by construction).
    """
    r = X_raw.shape[1]
    if a_symb is None:
        a_symb = list(sp.symbols(" ".join(f"a{i + 1}" for i in range(r))))

    X_scale = np.max(np.abs(X_raw))
    X = X_raw / X_scale

    Theta = ps.PolynomialLibrary(degree=degree, include_interaction=True, include_bias=True)
    Theta.fit(X)
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=max_iter)
    model = ps.SINDy(differentiation_method=ps.FiniteDifference(order=2),
                      feature_library=Theta, optimizer=optimizer)
    model.fit(X, t=t, feature_names=[s.name for s in a_symb])

    Xi = model.coefficients().T  # (n_lib, n_targets), normalised coords
    feat_names = Theta.get_feature_names([s.name for s in a_symb])
    full_library = [sp.sympify(" * ".join(f.split(" "))) for f in feat_names]
    Xi_raw = np.array(Xi)
    for j, m in enumerate(full_library):
        e = np.array(_exponent_tuple(m, a_symb), float)
        Xi_raw[j, :] = Xi[j, :] * X_scale / np.prod(X_scale ** e)

    return dict(model=model, X=X, X_scale=X_scale, Xi=Xi, Xi_raw=Xi_raw,
                feat_names=feat_names, a_symb=a_symb, augment=False, weak=False)


def xi_ratio_generic(Xi_raw, feat_names, a_symb):
    """Same xi2/xi6 ratio as `xi_ratio`, but computed directly from (Xi_raw, feat_names,
    a_symb) instead of a `constraints` bundle -- works for unconstrained fits too."""
    full_library = [sp.sympify(" * ".join(f.split(" "))) for f in feat_names]
    monomial_index = {_exponent_tuple(m, a_symb): idx for idx, m in enumerate(full_library)}
    a1, a2 = a_symb[0], a_symb[1]
    xi2 = Xi_raw[monomial_index[_exponent_tuple(a1 * a2, a_symb)], 1]
    xi6 = Xi_raw[monomial_index[_exponent_tuple(a1 ** 2 * a2, a_symb)], 1]
    ratio = xi2 / xi6 if xi6 != 0 else np.inf
    return dict(xi2=xi2, xi6=xi6, ratio=ratio)


def symmetry_residual_generic(Xi, feat_names, a_symb, group=G):
    """Same symmetry-violation check as `symmetry_residual`, but computed directly
    against an arbitrary Xi (e.g. an unconstrained fit's coefficients) instead of a
    `constraints` bundle -- how badly does this fit violate the known symmetry group,
    relative to the same threshold the constrained fits satisfy exactly (~1e-14)?"""
    full_library = [sp.sympify(" * ".join(f.split(" "))) for f in feat_names]
    monomial_index = {_exponent_tuple(m, a_symb): idx for idx, m in enumerate(full_library)}
    subs_rules = symmetry_subs_rules(group, a_symb)
    M_list = [_build_Mk(full_library, sr, a_symb, monomial_index) for sr in subs_rules]
    scale = np.max(np.abs(Xi)) or 1.0
    resids = [np.max(np.abs(np.array(Mk, dtype=float) @ Xi - Xi @ g.T))
              for Mk, g in zip(M_list, group)]
    return max(resids) / scale


def equation_strings(fit, raw=True, precision=4):
    """The fitted (a1', a2', a3') equations as a list of strings, e.g.
    "(a1)' = +0.77 a1 -0.017 a1^3" -- `raw=True` (default) uses the raw sigma*v
    amplitude units (Xi_raw), matching Guan et al.'s eq. (4.9) convention; `raw=False`
    uses the normalised-coordinate coefficients actually passed to the optimizer.
    """
    Xi = fit["Xi_raw"] if raw else fit["Xi"]
    names = [s.name for s in fit["constraints"]["a_symb"]]
    lines = []
    for i, nm in enumerate(names):
        terms = [f"{Xi[j, i]:+.{precision}g} {fit['feat_names'][j]}"
                 for j in range(len(fit["feat_names"])) if abs(Xi[j, i]) > 1e-8]
        lines.append(f"({nm})' = " + " ".join(terms))
    return lines


def print_model(fit, raw=True):
    for line in equation_strings(fit, raw=raw):
        print(line)


def symmetry_residual(fit):
    """max|M_k Xi - Xi g_k^T| relative to max|Xi|, re-derived independent of pysindy's
    internal constraint_order convention -- catches ordering/wiring bugs even if the
    fit "converged"."""
    Xi = fit["Xi"]
    scale = np.max(np.abs(Xi)) or 1.0
    group = fit["constraints"]["group"]
    resids = [np.max(np.abs(np.array(Mk, dtype=float) @ Xi - Xi @ g.T))
              for Mk, g in zip(fit["constraints"]["M_list"], group)]
    return max(resids) / scale


def xi_ratio(fit):
    """The ξ2 (a1*a2 coefficient in a2') / ξ6 (a1^2*a2 coefficient in a2') ratio.
    |ratio| >> 1 => the a1-axis fixed points are saddles (Guan et al.'s switching
    mechanism); |ratio| << 1 => they're sources/sinks and the fit just decays or
    blows up instead of switching.

    Generalized to any r >= 2: only ever looks at a1 (a_symb[0]) and a2 (a_symb[1]),
    so it means the same thing regardless of how many other state variables (a3, a4,
    ...) the fit has -- `_exponent_tuple` is passed the FULL a_symb list (not just
    [a1,a2,a3]) so its exponent tuples have the right length to key into `midx`, which
    was built over all r symbols.
    """
    a_symb = fit["constraints"]["a_symb"]
    a1, a2 = a_symb[0], a_symb[1]
    midx = fit["constraints"]["monomial_index"]
    Xi_raw = fit["Xi_raw"]
    xi2 = Xi_raw[midx[_exponent_tuple(a1 * a2, a_symb)], 1]
    xi6 = Xi_raw[midx[_exponent_tuple(a1 ** 2 * a2, a_symb)], 1]
    ratio = xi2 / xi6 if xi6 != 0 else np.inf
    return dict(xi2=xi2, xi6=xi6, ratio=ratio)


def check_bounded(fit, t_end=1000.0, n_eval=5000, max_step=0.2, ic=None):
    """Simulate the fitted model (normalised coords, via pysindy's own integrator) from
    the data's first sample (or `ic`) and report whether it stays finite plus a rough
    switching-count proxy (sign changes in a1)."""
    model = fit["model"]
    x0 = fit["X"][0] if ic is None else ic
    t_eval = np.linspace(0, t_end, n_eval)
    try:
        Y = model.simulate(x0, t_eval, integrator_kws=dict(max_step=max_step))
    except Exception as e:  # noqa: BLE001 -- report, don't crash a sweep over it
        return dict(finite=False, error=f"{type(e).__name__}: {e}")
    finite = bool(np.isfinite(Y).all())
    if not finite:
        return dict(finite=False)
    return dict(finite=True,
                range=np.round(np.ptp(Y, axis=0), 4).tolist(),
                a1_sign_changes=int(np.sum(np.diff(np.sign(Y[:, 0])) != 0)))


def trajectory_fit_quality(fit, X_raw, t, horizon=50.0, max_step=0.05):
    """How well the fitted model actually tracks the real trajectory it was fit on --
    not just whether it's bounded, and not the R^2-of-the-derivative ceiling check.

    Simulates the model (raw units) from the data's own first sample over only the
    first `horizon` time units of `t`, and compares pointwise to the real X_raw there:
    per-variable NRMSE (RMSE normalised by that variable's std over the *full* slice,
    so 0 = perfect, ~1 = no better than predicting the mean) and Pearson correlation.

    Short horizon by design: this is chaotic/switching data, so even a "correct"
    model's trajectory decorrelates from the real one after roughly one Lyapunov time
    -- comparing over the *whole* window would just measure chaos, not fit quality.
    `horizon` should be well under one switching period (see `check_bounded`'s
    `a1_sign_changes` for the period scale) to actually test short-term tracking.
    """
    model = fit["model"]
    X_scale = fit["X_scale"]
    x0 = X_raw[0] / X_scale
    mask = t <= (t[0] + horizon)
    t_eval = t[mask]
    X_true = X_raw[mask]
    std_full = X_raw.std(axis=0)
    try:
        Y = model.simulate(x0, t_eval, integrator_kws=dict(max_step=max_step)) * X_scale
    except Exception as e:  # noqa: BLE001 -- report, don't crash a sweep over it
        return dict(ok=False, error=f"{type(e).__name__}: {e}")
    if not np.isfinite(Y).all() or Y.shape[0] != X_true.shape[0]:
        return dict(ok=False, diverged=True)
    rmse = np.sqrt(np.mean((Y - X_true) ** 2, axis=0))
    nrmse = rmse / np.where(std_full > 0, std_full, 1.0)
    n_vars = X_true.shape[1]
    corr = [float(np.corrcoef(Y[:, i], X_true[:, i])[0, 1]) if X_true[:, i].std() > 0 else float("nan")
            for i in range(n_vars)]
    return dict(ok=True, n=int(mask.sum()), horizon=horizon,
                nrmse=np.round(nrmse, 3).tolist(), corr=np.round(corr, 3).tolist())


def diagnostics(X_raw, t):
    """R^2 ceiling check: how well each amplitude's derivative is explained by the
    FULL (unconstrained) degree<=3 polynomial library IN ALL OF X_raw's COLUMNS --
    independent of the symmetry or energy constraints, or of which optimizer is used
    above. If this is low for some variable, no SINDy fit at this r on this window can
    do better for that variable, regardless of constraints/optimizer. Generalized to
    any r = X_raw.shape[1] (not just r=3) -- the library always spans every retained
    mode, so this is the right ceiling check for whatever r the caller is using.
    """
    dt = float(np.diff(t).mean())
    n = X_raw.shape[1]
    V = [X_raw[:, i] for i in range(n)]
    cols = [np.ones_like(V[0])]
    for deg in (1, 2, 3):
        for combo in cwr(range(n), deg):
            p = np.ones_like(V[0])
            for k in combo:
                p = p * V[k]
            cols.append(p)
    lib = np.vstack(cols).T

    def r2(y):
        c, *_ = np.linalg.lstsq(lib, y, rcond=None)
        return 1 - np.sum((y - lib @ c) ** 2) / np.sum((y - y.mean()) ** 2)

    out = {"dt": dt, "n": len(t), "t_span": (float(t[0]), float(t[-1])),
           "std": np.round(X_raw.std(0), 4).tolist()}
    for i in range(n):
        nm = f"a{i + 1}"
        draw = np.gradient(V[i], dt)
        dsmooth = savgol_filter(V[i], 11, 3, deriv=1, delta=dt) if len(t) > 15 else draw
        out[f"R2_{nm}_raw"] = r2(draw)
        out[f"R2_{nm}_smooth"] = r2(dsmooth)
    return out


# ---------------------------------------------------------------------------
# Top-level: one slice, end to end
# ---------------------------------------------------------------------------
def run_slice(sigma, Vt, t_full, r=3, t_start=None, t_end=None, stride=1, label="",
              constraints=None, reg_weight_lam=0.0, check_sim=True, sim_kwargs=None,
              check_fit_quality=True, fit_quality_horizon=50.0, augment=True,
              weak=False, n_subdomains=100, weak_seed=0, max_iter=500, weak_retries=5,
              group=G, a_symb=A_SYMB):
    """Slice the amplitudes, run diagnostics + the constrained fit + symmetry check +
    a boundedness check + a short-horizon fit-quality check, and return one dict
    "row" summarizing the slice.

    `sim_kwargs` is passed through to `check_bounded` (e.g. dict(t_end=..., n_eval=...,
    max_step=...)) -- its defaults are already light (t_end=1000, n_eval=5000,
    max_step=0.2), but a sweep over many slices can still add up, so tune it down
    further here if a sweep is running long. `fit_quality_horizon` is the short-horizon
    window (in the same time units as `t`) used by `trajectory_fit_quality` -- keep it
    well under one switching period (see the printed `a1_sign_changes` for scale).
    `augment`, `weak`, `n_subdomains`, `weak_seed`, `max_iter` are passed through to
    `fit_symmetric_sindy` -- see there for what they do. `group`/`a_symb` MUST match
    whatever `constraints` was built with (if given) -- e.g. for r=6, build `constraints`
    with `build_symmetry_constraints(r=6, group=G6, a_symb=a_symb6)` and pass
    `group=G6, a_symb=a_symb6` here too, not just `r=6` -- `fit_symmetric_sindy` uses
    its own `group`/`a_symb` params directly (feature names, augmentation, the raw-unit
    conversion), not `constraints["a_symb"]`, so a mismatch here silently uses the wrong
    (module-default, r=3) names/dims even when `constraints` itself is correctly sized.

    When `weak=True`, pysindy's WeakSINDy can occasionally raise on a particular random
    subdomain placement (observed: "Extraneous calculation; truncate to domain [-1,1]"
    -- a floating-point boundary edge case, more likely on short windows) -- this
    retries with `weak_seed + 1000*(1..weak_retries)` on failure. If every retry fails,
    returns a row with `fit=None` and an `"error"` key instead of raising, so one bad
    window doesn't take down a whole `run_multi_slice` sweep; `run_multi_slice`
    excludes such rows from its summary statistics.
    """
    X_raw, t = slice_amplitudes(sigma, Vt, t_full, r=r, t_start=t_start, t_end=t_end, stride=stride)
    diag = diagnostics(X_raw, t)

    fit = None
    last_err = None
    n_attempts = (weak_retries + 1) if weak else 1
    for attempt in range(n_attempts):
        try:
            fit = fit_symmetric_sindy(X_raw, t, r=r, constraints=constraints,
                                       reg_weight_lam=reg_weight_lam, augment=augment, weak=weak,
                                       n_subdomains=n_subdomains, weak_seed=weak_seed + 1000 * attempt,
                                       max_iter=max_iter, group=group, a_symb=a_symb)
            break
        except Exception as e:  # noqa: BLE001 -- report, don't crash a sweep over one bad seed
            last_err = f"{type(e).__name__}: {e}"
    if fit is None:
        return dict(label=label, t_start=t_start, t_end=t_end, stride=stride,
                     diag=diag, fit=None, error=last_err)

    ratio = xi_ratio(fit)
    row = dict(label=label, t_start=t_start, t_end=t_end, stride=stride,
               diag=diag, fit=fit, sym_resid=symmetry_residual(fit), **ratio)
    if check_sim:
        row["bounded"] = check_bounded(fit, **(sim_kwargs or {}))
    if check_fit_quality:
        row["fit_quality"] = trajectory_fit_quality(fit, X_raw, t, horizon=fit_quality_horizon)
    return row


def print_row(row, show_equations=True):
    d = row["diag"]
    print(f"--- {row['label']} ---")
    print(f"  n={d['n']}  dt={d['dt']:.4g}  t in [{d['t_span'][0]:.4g}, {d['t_span'][1]:.4g}]")
    if row.get("fit") is None:
        print(f"  FIT FAILED (all retries exhausted): {row.get('error')}")
        return
    n_vars = sum(1 for k in d if k.startswith("R2_") and k.endswith("_raw"))
    names = [f"a{i + 1}" for i in range(n_vars)]
    r2_str = "  ".join(f"{nm}'={d[f'R2_{nm}_raw']:.3f}" for nm in names)
    print(f"  R2 raw: {r2_str}")
    print(f"  symmetry residual (rel.) = {row['sym_resid']:.2e}")
    print(f"  xi2 (a1*a2) = {row['xi2']:+.4g}   xi6 (a1^2*a2) = {row['xi6']:+.4g}"
          f"   ratio = {row['ratio']:.3g}")
    if show_equations:
        print("  equations (raw sigma*v units):")
        for line in equation_strings(row["fit"]):
            print(f"    {line}")
    if "bounded" in row:
        b = row["bounded"]
        if b.get("finite"):
            print(f"  simulated: finite, range={b['range']}, a1 sign changes={b['a1_sign_changes']}")
        else:
            print(f"  simulated: DIVERGED/FAILED {b.get('error', '')}")
    if "fit_quality" in row:
        q = row["fit_quality"]
        if q.get("ok"):
            print(f"  fit quality (first {q['horizon']:g} time units, n={q['n']}):"
                  f" NRMSE({','.join(names)})={q['nrmse']}  corr({','.join(names)})={q['corr']}")
        else:
            print(f"  fit quality: FAILED {q.get('error', 'diverged before horizon')}")


# ---------------------------------------------------------------------------
# Multiple independent samples per (dt, duration) cell
# ---------------------------------------------------------------------------
# A single window per (dt, duration) -- what run_slice/run_window_sweep.py did up to
# this point -- can't separate a real dt/duration trend from "which switching-cycle
# phase this one window happened to start at" (e.g. the length-sweep's ratio bouncing
# from -189 to -244 with no trend as t_start varies, job 40061710). Drawing several
# non-overlapping windows of the same duration and reporting mean +/- std fixes that,
# and as a side effect gives enough independent samples to eventually compare
# *distributions* (switching period, spike amplitude) between data and simulation
# instead of the doomed pointwise NRMSE comparison (see trajectory_fit_quality).

def multi_window_starts(t_min, t_max, duration, n_samples=None):
    """Non-overlapping t_start values for windows of length `duration` tiling
    [t_min, t_max]. With `n_samples=None`, uses every non-overlapping tile that fits
    (`floor((t_max-t_min)/duration)` of them). With `n_samples` less than that, picks
    that many tiles spread evenly across the *whole* range (not just the first N), for
    coverage rather than just early-trajectory bias.
    """
    span = t_max - t_min
    max_n = max(1, int(span // duration))
    tile_starts = [t_min + i * duration for i in range(max_n)]
    if n_samples is None or n_samples >= max_n:
        return tile_starts
    idx = sorted(set(np.linspace(0, max_n - 1, n_samples).round().astype(int).tolist()))
    return [tile_starts[i] for i in idx]


def _mean_std(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(values)), float(np.std(values)), len(values))


def run_multi_slice(sigma, Vt, t_full, r=3, duration=1000.0, stride=1, n_samples=None,
                     t_min=None, t_max=None, label="", constraints=None,
                     reg_weight_lam=0.0, check_sim=True, sim_kwargs=None,
                     check_fit_quality=True, fit_quality_horizon=50.0, augment=True,
                     weak=False, n_subdomains=100, weak_seed=0, max_iter=500,
                     group=G, a_symb=A_SYMB):
    """Run `run_slice` over several non-overlapping windows of the same `duration` and
    `stride`, tiling [t_min, t_max] (defaults to the full available range). Returns a
    dict with `rows` (each window's full run_slice result, for e.g. equation_strings
    or per-window plotting) and `summary` (mean, std, n_ok across the successful
    samples for R^2, xi ratio, and fit-quality NRMSE/corr -- "n_ok" can be less than
    the sample count if a window diverged or its fit_quality simulate() failed).
    `augment`, `n_subdomains`, `max_iter` are passed through to `fit_symmetric_sindy`
    for every window unchanged. `weak_seed` is offset by the window index (`weak_seed
    + k`) so each window gets its own reproducible-but-distinct subdomain placement
    rather than literally repeating the same random draw shape across windows.
    `group`/`a_symb` must match `constraints` if given -- see `run_slice`.
    """
    if t_min is None:
        t_min = t_full[0]
    if t_max is None:
        t_max = t_full[-1]
    starts = multi_window_starts(t_min, t_max, duration, n_samples)

    rows = []
    for k, t_start in enumerate(starts):
        row = run_slice(sigma, Vt, t_full, r=r, t_start=t_start, t_end=t_start + duration,
                         stride=stride, label=f"{label} sample{k} (t_start={t_start:.0f})",
                         constraints=constraints, reg_weight_lam=reg_weight_lam,
                         check_sim=check_sim, sim_kwargs=sim_kwargs,
                         check_fit_quality=check_fit_quality, fit_quality_horizon=fit_quality_horizon,
                         augment=augment, weak=weak, n_subdomains=n_subdomains,
                         weak_seed=weak_seed + k, max_iter=max_iter, group=group, a_symb=a_symb)
        rows.append(row)

    ok_rows = [r for r in rows if r.get("fit") is not None]
    summary = dict(label=label, duration=duration, stride=stride, n_windows=len(rows),
                    n_failed=len(rows) - len(ok_rows), t_starts=starts)
    summary["R2_a1"] = _mean_std([r["diag"]["R2_a1_raw"] for r in ok_rows])
    summary["R2_a2"] = _mean_std([r["diag"]["R2_a2_raw"] for r in ok_rows])
    summary["R2_a3"] = _mean_std([r["diag"]["R2_a3_raw"] for r in ok_rows])
    summary["ratio"] = _mean_std([r["ratio"] for r in ok_rows])
    if check_sim:
        summary["a1_sign_changes"] = _mean_std(
            [r["bounded"].get("a1_sign_changes") for r in ok_rows if r["bounded"].get("finite")])
    if check_fit_quality:
        for i, nm in enumerate(["a1", "a2", "a3"]):
            summary[f"nrmse_{nm}"] = _mean_std(
                [r["fit_quality"]["nrmse"][i] for r in ok_rows if r["fit_quality"].get("ok")])
            summary[f"corr_{nm}"] = _mean_std(
                [r["fit_quality"]["corr"][i] for r in ok_rows if r["fit_quality"].get("ok")])
    return dict(rows=rows, summary=summary)


def print_multi_summary(result):
    s = result["summary"]
    failed_note = f", {s['n_failed']} FAILED" if s.get("n_failed") else ""
    print(f"--- {s['label']}  ({s['n_windows']} windows, duration={s['duration']:g}, "
          f"stride={s['stride']}{failed_note}) ---")
    print(f"  t_starts = {[round(t) for t in s['t_starts']]}")

    def fmt(key, name):
        m, sd, n = s[key]
        print(f"  {name}: {m:+.4g} +/- {sd:.3g}  (n={n})")

    fmt("R2_a1", "R2(a1')"); fmt("R2_a2", "R2(a2')"); fmt("R2_a3", "R2(a3')")
    fmt("ratio", "xi2/xi6 ratio")
    if "a1_sign_changes" in s:
        fmt("a1_sign_changes", "a1 sign changes")
    if "nrmse_a1" in s:
        fmt("nrmse_a1", "NRMSE a1"); fmt("nrmse_a2", "NRMSE a2"); fmt("nrmse_a3", "NRMSE a3")
        fmt("corr_a1", "corr a1"); fmt("corr_a2", "corr a2"); fmt("corr_a3", "corr a3")
