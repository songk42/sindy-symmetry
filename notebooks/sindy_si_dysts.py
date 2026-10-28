# %% [markdown]
# # SINDy-SI (Machado & Jones, arXiv:2310.04227) on the dysts benchmark
#
# Same systems, data, noise draws, derivative estimates and metrics as `symmetric_sindy_dysts.py`, so the results
# can be paired row-by-row with that run (logs/symmetric_sindy_dysts/alpha0.05/fits.csv). The RNG calls happen in
# the same order as there, so every system/noise level sees the identical noisy trajectory.
#
# Side information per system, all read off the true dysts equations (as the symmetries were before):
#   sym  - signed-permutation equivariance f(g x) = g f(x)  -> linear equalities (same C as v1)
#   eq   - equilibria of the true vector field inside the data ball  -> f_hat(x_eq) = 0
#   mono - coordinate-wise monotonicity s * d f_i/d x_i >= 0 on the data ball, kept only if the true coefficients
#          admit the SOS certificate (the side information used in the paper's Lorenz example)
#
# Methods (all with unit-norm library columns and thresholds on the normalized coefficients, as in the paper):
#   SINDySI_none         sequential thresholding, no side info (the "SINDy" column of the paper, with refits)
#   SINDySI_sym          symmetry only (equality-constrained LS, solved in the constraint null space)
#   SINDySI_sym_eq       symmetry + equilibria
#   SINDySI_paper        equilibria + monotonicity (the paper's Lorenz side information, no symmetry)
#   SINDySI_full         symmetry + equilibria + monotonicity
#   SI_full              all side information, no pruning (the paper's "SI" column)
#
# Equiv-c (Yang et al., arXiv:2405.16756; see equiv_c.py), group-element form of the constraint, raw library,
# thresholds relative to max|Xi| of the dense lstsq fit (the same scale symmetric_sindy_dysts.py uses for STLSQ):
#   EquivC               symmetry null space, constant terms free (reference default), no ridge (paper config)
#   EquivC_cc            same with the constant terms constrained too (--constrain_constant)

# %%
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pysindy as ps
import scipy.linalg
import scipy.optimize
import sympy as sp

import dysts.flows as flows

from fit_metrics import Evaluator

from equiv_c import EquivC
from sindy_si import SINDySI, certify_monotone

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DYSTS_DATA_DIR", "/home/songk/data-songk/dysts_hf/fine")
OUT_DIR = os.path.join(os.environ.get("LOG_DIR", "/home/songk/sindy-symmetry/logs"), "sindy_si_dysts")
os.makedirs(OUT_DIR, exist_ok=True)

DEGREE = 3
NOISE_LEVELS = ["exact_xdot", 0.0, 0.01, 0.05]
# relative to max|W~| of the dense normalized fit. 0.003-0.1 is the grid of symmetric_sindy_dysts.py; 0.3 is added
# because spurious normalized coefficients of the unregularized fit are large (summaries report both grids)
REL_THRESHOLDS = [0.003, 0.01, 0.03, 0.1, 0.3]
RNG = np.random.default_rng(0)  # consumed exactly as in symmetric_sindy_dysts.py
SIDE_RNG = np.random.default_rng(1)  # equilibrium search only


# %% [markdown]
# ## Symmetry helpers (identical to symmetric_sindy_dysts.py)

# %%
def signed_permutations(n):
    for perm in itertools.permutations(range(n)):
        for signs in itertools.product([1, -1], repeat=n):
            g = np.zeros((n, n), dtype=int)
            g[np.arange(n), perm] = signs
            yield g


def eval_rhs(system, X):
    try:
        F = np.stack(system.rhs(X, 0.0), axis=-1)
        if F.shape == X.shape:
            return F
    except Exception:
        pass
    return np.array([system.rhs(x, 0.0) for x in X])


def detect_symmetries(system, X, rtol=1e-8):
    n = X.shape[1]
    pts = X[RNG.choice(len(X), size=200, replace=False)]
    F = eval_rhs(system, pts)
    scale = np.max(np.abs(F)) + 1e-12
    G = []
    for g in signed_permutations(n):
        if np.array_equal(g, np.eye(n, dtype=int)):
            continue
        resid = np.max(np.abs(eval_rhs(system, pts @ g.T) - F @ g.T))
        if resid / scale < rtol:
            G.append(g)
    return G


def build_Mk(library, subs_rule, vars_, monomial_index):
    n = len(library)
    M = sp.zeros(n, n)
    for j, theta_j in enumerate(library):
        transformed = sp.expand(theta_j.subs(subs_rule, simultaneous=True))
        poly = sp.Poly(transformed, *vars_)
        for exps, coeff in poly.terms():
            M[monomial_index[exps], j] += coeff
    return M


def exponent_tuple(monom, vars_):
    poly = sp.Poly(monom, *vars_)
    if poly.is_zero:
        return tuple([0] * len(vars_))
    return poly.monoms()[0]


def symmetry_constraints(G, feature_names, var_names):
    syms = sp.symbols(var_names)
    full_library = [sp.sympify(" * ".join(f.split(" "))) for f in feature_names]
    monomial_index = {exponent_tuple(m, syms): idx for idx, m in enumerate(full_library)}
    M_list, C_list = [], []
    for g in G:
        subs_rule = {a: sum(int(g[i, j]) * syms[j] for j in range(len(syms))) for i, a in enumerate(syms)}
        M = np.array(build_Mk(full_library, subs_rule, syms, monomial_index), dtype=float)
        M_list.append(M)
        C_list.append(np.kron(np.eye(g.shape[0]), M) - np.kron(g, np.eye(M.shape[0])))
    A = np.vstack(C_list)
    rank = np.linalg.matrix_rank(A)
    _, _, piv = scipy.linalg.qr(A.T, pivoting=True)
    C = A[piv[:rank]]
    C = C[~np.isclose(C, 0).all(axis=1)]
    return M_list, C, np.zeros(C.shape[0])


# %% [markdown]
# ## Non-symmetry side information from the true equations

# %%
def find_equilibria(system, X_raw, center, radius, s, n_starts=40):
    """Roots of the true RHS (raw coords) reachable from attractor points + origin; returned in scaled coords."""
    F_scale = np.max(np.abs(eval_rhs(system, X_raw))) + 1e-12
    starts = np.vstack([np.zeros(X_raw.shape[1]), X_raw[SIDE_RNG.choice(len(X_raw), n_starts, replace=False)]])
    found = []
    for x0 in starts:
        x, info, ier, _ = scipy.optimize.fsolve(lambda x: system.rhs(x, 0.0), x0, full_output=True, xtol=1e-13)
        if ier != 1 or np.max(np.abs(system.rhs(x, 0.0))) > 1e-9 * F_scale:
            continue
        y = x / s
        if np.linalg.norm(y - center) > radius:
            continue
        if not any(np.linalg.norm(y - z) < 1e-6 for z in found):
            found.append(y)
    return found


def certified_monotone(Xi_true, powers, center, radius):
    out = []
    for i in range(Xi_true.shape[1]):
        for sign in (-1, 1):
            if certify_monotone(Xi_true, powers, i, i, sign, center, radius):
                out.append((i, i, sign))
                break
    return out


# %% [markdown]
# ## Main loop

# %%
meta = json.load(open(os.path.join(os.path.dirname(flows.__file__), "data", "chaotic_attractors.json")))
names = sorted(f.replace("_fine.csv", "") for f in os.listdir(DATA_DIR) if f.endswith("_fine.csv"))
if len(sys.argv) > 1:
    names = [n for n in names if n in sys.argv[1:]]
suffix = "" if len(sys.argv) == 1 else "_" + "_".join(sys.argv[1:])

side_rows, fit_rows = [], []
for name in names:
    t0 = time.time()
    md = meta.get(name, {})
    if md.get("nonautonomous") or md.get("delay") or not hasattr(flows, name):
        continue
    system = getattr(flows, name)()
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}_fine.csv"))
    X_raw = df.iloc[:, 1:].to_numpy(float)
    n = X_raw.shape[1]
    if n > 4 or not np.all(np.isfinite(X_raw)):
        continue
    try:
        G = detect_symmetries(system, X_raw)
        F_raw = eval_rhs(system, X_raw)
    except Exception as e:
        print(f"{name}: rhs eval failed ({e})")
        continue

    dXdn = np.gradient(X_raw, axis=0)
    dt = np.sum(dXdn * F_raw) / np.sum(F_raw * F_raw)
    dt_r2 = 1 - np.sum((dXdn - dt * F_raw) ** 2) / np.sum((dXdn - dXdn.mean(0)) ** 2)

    s = np.max(np.abs(X_raw))
    X = X_raw / s
    t = np.arange(len(X)) * dt
    var_names = [f"x{i}" for i in range(n)]
    lib = ps.PolynomialLibrary(degree=DEGREE, include_bias=True)
    lib.fit(X)
    feature_names = lib.get_feature_names(var_names)
    powers = np.asarray(lib.powers_)
    Theta = np.asarray(lib.transform(X))
    Xi_true, *_ = np.linalg.lstsq(Theta, F_raw / s, rcond=None)
    poly_resid = np.linalg.norm(Theta @ Xi_true - F_raw / s) / np.linalg.norm(F_raw / s)
    is_poly = poly_resid < 1e-6
    Xi_true[np.abs(Xi_true) < 1e-8 * np.max(np.abs(Xi_true))] = 0.0

    # mirror symmetric_sindy_dysts.py: only symmetric, polynomial, well-calibrated systems are fitted
    if not G or not is_poly or dt_r2 < 0.99:
        continue
    evaluate = Evaluator(X, Xi_true, powers, dt, md.get("maximum_lyapunov_estimated"))
    M_list, C, d = symmetry_constraints(G, feature_names, var_names)

    center = X.mean(axis=0)
    radius = 1.1 * np.max(np.linalg.norm(X - center, axis=1))
    try:
        equilibria = find_equilibria(system, X_raw, center, radius, s)
    except Exception as e:
        print(f"  equilibrium search failed ({e})")
        equilibria = []
    monotone = certified_monotone(Xi_true, powers, center, radius)
    eq_resid = max((np.max(np.abs(np.prod(x[None] ** powers, axis=1) @ Xi_true)) for x in equilibria), default=0.0)
    side_rows.append(dict(system=name, dim=n, n_sym=len(G), n_constraints=C.shape[0], n_equilibria=len(equilibria),
                          equilibria=[x.tolist() for x in equilibria], eq_resid_true=eq_resid,
                          monotone=monotone, ball_center=center.tolist(), ball_radius=radius))
    print(f"{name}: |G|={len(G)} n_eq={len(equilibria)} (true resid {eq_resid:.1e}) monotone={monotone}", flush=True)

    side = {
        "SINDySI_none": dict(),
        "SINDySI_sym": dict(eq_lhs=C, eq_rhs=d),
        "SINDySI_sym_eq": dict(eq_lhs=C, eq_rhs=d, equilibria=equilibria),
        "SINDySI_paper": dict(equilibria=equilibria, monotone=monotone, ball=(center, radius)),
        "SINDySI_full": dict(eq_lhs=C, eq_rhs=d, equilibria=equilibria, monotone=monotone, ball=(center, radius)),
    }

    for noise in NOISE_LEVELS:
        exact = noise == "exact_xdot"
        Xn = X if exact else X + noise * X.std(axis=0) * RNG.standard_normal(X.shape)
        diff = ps.FiniteDifference(order=4) if noise in ("exact_xdot", 0.0) else ps.SmoothedFiniteDifference()
        Y = F_raw / s if exact else np.asarray(diff(Xn, t))
        # ps.SINDy.fit evaluates the library on the smoothed trajectory when the differentiator provides one
        # (SmoothedFiniteDifference, save_smooth=True); do the same so features match symmetric_sindy_dysts.py
        Phi = np.asarray(lib.transform(np.asarray(getattr(diff, "smoothed_x_", Xn)) if not exact else Xn))

        dense = SINDySI(threshold=0.0).fit(Phi, Y, powers)
        coef_scale = np.max(np.abs(dense.coef_normalized_))
        raw_scale = np.max(np.abs(dense.coef_))

        configs = [("SI_full", None, side["SINDySI_full"])]
        for rt in REL_THRESHOLDS:
            configs += [(m, rt, kw) for m, kw in side.items()]
            configs += [("EquivC", rt, dict(equiv_c=True, constrain_constant=False)),
                        ("EquivC_cc", rt, dict(equiv_c=True, constrain_constant=True))]
        cache = {}
        for method, rt, kw in configs:
            # with no monotone side info, "paper" == eq-only and "full" == sym_eq: don't solve twice
            key = (rt, tuple(sorted((k, id(v)) for k, v in kw.items() if k != "ball" and not (k == "monotone" and not v))))
            tic = time.time()
            try:
                if kw.get("equiv_c"):
                    opt = EquivC(threshold=rt * raw_scale, constraint=C,
                                 constrain_constant=kw["constrain_constant"]).fit(Phi, Y)
                    opt.history_ = [None] * opt.n_iter_
                elif key in cache:
                    opt = cache[key]
                else:
                    opt = SINDySI(threshold=0.0 if rt is None else rt * coef_scale, max_iter=1 if rt is None else None,
                                  **kw).fit(Phi, Y, powers)
                    cache[key] = opt
            except Exception as e:
                print(f"  {method} rt={rt} noise={noise}: fit failed ({e})")
                continue
            Xi = opt.coef_
            sym_resid = max(np.max(np.abs(M @ Xi - Xi @ g.T)) for g, M in zip(G, M_list)) / max(np.max(np.abs(Xi)), 1e-12)
            fit_rows.append(dict(system=name, noise=str(noise), method=method, rel_threshold=rt, **evaluate(Xi),
                                 sym_resid=sym_resid, n_iter=len(opt.history_), fit_time=time.time() - tic))
    print(f"  fits done ({time.time() - t0:.1f}s)", flush=True)
    pd.DataFrame(side_rows).to_csv(os.path.join(OUT_DIR, f"side_info{suffix}.csv"), index=False)
    pd.DataFrame(fit_rows).to_csv(os.path.join(OUT_DIR, f"fits{suffix}.csv"), index=False)

# %% [markdown]
# ## Summary

# %%
fit_df = pd.DataFrame(fit_rows)
pd.set_option("display.width", 250, "display.max_rows", 500)
if len(fit_df):
    best = fit_df.sort_values("coef_rel_err").groupby(["system", "noise", "method"], as_index=False).first()
    print("\n=== Aggregate (median rel. coef err / frac exact support / max sym resid) ===")
    print(best.groupby(["noise", "method"]).agg(med_err=("coef_rel_err", "median"), exact=("exact_support", "mean"),
                                                 max_sym_resid=("sym_resid", "max")).to_string(float_format="%.2e"))
