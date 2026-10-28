# %% [markdown]
# # Checking symmetric SINDy (v1) against the dysts benchmark
#
# Runs the constraint construction from `symmetric_sindy_v1.py` (sympy M_k, (I (x) M_k - g_k (x) I) vec(Xi) = 0,
# QR row pruning, augmentation with X g_k^T) on the pre-simulated dysts "fine" trajectories
# (`hf download williamgilpin/dysts --repo-type=dataset`), where the true equations are known.
#
# For each system:
#   1. detect every signed-permutation symmetry g (f(g x) = g f(x)) numerically from the dysts RHS
#   2. get the true Xi by regressing the true RHS onto the degree-3 polynomial library (exact iff polynomial)
#   3. sanity-check the construction: Theta(X g^T) = Theta(X) M_k and C vec(Xi_true) = 0
#   4. fit STLSQ (raw / augmented), ConstrainedSTLSQ, and ConstrainedSR3(reg_weight_lam=0, as in v1)
#      at several noise levels and compare coefficient error + support recovery to the truth

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
import sympy as sp

import dysts.flows as flows

from equiv_c import EquivC
from fit_metrics import Evaluator

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DYSTS_DATA_DIR", "/gscratch/stf/songk42/dysts_hf/fine")
ALPHA = float(os.environ.get("SINDY_ALPHA", 0.05))  # ridge for STLSQ variants; both refit with alpha=0 on the final support (unbias)
# column scaling: divide each library column by its L2 norm, threshold the scaled coefficients (as in SINDy-SI)
NORMALIZE = os.environ.get("SINDY_NORMALIZE", "0") == "1"
# "stlsq" (default): STLSQ family + dense v1 ConstrainedSR3; "sr3": sparse SR3 / ConstrainedSR3 (ridge-free, so
# SINDY_ALPHA is unused). Noise draws don't depend on this, so runs pair row-by-row.
METHODS = os.environ.get("SINDY_METHODS", "stlsq")
SR3_NU = 1.0
SR3_MAX_ITER = 500
# Extra side information for the constrained methods (as in SINDy-SI; read from logs/sindy_si_dysts/side_info.csv):
#   ""       symmetry only (all earlier runs)
#   "eq"     + equilibria f(x_eq) = 0 as extra equality rows
#   "eq_mono"+ sampled monotonicity s * d f_i/d x_i (x_k) >= 0 at N_MONO_PTS uniform points in the SINDy-SI ball, as
#            linear inequalities. Only ConstrainedSTLSQ (QP refit) and ConstrainedSR3's inequality path (l1, one
#            cvxpy solve, no unbias) can take inequalities; the other methods get the equilibria only.
# In side-info runs only the constrained methods are fitted (nothing changes for the others).
SIDE = os.environ.get("SINDY_SIDE", "")
N_MONO_PTS = 1000
OUT_DIR = os.path.join(os.environ.get("LOG_DIR", "/mmfs1/home/songk42/ehd-sindy/logs"), "symmetric_sindy_dysts",
                       (f"alpha{ALPHA:g}" if METHODS == "stlsq" else METHODS) + ("_norm" if NORMALIZE else "")
                       + (f"_{SIDE}" if SIDE else ""))
os.makedirs(OUT_DIR, exist_ok=True)

DEGREE = 3
NOISE_LEVELS = ["exact_xdot", 0.0, 0.01, 0.05]  # fraction of per-channel std; "exact_xdot" = clean x with true RHS as x_dot
REL_THRESHOLDS = [0.003, 0.01, 0.03, 0.1]  # STLSQ thresholds, relative to max|Xi| of a dense lstsq fit
if NORMALIZE:
    REL_THRESHOLDS = REL_THRESHOLDS + [0.3]  # same 5-point grid as sindy_si_dysts.py; relative to max|scaled Xi|
RNG = np.random.default_rng(0)


# %% [markdown]
# ## Helpers (same constructions as v1, generalized to n variables)

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
    """All non-identity signed permutations g with f(x g^T) = f(x) g^T on attractor points."""
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
    """Theta(g x) = Theta(x) @ M_k (identical to v1)."""
    n = len(library)
    M = sp.zeros(n, n)
    for j, theta_j in enumerate(library):
        transformed = sp.expand(theta_j.subs(subs_rule, simultaneous=True))
        poly = sp.Poly(transformed, *vars_)
        for exps, coeff in poly.terms():
            i = monomial_index[exps]
            M[i, j] += coeff
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


def equilibrium_rows(equilibria, powers):
    """f_hat(x_eq) = 0: for each equilibrium, one row per equation, target order."""
    n = powers.shape[1]
    rows = [np.kron(np.eye(n), np.prod(np.asarray(x)[None, :] ** powers, axis=1)[None, :]) for x in equilibria]
    return np.vstack(rows) if rows else np.zeros((0, powers.shape[0] * n))


def prune_rows(A):
    """Full-row-rank subset of the constraint rows (same pivoted-QR pruning as symmetry_constraints)."""
    if not len(A):
        return A
    rank = np.linalg.matrix_rank(A)
    _, _, piv = scipy.linalg.qr(A.T, pivoting=True)
    A = A[np.sort(piv[:rank])]
    return A[~np.isclose(A, 0).all(axis=1)]


def monotone_rows(monotone, powers, center, radius, rng, n_pts=N_MONO_PTS):
    """A w <= 0 rows for -s * d f_hat_i / d x_i (x_k) <= 0 at n_pts uniform points in the ball (target order)."""
    h, n = powers.shape
    z = rng.standard_normal((n_pts, n))
    pts = center + radius * z / np.linalg.norm(z, axis=1, keepdims=True) * rng.random((n_pts, 1)) ** (1 / n)
    rows = []
    for i, j, sgn in monotone:
        e = powers.copy()
        coef = e[:, j].astype(float)
        e[:, j] = np.maximum(e[:, j] - 1, 0)
        dtheta = coef[None, :] * np.prod(pts[:, None, :] ** e[None, :, :], axis=2)  # (n_pts, h)
        block = np.zeros((n_pts, h * n))
        block[:, i * h:(i + 1) * h] = -sgn * dtheta
        rows.append(block)
    return np.vstack(rows) if rows else np.zeros((0, h * n))


side_info = None
if SIDE:
    import ast
    import zlib
    side_info = pd.read_csv(os.path.join(os.path.dirname(OUT_DIR), "..", "sindy_si_dysts", "side_info.csv"))
    side_info = side_info.set_index("system")


# %% [markdown]
# ## Main loop

# %%
meta = json.load(open(os.path.join(os.path.dirname(flows.__file__), "data", "chaotic_attractors.json")))
names = sorted(f.replace("_fine.csv", "") for f in os.listdir(DATA_DIR) if f.endswith("_fine.csv"))
if len(sys.argv) > 1:
    names = [n for n in names if n in sys.argv[1:]]

sym_rows, fit_rows = [], []
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

    # The CSV time column is a sample index; recover the physical step by regressing dX/dn on f(X).
    dXdn = np.gradient(X_raw, axis=0)
    dt = np.sum(dXdn * F_raw) / np.sum(F_raw * F_raw)
    dt_r2 = 1 - np.sum((dXdn - dt * F_raw) ** 2) / np.sum((dXdn - dXdn.mean(0)) ** 2)

    # Uniform scaling commutes with every signed permutation (same argument as v1).
    s = np.max(np.abs(X_raw))
    X = X_raw / s
    t = np.arange(len(X)) * dt
    var_names = [f"x{i}" for i in range(n)]
    lib = ps.PolynomialLibrary(degree=DEGREE, include_bias=True)
    lib.fit(X)
    feature_names = lib.get_feature_names(var_names)
    Theta = lib.transform(X)
    Theta = np.asarray(Theta)

    # True Xi in scaled coords: dY/dt = f(sY)/s. Exact iff the RHS lies in the library.
    Xi_true, *_ = np.linalg.lstsq(Theta, F_raw / s, rcond=None)
    poly_resid = np.linalg.norm(Theta @ Xi_true - F_raw / s) / np.linalg.norm(F_raw / s)
    is_poly = poly_resid < 1e-6
    Xi_true[np.abs(Xi_true) < 1e-8 * np.max(np.abs(Xi_true))] = 0.0

    row = dict(system=name, dim=n, n_sym=len(G), dt=dt, dt_r2=dt_r2, poly_resid=poly_resid,
               is_poly=is_poly, G=[g.tolist() for g in G])
    if not G:
        sym_rows.append(row)
        print(f"{name}: no signed-permutation symmetry ({time.time() - t0:.1f}s)")
        continue

    M_list, C, d = symmetry_constraints(G, feature_names, var_names)
    row["n_constraints"] = C.shape[0]
    row["Mk_resid"] = max(np.max(np.abs(np.asarray(lib.transform(X @ g.T)) - Theta @ M)) for g, M in zip(G, M_list))
    vec_true = Xi_true.flatten(order="F")
    row["true_constraint_resid"] = np.max(np.abs(C @ vec_true)) / np.max(np.abs(vec_true)) if is_poly else np.nan
    sym_rows.append(row)
    print(f"{name}: |G|={len(G)} poly={is_poly} dt_r2={dt_r2:.6f} Mk_resid={row['Mk_resid']:.1e} "
          f"C@Xi_true={row['true_constraint_resid']:.1e}")
    if not is_poly or dt_r2 < 0.99:  # dt_r2 < 0.99: CSV sampling doesn't match the dysts RHS, truth unusable
        continue
    evaluate = Evaluator(X, Xi_true, np.asarray(lib.powers_), dt, md.get("maximum_lyapunov_estimated"))
    A_ineq = np.zeros((0, C.shape[1]))
    if SIDE:
        si = side_info.loc[name]
        powers = np.asarray(lib.powers_)
        C = prune_rows(np.vstack([C, equilibrium_rows([np.array(x) for x in ast.literal_eval(si["equilibria"])], powers)]))
        d = np.zeros(C.shape[0])
        if SIDE == "eq_mono":
            A_ineq = monotone_rows([tuple(m) for m in ast.literal_eval(si["monotone"])], powers,
                                   np.array(ast.literal_eval(si["ball_center"])), float(si["ball_radius"]),
                                   np.random.default_rng(zlib.crc32(name.encode())))
        b_ineq = np.zeros(A_ineq.shape[0])
        viol = np.max(A_ineq @ vec_true) / np.max(np.abs(vec_true)) if len(A_ineq) else 0.0
        print(f"  side info {SIDE}: {C.shape[0]} equality rows (C@Xi_true={np.max(np.abs(C @ vec_true)):.1e}), "
              f"{A_ineq.shape[0]} inequality rows (max violation by Xi_true {viol:.1e})")

    for noise in NOISE_LEVELS:
        # dysts "fine" data is ~100 samples/period, so 2nd-order FD alone leaves a few-% derivative bias
        exact = noise == "exact_xdot"
        Xn = X if exact else X + noise * X.std(axis=0) * RNG.standard_normal(X.shape)
        X_aug = [Xn] + [Xn @ g.T for g in G]
        t_aug = [t] * len(X_aug)
        xdot = F_raw / s if exact else None
        xdot_aug = [xdot @ g.T for g in [np.eye(n)] + G] if exact else None
        diff = ps.FiniteDifference(order=4) if noise in ("exact_xdot", 0.0) else ps.SmoothedFiniteDifference()

        # threshold scale from a dense unconstrained fit
        dense = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=DEGREE), optimizer=ps.STLSQ(threshold=0.0, alpha=0.0),
                         differentiation_method=diff)
        dense.fit(Xn, t=t, x_dot=xdot)
        coef_scale = np.max(np.abs(dense.coefficients()))
        if NORMALIZE:  # threshold scale of the column-scaled problem
            coef_scale = np.max(np.abs(dense.coefficients() * np.linalg.norm(dense.optimizer.Theta_, axis=0)))

        configs = []
        if SIDE and METHODS == "stlsq":
            # constrained methods only; v1 dense and Equiv-c are raw-library methods
            configs = [] if NORMALIZE else [("ConstrainedSR3_v1_noaug", None, False)]
            for rt in REL_THRESHOLDS:
                configs += [("ConstrainedSTLSQ_noaug", rt, False)] + ([] if NORMALIZE else [("EquivC_cc", rt, False)])
        elif SIDE and METHODS == "sr3":
            for rt in REL_THRESHOLDS:
                configs += [("ConstrainedSR3_l0_noaug", rt, False), ("ConstrainedSR3_l1_noaug", rt, False)]
        elif METHODS == "stlsq":
            configs = [] if NORMALIZE else [("ConstrainedSR3_v1", None, True), ("ConstrainedSR3_v1_noaug", None, False)]
            for rt in REL_THRESHOLDS:
                configs += [("STLSQ", rt, False), ("STLSQ_aug", rt, True), ("ConstrainedSTLSQ", rt, True),
                            ("ConstrainedSTLSQ_noaug", rt, False)]
        elif METHODS == "sr3":
            # threshold tau = rt * coef_scale, as for STLSQ: l0 prox keeps |w| > sqrt(2 lam nu), l1 shrinks by lam nu
            for rt in REL_THRESHOLDS:
                configs += [("SR3_l0", rt, False), ("ConstrainedSR3_l0_noaug", rt, False),
                            ("ConstrainedSR3_l1_noaug", rt, False)]

        use_ineq = len(A_ineq) > 0
        # ConstrainedSR3's inequality path: stacked [A; C], separation index = number of inequality rows
        sr3_ineq = dict(constraint_lhs=np.vstack([A_ineq, C]), constraint_rhs=np.concatenate([b_ineq, d]),
                        constraint_order="target", inequality_constraints=True, equality_constraints=True,
                        constraint_separation_index=A_ineq.shape[0]) if use_ineq else None
        Phi_xdot = None
        for method, rt, augment in configs:
            if method == "EquivC_cc":
                if Phi_xdot is None:  # same features / targets as ps.SINDy.fit
                    Yd = xdot if exact else np.asarray(diff(Xn, t))
                    Xs = Xn if exact else np.asarray(getattr(diff, "smoothed_x_", Xn))
                    Phi_xdot = (np.asarray(lib.transform(Xs)), Yd)
                Xi = EquivC(threshold=rt * coef_scale, constraint=C, constrain_constant=True).fit(*Phi_xdot).coef_
                sym_resid = max(np.max(np.abs(M @ Xi - Xi @ g.T)) for g, M in zip(G, M_list)) / max(np.max(np.abs(Xi)), 1e-12)
                fit_rows.append(dict(system=name, noise=str(noise), method=method, rel_threshold=rt, **evaluate(Xi),
                                     sym_resid=sym_resid))
                continue
            if method.startswith("ConstrainedSR3_v1"):
                if use_ineq:  # dense constrained least squares with all side info (cvxpy)
                    opt = ps.ConstrainedSR3(reg_weight_lam=0, regularizer="l1", **sr3_ineq)
                else:
                    opt = ps.ConstrainedSR3(constraint_lhs=C, constraint_rhs=d, constraint_order="target",
                                            equality_constraints=True, reg_weight_lam=0)
            elif method.startswith("SR3") or method.startswith("ConstrainedSR3"):
                tau = rt * coef_scale
                reg = "l1" if "_l1" in method else "l0"
                kw = dict(reg_weight_lam=(tau if reg == "l1" else tau ** 2 / 2) / SR3_NU, regularizer=reg,
                          relax_coeff_nu=SR3_NU, max_iter=SR3_MAX_ITER, unbias=True, normalize_columns=NORMALIZE)
                if method.startswith("Constrained") and use_ineq and reg == "l1":
                    kw.pop("unbias")  # inequality path: one cvxpy solve, no thresholding loop, unbias not allowed
                    opt = ps.ConstrainedSR3(**sr3_ineq, **kw)
                elif method.startswith("Constrained"):
                    opt = ps.ConstrainedSR3(constraint_lhs=C, constraint_rhs=d, constraint_order="target",
                                            equality_constraints=True, **kw)
                else:
                    opt = ps.SR3(**kw)
            elif method.startswith("ConstrainedSTLSQ"):
                opt = ps.ConstrainedSTLSQ(threshold=rt * coef_scale, alpha=ALPHA, constraint_lhs=C,
                                          constraint_rhs=d, constraint_order="target", normalize_columns=NORMALIZE,
                                          **(dict(inequality_lhs=A_ineq, inequality_rhs=b_ineq) if use_ineq else {}))
            else:
                opt = ps.STLSQ(threshold=rt * coef_scale, alpha=ALPHA, normalize_columns=NORMALIZE)
            model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=DEGREE), optimizer=opt,
                             differentiation_method=diff)
            try:
                if augment:
                    model.fit(X_aug, t=t_aug, x_dot=xdot_aug, feature_names=var_names)
                else:
                    model.fit(Xn, t=t, x_dot=xdot, feature_names=var_names)
            except Exception as e:
                print(f"  {method} rt={rt} noise={noise}: fit failed ({e})")
                continue
            Xi = model.coefficients().T
            sym_resid = max(np.max(np.abs(M @ Xi - Xi @ g.T)) for g, M in zip(G, M_list)) / max(np.max(np.abs(Xi)), 1e-12)
            fit_rows.append(dict(system=name, noise=str(noise), method=method, rel_threshold=rt, **evaluate(Xi),
                                 sym_resid=sym_resid))
    print(f"  fits done ({time.time() - t0:.1f}s)")

sym_df = pd.DataFrame(sym_rows)
fit_df = pd.DataFrame(fit_rows)
suffix = "" if len(sys.argv) == 1 else "_" + "_".join(sys.argv[1:])
sym_df.to_csv(os.path.join(OUT_DIR, f"symmetries{suffix}.csv"), index=False)
fit_df.to_csv(os.path.join(OUT_DIR, f"fits{suffix}.csv"), index=False)

# %% [markdown]
# ## Summary

# %%
pd.set_option("display.width", 250, "display.max_rows", 500)
print("\n=== Systems with symmetries ===")
print(sym_df[sym_df.n_sym > 0].drop(columns="G").to_string(index=False))
if len(fit_df):
    best = (fit_df.sort_values("coef_rel_err").groupby(["system", "noise", "method"], as_index=False).first())
    print("\n=== Best-threshold coefficient error per system ===")
    print(best.pivot_table(index=["system", "noise"], columns="method", values="coef_rel_err").to_string(float_format="%.2e"))
    print("\n=== Best-threshold exact support recovery per system ===")
    print(best.pivot_table(index=["system", "noise"], columns="method", values="exact_support").to_string())
    print("\n=== Aggregate (median rel. coef err / frac exact support / max sym resid) ===")
    print(best.groupby(["noise", "method"]).agg(med_err=("coef_rel_err", "median"), exact=("exact_support", "mean"),
                                                 max_sym_resid=("sym_resid", "max")).to_string(float_format="%.2e"))
