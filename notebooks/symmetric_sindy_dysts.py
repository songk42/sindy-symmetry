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

warnings.filterwarnings("ignore")

DATA_DIR = "/gscratch/stf/songk42/dysts_hf/fine"
ALPHA = float(os.environ.get("SINDY_ALPHA", 0.05))  # ridge for STLSQ variants; both refit with alpha=0 on the final support (unbias)
OUT_DIR = f"/mmfs1/home/songk42/ehd-sindy/logs/symmetric_sindy_dysts/alpha{ALPHA:g}"
os.makedirs(OUT_DIR, exist_ok=True)

DEGREE = 3
NOISE_LEVELS = ["exact_xdot", 0.0, 0.01, 0.05]  # fraction of per-channel std; "exact_xdot" = clean x with true RHS as x_dot
REL_THRESHOLDS = [0.003, 0.01, 0.03, 0.1]  # STLSQ thresholds, relative to max|Xi| of a dense lstsq fit
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


def support_metrics(Xi, Xi_true, tol=1e-10):
    est, true = np.abs(Xi) > tol, np.abs(Xi_true) > tol
    tp = np.sum(est & true)
    prec = tp / max(est.sum(), 1)
    rec = tp / max(true.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return f1, bool(np.array_equal(est, true))


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

        configs = [("ConstrainedSR3_v1", None, True), ("ConstrainedSR3_v1_noaug", None, False)]
        for rt in REL_THRESHOLDS:
            configs += [("STLSQ", rt, False), ("STLSQ_aug", rt, True), ("ConstrainedSTLSQ", rt, True),
                        ("ConstrainedSTLSQ_noaug", rt, False)]

        for method, rt, augment in configs:
            if method.startswith("ConstrainedSR3_v1"):
                opt = ps.ConstrainedSR3(constraint_lhs=C, constraint_rhs=d, constraint_order="target",
                                        equality_constraints=True, reg_weight_lam=0)
            elif method.startswith("ConstrainedSTLSQ"):
                opt = ps.ConstrainedSTLSQ(threshold=rt * coef_scale, alpha=ALPHA, constraint_lhs=C,
                                          constraint_rhs=d, constraint_order="target")
            else:
                opt = ps.STLSQ(threshold=rt * coef_scale, alpha=ALPHA)
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
            f1, exact = support_metrics(Xi, Xi_true)
            sym_resid = max(np.max(np.abs(M @ Xi - Xi @ g.T)) for g, M in zip(G, M_list)) / max(np.max(np.abs(Xi)), 1e-12)
            fit_rows.append(dict(system=name, noise=str(noise), method=method, rel_threshold=rt,
                                 coef_rel_err=np.linalg.norm(Xi - Xi_true) / np.linalg.norm(Xi_true),
                                 support_f1=f1, exact_support=exact, n_terms=int(np.sum(np.abs(Xi) > 1e-10)),
                                 n_true_terms=int(np.sum(Xi_true != 0)), sym_resid=sym_resid))
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
