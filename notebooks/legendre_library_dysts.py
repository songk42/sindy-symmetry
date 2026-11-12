# %% [markdown]
# # Monomial vs tensor-product Legendre library (degree <= 3) on the dysts systems
#
# The state is already scaled by max|X| into [-1, 1]^n (uniform scale, so the signed-permutation symmetries survive);
# the Legendre library uses features prod_i P_{a_i}(x_i) with the same exponent tuples a as the monomial library.
# Both libraries span the same polynomials: Theta_leg = Theta_mono @ T, so a model Xi_leg maps to Xi_mono = T @ Xi_leg.
# P_k has the parity of x^k, so a signed permutation maps Legendre features to +- Legendre features exactly as it maps
# monomials, and the symmetry constraint matrix C is reused unchanged (checked below).
#
# Per system: (a) condition number of the column-normalized library on the clean attractor, both bases;
# (b) number of nonzero true coefficients in each basis;
# (c) paired fits on the same windows / noise draws as data_efficiency_dysts.py (same RNG seeds), N in N_LIST,
#     methods STLSQ_scaled / ConstrainedSTLSQ_scaled (thresholds on the same grid, relative to the window's
#     column-scaled dense fit in that basis) and dense least squares, each in both bases. Errors are measured on the
#     monomial-basis coefficients (T @ Xi_leg) so they are comparable: coefficient error and vector-field error on
#     the clean attractor. Thresholds picked per repeat by coefficient error (oracle), as elsewhere.
#
# Usage: python legendre_library_dysts.py <system> ...  or SLURM_ARRAY_TASK_ID=<i>. Output: LOG_DIR/legendre/.

# %%
import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd
import pysindy as ps
from numpy.polynomial import legendre as L

from dysts_common import load_system

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DYSTS_DATA_DIR", "/home/songk/data-songk/dysts_hf/fine")
LOG_DIR = os.environ.get("LOG_DIR", "/home/songk/sindy-symmetry/logs")
OUT_DIR = os.path.join(LOG_DIR, "legendre")
os.makedirs(OUT_DIR, exist_ok=True)
NOISE_LEVELS = [0.0, 0.01, 0.05]  # same order as data_efficiency_dysts.py (k_noise enters the seeds)
N_LIST = [200, 800, 10000]
N_REPEATS = 5
REL_THRESHOLDS = [0.003, 0.01, 0.03, 0.1, 0.3]


def legendre_features(X, powers):
    out = np.ones((X.shape[0], len(powers)))
    for k, a in enumerate(powers):
        for i, ai in enumerate(a):
            if ai:
                out[:, k] *= L.legval(X[:, i], np.eye(ai + 1)[ai])
    return out


def cond(P):
    P = P / np.linalg.norm(P, axis=0)
    s = np.linalg.svd(P, compute_uv=False)
    return s[0] / s[-1]


def fit(method, rt, Phi, Y, C, scaled_scale):
    cons = dict(constraint_lhs=C, constraint_rhs=np.zeros(C.shape[0]), constraint_order="target")
    if method == "LSQ_dense":
        return np.linalg.lstsq(Phi, Y, rcond=None)[0]
    if method == "STLSQ_scaled":
        opt = ps.STLSQ(threshold=rt * scaled_scale, alpha=0.0, normalize_columns=True)
    elif method == "ConstrainedSTLSQ_scaled":
        opt = ps.ConstrainedSTLSQ(threshold=rt * scaled_scale, alpha=0.0, normalize_columns=True, **cons)
    return np.asarray(opt.fit(Phi, Y).coef_).T


def run_system(name):
    t0 = time.time()
    s = load_system(name, DATA_DIR)
    X, dt, lib, powers, Xi_true, C = s["X"], s["dt"], s["lib"], s["powers"], s["Xi_true"], s["C"]
    Th_m = np.asarray(lib.transform(X))
    Th_l = legendre_features(X, powers)
    T = np.linalg.lstsq(Th_m[:2000], Th_l[:2000], rcond=None)[0]  # exact change of basis (same span)
    T[np.abs(T) < 1e-9] = 0.0
    T = np.round(T, 12)
    Tinv = np.linalg.inv(T)
    Xi_true_l = Tinv @ Xi_true
    Xi_true_l[np.abs(Xi_true_l) < 1e-8 * np.abs(Xi_true_l).max()] = 0.0
    # checks: change of basis exact; C is satisfied by the true model in both bases (same vectorization as pysindy)
    basis_err = np.abs(Th_m @ T - Th_l).max()
    c_m = np.abs(C @ Xi_true.T.ravel()).max() if C.size else 0.0
    c_l = np.abs(C @ Xi_true_l.T.ravel()).max() if C.size else 0.0
    info = dict(system=name, dim=X.shape[1], cond_mono=cond(Th_m), cond_leg=cond(Th_l),
                nnz_mono=int((Xi_true != 0).sum()), nnz_leg=int((Xi_true_l != 0).sum()),
                basis_err=basis_err, C_resid_mono=c_m, C_resid_leg=c_l)
    print(info, flush=True)
    norm_true = np.linalg.norm(Xi_true)
    F_true = Th_m @ Xi_true
    rows = []
    std = X.std(axis=0)
    for k_noise, noise in enumerate(NOISE_LEVELS):
        diff = ps.FiniteDifference(order=4) if noise == 0.0 else ps.SmoothedFiniteDifference()
        for N in N_LIST:
            for rep in range(N_REPEATS):
                rng = np.random.default_rng([zlib.crc32(name.encode()), k_noise, N, rep])  # as data_efficiency_dysts
                start = int(rng.integers(0, len(X) - N + 1))
                Xn = X[start:start + N] + noise * std * rng.standard_normal((N, X.shape[1]))
                Y = np.asarray(diff(Xn, np.arange(N) * dt))
                Xs = np.asarray(getattr(diff, "smoothed_x_", Xn)) if noise else Xn
                for basis, Phi, back in [("mono", np.asarray(lib.transform(Xs)), None),
                                         ("legendre", legendre_features(Xs, powers), T)]:
                    W = np.linalg.lstsq(Phi, Y, rcond=None)[0]
                    scaled_scale = np.max(np.abs(W * np.linalg.norm(Phi, axis=0)[:, None]))
                    for method in ["LSQ_dense", "STLSQ_scaled", "ConstrainedSTLSQ_scaled"]:
                        for rt in ([None] if method == "LSQ_dense" else REL_THRESHOLDS):
                            try:
                                Xi = fit(method, rt, Phi, Y, C, scaled_scale)
                            except Exception as e:
                                print(f"  {basis} {method} N={N} rep={rep} noise={noise} rt={rt}: failed ({e})")
                                continue
                            Xi_m = Xi if back is None else back @ Xi
                            rows.append(dict(system=name, noise=noise, N=N, rep=rep, start=start, basis=basis,
                                             method=method, rel_threshold=rt,
                                             coef_rel_err=np.linalg.norm(Xi_m - Xi_true) / norm_true,
                                             vf_rel_err=np.linalg.norm(Th_m @ Xi_m - F_true) / np.linalg.norm(F_true),
                                             n_terms=int((np.abs(Xi) > 0).sum())))
        print(f"{name} noise={noise} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"fits_{name}.csv"), index=False)
    pd.DataFrame([info]).to_csv(os.path.join(OUT_DIR, f"info_{name}.csv"), index=False)


# %%
if __name__ == "__main__":
    side = pd.read_csv(os.path.join(LOG_DIR, "sindy_si_dysts", "side_info.csv")).set_index("system")
    names = sys.argv[1:] or [sorted(side.index)[int(os.environ["SLURM_ARRAY_TASK_ID"])]]
    for name in names:
        run_system(name)
