"""Shared dysts benchmark setup: system loading, signed-permutation symmetries and the v1 constraint matrix.

Same constructions as symmetric_sindy_dysts.py / sindy_si_dysts.py (which keep their own copies so the past runs
stay reproducible as-is). Used by data_efficiency_dysts.py.
"""
import ast
import itertools
import json
import os

import numpy as np
import pandas as pd
import pysindy as ps
import scipy.linalg
import sympy as sp

import dysts.flows as flows

DEGREE = 3
META = json.load(open(os.path.join(os.path.dirname(flows.__file__), "data", "chaotic_attractors.json")))


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


def detect_symmetries(system, X, rng, rtol=1e-8):
    """All non-identity signed permutations g with f(x g^T) = f(x) g^T on attractor points."""
    n = X.shape[1]
    pts = X[rng.choice(len(X), size=200, replace=False)]
    F = eval_rhs(system, pts)
    scale = np.max(np.abs(F)) + 1e-12
    G = []
    for g in signed_permutations(n):
        if np.array_equal(g, np.eye(n, dtype=int)):
            continue
        if np.max(np.abs(eval_rhs(system, pts @ g.T) - F @ g.T)) / scale < rtol:
            G.append(g)
    return G


def _build_Mk(library, subs_rule, vars_, monomial_index):
    M = sp.zeros(len(library), len(library))
    for j, theta_j in enumerate(library):
        poly = sp.Poly(sp.expand(theta_j.subs(subs_rule, simultaneous=True)), *vars_)
        for exps, coeff in poly.terms():
            M[monomial_index[exps], j] += coeff
    return M


def _exponent_tuple(monom, vars_):
    poly = sp.Poly(monom, *vars_)
    return tuple([0] * len(vars_)) if poly.is_zero else poly.monoms()[0]


def symmetry_constraints(G, feature_names, var_names):
    """M_g with Theta(x g^T) = Theta(x) M_g, and the QR-pruned C with C vec(Xi) = 0 (target order)."""
    syms = sp.symbols(var_names)
    full_library = [sp.sympify(" * ".join(f.split(" "))) for f in feature_names]
    monomial_index = {_exponent_tuple(m, syms): idx for idx, m in enumerate(full_library)}
    M_list, C_list = [], []
    for g in G:
        subs_rule = {a: sum(int(g[i, j]) * syms[j] for j in range(len(syms))) for i, a in enumerate(syms)}
        M = np.array(_build_Mk(full_library, subs_rule, syms, monomial_index), dtype=float)
        M_list.append(M)
        C_list.append(np.kron(np.eye(g.shape[0]), M) - np.kron(g, np.eye(M.shape[0])))
    A = np.vstack(C_list)
    rank = np.linalg.matrix_rank(A)
    _, _, piv = scipy.linalg.qr(A.T, pivoting=True)
    C = A[piv[:rank]]
    C = C[~np.isclose(C, 0).all(axis=1)]
    return M_list, C


def load_system(name, data_dir, side_info=None):
    """Everything about one system that doesn't depend on the training window or noise.

    Returns a dict with the scaled clean trajectory X, time step dt, true coefficients Xi_true, library exponents,
    symmetry group G with M_list and C, and (if a side_info row from sindy_si_dysts is given) equilibria, monotone
    sign constraints and the ball used for them.
    """
    system = getattr(flows, name)()
    X_raw = pd.read_csv(os.path.join(data_dir, f"{name}_fine.csv")).iloc[:, 1:].to_numpy(float)
    n = X_raw.shape[1]
    G = detect_symmetries(system, X_raw, np.random.default_rng(0))
    F_raw = eval_rhs(system, X_raw)
    dXdn = np.gradient(X_raw, axis=0)
    dt = np.sum(dXdn * F_raw) / np.sum(F_raw * F_raw)
    s = np.max(np.abs(X_raw))
    X = X_raw / s
    var_names = [f"x{i}" for i in range(n)]
    lib = ps.PolynomialLibrary(degree=DEGREE, include_bias=True)
    lib.fit(X)
    feature_names = lib.get_feature_names(var_names)
    Theta = np.asarray(lib.transform(X))
    Xi_true, *_ = np.linalg.lstsq(Theta, F_raw / s, rcond=None)
    Xi_true[np.abs(Xi_true) < 1e-8 * np.max(np.abs(Xi_true))] = 0.0
    M_list, C = symmetry_constraints(G, feature_names, var_names) if G else ([], np.zeros((0, Xi_true.size)))
    out = dict(name=name, X=X, dt=dt, n=n, lib=lib, powers=np.asarray(lib.powers_), Xi_true=Xi_true, G=G,
               M_list=M_list, C=C, lyap=META.get(name, {}).get("maximum_lyapunov_estimated"))
    if side_info is not None:
        out["equilibria"] = [np.array(x) for x in ast.literal_eval(side_info["equilibria"])]
        out["monotone"] = [tuple(m) for m in ast.literal_eval(side_info["monotone"])]
        out["ball"] = (np.array(ast.literal_eval(side_info["ball_center"])), float(side_info["ball_radius"]))
    return out
