# %% [markdown]
# # Weak-form SINDy and orbit thresholding on the dysts benchmark
#
# Two changes aimed at the noise floor and at structure recovery, tested in a 2 x 3 grid against the
# column-scaled, alpha = 0 setup of symmetric_sindy_dysts.py (run alpha0_norm):
#
#   form   strong  regression of finite-difference derivatives on Theta(x) (as before; noise 0 uses 4th-order
#                  FiniteDifference, noisy data SmoothedFiniteDifference, exactly like the harness)
#          weak    integral form: for compactly supported test functions phi_k(t) = (1 - s^2)^p, s = (t - c_k)/(m dt),
#                  int phi_k xdot dt = -int phi_k' x dt, so b = -V' X (K, n) is regressed on G = V Theta(X) (K, h).
#                  No derivatives of the noisy data are taken. Quadrature = trapezoid (phi vanishes at the ends).
#   opt    STLSQ             no symmetry
#          ConstrainedSTLSQ  symmetry as equality constraints C vec(Xi) = 0, coefficient-wise thresholding
#          OrbitSTLSQ        symmetry built into the basis: vec(Xi) = B v, one column of B per orbit of the group
#                            acting on (feature, target) pairs, so v has one entry per orbit and thresholding acts
#                            on whole orbits (on the orbit's scaled contribution, with column scaling). The
#                            regression on (I (x) Theta) B is unconstrained.
#
# All column-scaled, alpha = 0, same 5-point relative threshold grid as alpha0_norm. Noise 0 / 1% / 5% reuse the
# exact draws of symmetric_sindy_dysts.py (its RNG(0) is replayed), so rows pair with alpha0_norm/fits.csv, and the
# strong STLSQ / ConstrainedSTLSQ rows are checked against it. 10% noise is new (own seed per system).
#
# One Slurm array task per (system, noise): SLURM_ARRAY_TASK_ID indexes systems x NOISES; or pass systems on argv.
# Output: LOG_DIR/weak_orbit_dysts/<tag>/fits_<system>_noise<noise>.csv

# %%
import json
import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd
import pysindy as ps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dysts.flows as flows
from dysts_common import load_system
from fit_metrics import Evaluator

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DYSTS_DATA_DIR", "/gscratch/stf/songk42/dysts_hf/fine")
LOG_DIR = os.environ.get("LOG_DIR", "/mmfs1/home/songk42/ehd-sindy/logs/dysts_traj")
WEAK_M = int(os.environ.get("WEAK_M", 25))  # test-function half-width in samples (dysts fine: ~100 samples/period)
WEAK_P = int(os.environ.get("WEAK_P", 8))  # test-function exponent
WEAK_STRIDE = int(os.environ.get("WEAK_STRIDE", 0))  # spacing of test-function centers; 0 = max(1, m // 4)
FORMS = os.environ.get("WO_FORMS", "strong,weak").split(",")
TAG = os.environ.get("WO_TAG", f"m{WEAK_M}_p{WEAK_P}")
REF_RUN = "alpha0_norm"
HARNESS_NOISES = ["exact_xdot", 0.0, 0.01, 0.05]  # order of RNG use in symmetric_sindy_dysts.py
NOISES = [0.0, 0.01, 0.05, 0.1]
REL_THRESHOLDS = [0.003, 0.01, 0.03, 0.1, 0.3]
DEGREE = 3
OUT = os.path.join(LOG_DIR, "weak_orbit_dysts", TAG)
os.makedirs(OUT, exist_ok=True)

SYSTEMS = sorted(pd.read_csv(os.path.join(LOG_DIR, "sindy_si_dysts", "side_info.csv")).system)  # the 42 systems


# %% [markdown]
# ## Pieces

# %%
def replay_noise(names):
    """Noisy trajectories of symmetric_sindy_dysts.py for `names`, replaying its RNG(0) in its exact order of use."""
    meta = json.load(open(os.path.join(os.path.dirname(flows.__file__), "data", "chaotic_attractors.json")))
    rng = np.random.default_rng(0)
    draws = {}
    for name in sorted(f.replace("_fine.csv", "") for f in os.listdir(DATA_DIR) if f.endswith("_fine.csv")):
        md = meta.get(name, {})
        if md.get("nonautonomous") or md.get("delay") or not hasattr(flows, name):
            continue
        X_raw = pd.read_csv(os.path.join(DATA_DIR, f"{name}_fine.csv")).iloc[:, 1:].to_numpy(float)
        if X_raw.shape[1] > 4 or not np.all(np.isfinite(X_raw)):
            continue
        rng.choice(len(X_raw), size=200, replace=False)  # detect_symmetries' sample points
        if name in SYSTEMS:
            X = X_raw / np.max(np.abs(X_raw))
            d = {nz: X + nz * X.std(axis=0) * rng.standard_normal(X.shape) for nz in HARNESS_NOISES if nz != "exact_xdot"}
            if name in names:
                draws[name] = d
    return draws


def orbit_basis(G, M_list, h, n):
    """B (h n, r), target order: one column per orbit of the symmetry group on (feature, target) pairs.

    The group acts on coefficient matrices by Xi -> M_g^T Xi g^T (M_g signed permutations of the monomials, with
    Theta(x g^T) = Theta(x) M_g); its fixed points are exactly the equivariant Xi (M_g Xi = Xi g^T). Averaging that
    action over the group maps each unit matrix E_jk to a signed indicator of its orbit, or to 0 when the orbit
    forces the coefficient to vanish (e.g. even terms in an odd-symmetric equation).
    """
    group = [(np.eye(n), np.eye(h))] + [(np.asarray(g, float), M) for g, M in zip(G, M_list)]
    cols, seen = [], set()
    for k in range(n):
        for j in range(h):
            if (j, k) in seen:
                continue
            P = np.zeros((h, n))
            for g, M in group:
                P += M[j][:, None] * g[:, k][None, :]  # M^T E_jk g^T
            P /= len(group)
            if np.max(np.abs(P)) < 1e-12:
                seen.add((j, k))
                continue
            P /= np.max(np.abs(P))
            P[np.abs(P) < 1e-12] = 0.0
            seen.update(zip(*np.nonzero(P)))
            cols.append(P.flatten(order="F"))
    return np.array(cols).T


def test_functions(N, dt, m=WEAK_M, p=WEAK_P, stride=WEAK_STRIDE):
    """phi and phi' on a window of 2m+1 samples (quadrature weight dt folded in), and the window starts.

    Short series shrink the half-width to (N - 1) // 4, so at least ~N/2 test functions fit."""
    m = max(min(m, (N - 1) // 4), 2)
    stride = stride or max(1, m // 4)
    s = np.linspace(-1, 1, 2 * m + 1)
    phi = (1 - s ** 2) ** p * dt
    dphi = -2 * p * s * (1 - s ** 2) ** (p - 1) / (m * dt) * dt
    starts = np.arange(0, N - 2 * m, stride)
    return phi, dphi, starts


def weak_system(Xn, Theta, dt):
    """G = V Theta (K, h) and b = -V' X (K, n)."""
    phi, dphi, starts = test_functions(len(Xn), dt)
    win = lambda A: np.lib.stride_tricks.sliding_window_view(A, len(phi), axis=0)[starts]  # (K, cols, L)
    return win(Theta) @ phi, -(win(Xn) @ dphi)


def strong_system(Xn, t, lib, noise):
    """Features and targets exactly as ps.SINDy.fit builds them in symmetric_sindy_dysts.py."""
    diff = ps.FiniteDifference(order=4) if noise == 0.0 else ps.SmoothedFiniteDifference()
    Yd = np.asarray(diff(Xn, t))
    Xs = np.asarray(getattr(diff, "smoothed_x_", Xn))
    return np.asarray(lib.transform(Xs)), Yd


def fit_all(Phi, Y, C, B):
    """{method: [(rel_threshold, Xi)]} for STLSQ / ConstrainedSTLSQ / OrbitSTLSQ, column-scaled, alpha 0."""
    h, n = Phi.shape[1], Y.shape[1]
    norms = np.linalg.norm(Phi, axis=0)
    dense = np.linalg.lstsq(Phi, Y, rcond=None)[0]
    scale = np.max(np.abs(dense * norms[:, None]))
    A = np.kron(np.eye(n), Phi) @ B  # (K n, r)
    y = Y.flatten(order="F")[:, None]
    v_dense = np.linalg.lstsq(A, y, rcond=None)[0][:, 0]
    scale_orbit = np.max(np.abs(v_dense * np.linalg.norm(A, axis=0)))
    out = {"STLSQ": [], "ConstrainedSTLSQ": [], "OrbitSTLSQ": []}
    for rt in REL_THRESHOLDS:
        opt = ps.STLSQ(threshold=rt * scale, alpha=0.0, normalize_columns=True).fit(Phi, Y)
        out["STLSQ"].append((rt, np.asarray(opt.coef_).T))
        opt = ps.ConstrainedSTLSQ(threshold=rt * scale, alpha=0.0, constraint_lhs=C, constraint_rhs=np.zeros(len(C)),
                                  constraint_order="target", normalize_columns=True).fit(Phi, Y)
        out["ConstrainedSTLSQ"].append((rt, np.asarray(opt.coef_).T))
        opt = ps.STLSQ(threshold=rt * scale_orbit, alpha=0.0, normalize_columns=True).fit(A, y)
        out["OrbitSTLSQ"].append((rt, (B @ np.asarray(opt.coef_)[0]).reshape(h, n, order="F")))
    return out


# %% [markdown]
# ## Main

# %%
if __name__ == "__main__":
    ref = pd.read_csv(os.path.join(LOG_DIR, "symmetric_sindy_dysts", REF_RUN, "fits.csv"))
    ref["noise"] = ref.noise.astype(str)
    if len(sys.argv) > 1:
        tasks = [(s, nz) for s in sys.argv[1:] for nz in NOISES]
    else:
        i = int(os.environ["SLURM_ARRAY_TASK_ID"])
        tasks = [(SYSTEMS[i // len(NOISES)], NOISES[i % len(NOISES)])]
    draws = replay_noise({s for s, _ in tasks})
    print(f"weak: m={WEAK_M} p={WEAK_P} stride={WEAK_STRIDE}; forms={FORMS}; tasks={tasks}", flush=True)

    for name, noise in tasks:
        t0 = time.time()
        sysd = load_system(name, DATA_DIR)
        X, n, dt, lib, Xi_true = sysd["X"], sysd["n"], sysd["dt"], sysd["lib"], sysd["Xi_true"]
        h = Xi_true.shape[0]
        C, G, M_list = sysd["C"], sysd["G"], sysd["M_list"]
        B = orbit_basis(G, M_list, h, n)
        assert np.max(np.abs(C @ B)) < 1e-10 and B.shape[1] == h * n - np.linalg.matrix_rank(C), "orbit basis != null(C)"
        vec_true = Xi_true.flatten(order="F")
        true_in_span = np.linalg.norm(B @ np.linalg.lstsq(B, vec_true, rcond=None)[0] - vec_true) / np.linalg.norm(vec_true)

        if noise in (0.0, 0.01, 0.05):
            Xn = draws[name][noise]
        else:
            Xn = X + noise * X.std(axis=0) * np.random.default_rng([zlib.crc32(name.encode()), int(round(noise * 1000))]
                                                                   ).standard_normal(X.shape)
        t = np.arange(len(X)) * dt
        ev = Evaluator(X, Xi_true, sysd["powers"], dt, sysd["lyap"])
        ev.set_noisy(Xn)

        rows = []
        for form in FORMS:
            if form == "strong":
                Phi, Y = strong_system(Xn, t, lib, noise)
            else:
                Phi, Y = weak_system(Xn, np.asarray(lib.transform(Xn)), dt)
            for method, fits in fit_all(Phi, Y, C, B).items():
                for rt, Xi in fits:
                    sym_resid = (max(np.max(np.abs(M @ Xi - Xi @ g.T)) for g, M in zip(G, M_list))
                                 / max(np.max(np.abs(Xi)), 1e-12))
                    rows.append(dict(system=name, noise=str(noise), form=form, method=f"{form}_{method}",
                                     rel_threshold=rt, n_orbits=B.shape[1], n_test_fns=len(Phi) if form == "weak" else 0,
                                     sym_resid=sym_resid, **ev(Xi)))
        df = pd.DataFrame(rows)

        # pairing check: strong STLSQ / ConstrainedSTLSQ must reproduce alpha0_norm (same data, same fit)
        if "strong" in FORMS and noise in (0.0, 0.01, 0.05):
            for mine, theirs in [("strong_STLSQ", "STLSQ"), ("strong_ConstrainedSTLSQ", "ConstrainedSTLSQ_noaug")]:
                a = df[df.method == mine].set_index("rel_threshold").coef_rel_err
                b = ref[(ref.system == name) & (ref.noise == str(noise)) & (ref.method == theirs)].set_index(
                    "rel_threshold").coef_rel_err
                bad = (~np.isclose(a, b.reindex(a.index), rtol=1e-6, atol=1e-12)).sum()
                print(f"  check {mine} vs {REF_RUN}/{theirs}: {'OK' if bad == 0 else f'{bad} MISMATCHES'}")
        df.to_csv(os.path.join(OUT, f"fits_{name}_noise{noise}.csv"), index=False)
        best = df.sort_values("coef_rel_err").groupby("method").first()
        print(f"{name} noise={noise}: |orbits|={B.shape[1]} true-in-span={true_in_span:.1e} ({time.time() - t0:.0f}s)\n"
              + best[["coef_rel_err", "exact_support_rel", "vf_rel_err", "traj_nrmse_clean"]].to_string(
                  float_format="%.3g"), flush=True)
