# %% [markdown]
# # Data efficiency of weak-form SINDy and orbit thresholding (dysts, 42 symmetric polynomial systems)
#
# Same training windows, noise draws, evaluation and threshold grid as data_efficiency_dysts.py (seeds
# [crc32(name), k_noise, N, rep]), with the 2 x 3 grid of weak_orbit_dysts.py:
#   strong_{STLSQ, ConstrainedSTLSQ, OrbitSTLSQ}   finite-difference derivatives on the window (strong_STLSQ and
#                                                  strong_ConstrainedSTLSQ = STLSQ_scaled / ConstrainedSTLSQ_scaled
#                                                  of data_efficiency_dysts.py; checked row by row)
#   weak_{STLSQ, ConstrainedSTLSQ, OrbitSTLSQ}     integral form on the noisy window (half-width min(WEAK_M, (N-1)//4))
# Noise 0 / 1% / 5% keep their k_noise = 0 / 1 / 2; 10% is new (k_noise = 3).
#
# One Slurm array task per (system, noise): SLURM_ARRAY_TASK_ID indexes the 42 systems x NOISES; or
# python weak_orbit_data_eff.py <system> <noise>. Output: LOG_DIR/weak_orbit_data_eff/<tag>/fits_<system>_noise<n>.csv

# %%
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import pysindy as ps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dysts_common import load_system
from fit_metrics import Evaluator
from weak_orbit_dysts import DATA_DIR, LOG_DIR, SYSTEMS, TAG, fit_all, orbit_basis, weak_system

WINDOWS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 10000]
N_REPEATS = 5
NOISES = [0.0, 0.01, 0.05, 0.1]  # index = k_noise of data_efficiency_dysts.py
OUT = os.path.join(LOG_DIR, "weak_orbit_data_eff", TAG)
os.makedirs(OUT, exist_ok=True)


def run(name, k_noise):
    noise = NOISES[k_noise]
    t0 = time.time()
    sysd = load_system(name, DATA_DIR)
    X, dt, lib, Xi_true, C = sysd["X"], sysd["dt"], sysd["lib"], sysd["Xi_true"], sysd["C"]
    h, n = Xi_true.shape
    B = orbit_basis(sysd["G"], sysd["M_list"], h, n)
    assert np.max(np.abs(C @ B)) < 1e-10 and B.shape[1] == h * n - np.linalg.matrix_rank(C)
    std = X.std(axis=0)
    ev = Evaluator(X, Xi_true, sysd["powers"], dt, sysd["lyap"])
    ev.set_noisy(X + noise * std * np.random.default_rng([zlib.crc32(name.encode()), k_noise]).standard_normal(X.shape))
    diff = ps.FiniteDifference(order=4) if noise == 0.0 else ps.SmoothedFiniteDifference()
    rows = []
    for N in WINDOWS:
        for rep in range(N_REPEATS):
            rng = np.random.default_rng([zlib.crc32(name.encode()), k_noise, N, rep])
            start = int(rng.integers(0, len(X) - N + 1))
            Xw = X[start:start + N]
            Xn = Xw + noise * std * rng.standard_normal(Xw.shape)
            t = np.arange(N) * dt
            Y = np.asarray(diff(Xn, t))
            Phi = np.asarray(lib.transform(np.asarray(getattr(diff, "smoothed_x_", Xn)) if noise else Xn))
            systems = {"strong": (Phi, Y), "weak": weak_system(Xn, np.asarray(lib.transform(Xn)), dt)}
            base = dict(system=name, noise=noise, N=N, rep=rep, start=start)
            for form, (P, T) in systems.items():
                try:
                    fits = fit_all(P, T, C, B)
                except Exception as e:
                    print(f"  {form} N={N} rep={rep}: failed ({e})", flush=True)
                    continue
                for method, lst in fits.items():
                    for rt, Xi in lst:
                        rows.append(dict(base, form=form, method=f"{form}_{method}", rel_threshold=rt,
                                         n_eq=len(P), **ev(Xi, forecast=False)))
        print(f"{name} noise={noise} N={N} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"fits_{name}_noise{noise:g}.csv"), index=False)

    # pairing check against data_efficiency_dysts.py
    ref_path = os.path.join(LOG_DIR, "data_efficiency", f"fits_{name}.csv")
    alt = os.path.join(LOG_DIR, "data_efficiency", f"fits_{name}_noise{noise:g}.csv")
    if noise != 0.1 and (os.path.exists(ref_path) or os.path.exists(alt)):
        ref = pd.concat([pd.read_csv(p) for p in (ref_path, alt) if os.path.exists(p)])
        ref = ref[np.isclose(ref.noise, noise)]
        for mine, theirs in [("strong_STLSQ", "STLSQ_scaled"), ("strong_ConstrainedSTLSQ", "ConstrainedSTLSQ_scaled")]:
            key = ["N", "rep", "rel_threshold"]
            m = df[df.method == mine].merge(ref[ref.method == theirs], on=key, suffixes=("", "_ref"))
            bad = (~np.isclose(m.coef_rel_err, m.coef_rel_err_ref, rtol=1e-6, atol=1e-12)).sum()
            print(f"  check {mine} vs {theirs}: {len(m)} rows, {'OK' if bad == 0 else f'{bad} MISMATCHES'}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        run(sys.argv[1], NOISES.index(float(sys.argv[2])))
    else:
        i = int(os.environ["SLURM_ARRAY_TASK_ID"])
        run(SYSTEMS[i // len(NOISES)], i % len(NOISES))
