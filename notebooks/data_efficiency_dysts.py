# %% [markdown]
# # How much data does each method need for a reliable fit? (dysts, 42 symmetric polynomial systems)
#
# For each system: contiguous training windows of N samples (random start), 5 repeats per N with fresh noise,
# noise 0% / 1% / 5% of each channel's std (std of the full clean trajectory). Derivatives, scaling, library, true
# coefficients and side information are exactly as in symmetric_sindy_dysts.py / sindy_si_dysts.py; the library is
# evaluated on the smoothed window, as ps.SINDy.fit does.
#
# Methods (all thresholds on the same 5-point grid, relative to the largest coefficient of the window's dense fit,
# raw or column-scaled to match the method):
#   STLSQ_a0.05               old default (ridge 0.05 + unbias), raw library
#   ConstrainedSTLSQ_a0.05    same + symmetry constraints (no augmentation)
#   STLSQ_scaled              column scaling, alpha = 0
#   ConstrainedSTLSQ_scaled   column scaling, alpha = 0, symmetry constraints (recommended default)
#   ConstrainedSR3_l0_scaled  ConstrainedSR3 l0, nu = 1, column scaling, constrained unbias refit
#   ConstrainedSR3_v1         v1: dense constrained least squares (reg_weight_lam = 0), no threshold
#   SINDySI_full              SINDy-SI with symmetry + equilibria + monotonicity (SOS)
#
# Usage: python data_efficiency_dysts.py <system> [<system> ...]   or   SLURM_ARRAY_TASK_ID=<i> (index into the
# 42 systems of logs/sindy_si_dysts/side_info.csv). One CSV per system in logs/data_efficiency/.

# %%
import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd
import pysindy as ps

from dysts_common import META, load_system
from fit_metrics import Evaluator
from sindy_si import SINDySI

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DYSTS_DATA_DIR", "/home/songk/data-songk/dysts_hf/fine")
LOG_DIR = os.environ.get("LOG_DIR", "/home/songk/sindy-symmetry/logs")
OUT_DIR = os.path.join(LOG_DIR, "data_efficiency")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOWS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 10000]  # samples (~100 per period)
N_REPEATS = 5
NOISE_LEVELS = [0.0, 0.01, 0.05]
REL_THRESHOLDS = [0.003, 0.01, 0.03, 0.1, 0.3]
ALPHA_OLD = 0.05
SR3_NU, SR3_MAX_ITER = 1.0, 500
METHODS = ["STLSQ_a0.05", "ConstrainedSTLSQ_a0.05", "STLSQ_scaled", "ConstrainedSTLSQ_scaled",
           "ConstrainedSR3_l0_scaled", "ConstrainedSR3_v1", "SINDySI_full"]


def fit_method(method, rt, Phi, Y, sysd, raw_scale, scaled_scale):
    C = sysd["C"]
    d = np.zeros(C.shape[0])
    cons = dict(constraint_lhs=C, constraint_rhs=d, constraint_order="target")
    if method == "STLSQ_a0.05":
        opt = ps.STLSQ(threshold=rt * raw_scale, alpha=ALPHA_OLD)
    elif method == "ConstrainedSTLSQ_a0.05":
        opt = ps.ConstrainedSTLSQ(threshold=rt * raw_scale, alpha=ALPHA_OLD, **cons)
    elif method == "STLSQ_scaled":
        opt = ps.STLSQ(threshold=rt * scaled_scale, alpha=0.0, normalize_columns=True)
    elif method == "ConstrainedSTLSQ_scaled":
        opt = ps.ConstrainedSTLSQ(threshold=rt * scaled_scale, alpha=0.0, normalize_columns=True, **cons)
    elif method == "ConstrainedSR3_l0_scaled":
        tau = rt * scaled_scale
        opt = ps.ConstrainedSR3(reg_weight_lam=tau ** 2 / 2 / SR3_NU, regularizer="l0", relax_coeff_nu=SR3_NU,
                                max_iter=SR3_MAX_ITER, unbias=True, normalize_columns=True,
                                equality_constraints=True, **cons)
    elif method == "ConstrainedSR3_v1":
        opt = ps.ConstrainedSR3(reg_weight_lam=0, equality_constraints=True, **cons)
    elif method == "SINDySI_full":
        return SINDySI(threshold=rt * scaled_scale, eq_lhs=C, eq_rhs=d, equilibria=sysd["equilibria"],
                       monotone=sysd["monotone"], ball=sysd["ball"]).fit(Phi, Y, sysd["powers"]).coef_
    else:
        raise ValueError(method)
    return np.asarray(opt.fit(Phi, Y).coef_).T  # (h, n)


def run_system(name, side_row):
    t0 = time.time()
    sysd = load_system(name, DATA_DIR, side_row)
    X, dt, lib = sysd["X"], sysd["dt"], sysd["lib"]
    evaluate = Evaluator(X, sysd["Xi_true"], sysd["powers"], dt, sysd["lyap"])
    std = X.std(axis=0)
    period = META.get(name, {}).get("period")
    rows = []
    for k_noise, noise in enumerate(NOISE_LEVELS):
        diff = ps.FiniteDifference(order=4) if noise == 0.0 else ps.SmoothedFiniteDifference()
        for N in WINDOWS:
            for rep in range(N_REPEATS):
                rng = np.random.default_rng([zlib.crc32(name.encode()), k_noise, N, rep])
                start = int(rng.integers(0, len(X) - N + 1))
                Xw = X[start:start + N]
                Xn = Xw + noise * std * rng.standard_normal(Xw.shape)
                t = np.arange(N) * dt
                Y = np.asarray(diff(Xn, t))
                Phi = np.asarray(lib.transform(np.asarray(getattr(diff, "smoothed_x_", Xn)) if noise else Xn))
                W_dense, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
                raw_scale = np.max(np.abs(W_dense))
                scaled_scale = np.max(np.abs(W_dense * np.linalg.norm(Phi, axis=0)[:, None]))
                base = dict(system=name, noise=noise, N=N, rep=rep, start=start, n_periods=N * dt / period if period else np.nan,
                            n_lyap=N * dt * sysd["lyap"] if sysd["lyap"] else np.nan)
                for method in METHODS:
                    for rt in ([None] if method == "ConstrainedSR3_v1" else REL_THRESHOLDS):
                        tic = time.time()
                        try:
                            Xi = fit_method(method, rt, Phi, Y, sysd, raw_scale, scaled_scale)
                        except Exception as e:
                            print(f"  {method} N={N} rep={rep} noise={noise} rt={rt}: failed ({e})", flush=True)
                            continue
                        rows.append(dict(base, method=method, rel_threshold=rt, **evaluate(Xi, forecast=False),
                                         fit_time=time.time() - tic))
            print(f"{name} noise={noise} N={N} done ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"fits_{name}.csv"), index=False)
    return rows


# %%
if __name__ == "__main__":
    side = pd.read_csv(os.path.join(LOG_DIR, "sindy_si_dysts", "side_info.csv")).set_index("system")
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = [sorted(side.index)[int(os.environ["SLURM_ARRAY_TASK_ID"])]]
    for name in names:
        run_system(name, side.loc[name])
