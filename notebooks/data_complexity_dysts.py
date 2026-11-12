# %% [markdown]
# # Data needed vs system size and complexity (dysts data-efficiency sweep)
#
# Joins each method's N_min (as in summarize_data_efficiency.py: smallest N from which >= 4 of 5 repeats succeed at
# every larger N; thresholds picked by coefficient error) with per-system descriptors:
#   dim, n_features (degree-3 library: 20 in 3-D, 35 in 4-D), n_unknowns = dim * n_features, n_true_terms,
#   samples per Lyapunov time (dysts_dt.csv).
# Reports median N_min by dimension, N_min per library feature (N / n_features = rows per unknown of each
# per-equation regression), and Spearman correlations of N_min with each descriptor.
#
# Env: LOG_DIR (default logs), DE_DIRS (comma-separated subdirs of LOG_DIR pooled), DT_CSV (default dysts_dt.csv).

# %%
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LOG_DIR = os.environ.get("LOG_DIR", "/home/songk/sindy-symmetry/logs")
DE_DIRS = os.environ.get("DE_DIRS", "data_efficiency").split(",")
DT_CSV = os.environ.get("DT_CSV", "/home/songk/sindy-symmetry/dysts_dt.csv")
METHODS = ["LSQ_dense", "ConstrainedSR3_v1", "STLSQ_scaled", "ConstrainedSTLSQ_scaled", "SINDySI_full"]
CRITERIA = ["accuracy", "dynamics", "trajectory"]
RELIABLE_FRAC = 0.8

cols = ["system", "noise", "N", "rep", "method", "coef_rel_err", "vf_rel_err", "traj_nrmse_clean", "n_true_terms"]
files = sorted(f for d in DE_DIRS for f in glob.glob(os.path.join(LOG_DIR, d, "fits_*.csv")))
df = pd.concat([pd.read_csv(f, usecols=cols) for f in files])
df = df[df.method.isin(METHODS)]
best = df.sort_values("coef_rel_err").groupby(["system", "noise", "N", "rep", "method"], as_index=False).first()
best["accuracy"] = best.coef_rel_err < 0.1
best["dynamics"] = best.vf_rel_err < 0.05
best["trajectory"] = best.traj_nrmse_clean < 0.1
WINDOWS = sorted(best.N.unique())
NOISES = sorted(best.noise.unique())

desc = pd.read_csv(DT_CSV).set_index("system")
desc["n_features"] = desc.dim.map({3: 20, 4: 35})
desc["n_unknowns"] = desc.dim * desc.n_features
desc["n_true_terms"] = best.groupby("system").n_true_terms.first()
desc["sparsity"] = desc.n_true_terms / desc.n_unknowns


def n_min(rate):
    ok = (rate.reindex(WINDOWS).fillna(0) >= RELIABLE_FRAC).to_numpy()
    for i in range(len(WINDOWS)):
        if ok[i:].all():
            return WINDOWS[i]
    return np.inf


pd.set_option("display.width", 250, "display.max_rows", 500)
fmt = lambda v: ">max" if not np.isfinite(v) else f"{v:g}"
methods = [m for m in METHODS if m in best.method.unique()]
print(f"{best.system.nunique()} systems; methods {methods}\n")
print("-- descriptors by dimension")
print(desc.groupby("dim").agg(n=("dt", "size"), n_features=("n_features", "first"), n_unknowns=("n_unknowns", "first"),
                              true_terms_median=("n_true_terms", "median"), sparsity_median=("sparsity", "median"),
                              samples_per_lyap_median=("samples_per_lyap_time", "median")).to_string(float_format="%.3g"))

allnm = {}
for crit in CRITERIA:
    rate = best.groupby(["system", "noise", "method", "N"])[crit].mean()
    nm = rate.groupby(["system", "noise", "method"]).apply(lambda r: n_min(r.droplevel([0, 1, 2]))).unstack("method")
    allnm[crit] = nm
    print(f"\n################ criterion: {crit}")
    for nz in NOISES:
        x = nm.xs(nz, level="noise")[methods].join(desc)
        print(f"\nnoise {nz:g}: reliable systems / median N_min / median N_min per library feature, by dimension")
        rows = {}
        for m in methods:
            rows[m] = {}
            for d, g in x.groupby("dim"):
                v = g[m]
                rows[m][f"{d}-D (n={len(g)})"] = (f"{np.isfinite(v).sum():2d} / {fmt(v.median())} / "
                                                  f"{fmt((v / g.n_features).median()) if np.isfinite(v.median()) else '>max'}")
        print(pd.DataFrame(rows).T.to_string())
        print("Spearman rho of N_min (inf ranked last) with descriptors [all systems]")
        rows = {}
        for m in methods:
            v = x[m].replace(np.inf, 1e9)
            rows[m] = {k: f"{spearmanr(v, x[k]).statistic:+.2f}" for k in
                       ["dim", "n_true_terms", "sparsity", "samples_per_lyap_time"]}
            # within-dimension correlation with the number of true terms (removes the library-size effect)
            for d, g in x.groupby("dim"):
                rows[m][f"n_true_terms|{d}-D"] = f"{spearmanr(g[m].replace(np.inf, 1e9), g.n_true_terms).statistic:+.2f}"
        print(pd.DataFrame(rows).T.to_string())

print("\n################ per-system N_min, criterion = accuracy, noise 1% (with descriptors)")
x = allnm["accuracy"].xs(0.01, level="noise")[methods].map(fmt)
print(desc[["dim", "n_features", "n_true_terms", "samples_per_lyap_time"]].join(x)
      .sort_values(["dim", "n_true_terms"]).to_string(float_format="%.0f"))
print("\n################ per-system N_min, criterion = dynamics, noise 1% (with descriptors)")
x = allnm["dynamics"].xs(0.01, level="noise")[methods].map(fmt)
print(desc[["dim", "n_true_terms"]].join(x).sort_values(["dim", "n_true_terms"]).to_string())
