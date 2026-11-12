# %% [markdown]
# # How much do the data-efficiency results vary per system? (dysts sweep)
#
# Two kinds of variation, thresholds picked by coefficient error (oracle) as in summarize_data_efficiency.py:
#   across systems  spread of each system's median error (over its 5 repeats) and of N_min over the 42 systems
#   within system   spread over the 5 repeats (window start + noise draw) of one system
# 1. Variance decomposition of log10 error at fixed (method, noise, N): share of the variance that is between systems
#    vs between repeats of the same system.
# 2. Quantiles over systems of the per-system median error, and the typical within-system spread (max/min over
#    repeats).
# 3. Noise robustness per system: error at 5% noise / error at 0% noise (N = 10000), quantiles over systems.
# 4. N_min: quantiles over systems, and its within-system uncertainty from bootstrapping the 5 repeats.
# 5. Paired effect of the symmetry per system: N_min ratio quantiles.
#
# Env: LOG_DIR, DE_DIRS (as in summarize_data_efficiency.py).

# %%
import glob
import os

import numpy as np
import pandas as pd

LOG_DIR = os.environ.get("LOG_DIR", "/home/songk/sindy-symmetry/logs")
DE_DIRS = os.environ.get("DE_DIRS", "data_efficiency").split(",")
METHODS = ["LSQ_dense", "ConstrainedSR3_v1", "STLSQ_scaled", "ConstrainedSTLSQ_scaled", "SINDySI_full"]
METRICS = {"coef_rel_err": "accuracy", "vf_rel_err": "dynamics", "traj_nrmse_clean": "trajectory"}
CUT = {"accuracy": ("coef_rel_err", 0.1), "dynamics": ("vf_rel_err", 0.05), "trajectory": ("traj_nrmse_clean", 0.1)}
N_SHOW = [200, 800, 10000]
N_BOOT = 200
RELIABLE_FRAC = 0.8

cols = ["system", "noise", "N", "rep", "method"] + list(METRICS)
files = sorted(f for d in DE_DIRS for f in glob.glob(os.path.join(LOG_DIR, d, "fits_*.csv")))
df = pd.concat([pd.read_csv(f, usecols=cols) for f in files])
df = df[df.method.isin(METHODS)]
best = df.sort_values("coef_rel_err").groupby(["system", "noise", "N", "rep", "method"], as_index=False).first()
for m in METRICS:  # log10 error, clipped: blow-ups (inf/nan) -> 1e3, exact fits -> 1e-6
    best["l_" + m] = np.log10(best[m].replace([np.inf, -np.inf], np.nan).fillna(1e3).clip(1e-6, 1e3))
WINDOWS = sorted(best.N.unique())
NOISES = sorted(best.noise.unique())
methods = [m for m in METHODS if m in best.method.unique()]
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
q = lambda s, p: np.nanquantile(s, p)
pct = lambda v: f"{100 * v:.3g}%" if np.isfinite(v) else "inf"

# %% 1-2. variance decomposition and spreads
print("################ 1. share of the variance of log10(error) that is between systems (rest: between repeats)")
print("     and 2. across-system quantiles (10/50/90%) of each system's median error | median within-system max/min")
for metric in METRICS:
    print(f"\n==== {metric}")
    for nz in NOISES:
        rows = {}
        for m in methods:
            rows[m] = {}
            for N in N_SHOW:
                g = best[(best.method == m) & (best.noise == nz) & (best.N == N)]
                lv = g["l_" + metric]
                sys_mean = g.groupby("system")["l_" + metric].transform("mean")
                tot, within = lv.var(ddof=0), ((lv - sys_mean) ** 2).mean()
                share = 1 - within / tot if tot > 0 else np.nan
                med = g.groupby("system")[metric].median().replace(np.inf, np.nan)
                ratio = g.groupby("system")["l_" + metric].agg(lambda s: 10 ** (s.max() - s.min())).median()
                rows[m][f"N={N}"] = (f"{share:4.2f} | {pct(q(med, .1))} / {pct(q(med, .5))} / {pct(q(med, .9))}"
                                     f" | x{ratio:.3g}")
        print(f"-- noise {nz:g}")
        print(pd.DataFrame(rows).T.to_string())

# %% 3. noise robustness
print("\n################ 3. noise robustness per system: error at 5% noise / error at 0% noise (N = 10000, median "
      "over repeats); quantiles 10/50/90% over systems, and the error at 5% noise (10/50/90%)")
for metric in METRICS:
    rows = {}
    for m in methods:
        g = best[(best.method == m) & (best.N == 10000)].groupby(["system", "noise"])[metric].median().unstack()
        r = (g[0.05] / g[0.0].clip(lower=1e-6)).replace(np.inf, np.nan)
        rows[m] = {"ratio 5%/0%": f"x{q(r, .1):.3g} / x{q(r, .5):.3g} / x{q(r, .9):.3g}",
                   "error at 5%": f"{pct(q(g[0.05], .1))} / {pct(q(g[0.05], .5))} / {pct(q(g[0.05], .9))}"}
    print(f"-- {metric}")
    print(pd.DataFrame(rows).T.to_string())


# %% 4. N_min across systems and its bootstrap uncertainty within a system
def n_min_from(ok):
    """ok: (..., len(WINDOWS)) bool. Index of smallest N from which every larger N is reliable; len(WINDOWS) if never."""
    tail = np.flip(np.cumprod(np.flip(ok, -1), -1), -1).astype(bool)  # tail[i]: ok at i and every larger N
    return np.where(tail.any(-1), np.argmax(tail, -1), len(WINDOWS))


W = np.array(WINDOWS + [np.inf])
rng = np.random.default_rng(0)
print("\n################ 4. N_min: quantiles over systems (10/25/50/75/90%, samples; inf = never reliable) | "
      "within-system bootstrap over the 5 repeats: share of systems whose N_min moves by > 2x in >= 20% of resamples")
nmin_all = {}
for crit, (metric, cut) in CUT.items():
    print(f"\n==== {crit}")
    for nz in NOISES:
        rows = {}
        for m in methods:
            g = best[(best.method == m) & (best.noise == nz)]
            S = (g.assign(s=g[metric] < cut).pivot_table(index=["system", "N"], columns="rep", values="s", aggfunc="first")
                 .unstack("N").reindex(columns=WINDOWS, level="N"))  # (system, rep x N)
            systems = S.index
            succ = S.to_numpy(dtype=float).reshape(len(systems), -1, len(WINDOWS))  # (sys, rep, N)
            succ = np.nan_to_num(succ)
            nrep = succ.shape[1]
            idx0 = n_min_from(succ.mean(1) >= RELIABLE_FRAC)
            nmin = pd.Series(W[idx0], index=systems)
            nmin_all[(crit, nz, m)] = nmin
            boot = rng.integers(0, nrep, size=(N_BOOT, nrep))
            bidx = n_min_from(succ[:, boot, :].mean(2) >= RELIABLE_FRAC)  # (sys, boot)
            moved = np.abs(np.log2(W[bidx].clip(max=2e4) / W[idx0][:, None].clip(max=2e4))) > 1
            unstable = (moved.mean(1) >= 0.2).mean()
            qs = [W[int(np.nanquantile(idx0, p, method="nearest"))] for p in (.1, .25, .5, .75, .9)]
            rows[m] = {"N_min 10/25/50/75/90%": " / ".join("inf" if not np.isfinite(v) else f"{v:g}" for v in qs),
                       "unstable share": f"{unstable:.2f}"}
        print(f"-- noise {nz:g}")
        print(pd.DataFrame(rows).T.to_string())

# %% 5. paired symmetry effect per system
print("\n################ 5. per-system N_min ratio, symmetric / plain (systems where both are reliable): "
      "10/50/90% | share of systems with ratio < 1 / = 1 / > 1")
for a, b in [("ConstrainedSTLSQ_scaled", "STLSQ_scaled"), ("ConstrainedSR3_v1", "LSQ_dense"),
             ("ConstrainedSTLSQ_scaled", "LSQ_dense")]:
    if a not in methods or b not in methods:
        continue
    rows = {}
    for crit in CUT:
        rows[crit] = {}
        for nz in NOISES:
            x, y = nmin_all[(crit, nz, a)], nmin_all[(crit, nz, b)]
            both = np.isfinite(x) & np.isfinite(y)
            r = (x[both] / y[both])
            rows[crit][f"noise {nz:g}"] = (f"x{q(r, .1):.2g} / x{q(r, .5):.2g} / x{q(r, .9):.2g} | "
                                          f"{(r < 1).mean():.2f}/{(r == 1).mean():.2f}/{(r > 1).mean():.2f} (n={both.sum()})")
    print(f"-- {a} vs {b}")
    print(pd.DataFrame(rows).T.to_string())

# %% 6. which systems are the outliers (accuracy, 1% noise, symmetric SINDy)
print("\n################ 6. per-system median coefficient error over repeats, 1% noise, N = 800 and 10000 "
      "(sorted by ConstrainedSTLSQ_scaled at 10000)")
g = best[(best.noise == 0.01) & best.N.isin([800, 10000])]
t = g.pivot_table(index="system", columns=["N", "method"], values="coef_rel_err", aggfunc="median")
t = t.sort_values((10000, "ConstrainedSTLSQ_scaled"))
print(t.map(pct).to_string())
