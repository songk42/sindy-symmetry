# %% [markdown]
# # Summarize weak form x orbit thresholding (weak_orbit_dysts.py, weak_orbit_data_eff.py)
#
# Full data: per (system, noise, method) the best threshold by SELECT (coef_rel_err = oracle, or traj_nrmse_noisy =
# no ground truth); medians over the 42 systems, exact-support fraction, blow-up fraction, and paired ratios.
# Data efficiency: same criteria and N_min rule as summarize_data_efficiency.py (>= 4 of 5 repeats succeed at N and
# at every larger N), with paired N_min ratios over systems where both are reliable.

# %%
import glob
import os

import numpy as np
import pandas as pd

LOG_DIR = os.environ.get("LOG_DIR", "/mmfs1/home/songk42/ehd-sindy/logs/dysts_traj")
TAG = os.environ.get("WO_TAG", "m25_p8")
SELECT = os.environ.get("SELECT", "coef_rel_err")
RELIABLE_FRAC = 0.8
METHODS = [f"{f}_{m}" for f in ["strong", "weak"] for m in ["STLSQ", "ConstrainedSTLSQ", "OrbitSTLSQ"]]
PAIRS = [("weak_STLSQ", "strong_STLSQ"), ("weak_ConstrainedSTLSQ", "strong_ConstrainedSTLSQ"),
         ("weak_OrbitSTLSQ", "strong_OrbitSTLSQ"),
         ("strong_ConstrainedSTLSQ", "strong_STLSQ"), ("weak_ConstrainedSTLSQ", "weak_STLSQ"),
         ("strong_OrbitSTLSQ", "strong_ConstrainedSTLSQ"), ("weak_OrbitSTLSQ", "weak_ConstrainedSTLSQ"),
         ("weak_OrbitSTLSQ", "strong_ConstrainedSTLSQ")]
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
pct = lambda v: "inf" if not np.isfinite(v) else f"{100 * v:.3g}%"
fmt_n = lambda v: ">max" if not np.isfinite(v) else f"{v:g}"


def paired(best, metric, key):
    """Median ratio a/b and wins/ties/losses (10% band) per noise level."""
    rows = {}
    for a, b in PAIRS:
        x = best.pivot_table(index=key, columns="method", values=metric, aggfunc="first")
        r = {}
        for nz, g in x.groupby(level="noise"):
            ga, gb = g[a], g[b]
            ok = np.isfinite(ga) & np.isfinite(gb) & (gb > 0)
            ratio = (ga[ok] / gb[ok])
            win = ((ga < gb / 1.1) | (np.isfinite(gb) == False) & np.isfinite(ga)).sum()
            lose = ((ga > gb * 1.1) | (np.isfinite(ga) == False) & np.isfinite(gb)).sum()
            r[f"noise {nz}"] = f"x{ratio.median():.2f}  {win}/{len(g) - win - lose}/{lose}"
        rows[f"{a} vs {b}"] = r
    return pd.DataFrame(rows).T


# %% full data
files = glob.glob(os.path.join(LOG_DIR, "weak_orbit_dysts", TAG, "fits_*.csv"))
if files:
    df = pd.concat([pd.read_csv(f) for f in files])
    df["noise"] = df.noise.astype(float)
    best = df.sort_values(SELECT).groupby(["system", "noise", "method"], as_index=False).first()
    n_sys = best.system.nunique()
    print(f"######## FULL DATA: {n_sys} systems, thresholds selected by {SELECT}\n")
    for value in ["coef_rel_err", "vf_rel_err", "traj_nrmse_clean", "traj_nrmse_noisy"]:
        t = best.pivot_table(index="method", columns="noise", values=value, aggfunc="median").reindex(METHODS)
        print(f"-- median {value}")
        print(t.map(pct).to_string(), "\n")
    for value, lab in [("exact_support_rel", "exact support fraction"),
                       ("traj_nrmse_clean", "blow-up fraction (traj inf)")]:
        v = best[value].astype(float) if value == "exact_support_rel" else (~np.isfinite(best[value])).astype(float)
        t = best.assign(v=v).pivot_table(index="method", columns="noise", values="v", aggfunc="mean").reindex(METHODS)
        print(f"-- {lab}")
        print(t.to_string(float_format="%.2f"), "\n")
    for value in ["coef_rel_err", "traj_nrmse_clean"]:
        print(f"-- paired {value}: median ratio a/b, a better / within 10% / worse")
        print(paired(best, value, ["system", "noise"]).to_string(), "\n")
    print("-- per-system coef_rel_err at 5% noise")
    t = best[best.noise == 0.05].pivot_table(index="system", columns="method", values="coef_rel_err").reindex(
        columns=METHODS)
    print(t.map(pct).to_string(), "\n")

# %% data efficiency
files = glob.glob(os.path.join(LOG_DIR, "weak_orbit_data_eff", TAG, "fits_*.csv"))
if files:
    df = pd.concat([pd.read_csv(f) for f in files])
    best = df.sort_values(SELECT).groupby(["system", "noise", "N", "rep", "method"], as_index=False).first()
    best["structure"] = best.exact_support_rel.astype(bool) & (best.coef_rel_err < 0.1)
    best["accuracy"] = best.coef_rel_err < 0.1
    best["dynamics"] = best.vf_rel_err < 0.05
    best["trajectory"] = best.traj_nrmse_clean < 0.1
    WINDOWS = sorted(best.N.unique())
    NOISES = sorted(best.noise.unique())
    systems = sorted(best.system.unique())
    print(f"######## DATA EFFICIENCY: {len(systems)} systems, windows {WINDOWS}, noise {NOISES}; select {SELECT}\n")

    def n_min(rate):
        ok = (rate.reindex(WINDOWS).fillna(0) >= RELIABLE_FRAC).to_numpy()
        for i in range(len(WINDOWS)):
            if ok[i:].all():
                return WINDOWS[i]
        return np.inf

    for crit in ["structure", "accuracy", "dynamics", "trajectory"]:
        rate = best.groupby(["system", "noise", "method", "N"])[crit].mean()
        nm = rate.groupby(["system", "noise", "method"]).apply(lambda r: n_min(r.droplevel([0, 1, 2]))).unstack("method")
        print(f"################ criterion: {crit}")
        print("-- systems reliable (of %d) / median N_min" % len(systems))
        rows = {}
        for m in [m for m in METHODS if m in nm.columns]:
            rows[m] = {f"noise {nz:g}": f"{np.isfinite(nm.xs(nz, level='noise')[m]).sum():2d} / "
                                        f"{fmt_n(nm.xs(nz, level='noise')[m].median())}" for nz in NOISES}
        print(pd.DataFrame(rows).T.to_string())
        print("-- paired N_min: median ratio a/b (both reliable), a needs less / same / more (either reliable)")
        rows = {}
        for a, b in PAIRS:
            rows[f"{a} vs {b}"] = {}
            for nz in NOISES:
                x = nm.xs(nz, level="noise")
                both = x[np.isfinite(x[a]) & np.isfinite(x[b])]
                x = x[np.isfinite(x[a]) | np.isfinite(x[b])]
                ratio = (both[a] / both[b]).median() if len(both) else np.nan
                rows[f"{a} vs {b}"][f"noise {nz:g}"] = (f"x{ratio:.2f} (n={len(both)}), "
                                                        f"{(x[a] < x[b]).sum()}/{(x[a] == x[b]).sum()}/{(x[a] > x[b]).sum()}")
        print(pd.DataFrame(rows).T.to_string(), "\n")

    for value in ["coef_rel_err", "traj_nrmse_clean"]:
        print(f"################ median {value} by N")
        for nz in NOISES:
            t = best[best.noise == nz].pivot_table(index="method", columns="N", values=value, aggfunc="median")
            print(f"noise {nz:g}")
            print(t.reindex([m for m in METHODS if m in t.index]).map(pct).to_string())
        print()
