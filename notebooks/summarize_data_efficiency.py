# %% [markdown]
# # Summarize the data-efficiency sweep (logs/data_efficiency/fits_*.csv)
#
# Each (system, noise, N, repeat, method) is scored at its best threshold by coefficient error (oracle, as in the
# other dysts comparisons). A repeat "succeeds" under three criteria:
#   structure  exact terms (relative cutoff 1e-3 max|xi|) and coefficient error < 10%
#   accuracy   coefficient error < 10%
#   dynamics   vector-field error on the attractor < 5%
# A method is reliable for a system at N if >= 4 of 5 repeats succeed at N and at every larger N; N_min is the
# smallest such N (inf if never).

# %%
import glob
import os

import numpy as np
import pandas as pd

LOG_DIR = os.environ.get("LOG_DIR", "/home/songk/sindy-symmetry/logs")
RELIABLE_FRAC = 0.8
METHODS = ["STLSQ_a0.05", "ConstrainedSTLSQ_a0.05", "STLSQ_scaled", "ConstrainedSTLSQ_scaled",
           "ConstrainedSR3_l0_scaled", "ConstrainedSR3_v1", "SINDySI_full"]
PAIRS = [("ConstrainedSTLSQ_a0.05", "STLSQ_a0.05"), ("ConstrainedSTLSQ_scaled", "STLSQ_scaled"),
         ("ConstrainedSTLSQ_scaled", "STLSQ_a0.05"), ("SINDySI_full", "ConstrainedSTLSQ_scaled"),
         ("ConstrainedSR3_l0_scaled", "ConstrainedSTLSQ_scaled")]

files = sorted(glob.glob(os.path.join(LOG_DIR, "data_efficiency", "fits_*.csv")))
df = pd.concat([pd.read_csv(f) for f in files])
best = df.sort_values("coef_rel_err").groupby(["system", "noise", "N", "rep", "method"], as_index=False).first()
best["structure"] = best.exact_support_rel.astype(bool) & (best.coef_rel_err < 0.1)
best["accuracy"] = best.coef_rel_err < 0.1
best["dynamics"] = best.vf_rel_err < 0.05
WINDOWS = sorted(best.N.unique())
NOISES = sorted(best.noise.unique())
systems = sorted(best.system.unique())
periods_per_sample = best.groupby("system").apply(lambda g: (g.n_periods / g.N).iloc[0])
print(f"{len(systems)} systems, windows {WINDOWS}, noise {NOISES}\n")


def n_min(rate):
    """rate: Series indexed by N (sorted). Smallest N from which every larger N is reliable."""
    ok = (rate.reindex(WINDOWS).fillna(0) >= RELIABLE_FRAC).to_numpy()
    for i in range(len(WINDOWS)):
        if ok[i:].all():
            return WINDOWS[i]
    return np.inf


pd.set_option("display.width", 250, "display.max_rows", 500)
fmt_n = lambda v: ">max" if not np.isfinite(v) else f"{v:g}"
out = {}
for crit in ["structure", "accuracy", "dynamics"]:
    rate = best.groupby(["system", "noise", "method", "N"])[crit].mean()
    nm = rate.groupby(["system", "noise", "method"]).apply(lambda r: n_min(r.droplevel([0, 1, 2]))).unstack("method")
    out[crit] = nm
    print(f"################ criterion: {crit}")
    print("\n-- success rate over all systems x repeats, by N (rows: method; per noise)")
    for nz in NOISES:
        t = best[best.noise == nz].pivot_table(index="method", columns="N", values=crit, aggfunc="mean")
        print(f"noise {nz:g}")
        print(t.reindex([m for m in METHODS if m in t.index]).to_string(float_format="%.2f"))
    print("\n-- systems that become reliable (of %d) / median N_min in samples / median N_min in periods" % len(systems))
    rows = {}
    for m in [m for m in METHODS if m in nm.columns]:
        rows[m] = {}
        for nz in NOISES:
            v = nm.xs(nz, level="noise")[m]
            per = v * periods_per_sample.reindex(v.index)
            rows[m][f"noise {nz:g}"] = f"{np.isfinite(v).sum():2d} / {fmt_n(v.median())} / {fmt_n(per.median()) if np.isfinite(per.median()) else '>max'}"
    print(pd.DataFrame(rows).T.to_string())
    print("\n-- paired N_min: a needs less / same / more data than b (systems where at least one is reliable)")
    rows = {}
    for a, b in PAIRS:
        rows[f"{a} vs {b}"] = {}
        for nz in NOISES:
            x = nm.xs(nz, level="noise")
            x = x[np.isfinite(x[a]) | np.isfinite(x[b])]
            rows[f"{a} vs {b}"][f"noise {nz:g}"] = f"{(x[a] < x[b]).sum()}/{(x[a] == x[b]).sum()}/{(x[a] > x[b]).sum()}"
    print(pd.DataFrame(rows).T.to_string())
    print()

print("################ median coefficient error by N")
for nz in NOISES:
    t = best[best.noise == nz].pivot_table(index="method", columns="N", values="coef_rel_err", aggfunc="median")
    print(f"noise {nz:g}")
    print(t.reindex([m for m in METHODS if m in t.index]).map(lambda v: f"{100 * v:.3g}%").to_string())

print("\n################ per-system N_min (samples), criterion = accuracy, noise 1%")
x = out["accuracy"].xs(0.01, level="noise")[[m for m in METHODS if m in out["accuracy"].columns]]
print(x.map(fmt_n).to_string())
print("\n################ per-system N_min (samples), criterion = structure, noise 1%")
x = out["structure"].xs(0.01, level="noise")[[m for m in METHODS if m in out["structure"].columns]]
print(x.map(fmt_n).to_string())
