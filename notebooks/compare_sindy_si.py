# %% [markdown]
# # Compare all symmetric-SINDy variants on dysts with several correctness metrics
#
# Inputs (same 42 systems and noise draws):
#   logs/symmetric_sindy_dysts/alpha{0,0.05}[_norm]/fits.csv  STLSQ / ConstrainedSTLSQ, raw or column-scaled library
#   logs/sindy_si_dysts/fits.csv                              SINDy-SI variants and Equiv-c
# Each method is scored at the threshold with the lowest coefficient error (oracle, as before), on the original 4-point
# grid and on the 5-point grid that adds rel_threshold=0.3 (only run for column-scaled methods). All metrics are then
# read off that one selected fit; see fit_metrics.py for definitions.

# %%
import os

import numpy as np
import pandas as pd

LOG_DIR = os.environ.get("LOG_DIR", "/home/songk/sindy-symmetry/logs")
OLD_GRID = [0.003, 0.01, 0.03, 0.1]
NOISES = ["exact_xdot", "0.0", "0.01", "0.05"]
RUNS = [("alpha0.05", ""), ("alpha0", "_a0"), ("alpha0.05_norm", "_norm"), ("alpha0_norm", "_a0_norm"),
        ("sr3", ""), ("sr3_norm", "_norm")]  # sparse SR3 runs (SINDY_METHODS=sr3)
SR3_METHODS = ["SR3_l0", "ConstrainedSR3_l0_noaug", "ConstrainedSR3_l1_noaug"]
METHODS = [
    "STLSQ", "ConstrainedSTLSQ_noaug",  # previous defaults (ridge 0.05, raw library)
    "STLSQ_a0", "ConstrainedSTLSQ_noaug_a0",
    "STLSQ_norm", "ConstrainedSTLSQ_noaug_norm",
    "STLSQ_a0_norm", "ConstrainedSTLSQ_noaug_a0_norm",
    "ConstrainedSR3_v1_noaug",
    "SR3_l0", "ConstrainedSR3_l0_noaug", "ConstrainedSR3_l1_noaug",
    "SR3_l0_norm", "ConstrainedSR3_l0_noaug_norm", "ConstrainedSR3_l1_noaug_norm",
    "EquivC",
    "SINDySI_none", "SINDySI_sym", "SINDySI_paper", "SINDySI_full", "SI_full",
]

frames = []
for sub, tag in RUNS:
    f = os.path.join(LOG_DIR, f"symmetric_sindy_dysts/{sub}/fits.csv")
    if os.path.exists(f):
        o = pd.read_csv(f)
        if sub.startswith("sr3"):
            keep = SR3_METHODS
        else:
            keep = ["STLSQ", "ConstrainedSTLSQ_noaug"] + (["ConstrainedSR3_v1_noaug"] if tag == "" else [])
        frames.append(o[o.method.isin(keep)].assign(method=lambda d: d.method + tag))
new = pd.read_csv(os.path.join(LOG_DIR, "sindy_si_dysts/fits.csv"))
frames.append(new)
side = pd.read_csv(os.path.join(LOG_DIR, "sindy_si_dysts/side_info.csv"))
allfits = pd.concat(frames)
allfits["noise"] = allfits["noise"].astype(str)
systems = sorted(set(new.system))
allfits = allfits[allfits.system.isin(systems)]
print(f"{len(systems)} systems; monotone side info certified on {(side.monotone != '[]').sum()}, "
      f">=1 equilibrium on {(side.n_equilibria > 0).sum()}")


def best(df):
    return df.sort_values("coef_rel_err").groupby(["system", "noise", "method"], as_index=False).first()


def table(b, value, agg="median", fmt="{:.2f}"):
    t = b.pivot_table(index="method", columns="noise", values=value, aggfunc=agg)
    t = t.reindex([m for m in METHODS if m in t.index])[[n for n in NOISES if n in t.columns]]
    return t.map(lambda v: "" if pd.isna(v) else fmt.format(v)).to_string()


pct = lambda v: f"{100 * v:.3g}%"
pd.set_option("display.width", 250, "display.max_rows", 500)
grids = {
    "4-pt grid": best(allfits[allfits.rel_threshold.isin(OLD_GRID) | allfits.rel_threshold.isna()]),
    "5-pt grid (+0.3 for the column-scaled methods)": best(allfits),
}
for grid, b in grids.items():
    b = b.assign(exact_support=b.exact_support.astype(float), exact_support_rel=b.exact_support_rel.astype(float))
    print(f"\n######## {grid}")
    print("\n-- median relative coefficient error")
    print(table(b, "coef_rel_err", fmt="{:.3g}"))
    print("\n-- fraction exact support, strict (|xi| > 1e-10)")
    print(table(b, "exact_support", "mean"))
    print("\n-- fraction exact support, relative (|xi| > 1e-3 max|xi|)")
    print(table(b, "exact_support_rel", "mean"))
    print("\n-- median support F1, relative cutoff")
    print(table(b, "support_f1_rel"))
    print("\n-- median error on the true terms only (all runs) / (relative-exact-support runs only, n)")
    on_all = b.pivot_table(index="method", columns="noise", values="on_support_err", aggfunc="median")
    succ = b[b.exact_support_rel > 0]
    on_s = succ.pivot_table(index="method", columns="noise", values="on_support_err", aggfunc="median")
    n_s = succ.pivot_table(index="method", columns="noise", values="on_support_err", aggfunc="count")
    rows = {}
    for m in [m for m in METHODS if m in on_all.index]:
        rows[m] = {nz: f"{pct(on_all.loc[m, nz])} / " + (f"{pct(on_s.loc[m, nz])} ({int(n_s.loc[m, nz])})"
                                                          if m in on_s.index and nz in on_s.columns and not pd.isna(on_s.loc[m, nz]) else "- (0)")
                   for nz in NOISES if nz in on_all.columns}
    print(pd.DataFrame(rows).T.to_string())
    print("\n-- median relative vector-field error on the attractor")
    print(table(b, "vf_rel_err", fmt="{:.3g}"))
    print("\n-- median valid prediction time [Lyapunov times]  (cap: median horizon "
          f"{b.vpt_horizon.median():.1f}, min {b.vpt_horizon.min():.2f})")
    print(table(b, "vpt_lyap"))
    print("\n-- mean VPT as a fraction of each system's horizon")
    print(table(b.assign(vpt_frac=b.vpt_lyap / b.vpt_horizon), "vpt_frac", "mean"))

# %% rank agreement between metrics (4-pt grid, noisy fits): does low coefficient error mean good dynamics?
b = grids["4-pt grid"]
b = b[b.noise.isin(["0.01", "0.05"])]
print("\nSpearman correlation across all (system, noise, method) fits, 1% and 5% noise:")
print(b[["coef_rel_err", "on_support_err", "support_f1_rel", "vf_rel_err", "vpt_lyap"]].corr("spearman").round(2).to_string())
