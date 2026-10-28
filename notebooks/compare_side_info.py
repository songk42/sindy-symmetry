# %% [markdown]
# # Does SINDy-SI's extra side information help the other constrained methods too?
#
# Pairs each constrained method's symmetry-only run with the same method + equilibria ("eq") and + equilibria +
# sampled monotonicity ("eq_mono"; SINDY_SIDE in symmetric_sindy_dysts.py). Same systems, noise draws and
# derivatives; each method at its best threshold by coefficient error (oracle), using the same threshold grid in
# all three variants. SINDy-SI rows (sum-of-squares monotonicity) are included for reference.

# %%
import os

import numpy as np
import pandas as pd

LOG_DIR = os.environ.get("LOG_DIR", "/home/songk/sindy-symmetry/logs")
D = os.path.join(LOG_DIR, "symmetric_sindy_dysts")
NOISES = ["0.0", "0.01", "0.05"]
GRID4 = [0.003, 0.01, 0.03, 0.1]

# (label, run directory without side suffix, method name in that run)
BASES = [
    ("ConstrainedSTLSQ α=0.05", "alpha0.05", "ConstrainedSTLSQ_noaug"),
    ("ConstrainedSTLSQ α=0", "alpha0", "ConstrainedSTLSQ_noaug"),
    ("ConstrainedSTLSQ α=0, scaled", "alpha0_norm", "ConstrainedSTLSQ_noaug"),
    ("v1 dense ConstrainedSR3", "alpha0", "ConstrainedSR3_v1_noaug"),
    ("ConstrainedSR3 l0", "sr3", "ConstrainedSR3_l0_noaug"),
    ("ConstrainedSR3 l1", "sr3", "ConstrainedSR3_l1_noaug"),
    ("ConstrainedSR3 l0, scaled", "sr3_norm", "ConstrainedSR3_l0_noaug"),
    ("ConstrainedSR3 l1, scaled", "sr3_norm", "ConstrainedSR3_l1_noaug"),
]
SIDES = [("sym", ""), ("+eq", "_eq"), ("+eq+mono", "_eq_mono")]


def load(run, method, label):
    f = os.path.join(D, run, "fits.csv")
    if not os.path.exists(f):
        return None
    o = pd.read_csv(f)
    o = o[o.method == method].copy()
    o["method"] = label
    return o


frames = []
for label, run, method in BASES:
    for side, suf in SIDES:
        # Equiv-c (constants constrained) only exists in the side-info raw runs; its symmetry-only twin is
        # ConstrainedSTLSQ α=0 (identical to every digit)
        o = load(run + suf, method, f"{label} | {side}")
        if o is not None:
            frames.append(o)
for side, suf in SIDES[1:]:
    o = load("alpha0" + suf, "EquivC_cc", f"Equiv-c | {side}")
    if o is not None:
        frames.append(o)
si = pd.read_csv(os.path.join(LOG_DIR, "sindy_si_dysts", "fits.csv"))
for m, side in [("SINDySI_sym", "sym"), ("SINDySI_sym_eq", "+eq"), ("SINDySI_full", "+eq+mono (SOS)")]:
    frames.append(si[si.method == m].assign(method=f"SINDy-SI | {side}"))
df = pd.concat(frames)
df["noise"] = df.noise.astype(str)
df = df[df.noise.isin(NOISES)]


def best(d):
    return d.sort_values("coef_rel_err").groupby(["system", "noise", "method"], as_index=False).first()


order = list(dict.fromkeys(df.method))
pd.set_option("display.width", 250, "display.max_rows", 500)
for grid_name, d in [("4-pt grid", df[df.rel_threshold.isin(GRID4) | df.rel_threshold.isna()]), ("5-pt grid", df)]:
    b = best(d)
    b["exact_support_rel"] = b.exact_support_rel.astype(float)
    print(f"\n######## {grid_name} (5-pt only differs for column-scaled methods and SINDy-SI)")
    for value, agg, fmt in [("coef_rel_err", "median", lambda v: f"{100 * v:.3g}%"),
                            ("exact_support_rel", "mean", lambda v: f"{v:.2f}"),
                            ("vf_rel_err", "median", lambda v: f"{100 * v:.3g}%"),
                            ("vpt_lyap", "median", lambda v: f"{v:.2f}")]:
        t = b.pivot_table(index="method", columns="noise", values=value, aggfunc=agg).reindex(order)[NOISES]
        print(f"\n-- {agg} {value}")
        print(t.map(lambda v: "" if pd.isna(v) else fmt(v)).to_string())

    print("\n-- paired coefficient error vs the same method with symmetry only: better/same/worse, geo-mean ratio")
    e = b.pivot_table(index=["system", "noise"], columns="method", values="coef_rel_err")
    rows = {}
    for label in [l for l, _, _ in BASES] + ["SINDy-SI"]:
        ref = f"{label} | sym"
        for side in ["+eq", "+eq+mono", "+eq+mono (SOS)"]:
            m = f"{label} | {side}"
            if m not in e.columns or ref not in e.columns:
                continue
            r = (e[m] / e[ref]).replace([np.inf, 0], np.nan).dropna()
            rows[m] = {nz: f"{(r.xs(nz, level='noise') < 0.9).sum()}/{((r.xs(nz, level='noise') >= 0.9) & (r.xs(nz, level='noise') <= 1.1)).sum()}/"
                           f"{(r.xs(nz, level='noise') > 1.1).sum()} x{np.exp(np.log(r.xs(nz, level='noise')).mean()):.2f}"
                       for nz in NOISES}
    print(pd.DataFrame(rows).T.to_string())

    print("\n-- paired coefficient error vs SINDy-SI (sym + eq + mono, SOS)")
    ref = "SINDy-SI | +eq+mono (SOS)"
    rows = {}
    for m in [c for c in e.columns if c.endswith("+eq+mono") or c.endswith("+eq") and c.startswith("Equiv")]:
        r = (e[m] / e[ref]).replace([np.inf, 0], np.nan).dropna()
        rows[m] = {nz: f"{(r.xs(nz, level='noise') < 0.9).sum()}/{((r.xs(nz, level='noise') >= 0.9) & (r.xs(nz, level='noise') <= 1.1)).sum()}/"
                       f"{(r.xs(nz, level='noise') > 1.1).sum()} x{np.exp(np.log(r.xs(nz, level='noise')).mean()):.2f}"
                   for nz in NOISES}
    print(pd.DataFrame(rows).T.to_string())
