# %% [markdown]
# # Summarize legendre_library_dysts.py: monomial vs Legendre library, paired per system
# Thresholds picked per repeat by coefficient error (oracle). Errors are on monomial-basis coefficients.

# %%
import glob
import os

import numpy as np
import pandas as pd

LOG_DIR = os.environ.get("LOG_DIR", "/home/songk/sindy-symmetry/logs")
D = os.path.join(LOG_DIR, "legendre")
pd.set_option("display.width", 250, "display.max_rows", 500)
info = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "info_*.csv")))]).set_index("system")
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "fits_*.csv")))])
best = df.sort_values("coef_rel_err").groupby(["system", "noise", "N", "rep", "basis", "method"], as_index=False).first()

print("################ per-system library facts (clean attractor)")
info["cond_ratio"] = info.cond_leg / info.cond_mono
print(info[["dim", "cond_mono", "cond_leg", "cond_ratio", "nnz_mono", "nnz_leg"]].sort_values("cond_ratio")
      .to_string(float_format="%.3g"))
print(f"\nchecks: max basis error {info.basis_err.max():.1e}, max |C xi_true| mono {info.C_resid_mono.max():.1e}, "
      f"legendre {info.C_resid_leg.max():.1e}")
print(f"cond legendre/mono: median x{info.cond_ratio.median():.2f}, better on {(info.cond_ratio < 1).sum()} of {len(info)}"
      f"; true model has more terms in Legendre on {(info.nnz_leg > info.nnz_mono).sum()} systems "
      f"(median {info.nnz_mono.median():g} -> {info.nnz_leg.median():g})")

for metric, cut in [("coef_rel_err", 0.1), ("vf_rel_err", 0.05)]:
    print(f"\n################ {metric}: success rate (< {cut:g}) over systems x repeats, mono -> legendre")
    best["ok"] = best[metric] < cut
    t = best.pivot_table(index=["method", "noise"], columns=["N", "basis"], values="ok", aggfunc="mean")
    out = pd.DataFrame({N: t[(N, "mono")].map("{:.2f}".format) + " -> " + t[(N, "legendre")].map("{:.2f}".format)
                        for N in sorted(best.N.unique())})
    print(out.to_string())
    print(f"\n-- per system (median over repeats): legendre/mono error ratio; systems better by >2x / within 2x / worse "
          f"by >2x; median ratio")
    med = best.groupby(["system", "noise", "N", "method", "basis"])[metric].median().unstack("basis")
    r = (med.legendre.clip(lower=1e-9) / med.mono.clip(lower=1e-9))
    g = r.groupby(["method", "noise", "N"])
    summ = pd.DataFrame({"better": g.apply(lambda x: (x < 0.5).sum()), "same": g.apply(lambda x: x.between(0.5, 2).sum()),
                         "worse": g.apply(lambda x: (x > 2).sum()), "median": g.median().map("x{:.2f}".format)})
    print(summ.to_string())
    for meth in ["STLSQ_scaled", "ConstrainedSTLSQ_scaled"]:
        x = r.xs(meth, level="method")
        big = x[(x < 0.5) | (x > 2)]
        if len(big):
            print(f"\n-- {meth}: systems where the basis changes {metric} by > 2x (noise, N: mono -> legendre)")
            m = med.xs(meth, level="method").loc[big.index]
            print(m.map(lambda v: f"{100 * v:.3g}%").join(big.rename("ratio").map("x{:.2g}".format))
                  .to_string())
