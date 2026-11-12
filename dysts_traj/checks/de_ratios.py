import glob, os, numpy as np, pandas as pd
L = "/mmfs1/home/songk42/ehd-sindy/logs/dysts_traj"
df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(L, "data_efficiency", "fits_*.csv"))])
W = sorted(df.N.unique())
def n_min(r):
    ok = (r.reindex(W).fillna(0) >= 0.8).to_numpy()
    for i in range(len(W)):
        if ok[i:].all(): return W[i]
    return np.inf
PAIRS = [("ConstrainedSTLSQ_scaled", "STLSQ_scaled", "Symmetry, scaled STLSQ"), ("ConstrainedSTLSQ_a0.05", "STLSQ_a0.05", "Symmetry, STLSQ α=0.05"),
         ("ConstrainedSTLSQ_scaled", "STLSQ_a0.05", "Scaled constrained vs old default"), ("SINDySI_full", "ConstrainedSTLSQ_scaled", "SINDy-SI vs scaled constrained"),
         ("ConstrainedSR3_l0_scaled", "ConstrainedSTLSQ_scaled", "ConstrainedSR3 l0 vs ConstrainedSTLSQ, scaled"), ("ConstrainedSR3_v1", "ConstrainedSTLSQ_scaled", "v1 dense vs scaled constrained")]
for sel in ["coef_rel_err", "traj_nrmse_noisy"]:
    b = df.sort_values(sel).groupby(["system", "noise", "N", "rep", "method"], as_index=False).first()
    b["structure"] = b.exact_support_rel.astype(bool) & (b.coef_rel_err < 0.1)
    b["accuracy"] = b.coef_rel_err < 0.1
    b["dynamics"] = b.vf_rel_err < 0.05
    b["trajectory"] = b.traj_nrmse_clean < 0.1
    print(f"\n### selection {sel}")
    for crit in ["accuracy", "structure", "dynamics", "trajectory"]:
        rate = b.groupby(["system", "noise", "method", "N"])[crit].mean()
        nm = rate.groupby(["system", "noise", "method"]).apply(lambda r: n_min(r.droplevel([0, 1, 2]))).unstack("method")
        for a, r, lab in PAIRS:
            cells = []
            for nz in [0.0, 0.01, 0.05]:
                x = nm.xs(nz, level="noise")
                both = x[np.isfinite(x[a]) & np.isfinite(x[r])]
                any_ = x[np.isfinite(x[a]) | np.isfinite(x[r])]
                g = np.exp(np.log(both[a] / both[r]).mean()) if len(both) else np.nan
                cells.append(f"×{g:.2f} (n={len(both)}), {(any_[a] < any_[r]).sum()}/{(any_[a] == any_[r]).sum()}/{(any_[a] > any_[r]).sum()}")
            print(f"{crit:10s} | {lab} | " + " | ".join(cells))
    # success-rate at N=200 / 10000, 1% noise, symmetry pair
    t = b[b.noise == 0.01].pivot_table(index="method", columns="N", values="accuracy", aggfunc="mean")
    print("accuracy rate 1%:", {m: (round(t.loc[m, 200], 2), round(t.loc[m, 10000], 2)) for m in ["STLSQ_scaled", "ConstrainedSTLSQ_scaled", "SINDySI_full"]})
    t = b[b.noise == 0.01].pivot_table(index="method", columns="N", values="trajectory", aggfunc="mean")
    print("trajectory rate 1%:", {m: [round(t.loc[m, n], 2) for n in [200, 800, 3200, 10000]] for m in t.index})
