# Markdown tables for symmetric_sindy_dysts.md from logs/dysts_traj (sections 5 and 7).
import os, numpy as np, pandas as pd
L = "/mmfs1/home/songk42/ehd-sindy/logs/dysts_traj"
D = os.path.join(L, "symmetric_sindy_dysts")
RUNS = [("alpha0.05", ""), ("alpha0", "_a0"), ("alpha0.05_norm", "_norm"), ("alpha0_norm", "_a0_norm"), ("sr3", ""), ("sr3_norm", "_norm")]
SR3 = ["SR3_l0", "ConstrainedSR3_l0_noaug", "ConstrainedSR3_l1_noaug"]
fr = []
for sub, tag in RUNS:
    o = pd.read_csv(os.path.join(D, sub, "fits.csv"))
    keep = SR3 if sub.startswith("sr3") else ["STLSQ", "ConstrainedSTLSQ_noaug"] + (["ConstrainedSR3_v1_noaug"] if tag == "" else [])
    fr.append(o[o.method.isin(keep)].assign(method=lambda d: d.method + tag))
fr.append(pd.read_csv(os.path.join(L, "sindy_si_dysts", "fits.csv")))
A = pd.concat(fr); A["noise"] = A.noise.astype(str)
NAMES = [("STLSQ", "STLSQ α=0.05 (old default)"), ("ConstrainedSTLSQ_noaug", "ConstrainedSTLSQ α=0.05"),
         ("STLSQ_a0", "STLSQ α=0"), ("ConstrainedSTLSQ_noaug_a0", "ConstrainedSTLSQ α=0"), ("EquivC", "Equiv-c"),
         ("ConstrainedSR3_v1_noaug", "v1 dense ConstrainedSR3"), ("STLSQ_norm", "STLSQ α=0.05, scaled"),
         ("ConstrainedSTLSQ_noaug_norm", "ConstrainedSTLSQ α=0.05, scaled"), ("STLSQ_a0_norm", "**STLSQ α=0, scaled**"),
         ("ConstrainedSTLSQ_noaug_a0_norm", "**ConstrainedSTLSQ α=0, scaled**"), ("SR3_l0", "SR3 l0"),
         ("ConstrainedSR3_l0_noaug", "ConstrainedSR3 l0"), ("ConstrainedSR3_l1_noaug", "ConstrainedSR3 l1"),
         ("SR3_l0_norm", "SR3 l0, scaled"), ("ConstrainedSR3_l0_noaug_norm", "ConstrainedSR3 l0, scaled"),
         ("ConstrainedSR3_l1_noaug_norm", "ConstrainedSR3 l1, scaled"), ("SINDySI_none", "SINDy-SI, no side info"),
         ("SINDySI_sym", "SINDy-SI, sym"), ("SINDySI_paper", "SINDy-SI, eq + mono (paper)"),
         ("SINDySI_full", "**SINDy-SI, sym + eq + mono**"), ("SI_full", "SI (all side info, no pruning)")]
pct = lambda v: "∞" if not np.isfinite(v) else (f"{100*v:.2g}%" if v < 0.001 else f"{100*v:.3g}%")


def best(d, sel):
    return d.sort_values(sel).groupby(["system", "noise", "method"], as_index=False).first()


def med(b, m, nz, col, agg="median"):
    x = b[(b.method == m) & (b.noise == nz)][col]
    return getattr(x.astype(float), agg)()


for sel in ["coef_rel_err", "traj_nrmse_noisy"]:
    b = best(A, sel)
    print(f"\n#### selection by {sel}: coefficient table")
    print("| Method | Coef. err 0% | 1% | 5% | Exact 1% | 5% | F1 1% | 5% |\n|---|---|---|---|---|---|---|---|")
    for m, nm in NAMES:
        print(f"| {nm} | " + " | ".join([pct(med(b, m, nz, "coef_rel_err")) for nz in ["0.0", "0.01", "0.05"]] +
              [f"{med(b, m, nz, 'exact_support_rel', 'mean'):.2f}" for nz in ["0.01", "0.05"]] +
              [f"{med(b, m, nz, 'support_f1_rel'):.2f}" for nz in ["0.01", "0.05"]]) + " |")
    print(f"\n#### selection by {sel}: dynamics table")
    print("| Method | VF err 1% | 5% | VPT 1% | 5% | Traj RMSE clean 1% | 5% | Traj RMSE noisy 1% | 5% | Blow-ups 1% | 5% |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for m, nm in NAMES:
        bl = [int((~np.isfinite(b[(b.method == m) & (b.noise == nz)].traj_nrmse_clean)).sum()) for nz in ["0.01", "0.05"]]
        print(f"| {nm} | " + " | ".join([pct(med(b, m, nz, "vf_rel_err")) for nz in ["0.01", "0.05"]] +
              [f"{med(b, m, nz, 'vpt_lyap'):.2f}" for nz in ["0.01", "0.05"]] +
              [pct(med(b, m, nz, "traj_nrmse_clean")) for nz in ["0.01", "0.05"]] +
              [pct(med(b, m, nz, "traj_nrmse_noisy")) for nz in ["0.01", "0.05"]] + [str(x) for x in bl]) + " |")
    fl = b.groupby("noise").traj_nrmse_noisy_true.median()
    print(f"floors noisy: {fl.to_dict()}; clean max {b.traj_nrmse_clean_true.max():.2g}")


def paired(b, a, r, col):
    e = b.pivot_table(index=["system", "noise"], columns="method", values=col)
    out = []
    for nz in ["0.01", "0.05"]:
        x = e.xs(nz, level="noise")[[a, r]].replace([np.inf], np.nan)
        # treat blow-up as worse; ratio only on finite pairs
        ea, er = e.xs(nz, level="noise")[a], e.xs(nz, level="noise")[r]
        better = ((ea < 0.9 * er) | (np.isfinite(er) == False) & np.isfinite(ea)).sum()
        worse = ((ea > 1.1 * er) | (np.isfinite(ea) == False) & np.isfinite(er)).sum()
        same = len(ea) - better - worse
        rr = (ea / er).replace([np.inf, 0], np.nan).dropna()
        out.append(f"{better}/{same}/{worse}, ×{np.exp(np.log(rr).mean()):.2f}")
    return out


PAIRS = [("ConstrainedSTLSQ_noaug", "STLSQ", "**Symmetry constraints**, STLSQ α=0.05"),
         ("ConstrainedSTLSQ_noaug_a0", "STLSQ_a0", "Symmetry constraints, STLSQ α=0"),
         ("ConstrainedSTLSQ_noaug_a0_norm", "STLSQ_a0_norm", "Symmetry constraints, STLSQ α=0 scaled"),
         ("ConstrainedSR3_l0_noaug", "SR3_l0", "Symmetry constraints, SR3 l0"),
         ("ConstrainedSR3_l0_noaug_norm", "SR3_l0_norm", "Symmetry constraints, SR3 l0 scaled"),
         ("SINDySI_full", "SINDySI_paper", "Symmetry added to SINDy-SI eq + mono"),
         ("ConstrainedSTLSQ_noaug_a0_norm", "STLSQ", "**Column scaling + α=0** (constrained) vs old default STLSQ"),
         ("SINDySI_paper", "SINDySI_none", "**eq + mono side info** vs SINDy-SI with none"),
         ("ConstrainedSR3_l0_noaug_norm", "ConstrainedSTLSQ_noaug_a0_norm", "ConstrainedSR3 l0 scaled vs ConstrainedSTLSQ scaled"),
         ("ConstrainedSR3_l1_noaug", "ConstrainedSR3_l0_noaug", "ConstrainedSR3 l1 vs l0, unscaled"),
         ("ConstrainedSR3_v1_noaug", "ConstrainedSTLSQ_noaug_a0_norm", "v1 dense vs ConstrainedSTLSQ scaled")]
b = best(A, "coef_rel_err")
print("\n#### paired (coef selection): coef err | traj clean")
print("| Comparison | Coef 1% | Coef 5% | Traj 1% | Traj 5% |\n|---|---|---|---|---|")
for a, r, nm in PAIRS:
    print(f"| {nm} | " + " | ".join(paired(b, a, r, "coef_rel_err") + paired(b, a, r, "traj_nrmse_clean")) + " |")
# augmentation vs constraints (scaled) from the alpha0_norm run
o = pd.read_csv(os.path.join(D, "alpha0_norm", "fits.csv")); o["noise"] = o.noise.astype(str)
bo = best(o, "coef_rel_err")
print("| Augmentation vs constraints, scaled | " + " | ".join(paired(bo, "STLSQ_aug", "ConstrainedSTLSQ_noaug", "coef_rel_err") + paired(bo, "STLSQ_aug", "ConstrainedSTLSQ_noaug", "traj_nrmse_clean")) + " |")

print("\n#### oracle vs honest selection: cost of picking by traj RMSE on noisy data (ratio of medians / paired)")
bt = best(A, "traj_nrmse_noisy")
for m in ["STLSQ", "STLSQ_a0_norm", "ConstrainedSTLSQ_noaug_a0_norm", "ConstrainedSR3_l0_noaug_norm", "SINDySI_full", "ConstrainedSR3_v1_noaug"]:
    row = []
    for nz in ["0.01", "0.05"]:
        same = (b[(b.method == m) & (b.noise == nz)].set_index("system").rel_threshold ==
                bt[(bt.method == m) & (bt.noise == nz)].set_index("system").rel_threshold)
        row.append(f"same threshold {same.mean():.2f}; coef {pct(med(b, m, nz, 'coef_rel_err'))}→{pct(med(bt, m, nz, 'coef_rel_err'))}; "
                   f"traj clean {pct(med(b, m, nz, 'traj_nrmse_clean'))}→{pct(med(bt, m, nz, 'traj_nrmse_clean'))}")
    print(m, " || ".join(row))

print("\n#### per-system symmetry effect, biggest (coef selection), scaled STLSQ and SINDy-SI")
e = b.pivot_table(index=["system", "noise"], columns="method", values="coef_rel_err")
for a, r in [("ConstrainedSTLSQ_noaug_a0_norm", "STLSQ_a0_norm"), ("ConstrainedSTLSQ_noaug", "STLSQ"), ("SINDySI_full", "SINDySI_paper"), ("ConstrainedSTLSQ_noaug_a0", "STLSQ_a0")]:
    x = (e[a] / e[r]).dropna()
    x = x[x.index.get_level_values(1).isin(["0.01", "0.05"])].sort_values()
    print(a, "vs", r, [(i, f"{pct(e.loc[i, r])}→{pct(e.loc[i, a])}") for i in list(x.index[:4]) + list(x.index[-2:])])
