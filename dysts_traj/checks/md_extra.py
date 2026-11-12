import sys
sys.argv = ["x"]
exec(open("/mmfs1/home/songk42/ehd-sindy/logs/dysts_traj/checks/md_tables.py").read().split("for sel in")[0])
exec("def paired" + open("/mmfs1/home/songk42/ehd-sindy/logs/dysts_traj/checks/md_tables.py").read().split("def paired")[1].split("PAIRS = ")[0])
PAIRS = [("ConstrainedSTLSQ_noaug", "STLSQ", "Symmetry, STLSQ α=0.05"), ("ConstrainedSTLSQ_noaug_a0", "STLSQ_a0", "Symmetry, STLSQ α=0"),
         ("ConstrainedSTLSQ_noaug_a0_norm", "STLSQ_a0_norm", "Symmetry, STLSQ α=0 scaled"), ("ConstrainedSR3_l0_noaug_norm", "SR3_l0_norm", "Symmetry, SR3 l0 scaled"),
         ("SINDySI_full", "SINDySI_paper", "Symmetry added to SINDy-SI eq+mono"), ("SINDySI_paper", "SINDySI_none", "eq+mono vs none (SINDy-SI)"),
         ("ConstrainedSTLSQ_noaug_a0_norm", "STLSQ", "scaled constrained vs old default"), ("ConstrainedSR3_v1_noaug", "ConstrainedSTLSQ_noaug_a0_norm", "v1 dense vs scaled ConstrainedSTLSQ")]
bt = best(A, "traj_nrmse_noisy")
print("| Comparison (traj selection) | Coef 1% | Coef 5% | Traj 1% | Traj 5% |")
for a, r, nm in PAIRS:
    print(f"| {nm} | " + " | ".join(paired(bt, a, r, "coef_rel_err") + paired(bt, a, r, "traj_nrmse_clean")) + " |")
b = best(A, "coef_rel_err")
for s in ["PanXuZhou", "Qi", "ChenLee", "LuChenCheng"]:
    x = b[(b.system == s) & b.noise.isin(["0.01", "0.05"]) & b.method.isin(["STLSQ_a0_norm", "ConstrainedSTLSQ_noaug_a0_norm", "SINDySI_paper", "SINDySI_full"])]
    print(s, x[["noise", "method", "coef_rel_err", "traj_nrmse_clean"]].round(4).values.tolist())
# how often is the oracle-coef threshold also the traj-optimal threshold, and are the traj metrics' best thresholds larger?
for m in ["ConstrainedSTLSQ_noaug_a0_norm", "SINDySI_full"]:
    for nz in ["0.01", "0.05"]:
        c = b[(b.method == m) & (b.noise == nz)].set_index("system").rel_threshold
        t = bt[(bt.method == m) & (bt.noise == nz)].set_index("system").rel_threshold
        print(m, nz, "traj-selected threshold larger/equal/smaller:", (t > c).sum(), (t == c).sum(), (t < c).sum())
