# Does the old (2nd-order, biased-low) dt offset SmoothedFiniteDifference's amplitude bias at 1% noise?
import sys, warnings
import numpy as np, pysindy as ps
sys.path.insert(0, "/mmfs1/home/songk42/ehd-sindy/notebooks")
from dysts_common import load_system
warnings.filterwarnings("ignore")
D = "/gscratch/stf/songk42/dysts_hf/fine"
for name in ["Lorenz", "Chen", "Arneodo", "Halvorsen", "HyperPang", "SprottB", "RayleighBenard", "Rucklidge"]:
    s = load_system(name, D)
    X, dt6, Xt = s["X"], s["dt"], s["Xi_true"]
    X_raw = X  # dt2 estimate from the scaled data is the same ratio
    g2 = np.gradient(X, axis=0)
    F = np.asarray(s["lib"].transform(X)) @ Xt
    dt2 = np.sum(g2 * F) / np.sum(F * F)
    Xn = X + 0.01 * X.std(0) * np.random.default_rng(1).standard_normal(X.shape)
    out = []
    for dt in (dt6, dt2):
        t = np.arange(len(X)) * dt
        best = None
        for rt in [0.003, 0.01, 0.03, 0.1, 0.3]:
            d = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3), optimizer=ps.STLSQ(threshold=0, alpha=0),
                         differentiation_method=ps.SmoothedFiniteDifference()).fit(Xn, t=t)
            sc = np.max(np.abs(d.coefficients() * np.linalg.norm(d.optimizer.Theta_, axis=0)))
            m = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3),
                         optimizer=ps.STLSQ(threshold=rt * sc, alpha=0, normalize_columns=True),
                         differentiation_method=ps.SmoothedFiniteDifference()).fit(Xn, t=t)
            Xi = m.coefficients().T
            e = np.linalg.norm(Xi - Xt) / np.linalg.norm(Xt)
            if best is None or e < best[0]:
                best = (e, np.sum(Xi * Xt) / np.sum(Xt * Xt))
        out.append(best)
    print(f"{name:15s} dt2/dt6-1={dt2/dt6-1:+.2%}  dt6: err {out[0][0]:.2%} scale {out[0][1]:.4f} | dt2: err {out[1][0]:.2%} scale {out[1][1]:.4f}", flush=True)
