# %% [markdown]
# # Diagrams of every dysts system as identified by ConstrainedSTLSQ
#
# Recommended setup from symmetric_sindy_dysts.md: ConstrainedSTLSQ, column-scaled, alpha = 0, symmetry constraints,
# no augmentation (run alpha0_norm, method ConstrainedSTLSQ_noaug). fits.csv stores metrics only, so each
# (system, noise) is refit at its selected threshold with the exact noise draw of symmetric_sindy_dysts.py (its
# RNG(0) is replayed: 200 symmetry-detection points for every scanned system, then the noise draws of the fitted ones,
# in the harness's order); the refit's coefficient error is checked against fits.csv.
#
# Per system: columns = noise 0% / 1% / 5%; top row = attractor of the data vs a free run of the learned model from
# the same state (first three coordinates); bottom = each coordinate vs time over 5 Lyapunov times, with the 1-Lyapunov-
# time trajectory-RMSE horizon marked. Plus one overview sheet of all attractors per noise level.
# SELECT=traj_nrmse_noisy picks thresholds by trajectory RMSE on the noisy data instead of coefficient error.

# %%
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dysts.flows as flows
from dysts_common import load_system
from fit_metrics import Evaluator

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DYSTS_DATA_DIR", "/gscratch/stf/songk42/dysts_hf/fine")
LOG_DIR = os.environ.get("LOG_DIR", "/mmfs1/home/songk42/ehd-sindy/logs/dysts_traj")
SELECT = os.environ.get("SELECT", "coef_rel_err")
RUN, METHOD = "alpha0_norm", "ConstrainedSTLSQ_noaug"
HARNESS_NOISES = ["exact_xdot", 0.0, 0.01, 0.05]  # order of RNG use in symmetric_sindy_dysts.py
PLOT_NOISES = [0.0, 0.01, 0.05]
DEGREE = 3
FREE_RUN = 3000  # samples (~30 periods) for the attractor panels
TS_LYAP = 5.0  # Lyapunov times in the time-series panels
OUT = os.path.join(LOG_DIR, "figures", "constrained_stlsq" + ("" if SELECT == "coef_rel_err" else f"_{SELECT}"))
os.makedirs(OUT, exist_ok=True)

# reference palette (dataviz skill), light mode: data = slot 1, model = slot 2; ink and grid recessive
DATA_C, MODEL_C = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a65", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "font.size": 9, "axes.titlesize": 9, "legend.frameon": False})

fits = pd.read_csv(os.path.join(LOG_DIR, "symmetric_sindy_dysts", RUN, "fits.csv"))
fits["noise"] = fits.noise.astype(str)
systems = sorted(fits.system.unique())  # the harness fits systems in sorted order
sel = (fits[fits.method == METHOD].sort_values(SELECT).groupby(["system", "noise"]).first())


def refit(sysd, Xn, noise):
    """symmetric_sindy_dysts.py's ConstrainedSTLSQ_noaug fit (alpha0_norm) at the selected threshold."""
    row = sel.loc[(sysd["name"], str(noise))]
    t = np.arange(len(Xn)) * sysd["dt"]
    diff = ps.FiniteDifference(order=4) if noise == 0.0 else ps.SmoothedFiniteDifference()
    dense = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=DEGREE), optimizer=ps.STLSQ(threshold=0.0, alpha=0.0),
                     differentiation_method=diff)
    dense.fit(Xn, t=t)
    scale = np.max(np.abs(dense.coefficients() * np.linalg.norm(dense.optimizer.Theta_, axis=0)))
    C = sysd["C"]
    opt = ps.ConstrainedSTLSQ(threshold=row.rel_threshold * scale, alpha=0.0, constraint_lhs=C,
                              constraint_rhs=np.zeros(C.shape[0]), constraint_order="target", normalize_columns=True)
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=DEGREE), optimizer=opt, differentiation_method=diff)
    model.fit(Xn, t=t, feature_names=[f"x{i}" for i in range(sysd["n"])])
    Xi = model.coefficients().T
    err = np.linalg.norm(Xi - sysd["Xi_true"]) / np.linalg.norm(sysd["Xi_true"])
    if not np.isclose(err, row.coef_rel_err, rtol=1e-6, atol=1e-12):
        print(f"  WARNING {sysd['name']} noise={noise}: refit coef err {err:.4g} != fits.csv {row.coef_rel_err:.4g}")
    return Xi, row


def run(ev, Xi, x0, n_samples):
    """RK4 run of Xi from x0, at the data sampling; NaN after blow-up."""
    return ev._integrate(Xi, x0[None], n_samples * 4)[::4, 0]


def attractor(ax, X, Y, n):
    ax.plot(*X[:, :3].T, color=DATA_C, lw=0.4, alpha=0.55)
    ok = np.isfinite(Y).all(axis=1)
    if ok.any():
        ax.plot(*Y[ok, :3].T, color=MODEL_C, lw=0.6)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel("x0", labelpad=-12); ax.set_ylabel("x1", labelpad=-12); ax.set_zlabel("x2", labelpad=-12)
    lo, hi = X[:, :3].min(0), X[:, :3].max(0)
    pad = 0.15 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    if not ok.all():
        ax.text2D(0.02, 0.92, "model blows up", transform=ax.transAxes, color=INK, fontsize=8)


def fmt(v):
    return "inf" if not np.isfinite(v) else f"{100 * v:.3g}%"


def replay_noise():
    """Noisy trajectories of symmetric_sindy_dysts.py, by replaying its RNG(0) in its exact order of use."""
    meta = json.load(open(os.path.join(os.path.dirname(flows.__file__), "data", "chaotic_attractors.json")))
    rng = np.random.default_rng(0)
    draws = {}
    for name in sorted(f.replace("_fine.csv", "") for f in os.listdir(DATA_DIR) if f.endswith("_fine.csv")):
        md = meta.get(name, {})
        if md.get("nonautonomous") or md.get("delay") or not hasattr(flows, name):
            continue
        X_raw = pd.read_csv(os.path.join(DATA_DIR, f"{name}_fine.csv")).iloc[:, 1:].to_numpy(float)
        if X_raw.shape[1] > 4 or not np.all(np.isfinite(X_raw)):
            continue
        rng.choice(len(X_raw), size=200, replace=False)  # detect_symmetries' sample points
        if name in systems:
            X = X_raw / np.max(np.abs(X_raw))
            draws[name] = {nz: X + nz * X.std(axis=0) * rng.standard_normal(X.shape)
                           for nz in HARNESS_NOISES if nz != "exact_xdot"}
    return draws


all_draws = replay_noise()
overview = {nz: [] for nz in PLOT_NOISES}
for name in systems:
    sysd = load_system(name, DATA_DIR)
    X, n = sysd["X"], sysd["n"]
    ev = Evaluator(X, sysd["Xi_true"], sysd["powers"], sysd["dt"], sysd["lyap"])
    draws = all_draws[name]
    t_lyap = ev.t_lyap
    n_ts = int(min(np.ceil(TS_LYAP * t_lyap / sysd["dt"]), 2000, len(X) - 1))
    i0 = len(X) // 2
    fig = plt.figure(figsize=(12, 4.2 + 1.1 * n))
    gs = fig.add_gridspec(1 + n, len(PLOT_NOISES), height_ratios=[4.2] + [1.1] * n, hspace=0.35, wspace=0.12)
    for j, noise in enumerate(PLOT_NOISES):
        Xi, row = refit(sysd, draws[noise], noise)
        Y_free = run(ev, Xi, X[i0], FREE_RUN)
        overview[noise].append((name, X, Y_free, row))
        ax = fig.add_subplot(gs[0, j], projection="3d")
        attractor(ax, X[i0:i0 + FREE_RUN], Y_free, n)
        ax.set_title(f"noise {noise:.0%} · threshold {row.rel_threshold:g}\ncoef err {fmt(row.coef_rel_err)}\n"
                     f"traj RMSE {fmt(row.traj_nrmse_clean)} clean / {fmt(row.traj_nrmse_noisy)} noisy", pad=0)
        Y_ts = run(ev, Xi, X[i0], n_ts)
        tt = np.arange(n_ts + 1) * sysd["dt"] / t_lyap
        for c in range(n):
            a = fig.add_subplot(gs[1 + c, j])
            a.plot(tt, X[i0:i0 + n_ts + 1, c], color=DATA_C, lw=1.6, label="data (clean)")
            a.plot(tt, Y_ts[:, c], color=MODEL_C, lw=1.2, ls=(0, (4, 2)), label="ConstrainedSTLSQ model")
            a.axvline(1.0, color=MUTED, lw=0.8, ls=":")
            lo, hi = X[:, c].min(), X[:, c].max()
            a.set_ylim(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo))
            a.set_xlim(0, tt[-1])
            if j == 0:
                a.set_ylabel(f"x{c}")
            else:
                a.set_yticklabels([])
            if c < n - 1:
                a.set_xticklabels([])
            elif j == len(PLOT_NOISES) // 2:
                a.set_xlabel("time [Lyapunov times]")
    handles = [plt.Line2D([], [], color=DATA_C, lw=1.6), plt.Line2D([], [], color=MODEL_C, lw=1.2, ls=(0, (4, 2))),
               plt.Line2D([], [], color=MUTED, lw=0.8, ls=":")]
    fig.legend(handles, ["data (clean)", "ConstrainedSTLSQ model (column-scaled, α=0, symmetry)",
                         "1 Lyapunov time (trajectory-RMSE horizon)"], loc="upper right", ncol=3,
               bbox_to_anchor=(0.98, 1.0))
    fig.suptitle(f"{name}  ({n}-D, |G| = {len(sysd['G']) + 1})", x=0.02, ha="left", fontsize=12, y=1.0)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"{name} done", flush=True)

# overview sheets: every attractor at one noise level
cols = 7
for noise, items in overview.items():
    rows = int(np.ceil(len(items) / cols))
    fig = plt.figure(figsize=(cols * 2.3, rows * 2.5))
    fig.subplots_adjust(top=0.93, bottom=0.01, left=0.01, right=0.99, hspace=0.25, wspace=0.05)
    for k, (name, X, Y, row) in enumerate(items):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        i0 = len(X) // 2
        attractor(ax, X[i0:i0 + FREE_RUN], Y, X.shape[1])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        ax.set_title(f"{name}\ntraj {fmt(row.traj_nrmse_clean)}", fontsize=8, pad=-4)
    handles = [plt.Line2D([], [], color=DATA_C, lw=1.6), plt.Line2D([], [], color=MODEL_C, lw=1.6)]
    fig.legend(handles, ["data (clean)", "ConstrainedSTLSQ model, free run from the same state"], loc="upper left",
               ncol=2, bbox_to_anchor=(0.005, 0.975))
    fig.suptitle(f"All {len(items)} systems, noise {noise:.0%}: attractor of the data vs the identified model "
                 f"(title: trajectory RMSE vs clean data, 1 Lyapunov time)", x=0.01, y=0.995, ha="left", fontsize=12)
    fig.savefig(os.path.join(OUT, f"overview_noise{noise:g}.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
print(f"figures in {OUT}")
