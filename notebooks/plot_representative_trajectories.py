"""Plot real-vs-simulated trajectories for representative fitted models: r=3,4,5,6
(symmetry+energy-constrained SINDy), weak SINDy at r=3 (also constrained), and PLAIN
unconstrained SINDy (textbook STLSQ, no symmetry/energy constraints) at r=3 -- all on
the SAME window (t_start=800, duration=800) so the comparison is apples-to-apples --
this is the same window used in logs/r6_extension_summary.md's single-window r=6 deep
dive and the weak-vs-standard sweep's duration=800 cell (0 weak-fit failures there).

The plain-SINDy row answers a different question than r/weak: does imposing the known
symmetry + energy constraints actually help (vs. just running vanilla SINDy on the
same amplitudes), or is the switching-fidelity failure something even an unconstrained
fit can't avoid either?

For each model: fit on the window (augment=False; reg_weight_lam=0 for the constrained
fits), then simulate from the window's own first data point over (a) the short
fit-quality horizon (50 time units, well under one switching period) and (b) the full
window duration (800 time units), and plot both against the real data.

Output: notebooks/plots/traj_representatives_r3-6_weak_plain.png (one row per model)
plus a per-model summary printed to stdout and saved to
logs/representative_traj_summary.md.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import sympy as sp

from sindy_pipeline import (
    compute_pod, build_symmetry_constraints, slice_amplitudes, fit_symmetric_sindy,
    fit_plain_sindy, xi_ratio_generic, symmetry_residual_generic, G as G3,
)

POD_NPZ = "/gscratch/stf/songk42/ehd/sindy_data_fine.npz"
POD_CACHE = "/gscratch/stf/songk42/ehd/pod_cache_trimmed_t800.npz"
T_START, DURATION = 800.0, 800.0
T_END = T_START + DURATION
SHORT_HORIZON = 50.0
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

print("Loading POD (r=6 cache, sliced down per model)...", flush=True)
Phi, sigma, Vt, t_full = compute_pod(POD_NPZ, r=6, t_start=800.0, method="svds",
                                      cache_path=POD_CACHE)
print(f"  n_t={len(t_full)}, t in [{t_full[0]:.4g}, {t_full[-1]:.4g}]", flush=True)


def group_and_symbols(r):
    extra = r - 3
    Gr = np.array([sla.block_diag(g, np.eye(extra)) for g in G3]) if extra > 0 else G3
    a_symb_r = list(sp.symbols(" ".join(f"a{i + 1}" for i in range(r))))
    return Gr, a_symb_r


def simulate_from(fit, X_raw, duration, n_eval, max_step):
    """pysindy's simulate() can return fewer points than requested if the integrator
    blows up partway (observed for the unconstrained plain fit) -- trim t_eval to
    match rather than crash the caller."""
    X_scale = fit["X_scale"]
    x0 = X_raw[0] / X_scale
    t_eval = np.linspace(0.0, duration, n_eval)
    Y = fit["model"].simulate(x0, t_eval, integrator_kws=dict(max_step=max_step)) * X_scale
    return t_eval[:Y.shape[0]], Y


def fit_one(r, weak, label, plain=False):
    Gr, a_symb_r = group_and_symbols(r)
    X_raw, t_slice = slice_amplitudes(sigma, Vt, t_full, r=r, t_start=T_START, t_end=T_END, stride=1)

    t0 = time.time()
    if plain:
        fit = fit_plain_sindy(X_raw, t_slice, a_symb=a_symb_r)
    else:
        constraints_r = build_symmetry_constraints(r=r, group=Gr, a_symb=a_symb_r)
        fit = fit_symmetric_sindy(X_raw, t_slice, r=r, constraints=constraints_r, augment=False,
                                   group=Gr, a_symb=a_symb_r, weak=weak, weak_seed=0, max_iter=500)
    fit_time = time.time() - t0

    # generic helpers work for both the constrained fits and the plain (unconstrained)
    # one -- for `plain`, symmetry_residual_generic checks how badly a fit that was
    # NEVER told about the symmetry group happens to violate it anyway.
    ratio = xi_ratio_generic(fit["Xi_raw"], fit["feat_names"], a_symb_r)
    resid = symmetry_residual_generic(fit["Xi"], fit["feat_names"], a_symb_r, group=Gr)

    real_a1 = X_raw[:, 0]
    real_rate = int(np.sum(np.diff(np.sign(real_a1)) != 0)) / DURATION * 1000.0

    t0 = time.time()
    t_short, Y_short = simulate_from(fit, X_raw, SHORT_HORIZON, n_eval=500, max_step=0.05)
    t_long, Y_long = simulate_from(fit, X_raw, DURATION, n_eval=2000, max_step=0.2)
    sim_time = time.time() - t0

    sim_rate = int(np.sum(np.diff(np.sign(Y_long[:, 0])) != 0)) / DURATION * 1000.0
    fidelity = sim_rate / real_rate if real_rate > 0 else float("nan")

    nnz = int(np.sum(np.abs(fit["Xi"]) > 1e-10))
    print(f"[{label}] fit {fit_time:.1f}s, sim {sim_time:.1f}s, nnz={nnz}, "
          f"ratio={ratio['ratio']:.3g}, sym_resid={resid:.2e}, "
          f"real_rate={real_rate:.1f}/1000, sim_rate={sim_rate:.1f}/1000, "
          f"fidelity={fidelity:.3g}, a1 range real/sim={np.ptp(real_a1):.3g}/{np.ptp(Y_long[:, 0]):.3g}",
          flush=True)

    return dict(label=label, r=r, weak=weak, X_raw=X_raw, t_slice=t_slice,
                t_short=t_short, Y_short=Y_short, t_long=t_long, Y_long=Y_long,
                ratio=ratio["ratio"], sym_resid=resid, real_rate=real_rate,
                sim_rate=sim_rate, fidelity=fidelity)


models = []
for r in (3, 4, 5, 6):
    models.append(fit_one(r, weak=False, label=f"r={r} standard"))
models.append(fit_one(3, weak=True, label="r=3 weak"))
models.append(fit_one(3, weak=False, plain=True, label="r=3 plain (unconstrained STLSQ)"))

# ---------------------------------------------------------------------------
# Plot: one row per model, columns = [a1(t) full window, short-horizon zoom
# (a1 & a2), phase portrait a1 vs a2]
# ---------------------------------------------------------------------------
n_models = len(models)
fig, axes = plt.subplots(n_models, 3, figsize=(15, 3.2 * n_models))

for i, m in enumerate(models):
    t_rel = m["t_slice"] - T_START
    ax = axes[i, 0]
    ax.plot(t_rel, m["X_raw"][:, 0], color="tab:blue", lw=0.8, label="real a1")
    ax.plot(m["t_long"], m["Y_long"][:, 0], color="tab:orange", lw=0.8, label="sim a1")
    ax.set_title(f"{m['label']}: a1(t), full window (fidelity={m['fidelity']:.2g})", fontsize=9)
    ax.set_xlabel("t - t_start")
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

    ax = axes[i, 1]
    mask = t_rel <= SHORT_HORIZON
    ax.plot(t_rel[mask], m["X_raw"][mask, 0], color="tab:blue", lw=1.2, label="real a1")
    ax.plot(m["t_short"], m["Y_short"][:, 0], color="tab:blue", lw=1.2, ls="--", label="sim a1")
    ax.plot(t_rel[mask], m["X_raw"][mask, 1], color="tab:orange", lw=1.2, label="real a2")
    ax.plot(m["t_short"], m["Y_short"][:, 1], color="tab:orange", lw=1.2, ls="--", label="sim a2")
    ax.set_title(f"{m['label']}: short-horizon ({SHORT_HORIZON:g} tu) tracking", fontsize=9)
    ax.set_xlabel("t - t_start")
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

    ax = axes[i, 2]
    ax.plot(m["X_raw"][:, 0], m["X_raw"][:, 1], color="tab:blue", lw=0.6, alpha=0.8, label="real")
    ax.plot(m["Y_long"][:, 0], m["Y_long"][:, 1], color="tab:orange", lw=0.6, alpha=0.8, label="sim")
    ax.set_title(f"{m['label']}: phase portrait (a1, a2)", fontsize=9)
    ax.set_xlabel("a1")
    ax.set_ylabel("a2")
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

fig.suptitle(f"Representative window t_start={T_START:g}, duration={DURATION:g} -- "
             f"real vs. simulated across r=3,4,5,6 (standard), r=3 (weak), "
             f"r=3 (plain unconstrained)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])

PLOTS_DIR.mkdir(exist_ok=True)
out_path = PLOTS_DIR / "traj_representatives_r3-6_weak_plain.png"
fig.savefig(out_path, dpi=130)
print(f"\nSaved {out_path}", flush=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
lines = [
    "# Representative-model trajectory comparison (t_start=800, duration=800)",
    "",
    "Same window for every model, so differences are attributable to r / weak-vs-standard /",
    "constrained-vs-plain, not to which part of the trajectory was sampled. Plot: "
    "`notebooks/plots/traj_representatives_r3-6_weak_plain.png`.",
    "",
    "| model | xi2/xi6 | sym. resid. | real rate /1000 | sim rate /1000 | fidelity |",
    "|---|---|---|---|---|---|",
]
for m in models:
    lines.append(f"| {m['label']} | {m['ratio']:.3g} | {m['sym_resid']:.2e} | "
                 f"{m['real_rate']:.1f} | {m['sim_rate']:.1f} | {m['fidelity']:.3g} |")

LOGS_DIR.mkdir(exist_ok=True)
summary_path = LOGS_DIR / "representative_traj_summary.md"
summary_path.write_text("\n".join(lines) + "\n")
print(f"Saved {summary_path}", flush=True)
