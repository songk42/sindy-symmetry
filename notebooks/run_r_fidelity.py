"""Fidelity sweep for one r value (see /tmp scratchpad sweep_r_fidelity.py for the
original, killed by login-node memory pressure). Usage: python run_r_fidelity.py <r>
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import scipy.linalg as sla
import sympy as sp
from sindy_pipeline import compute_pod, build_symmetry_constraints, run_multi_slice, G as G3

r = int(sys.argv[1])

Phi, sigma, Vt, t_full = compute_pod(
    "/gscratch/stf/songk42/ehd/sindy_data_fine.npz", r=6, t_start=800.0, method="svds",
    cache_path="/gscratch/stf/songk42/ehd/pod_cache_trimmed_t800.npz")
print(f"POD loaded: n_t={len(t_full)}, t in [{t_full[0]:.4g}, {t_full[-1]:.4g}]")


def real_rate_per_1000(t_start, duration):
    mask = (t_full >= t_start) & (t_full <= t_start + duration)
    a1 = (sigma[0] * Vt[0, mask])
    n_cross = int(np.sum(np.diff(np.sign(a1)) != 0))
    return n_cross / duration * 1000.0


cells = [dict(duration=800.0, stride=s, n_samples=4, label=f"stride={s} (dt~{0.05*s:g})")
         for s in (1, 2, 4, 8)]
cells += [dict(duration=d, stride=1, n_samples=n, label=f"duration={d:g}")
          for d, n in [(200.0, 8), (400.0, 8), (800.0, 4), (1200.0, 2)]]

extra = r - 3
Gr = np.array([sla.block_diag(g, np.eye(extra)) for g in G3]) if extra > 0 else G3
a_symb_r = list(sp.symbols(" ".join(f"a{i+1}" for i in range(r))))
constraints_r = build_symmetry_constraints(r=r, group=Gr, a_symb=a_symb_r)
print(f"\n### r={r} ###", flush=True)

fidelities = []
for cell in cells:
    t0 = time.time()
    result = run_multi_slice(sigma, Vt, t_full, r=r, constraints=constraints_r, augment=False,
                              group=Gr, a_symb=a_symb_r, check_sim=True, check_fit_quality=False,
                              **cell)
    row_fids = []
    for row in result["rows"]:
        if row.get("fit") is None or not row["bounded"].get("finite"):
            row_fids.append(0.0)
            continue
        sim_rate = row["bounded"]["a1_sign_changes"]
        real_rate = real_rate_per_1000(row["t_start"], row["t_end"] - row["t_start"])
        row_fids.append(sim_rate / real_rate if real_rate > 0 else float("nan"))
    fidelities.extend(row_fids)
    print(f"  {cell['label']}: fidelities={[round(f,2) for f in row_fids]}  ({time.time()-t0:.0f}s)",
          flush=True)
    del result

fidelities = np.array(fidelities)
print(f"### r={r}: fidelity (sim rate / real rate) across {len(fidelities)} windows: "
      f"mean={fidelities.mean():.3f} median={np.median(fidelities):.3f} "
      f"min={fidelities.min():.3f} max={fidelities.max():.3f} "
      f"frac_within_2x={(np.abs(np.log2(np.where(fidelities>0,fidelities,1e-9)))<1).mean():.3f} ###")
