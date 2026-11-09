# Symmetry-aware SINDy on the dysts benchmark

Do discrete symmetries help sparse identification of chaotic ODEs, and which way of using them works best?
This note covers every setup run so far: symmetry constraints and augmentation (v1), SINDy-SI (Machado & Jones,
arXiv:2310.04227), Equiv-c (Yang et al., arXiv:2405.16756), column scaling, and sparse ConstrainedSR3. All methods see
the same systems, the same noise draws and the same derivative estimates, so results pair system by system.
Section 6 adds a data-efficiency sweep: how much training data each method needs before its fits become reliable.
Section 7 gives SINDy-SI's extra side information to the other constrained methods.

**Oct 29 rerun.** Everything was rerun on klone with two changes: a new metric, the **trajectory RMSE** (simulate
the identified model from points on the data and compare with the data that follows, over one Lyapunov time; section
4), and a fixed time-step estimate (section 1). The fix changes every number from earlier versions of this note:
noise-free errors fell about 30×, while errors with noise roughly doubled for the best methods, because the old bias
had been hiding part of the derivative estimate's error (finding 5.4.1). Which methods win on coefficients is unchanged.

**Short answer.**

- **Recovering the equations (coefficient error).** Column scaling with no ridge is the biggest single factor
  (×0.18 / ×0.44 the error of the old default at 1% / 5% noise, about 0.3× the data). Known equilibria and signs of
  ∂fᵢ/∂xᵢ add ×0.80 / ×0.55. The symmetry barely moves the median (×0.95 / ×1.00) but fixes the few unevenly
  sampled systems (ChenLee 145% → 2.1% at 5% noise), and occasionally hurts (PanXuZhou 0.8% → 62%).
- **Reproducing the data (trajectory RMSE).** This is a different target: it correlates only weakly with
  coefficient error (Spearman 0.45). The fits with the best coefficients often forecast badly (scaled
  ConstrainedSTLSQ: 2.1% coefficient error, 56% trajectory RMSE at 5% noise), and a dense fit with 60% coefficient
  error reproduces the data better. Choosing each threshold by trajectory RMSE on the noisy training data, which
  needs no ground truth, gives the best trajectories (scaled ConstrainedSTLSQ: 8.9% / 12.8% at 1% / 5% noise, no
  blow-ups) at the cost of coefficient accuracy (10% / 37%). The symmetry doesn't change trajectory RMSE in aggregate
  with the full trajectory.
- **The symmetry's main benefit is data efficiency.** With column scaling it reaches a reliable fit with 0.50–0.62×
  the data at 1–5% noise on every criterion, including reproducing the data over the whole attractor from a short
  window.
- **Side information** (equilibria as equalities, signs as sampled inequalities) given to column-scaled
  ConstrainedSTLSQ matches SINDy-SI: at 1% noise the coefficient error is the same on all 42 systems and the
  trajectory RMSE on 40 of 41 (one system is worse).

The recommended default is `ConstrainedSTLSQ(normalize_columns=True, alpha=0)` with the symmetry constraints and no
augmentation, plus equilibria and sampled sign constraints when they're known. Pick the threshold by coefficient
error (or a proxy for it) when the equations are the goal, and by trajectory RMSE on held-out data when forecasting
is. Section 5.5 has diagrams of every system as identified by this default.

---

## 1. Data and ground truth

- **Source:** the `fine/` split of the HuggingFace dataset `williamgilpin/dysts`, stored locally at
  `/gscratch/stf/songk42/dysts_hf/fine` on klone (`/home/songk/data-songk/dysts_hf/fine` on engaging). Each system
  is one noise-free trajectory of 10,000 samples, at about 100 samples per period of the main oscillation.
- **Time step:** the CSV time column is a sample index. The physical step `dt` is recovered by regressing a
  6th-order centered finite difference on the true dysts right-hand side; systems with a fit R² < 0.99 are skipped.
  It matches dysts' `period/100` to within 1e-4 on 41 of 42 systems (Bouali2's metadata period is wrong).
  - Runs before Oct 29 used `np.gradient` (2nd order), which put `dt` 0.65% too low on average and up to 2.8% too
    low. That scaled every fitted coefficient by the same factor and, for the true model, pushed RK4
    forecasts off the data within a Lyapunov time (3.6% of the attractor spread on Lorenz, 32% on ChenLee).
  - All numbers in this note are from the rerun with the fixed `dt`, so they differ from earlier versions; see
    finding 5.4.1.
- **Scaling:** the state is divided by `max|X|` (a uniform scale, which commutes with every signed permutation).
  All coefficients are in these scaled coordinates.
- **Library:** polynomials up to degree 3 with a constant term (20 terms in 3-D, 35 in 4-D).
- **True coefficients Ξ_true:** least squares of the true right-hand side on the library. This is exact when the
  system is polynomial (residual < 1e-6); entries below 1e-8·max are set to 0.

### Systems and symmetries

- **Detection:** every signed permutation g (variable reorderings with sign flips) is tested numerically for
  f(gx) = g f(x) on 200 attractor points (relative tolerance 1e-8). Only systems up to 4-D are scanned, so other
  kinds of symmetry are not looked for.
- **Counts:**
  - 94 autonomous, non-delay systems up to 4-D scanned.
  - **57** have at least one symmetry.
  - **42** of those are fitted: 29 in 3-D and 13 in 4-D. The rest are non-polynomial (12) or have a bad time-step fit (3).
- **Groups among the 42:**
  - Z₂ on 38 systems, mostly (x,y,z) → (−x,−y,z) as in Lorenz.
  - Order-4 groups on 3: ChenLee and Qi (Z₂×Z₂), and Lorenz96 (Z₄ cyclic shift).
  - Z₃ cyclic shift on 1: Halvorsen.
- **Constraint check:** C·vec(Ξ_true) = 0 to about 1e-15 on all 57 symmetric systems, and Θ(Xgᵀ) = Θ(X)M_g to
  1e-16. With exact derivatives the constrained fits recover Ξ_true to machine precision.

## 2. Noise and derivatives

The same four noise levels are used everywhere:

| Level | State | Derivative |
|---|---|---|
| `exact_xdot` | clean | true right-hand side (tests the optimizer only) |
| 0% | clean | 4th-order finite difference |
| 1% | + Gaussian noise, σ = 0.01·std of each channel | `SmoothedFiniteDifference` |
| 5% | + Gaussian noise, σ = 0.05·std of each channel | `SmoothedFiniteDifference` |

The library is evaluated on the **smoothed** trajectory whenever the differentiator produces one, as
`ps.SINDy.fit` does. The hand-written harness originally used the raw noisy state, which handicapped every method in
it until fixed. The noise RNG (seed 0) is used in the same order in both harnesses, so every method sees identical
noisy data.

## 3. Methods

**Thresholds.** Every sparse method is run on a threshold grid **relative to the largest coefficient of the dense
least-squares fit**. That is the raw fit for unscaled methods and the column-scaled fit for scaled ones. The main
grid is {0.003, 0.01, 0.03, 0.1}. The 5-point grid adds 0.3 and was only run for column-scaled methods.

| Family | Variant (name in CSVs) | Description |
|---|---|---|
| Baseline | `STLSQ` | pysindy STLSQ, ridge α = 0.05 with the final unbiased refit (earlier default); `_a0` = α = 0 |
| Augmentation | `STLSQ_aug` | fit on [X; Xg₁ᵀ; …], derivatives transformed the same way |
| v1 constraints | `ConstrainedSTLSQ` (+aug), `ConstrainedSTLSQ_noaug` | (I⊗M_g − g⊗I)vec(Ξ) = 0, rows pruned with QR; fork's `ConstrainedSTLSQ` with a constrained refit |
| v1 dense | `ConstrainedSR3_v1[_noaug]` | `ConstrainedSR3(reg_weight_lam=0)`: constrained least squares with no sparsity, as in `symmetric_sindy_v1` |
| Column scaling | `*_norm` | `normalize_columns=True`: columns divided by their L2 norm, threshold applied to the scaled coefficients |
| Sparse SR3 | `SR3_l0`, `ConstrainedSR3_l0_noaug`, `ConstrainedSR3_l1_noaug` (+`_norm`) | ν = 1, max_iter = 500, `unbias=True` (constrained refit); λ = τ²/2 for l0, λ = τ for l1, so the threshold equals τ |
| Equiv-c | `EquivC`, `EquivC_cc` | see 3.1 |
| SINDy-SI | `SINDySI_{none,sym,sym_eq,paper,full}`, `SI_full` | see 3.2 |

All symmetry-constrained variants enforce the constraints exactly (symmetry residual 0).

### 3.1 Equiv-c (Yang et al. 2024)

The paper builds its constraint from Lie-algebra generators, which a finite group doesn't have. Here it uses the
**group-element form** f(gx) = g f(x) with M_g computed by substitution, which gives the same constraint subspace
as v1.

Other details follow the reference repo's sparse-regression path:
- vec(Ξ) = Qβ, with Q the SVD null-space basis at the repo's 5e-3 singular-value cutoff;
- a thresholding mask on Ξ, refit by least squares on Θ[:,mask]·Q[mask], at most 5 iterations;
- constant terms left free (`EquivC_cc` constrains them too); no ridge.

Note: feeding g directly into the reference code would be wrong. Its M comes from J_Θ(z)·L·z, which matches Θ(gz)
only for the linear terms.

### 3.2 SINDy-SI (Machado & Jones 2023)

Unit-norm library columns; repeated pruning of entries with |w̃| < λ, each followed by a refit under the side
information. "Side information" is extra knowledge imposed as constraints:
- **sym:** the symmetry equalities above.
- **eq:** f̂(x_eq) = 0 at the true equilibria inside the data ball, found by root-finding the true equations.
  The ball is the data mean ± 1.1× the largest distance from it; 40 of 42 systems have at least one equilibrium.
- **mono:** s·∂f̂ᵢ/∂xᵢ ≥ 0 on the ball, enforced as a sum-of-squares certificate. It is kept only where the true
  coefficients pass the same certificate, which happened on all 42 systems.

Solver details:
- With only equalities, the fit is solved directly in the constraint null space.
- With sum-of-squares constraints, it goes through cvxpy (Clarabel, with SCS as fallback). The paper used
  SOSTOOLS and MOSEK.
- ℓ1 weight 0 and degree 3 throughout.

`SINDySI_paper` uses the paper's Lorenz side information (eq + mono, no symmetry). `SI_full` uses all side
information with no pruning (the paper's "SI" column).

## 4. Metrics

Defined in `notebooks/fit_metrics.py`. Each method is scored at its **best threshold by coefficient error**, chosen
using the true answer; this oracle choice applies equally to all methods. All other metrics are read off that
same fit.

| Metric | Definition | Why |
|---|---|---|
| Coefficient error | ‖Ξ − Ξ_true‖_F / ‖Ξ_true‖_F | headline, scale-free |
| Exact terms, strict | estimated term set = true set at \|ξ\| > 1e-10 | Yang et al.'s "success probability"; harsh, since one leftover 1e-11 term fails it |
| Exact terms, relative | same with cutoff \|ξ\| > 1e-3·max\|ξ\| | ignores negligible leftovers |
| Support F1 (relative cutoff) | F1 score of the estimated vs true term sets | partial credit |
| Error on true terms | ‖(Ξ − Ξ_true) on the true term set‖ / ‖Ξ_true‖, over all fits and over exact-term fits only | separates wrong values from wrong terms (the paper's "RMSE on successful runs") |
| Vector-field error | ‖Θ(X)Ξ − Θ(X)Ξ_true‖ / ‖Θ(X)Ξ_true‖ on the clean attractor states | is the dynamics right where the system lives? |
| Valid prediction time (VPT) | RK4 forecast of the learned vs true model from 8 attractor states, until the error exceeds 0.3× the attractor's RMS spread; mean, in Lyapunov times (dysts' largest exponent) | dynamics-level check that still makes sense for chaos |
| **Trajectory RMSE, clean** | simulate the learned model (RK4, 4 substeps per sample) from 32 states spread over the clean trajectory for 1 Lyapunov time (at most 2000 samples), RMSE against the clean data samples that follow, over samples, starts and coordinates, divided by the attractor's RMS spread | direct "does the model reproduce the data" check |
| **Trajectory RMSE, noisy** | the same, but starting from and comparing with the noisy training data | what you can measure without knowing the clean data; also usable to pick thresholds without ground truth |

The VPT forecast stops at min(10 Lyapunov times, 2000 samples); the median cap is 10 and the smallest is 0.94
(systems with long Lyapunov times). Divergent models score the time at which they blow up.

**Trajectory RMSE details.**
- The horizon is 1 Lyapunov time because the systems are chaotic. Over longer runs even a near-perfect model
  drifts off the data and the RMSE settles at the attractor's spread, so it would stop telling methods apart.
- A forecast that blows up within the horizon scores ∞. Medians are unaffected; blow-ups are counted separately.
- **Floor.** The true model scores 0.002% median (0.43% at worst) against the clean data. Against the noisy data it
  scores 9.6% at 1% noise and 38% at 5%: the start is off by the noise, and chaos amplifies that over the Lyapunov
  time. The noisy version therefore has a high floor and separates methods poorly at 5% noise.
- In the data-efficiency sweep (section 6) the windows are too short for these forecasts, so both versions are
  computed on the full trajectory, the noisy one with a fixed noise draw per system and noise level.

**Threshold selection.** By default every method is scored at its best threshold by coefficient error, which needs
the true coefficients. As an alternative that needs no ground truth, section 5.2 picks each method's threshold by
its trajectory RMSE against the noisy training data (`SELECT=traj_nrmse_noisy` in the summary scripts).

**How well the metrics agree.** Spearman rank correlations over all best-threshold fits at 1% and 5% noise:
- coefficient error vs support F1: −0.86
- coefficient error vs vector-field error: 0.74
- coefficient error vs VPT: −0.39
- coefficient error vs trajectory RMSE (clean): **0.45**
- vector-field error vs trajectory RMSE (clean): 0.72
- VPT vs trajectory RMSE (clean): −0.90
- trajectory RMSE, clean vs noisy: 0.91

Coefficient-level and dynamics-level correctness can therefore disagree. Dense fits with many wrong terms can match
the vector field on the attractor closely.

## 5. Results

### 5.1 Main table (thresholds picked by coefficient error)

Medians over the 42 systems, each method at its best threshold by coefficient error on its grid (5 points for the
column-scaled methods and SINDy-SI, 4 otherwise). "Exact" uses the relative cutoff.

| Method | Coef. err 0% | 1% | 5% | Exact 1% | 5% | F1 1% | 5% |
|---|---|---|---|---|---|---|---|
| STLSQ α=0.05 (old default) | 0.31% | 18% | 27% | 0.19 | 0.17 | 0.63 | 0.65 |
| ConstrainedSTLSQ α=0.05 | 0.19% | 14% | 26.7% | 0.19 | 0.14 | 0.67 | 0.62 |
| STLSQ α=0 | 0.021% | 39.9% | 75.2% | 0.19 | 0.19 | 0.47 | 0.54 |
| ConstrainedSTLSQ α=0 | 0.020% | 33.2% | 59.1% | 0.19 | 0.19 | 0.56 | 0.58 |
| Equiv-c | 0.020% | 33.2% | 52.3% | 0.19 | 0.19 | 0.60 | 0.58 |
| v1 dense ConstrainedSR3 | 0.65% | 27.8% | 62.3% | 0 | 0 | 0.36 | 0.34 |
| STLSQ α=0.05, scaled | 0.24% | 11.8% | 16.7% | 0.10 | 0.07 | 0.57 | 0.57 |
| ConstrainedSTLSQ α=0.05, scaled | 0.18% | 9.35% | 15% | 0.12 | 0.10 | 0.60 | 0.58 |
| **STLSQ α=0, scaled** | 0.011% | 1.23% | 2.14% | 0.64 | 0.55 | 1.00 | 1.00 |
| **ConstrainedSTLSQ α=0, scaled** | 0.011% | 1.23% | 2.14% | 0.67 | 0.55 | 1.00 | 1.00 |
| SR3 l0 | 0.030% | 34.1% | 72.8% | 0.17 | 0.07 | 0.46 | 0.40 |
| ConstrainedSR3 l0 | 0.025% | 33.1% | 58.9% | 0.19 | 0.12 | 0.53 | 0.49 |
| ConstrainedSR3 l1 | 0.028% | 6.49% | 7.15% | 0.24 | 0.19 | 0.88 | 0.81 |
| SR3 l0, scaled | 0.011% | 1.6% | 2.39% | 0.55 | 0.48 | 1.00 | 0.95 |
| ConstrainedSR3 l0, scaled | 0.011% | 1.56% | 2.39% | 0.60 | 0.50 | 1.00 | 0.98 |
| ConstrainedSR3 l1, scaled | 0.083% | 3.6% | 6.02% | 0.26 | 0.17 | 0.80 | 0.76 |
| SINDy-SI, no side info | 0.012% | 1.23% | 2.14% | 0.64 | 0.55 | 1.00 | 1.00 |
| SINDy-SI, sym | 0.012% | 1.23% | 2.14% | 0.67 | 0.55 | 1.00 | 1.00 |
| SINDy-SI, eq + mono (paper) | 0.011% | 0.96% | 1.63% | 0.76 | 0.62 | 1.00 | 1.00 |
| **SINDy-SI, sym + eq + mono** | 0.011% | 0.96% | 1.63% | 0.79 | 0.64 | 1.00 | 1.00 |
| SI (all side info, no pruning) | 0.44% | 21.9% | 24% | 0 | 0 | 0.39 | 0.38 |

The same fits, dynamics metrics. Trajectory RMSE is relative to the attractor's spread; its true-model floor is 0.002%
(clean), and 9.6% / 38% (noisy, at 1% / 5% noise). Blow-ups: systems (of 42) whose trajectory-RMSE forecast diverges.

| Method | VF err 1% | 5% | VPT 1% | 5% | Traj RMSE clean 1% | 5% | Traj RMSE noisy 1% | 5% | Blow-ups 1% | 5% |
|---|---|---|---|---|---|---|---|---|---|---|
| STLSQ α=0.05 (old default) | 2.63% | 11.8% | 0.60 | 0.14 | 26.5% | 85.5% | 37.1% | 85.1% | 0 | 2 |
| ConstrainedSTLSQ α=0.05 | 3.26% | 6.87% | 0.73 | 0.17 | 27.6% | 69.7% | 37.1% | 79% | 0 | 1 |
| STLSQ α=0 | 2.09% | 3.58% | 1.08 | 0.70 | 16.6% | 38.7% | 25.4% | 55.4% | 5 | 9 |
| ConstrainedSTLSQ α=0 | 2.24% | 3.38% | 1.07 | 0.65 | 18.9% | 34.3% | 28.1% | 57.5% | 6 | 7 |
| Equiv-c | 3% | 3.15% | 0.75 | 0.69 | 25.4% | 32.4% | 35.9% | 61.6% | 5 | 4 |
| **v1 dense ConstrainedSR3** | 1.58% | 2.14% | 1.86 | 1.55 | **10%** | **15.6%** | 17.6% | 39.3% | 1 | 3 |
| STLSQ α=0.05, scaled | 2.12% | 2.26% | 0.84 | 0.58 | 20.7% | 28.2% | 32.9% | 53.6% | 2 | 1 |
| ConstrainedSTLSQ α=0.05, scaled | 1.85% | 2.21% | 1.22 | 0.51 | 13.1% | 31.6% | 30.4% | 71.5% | 0 | 1 |
| STLSQ α=0, scaled | 0.98% | 1.88% | 1.11 | 0.49 | 11.1% | 46.1% | 22.6% | 72.3% | 2 | 3 |
| ConstrainedSTLSQ α=0, scaled | 0.98% | 2% | 1.12 | 0.49 | 11.1% | 56.2% | 22.6% | 73.8% | 2 | 3 |
| SR3 l0 | 2.26% | 3.65% | 1.14 | 0.66 | 17.2% | 35.9% | 29.2% | 64.9% | 5 | 9 |
| ConstrainedSR3 l0 | 2.26% | 3.65% | 1.18 | 0.51 | 17.4% | 36.1% | 29.3% | 68.8% | 5 | 7 |
| ConstrainedSR3 l1 | 1.69% | 2.2% | 0.80 | 0.45 | 25.6% | 42.6% | 31.5% | 62.2% | 4 | 0 |
| SR3 l0, scaled | 1.51% | 1.97% | 0.73 | 0.57 | 28.4% | 33.8% | 35.7% | 67.4% | 4 | 3 |
| ConstrainedSR3 l0, scaled | 1.4% | 1.72% | 0.73 | 0.57 | 26.5% | 30.3% | 33.4% | 62.4% | 2 | 1 |
| ConstrainedSR3 l1, scaled | 1.09% | 1.59% | 1.27 | 0.59 | 16.3% | 29.4% | 27% | 62.7% | 0 | 1 |
| SINDy-SI, no side info | 0.98% | 1.88% | 1.11 | 0.49 | 11.1% | 46.1% | 22.6% | 72.3% | 2 | 3 |
| SINDy-SI, sym | 0.98% | 2% | 1.12 | 0.49 | 11.1% | 56.2% | 22.6% | 73.8% | 2 | 3 |
| SINDy-SI, eq + mono (paper) | 0.77% | 1.24% | 1.62 | 0.90 | 8.38% | 15.6% | 13.6% | 47.1% | 2 | 4 |
| **SINDy-SI, sym + eq + mono** | 0.77% | 1.24% | 1.62 | 0.90 | **8.37%** | **15.6%** | 13.6% | 47.1% | 1 | 3 |
| SI (all side info, no pruning) | 1.43% | 1.69% | 1.70 | 1.41 | 11% | 17.2% | 17.9% | 38.2% | 0 | 0 |

With exact derivatives every method recovers Ξ_true to between 1e-15 and 1e-12 (about 1e-6 for SI), and its
trajectory RMSE equals the true model's (0.002%).

### 5.2 Thresholds picked by trajectory RMSE on the noisy data (no ground truth)

Same fits, but each method's threshold is the one whose model best reproduces the **noisy training data** over 1
Lyapunov time. Nothing here uses the true coefficients. (Dense methods have no threshold and are unchanged.)

| Method | Coef. err 1% | 5% | Exact 1% | 5% | VPT 1% | 5% | Traj RMSE clean 1% | 5% | Traj RMSE noisy 1% | 5% | Blow-ups 1% | 5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STLSQ α=0.05 (old default) | 23% | 40.7% | 0.10 | 0.10 | 1.09 | 0.81 | 14.7% | 24.3% | 18% | 48% | 0 | 0 |
| ConstrainedSTLSQ α=0.05 | 21.4% | 32.9% | 0.10 | 0.07 | 1.07 | 0.81 | 16.3% | 27.1% | 21% | 48.1% | 0 | 0 |
| STLSQ α=0 | 56.9% | 79.1% | 0.07 | 0.10 | 1.16 | 0.89 | 16.6% | 22.7% | 20.8% | 49.4% | 1 | 2 |
| ConstrainedSTLSQ α=0 | 40.5% | 60.7% | 0.05 | 0.10 | 1.17 | 0.93 | 17.2% | 21.9% | 21.1% | 46.9% | 1 | 2 |
| Equiv-c | 40.5% | 60.7% | 0.05 | 0.10 | 1.17 | 0.93 | 16.6% | 22% | 20.5% | 46.9% | 1 | 1 |
| v1 dense ConstrainedSR3 | 27.8% | 62.3% | 0 | 0 | 1.86 | 1.55 | 10% | 15.6% | 17.6% | 39.3% | 1 | 3 |
| STLSQ α=0.05, scaled | 26.7% | 38.5% | 0 | 0.02 | 1.61 | 1.09 | 9.82% | 16.3% | 16% | 38.9% | 0 | 0 |
| ConstrainedSTLSQ α=0.05, scaled | 17.2% | 36.1% | 0 | 0.05 | 1.61 | 1.16 | 8.24% | 13.6% | 15.7% | 38.9% | 0 | 0 |
| **STLSQ α=0, scaled** | 10.3% | 36.5% | 0.33 | 0.17 | 1.89 | 1.29 | 9.07% | 12.8% | 12.9% | 38.7% | 0 | 0 |
| **ConstrainedSTLSQ α=0, scaled** | 10.3% | 36.7% | 0.31 | 0.19 | 1.85 | 1.33 | 8.91% | 12.8% | 12.8% | 38.7% | 0 | 0 |
| SR3 l0 | 48.8% | 76.8% | 0.02 | 0.05 | 1.32 | 0.91 | 16.6% | 22.6% | 21.4% | 50.5% | 2 | 1 |
| ConstrainedSR3 l0 | 39.8% | 58.9% | 0.05 | 0.07 | 1.19 | 1.07 | 17.3% | 21.4% | 26.1% | 48.9% | 2 | 1 |
| ConstrainedSR3 l1 | 10.6% | 11.2% | 0.10 | 0.10 | 1.21 | 0.89 | 11.3% | 16.5% | 19% | 51.9% | 0 | 0 |
| SR3 l0, scaled | 19.3% | 53.5% | 0.26 | 0.17 | 1.98 | 1.34 | 7.32% | 13.8% | 11.6% | 39.1% | 0 | 1 |
| ConstrainedSR3 l0, scaled | 16.8% | 34.9% | 0.26 | 0.17 | 2.04 | 1.33 | 8.91% | 13.1% | 12.5% | 37.5% | 0 | 0 |
| ConstrainedSR3 l1, scaled | 4.69% | 14.3% | 0.19 | 0.10 | 1.27 | 0.85 | 13.3% | 23.5% | 17.1% | 46.7% | 0 | 0 |
| SINDy-SI, no side info | 10.3% | 36.5% | 0.33 | 0.17 | 1.92 | 1.29 | 9.07% | 12.8% | 12.9% | 38.7% | 0 | 0 |
| SINDy-SI, sym | 10.3% | 36.7% | 0.31 | 0.19 | 1.90 | 1.36 | 8.91% | 12.8% | 12.8% | 38.7% | 0 | 0 |
| SINDy-SI, eq + mono (paper) | 4.2% | 16.4% | 0.43 | 0.26 | 1.84 | 1.43 | 7.61% | 9.3% | 12.4% | 36.5% | 0 | 0 |
| **SINDy-SI, sym + eq + mono** | 3.34% | 13.1% | 0.43 | 0.26 | 1.84 | 1.44 | 7.38% | 13.5% | 12.6% | 37.3% | 0 | 0 |
| SI (all side info, no pruning) | 21.9% | 24% | 0 | 0 | 1.70 | 1.41 | 11% | 17.2% | 17.9% | 38.2% | 0 | 0 |

**The two selections want different models.** For scaled ConstrainedSTLSQ the trajectory-selected threshold is
*smaller* than the coefficient-selected one on 31 of 42 systems at 1% noise and 34 of 42 at 5% (equal on 9 / 6). A
denser model with a few small wrong terms forecasts better over a Lyapunov time than the sparsest correct-looking one.
The coefficient error goes up (1.2% → 10%, 2.1% → 37%), the trajectory RMSE comes down (11% → 8.9%, 56% → 13%), and
blow-ups disappear. At 5% noise the noisy-data RMSE of the best methods (37–39%) is at its true-model floor (38%),
so the selection there is close to a coin toss among reasonable thresholds.

### 5.3 Paired effects

Format: better / same / worse system counts, where "same" means the errors are within 10% of each other, followed by
the geometric-mean error ratio (over systems where both values are finite). A blow-up counts as worse.

Thresholds picked by coefficient error:

| Comparison | Coef. err 1% | 5% | Traj RMSE clean 1% | 5% |
|---|---|---|---|---|
| **Symmetry constraints**, STLSQ α=0.05 | 5/36/1, ×0.91 | 3/36/3, ×0.98 | 4/36/2, ×1.00 | 6/34/2, ×0.95 |
| Symmetry constraints, STLSQ α=0 | 7/34/1, ×0.90 | 7/34/1, ×0.87 | 2/37/3, ×1.07 | 8/31/3, ×1.03 |
| Symmetry constraints, STLSQ α=0 scaled | 1/41/0, ×0.95 | 1/40/1, ×1.00 | 1/41/0, ×0.99 | 1/40/1, ×1.00 |
| Symmetry constraints, SR3 l0 | 8/34/0, ×0.87 | 9/32/1, ×0.78 | 4/38/0, ×0.99 | 7/31/4, ×1.01 |
| Symmetry constraints, SR3 l0 scaled | 3/39/0, ×0.89 | 3/39/0, ×0.86 | 3/39/0, ×0.93 | 3/39/0, ×0.95 |
| Symmetry added to SINDy-SI eq + mono | 2/40/0, ×0.93 | 2/40/0, ×0.93 | 2/40/0, ×0.99 | 2/40/0, ×0.99 |
| **Column scaling + α=0** (constrained) vs old default STLSQ | 31/11/0, ×0.18 | 24/10/8, ×0.44 | 17/12/13, ×0.65 | 15/17/10, ×0.71 |
| **eq + mono side info** vs SINDy-SI with none | 13/25/4, ×0.80 | 17/22/3, ×0.55 | 13/26/3, ×0.77 | 14/21/7, ×0.70 |
| Augmentation vs constraints, scaled | 4/34/4, ×1.02 | 8/28/6, ×0.98 | 3/33/6, ×1.04 | 8/28/6, ×0.86 |
| ConstrainedSR3 l0 scaled vs ConstrainedSTLSQ scaled | 3/33/6, ×1.24 | 6/29/7, ×1.06 | 3/33/6, ×1.26 | 6/30/6, ×0.88 |
| ConstrainedSR3 l1 vs l0, unscaled | 32/9/1, ×0.17 | 33/8/1, ×0.13 | 10/18/14, ×1.13 | 17/12/13, ×0.94 |
| v1 dense vs ConstrainedSTLSQ scaled | 0/0/42, ×21.2 | 0/0/42, ×14.7 | **25/3/14, ×0.74** | **24/3/15, ×0.49** |

Thresholds picked by trajectory RMSE on the noisy data:

| Comparison | Coef. err 1% | 5% | Traj RMSE clean 1% | 5% |
|---|---|---|---|---|
| Symmetry constraints, STLSQ α=0.05 | 7/33/2, ×0.84 | 7/29/6, ×0.89 | 3/36/3, ×1.01 | 5/33/4, ×0.99 |
| Symmetry constraints, STLSQ α=0 | 7/29/6, ×0.98 | 9/32/1, ×0.83 | 4/33/5, ×1.02 | 7/32/3, ×0.92 |
| Symmetry constraints, STLSQ α=0 scaled | 5/32/5, ×0.99 | 8/32/2, ×0.88 | 3/34/5, ×1.00 | 10/29/3, ×0.96 |
| Symmetry constraints, SR3 l0 scaled | 5/37/0, ×0.84 | 11/26/5, ×0.78 | 2/35/5, ×1.10 | 10/26/6, ×0.99 |
| Symmetry added to SINDy-SI eq + mono | 5/37/0, ×0.86 | 11/30/1, ×0.82 | 1/40/1, ×0.99 | 3/33/6, ×1.05 |
| eq + mono side info vs SINDy-SI with none | 22/14/6, ×0.45 | 29/8/5, ×0.35 | 9/22/11, ×0.99 | 19/11/12, ×0.82 |
| Scaled ConstrainedSTLSQ α=0 vs old default STLSQ | 18/13/11, ×0.54 | 14/10/18, ×1.39 | 20/13/9, ×0.64 | 19/15/8, ×0.63 |
| v1 dense vs ConstrainedSTLSQ scaled | 3/12/27, ×3.16 | 1/16/25, ×2.21 | 10/16/16, ×1.07 | 8/17/17, ×1.08 |

**Where symmetry matters most** (coefficient error; thresholds by coefficient error):

| System | Without symmetry | With symmetry | Setting |
|---|---|---|---|
| ChenLee | 145% | 2.1% | ConstrainedSTLSQ α=0 scaled, 5% noise |
| ChenLee | 18.6% | 2.0% | same, 1% noise |
| ChenLee | 20.5% | 2.0% | SINDy-SI eq + mono → + sym, 1% noise |
| LuChenCheng | 41.3% | 3.8% | ConstrainedSTLSQ α=0, 1% noise |
| Qi | 100% | 45% | SINDy-SI eq + mono → + sym, 1% and 5% noise |
| PanXuZhou (**worse**) | 0.82% | 62% (forecast blows up) | ConstrainedSTLSQ α=0 scaled, 5% noise |
| Qi (**worse**) | 100% | 271% | ConstrainedSTLSQ α=0, 5% noise |

PanXuZhou's failure with the symmetry alone goes away once equilibria and monotonicity are added (SINDy-SI, all
side information: 0.80%).

### 5.4 Findings

1. **Fixing `dt` changed the absolute numbers, not the ranking.**
   - Without noise every sparse method is now near 0.01–0.03% coefficient error (was ~0.5%): most of the old floor
     was the time-step bias.
   - With noise, the column-scaled methods' errors roughly *doubled* (1% noise: 0.62% → 1.23%). The smoothing
     differentiator shrinks the derivative, so every fitted coefficient comes out 0.5–4% too small (checked on 8
     systems), and the old, too-small `dt` happened to cancel part of that. The new numbers are the honest ones; the
     remaining error of the best methods is now mostly this uniform shrinkage.
   - The symmetry's median effect shrank accordingly: ×0.95 / ×1.00 for scaled ConstrainedSTLSQ at 1% / 5% noise
     (was ×0.91 / ×0.84). The large per-system gains (ChenLee, LuChenCheng, Qi) remain.
2. **Coefficient accuracy and trajectory accuracy are different targets.**
   - They correlate only weakly across fits (Spearman 0.45). The fit with the best coefficients often forecasts
     badly: scaled ConstrainedSTLSQ has 2.1% coefficient error but 56% trajectory RMSE at 5% noise.
   - Among methods without equilibrium/sign information, the dense v1 fit reproduces the data best (10% / 15.6%
     trajectory RMSE, better than scaled ConstrainedSTLSQ on 25 and 24 of 42 systems) despite 28–62% coefficient
     error: fitting every term matches the vector field where the data lives, even when individual coefficients
     are wrong.
3. **Picking the threshold by trajectory RMSE on the noisy data works for forecasting, without ground truth.**
   Scaled ConstrainedSTLSQ then gives 8.9% / 12.8% trajectory RMSE with no blow-ups. That beats every method under
   coefficient-based selection, except SINDy-SI with all side information at 1% noise (8.4%). It costs coefficient accuracy (10% / 37%) because it prefers denser models.
4. **The symmetry doesn't change trajectory RMSE in aggregate** (×0.93–1.07 in every pairing, under either
   selection). Its value is in coefficient recovery on the few unevenly sampled systems, and in data efficiency
   (section 6).
5. **Symmetry: constraints and augmentation are interchangeable without a ridge or scaling.**
   - With α = 0 and raw columns, constraints only, augmentation only, and both give identical fits. Because g is
     orthogonal, the augmented least-squares loss restricted to symmetric models is |G| × the plain loss.
   - With a ridge or with column scaling they differ slightly (×1.02 / ×0.98 at 1% / 5% noise, scaled).
6. **The ridge must be off when columns are scaled.** With unit-norm columns, α = 0.05 is a strong penalty. It
   distorts which terms survive the threshold, and the final refit can't repair the term set (9–15% vs 1.2–2.1%).
7. **Column scaling explains most of SINDy-SI's gain.** SINDy-SI with no side information equals scaled STLSQ with
   α = 0 to every digit. Thresholding scaled coefficients removes spurious cubic terms whose raw coefficients are
   large but whose contribution to the fit is small.
8. **Sign and equilibrium constraints are the strongest extra information:** ×0.80 / ×0.55 coefficient error and
   ×0.77 / ×0.70 trajectory RMSE at 1% / 5% noise (×0.45 / ×0.35 coefficient error with trajectory-based
   selection). This is near-oracle knowledge, though: it is read off the true equations.
9. **Equiv-c on finite groups reduces to v1 constrained STLSQ with α = 0.** On the same Θ, Ẋ and threshold it
   matches `ConstrainedSTLSQ` to every digit, and freeing the constant terms changes nothing on these systems.
10. **ConstrainedSR3 doesn't beat constrained STLSQ** on coefficients (l0 scaled: ×1.24 / ×1.06). Without scaling,
    l1 is far better than l0 on coefficients (×0.17) but not on trajectories (×1.13 / ×0.94). ν = 1 was not tuned.

### 5.5 Diagrams

`notebooks/plot_constrained_stlsq_dysts.py` draws every system as identified by the recommended setup
(ConstrainedSTLSQ, column-scaled, α = 0, symmetry constraints, no augmentation; thresholds by coefficient error),
into `logs/dysts_traj/figures/constrained_stlsq/`:

- `<system>.png`: one column per noise level (0%, 1%, 5%). Top: the data's attractor (blue) and a 3000-sample free
  run of the identified model from the same state (orange), first three coordinates. Bottom: every coordinate over
  5 Lyapunov times, data vs model, with the 1-Lyapunov-time trajectory-RMSE horizon marked.
- `overview_noise{0,0.01,0.05}.png`: all 42 attractors on one sheet per noise level.

The fit results store metrics, not coefficients, so the script refits each system at its selected threshold with the
harness's exact noise draws (replaying its seed-0 generator) and checks that each refit's coefficient error matches
`fits.csv`. All 126 matched. `SELECT=traj_nrmse_noisy` draws the trajectory-selected models instead.

At 1% noise most attractors are reproduced closely. Clear failures: HyperXu and HyperYangChen blow up; Finance,
LorenzStenflo, LuChen, Qi and Tsucs2 settle onto a fixed point or a much smaller orbit; Bouali2 and NoseHoover end up
on a different, regular orbit. Several with large trajectory RMSE (Lorenz 65%, MooreSpiegel 72%, ChenLee 72%) still
have the right attractor: their forecasts split from the data near a lobe switch within the Lyapunov time.

## 6. How much data is needed

Script: `notebooks/data_efficiency_dysts.py`; summary: `notebooks/summarize_data_efficiency.py`, saved to
`logs/dysts_traj/data_efficiency_summary_<select>.txt`.

### 6.1 Setup

- **Systems, ground truth, side information:** the same 42 systems as above. The true coefficients and SINDy-SI's
  side information come from the full trajectory: they describe the system, not the training data.
- **Training data:** a contiguous window of N samples starting at a random point. N ∈ {50, 100, 200, 400, 800,
  1600, 3200, 6400, 10000}, about 0.5–100 oscillation periods at ~100 samples per period.
- **Repeats:** 5 per N, each with its own window start and noise draw (seeded per system, noise level, N and
  repeat). At N = 10000 only the noise differs.
- **Noise:** 0%, 1% and 5%, scaled to each channel's standard deviation over the **full** clean trajectory, so the
  noise level doesn't depend on the window.
- **Derivatives:** the same as above, computed on the window.
- **Thresholds:** every sparse method uses the same 5-point grid {0.003, 0.01, 0.03, 0.1, 0.3}, relative to the
  window's dense fit (raw or column-scaled to match the method). The best threshold by coefficient error is chosen
  per repeat, using the true answer, as before; section 6.6 repeats the analysis with thresholds picked by trajectory
  RMSE on noisy data instead.
- **Metrics:** VPT is skipped for cost. Trajectory RMSE is computed on the **full** trajectory (the windows are too
  short for 1-Lyapunov-time forecasts), so for short windows it mostly measures how well the model generalizes to
  parts of the attractor it hasn't seen. The noisy version uses one fixed noise draw per system and noise level.
- **Methods:**

| Name | Description |
|---|---|
| `STLSQ_a0.05` | old default: ridge 0.05 with unbias, raw library |
| `ConstrainedSTLSQ_a0.05` | same, with symmetry constraints, no augmentation |
| `STLSQ_scaled` | column scaling, α = 0 |
| `ConstrainedSTLSQ_scaled` | column scaling, α = 0, symmetry constraints (recommended default) |
| `ConstrainedSR3_l0_scaled` | ConstrainedSR3 l0, ν = 1, column scaling, constrained unbias refit |
| `ConstrainedSR3_v1` | v1 dense constrained least squares (λ = 0), no threshold |
| `SINDySI_full` | SINDy-SI with symmetry + equilibria + monotonicity |

**Success criteria for one repeat:**
- *accuracy*: coefficient error < 10%;
- *structure*: exactly the right terms (relative cutoff) **and** coefficient error < 10%;
- *dynamics*: vector-field error on the attractor < 5%;
- *trajectory*: trajectory RMSE against the clean data < 10% of the attractor's spread.

**Reliable.** A method is reliable for a system at N if at least 4 of 5 repeats succeed at N **and at every larger
N**. N_min is the smallest such N; it is infinite if the method never gets there.

No fit failed in this rerun (about 175,000 fits). Four systems with long Lyapunov times (DequanLi, NoseHoover,
SprottA, Tsucs2) were run one noise level per job, because their trajectory-RMSE forecasts are 1,200–2,000 samples long.

### 6.2 Data needed for a reliable fit

Each cell gives the number of systems (of 42) that ever become reliable, then the median N_min over all 42
systems. "More than 10,000" means fewer than half the systems ever get there. Periods are in parentheses.
Thresholds are picked by coefficient error.

**Accuracy criterion** (coefficient error < 10%):

| Method | 0% noise | 1% noise | 5% noise |
|---|---|---|---|
| STLSQ α=0.05 (old default) | 37, 200 (2) | 15, more than 10,000 | 12, more than 10,000 |
| ConstrainedSTLSQ α=0.05 | 38, 200 (2) | 17, more than 10,000 | 13, more than 10,000 |
| STLSQ, scaled | 42, 200 (2) | 34, 400 (4) | 28, 1600 (16) |
| **ConstrainedSTLSQ, scaled** | 42, **100 (1)** | 35, 400 (4) | 29, **1200 (8)** |
| ConstrainedSR3 l0, scaled | 42, 100 (1) | 35, 800 (7) | 27, 1600 (16) |
| v1 dense ConstrainedSR3 | 36, 200 (2) | 11, more than 10,000 | 7, more than 10,000 |
| **SINDy-SI, all side info** | 42, 100 (1) | 36, **200 (2)** | 31, **600 (5)** |

**Structure criterion** (exactly the right terms, and error < 10%):

| Method | 0% noise | 1% noise | 5% noise |
|---|---|---|---|
| STLSQ α=0.05 | 18, more than 10,000 | 8, more than 10,000 | 7, more than 10,000 |
| ConstrainedSTLSQ α=0.05 | 20, more than 10,000 | 8, more than 10,000 | 7, more than 10,000 |
| STLSQ, scaled | 41, 200 | 25, 2400 (16) | 21, more than 10,000 |
| **ConstrainedSTLSQ, scaled** | 41, 100 | 25, 1200 (8) | 21, more than 10,000 |
| ConstrainedSR3 l0, scaled | 40, 200 | 25, 3200 (32) | 18, more than 10,000 |
| v1 dense ConstrainedSR3 | 10, more than 10,000 | 0 | 0 |
| **SINDy-SI, all side info** | 41, 100 | 33, **400 (4)** | 26, **3200 (27)** |

**Dynamics criterion** (vector-field error < 5%):

| Method | 0% noise | 1% noise | 5% noise |
|---|---|---|---|
| STLSQ α=0.05 | 39, 200 | 24, 4800 | 19, more than 10,000 |
| ConstrainedSTLSQ α=0.05 | 40, 200 | 24, 6400 | 19, more than 10,000 |
| STLSQ, scaled | 42, 200 | 31, 800 | 25, 6400 |
| ConstrainedSTLSQ, scaled | 42, 100 | 31, 400 | 24, 4800 |
| ConstrainedSR3 l0, scaled | 42, 100 | 32, 1600 | 25, 8200 |
| **v1 dense ConstrainedSR3** | 42, 200 | **37**, 800 | **37**, 1200 |
| SINDy-SI, all side info | 42, 100 | 36, **200** | 29, **800** |

**Trajectory criterion** (trajectory RMSE vs the clean data < 10%), with thresholds picked by coefficient error →
by trajectory RMSE on the noisy data:

| Method | 0% noise | 1% noise | 5% noise |
|---|---|---|---|
| STLSQ α=0.05 | 35, 400 → 37, 300 | 13 → 18, more than 10,000 | 9 → 15, more than 10,000 |
| ConstrainedSTLSQ α=0.05 | 36, 200 → 37, 200 | 13 → 17, more than 10,000 | 9 → 15, more than 10,000 |
| STLSQ, scaled | 40, 200 → 40, 200 | 20, more than 10,000 → 24, 3200 | 11 → 19, more than 10,000 |
| **ConstrainedSTLSQ, scaled** | 40, 100 → 40, 100 | 19, more than 10,000 → 24, **2400** | 11 → 19, more than 10,000 |
| ConstrainedSR3 l0, scaled | 40, 150 → 40, 150 | 17, more than 10,000 → 24, 4000 | 10 → 18, more than 10,000 |
| v1 dense ConstrainedSR3 | 42, 200 | 20, more than 10,000 | 18, more than 10,000 |
| **SINDy-SI, all side info** | 40, 100 → 40, 100 | 24, 6400 → 25, **2000** | 16 → 20, more than 10,000 |

### 6.3 Success rate and error as data grows

Success rate on the accuracy criterion over all systems and repeats, at 1% noise:

| Method | N=50 | 100 | 200 | 400 | 800 | 1600 | 3200 | 6400 | 10000 |
|---|---|---|---|---|---|---|---|---|---|
| STLSQ α=0.05 | 0.00 | 0.01 | 0.13 | 0.27 | 0.29 | 0.32 | 0.39 | 0.40 | 0.37 |
| ConstrainedSTLSQ α=0.05 | 0.01 | 0.12 | 0.21 | 0.32 | 0.32 | 0.36 | 0.43 | 0.41 | 0.41 |
| STLSQ, scaled | 0.00 | 0.02 | 0.30 | 0.57 | 0.70 | 0.75 | 0.82 | 0.81 | 0.81 |
| ConstrainedSTLSQ, scaled | 0.03 | 0.29 | 0.55 | 0.68 | 0.78 | 0.81 | 0.86 | 0.85 | 0.84 |
| ConstrainedSR3 l0, scaled | 0.00 | 0.10 | 0.32 | 0.54 | 0.67 | 0.74 | 0.80 | 0.82 | 0.84 |
| v1 dense ConstrainedSR3 | 0.00 | 0.00 | 0.02 | 0.10 | 0.18 | 0.23 | 0.25 | 0.28 | 0.29 |
| SINDy-SI, all side info | 0.08 | 0.46 | 0.64 | 0.73 | 0.80 | 0.85 | 0.87 | 0.89 | 0.86 |

Median coefficient error at 1% / 5% noise (thresholds by coefficient error):

| Method | N=200 | 400 | 800 | 1600 | 10000 |
|---|---|---|---|---|---|
| STLSQ α=0.05 | 53% / 82% | 31% / 45% | 27% / 33% | 25% / 27% | 18% / 27% |
| ConstrainedSTLSQ α=0.05 | 42% / 64% | 28% / 37% | 25% / 31% | 23% / 24% | 14% / 23% |
| STLSQ, scaled | 100% / 100% | 5.6% / 60% | 1.9% / 8.7% | 1.8% / 5.5% | 1.2% / 1.9% |
| ConstrainedSTLSQ, scaled | 5.8% / 67% | 2.0% / 8.5% | 1.6% / 4.4% | 1.6% / 2.3% | 1.3% / 1.9% |
| ConstrainedSR3 l0, scaled | 64% / 100% | 7.4% / 30% | 2.1% / 9.7% | 1.9% / 5.4% | 1.6% / 2.2% |
| v1 dense ConstrainedSR3 | 530% / 1880% | 118% / 327% | 65% / 174% | 45% / 86% | 34% / 56% |
| SINDy-SI, all side info | 2.2% / 11% | 1.6% / 3.7% | 1.4% / 2.4% | 1.1% / 1.9% | 0.98% / 1.6% |

A median of 100% means more than half the windows' best fit was the all-zero model.

Median trajectory RMSE against the clean full trajectory at 1% / 5% noise, thresholds picked by trajectory RMSE on
the noisy data (the choice you'd make for forecasting):

| Method | N=200 | 400 | 800 | 1600 | 10000 |
|---|---|---|---|---|---|
| STLSQ α=0.05 | 94% / 110% | 40% / 83% | 29% / 54% | 20% / 31% | 14% / 24% |
| ConstrainedSTLSQ α=0.05 | 68% / 93% | 31% / 66% | 27% / 41% | 20% / 32% | 16% / 25% |
| STLSQ, scaled | 107% / 131% | 25% / 67% | 11% / 32% | 8.9% / 20% | 6.9% / 11% |
| ConstrainedSTLSQ, scaled | 29% / 84% | 10% / 32% | 9.2% / 20% | 8.7% / 16% | 7.9% / 11% |
| ConstrainedSR3 l0, scaled | 41% / 84% | 12% / 34% | 9.1% / 20% | 9.5% / 15% | 7.9% / 12% |
| v1 dense ConstrainedSR3 | ∞ / ∞ | 41% / 163% | 24% / 41% | 14% / 20% | 11% / 16% |
| SINDy-SI, all side info | 14% / 40% | 7.9% / 21% | 7.6% / 16% | 7.6% / 12% | 7.1% / 10% |

∞: more than half the windows' models blow up within the horizon.

### 6.4 Paired effect on the data needed

Geometric-mean ratio of N_min, on systems where both methods become reliable (n = number of such systems). The
less / same / more counts include systems where only one of the two becomes reliable. Thresholds by coefficient
error unless marked.

| Comparison | Criterion | 0% noise | 1% noise | 5% noise |
|---|---|---|---|---|
| **Symmetry**, scaled STLSQ | accuracy | ×0.76 (n=42), 14/28/0 | ×0.57 (n=34), 27/8/0 | ×0.58 (n=28), 16/13/0 |
| | structure | ×0.68 (n=41) | ×0.55 (n=24) | ×0.55 (n=21) |
| | dynamics | ×0.76 (n=42) | ×0.62 (n=30) | ×0.58 (n=24) |
| | trajectory | ×0.72 (n=40) | ×0.61 (n=19) | ×0.50 (n=11) |
| | trajectory, thresholds by trajectory RMSE | ×0.73 (n=40), 15/25/0 | ×0.51 (n=24), 18/6/0 | ×0.61 (n=19), 12/6/1 |
| Symmetry, STLSQ α=0.05 | accuracy | ×0.71 (n=37) | ×0.76 (n=15) | ×0.84 (n=12) |
| Scaled constrained vs old default STLSQ | accuracy | ×0.31 (n=37) | ×0.28 (n=15) | ×0.40 (n=12) |
| | structure | ×0.09 (n=18) | ×0.17 (n=8) | ×0.26 (n=7) |
| | trajectory | ×0.35 (n=35) | ×0.45 (n=12) | ×0.59 (n=8) |
| SINDy-SI vs scaled constrained STLSQ | accuracy | ×1.02 (n=42) | ×0.87 (n=35), 13/20/3 | ×0.65 (n=28), 17/12/3 |
| | trajectory, thresholds by trajectory RMSE | ×1.00 (n=40) | ×0.91 (n=22), 8/16/3 | ×0.51 (n=19), 15/4/1 |
| ConstrainedSR3 l0 vs ConstrainedSTLSQ, both scaled | accuracy | 0/34/8 | 1/9/26 | 2/11/17 |
| v1 dense vs scaled constrained STLSQ | dynamics | ×1.70 (n=42), 1/12/29 | ×1.52 (n=30), 12/4/22 | ×1.50 (n=23), 16/10/12 |
| | dynamics, thresholds by trajectory RMSE | ×1.70 (n=42) | ×2.18 (n=37), 0/3/36 | ×1.99 (n=36), 2/12/23 |
| | trajectory | ×2.14 (n=40), 2/5/35 | ×1.54 (n=16), 7/3/13 | ×2.52 (n=9), 11/0/9 |

### 6.5 Findings

1. **Symmetry pays off most when data is short.**
   - With column scaling it cuts the data needed for a reliable fit to about 0.55–0.6× at 1–5% noise, and never
     increases it on the accuracy criterion.
   - At 200 samples and 1% noise it raises the success rate from 0.30 to 0.55. With the full trajectory the gain is
     only 0.81 → 0.84.
   - **It also helps trajectories here** (×0.50–0.61 the data on the trajectory criterion), unlike with the full
     trajectory (finding 5.4.4). The trajectory RMSE is measured over the whole attractor, so this is
     generalization: a model fit on a short window that covers the attractor unevenly forecasts the unseen parts
     better when the symmetry is built in.
2. **Column scaling with α = 0 is the biggest single factor here too.** About 0.3–0.4× the data of the old default,
   or 0.1–0.26× when exactly the right terms are required. The old default never becomes reliable at 1% noise on 27
   of 42 systems.
3. **Side information adds another ×0.65 at 5% noise** (×0.51 on the trajectory criterion). SINDy-SI reaches 10%
   coefficient error with about 2 periods at 1% noise and 5 periods at 5%. It gets exactly the right terms on 7 of
   the 8 4-D Hyper* systems at 1% noise; no other method manages more than 3.
4. **ConstrainedSR3 needs more data than constrained STLSQ** on most systems (26 of 42 at 1% noise).
5. **The dense v1 fit's dynamics advantage disappears once sparse thresholds are tuned for dynamics.** With
   thresholds by coefficient error it is reliable on the dynamics criterion on 37 of 42 systems at both 1% and 5%
   noise, more than any sparse method. With thresholds by trajectory RMSE, scaled ConstrainedSTLSQ is reliable on 39
   / 36 systems from a median 200 / 600 samples, and the dense fit needs about twice the data (×2.18 / ×1.99).
6. **Reproducing the data over a Lyapunov time is the hardest criterion.** Even with trajectory-based thresholds,
   only 24–25 of 42 systems become reliable at 1% noise (after a median 2000–3200 samples), and fewer than half at
   5% noise with any method.
7. **Some systems never become reliable with any method.** On the accuracy criterion at 1% noise: Finance,
   LorenzStenflo, LuChen, Qi and QiChen; ConstrainedSR3 reaches Sakarya only at 10,000 samples, and SINDy-SI reaches
   HyperYangChen at 1600. On the structure criterion, Lorenz never gets exactly the right terms at 1% noise even with
   the full trajectory. Its error is below 10% from 200–400 samples, but small leftover terms remain.

### 6.6 Picking thresholds by trajectory RMSE instead

`SELECT=traj_nrmse_noisy python notebooks/summarize_data_efficiency.py` (saved as
`logs/dysts_traj/data_efficiency_summary_traj_nrmse_noisy.txt`). With this choice, which needs no ground truth:

- The accuracy and structure criteria get harder for every sparse method (scaled ConstrainedSTLSQ: 20 / 12 systems
  reliable at 1% noise, vs 35 / 25), because the trajectory picks denser models.
- The dynamics and trajectory criteria get easier: scaled ConstrainedSTLSQ is reliable on the dynamics criterion
  on 39 / 36 systems (31 / 24 before) and on the trajectory criterion on 24 / 19 (19 / 11).
- The symmetry still lowers the data needed on every criterion: ×0.40 / ×0.69 accuracy, ×0.55 / ×0.52 dynamics,
  ×0.51 / ×0.61 trajectory, at 1% / 5% noise.

## 7. Giving SINDy-SI's side information to the other methods

Until this section only the SINDy-SI variants used equilibria and monotonicity; every other constrained method used
the symmetry alone. Here the same side information is added to the other constrained methods, with the same
systems, noise draws, derivatives and threshold grids, so each method is paired with its own symmetry-only run.
Runs: `SINDY_SIDE=eq|eq_mono` in `symmetric_sindy_dysts.py`; summary: `notebooks/compare_side_info.py`, saved to
`logs/compare_side_info.txt`.

### 7.1 How the side information is passed

- **Equilibria (`eq`)** become linear equalities, one row per equilibrium and equation:
  θ(x_eq)ᵀ Ξ[:, i] = 0. They are stacked with the symmetry rows and pruned to full rank with pivoted QR, so every
  equality-constrained method can take them: ConstrainedSTLSQ, ConstrainedSR3 (l0, l1, v1 dense) and Equiv-c.
  - Equiv-c is run with its constant terms constrained (`EquivC_cc`). Its default free constants would leave the
    equilibrium constraint unenforced.
- **Monotonicity (`eq_mono`)** becomes sampled linear inequalities: s·∂f̂ᵢ/∂xᵢ(x_k) ≥ 0 at 1000 points drawn
  uniformly in SINDy-SI's ball (seeded per system), one row per point and constrained coordinate. This replaces
  SINDy-SI's sum-of-squares certificate over the whole ball.
  - The true coefficients satisfy every sampled inequality on all 42 systems.
- **Which methods can take the inequalities:**
  - `ConstrainedSTLSQ`: yes, via new `inequality_lhs`/`inequality_rhs` arguments in the fork. When inequalities are
    present, each refit, including the final unbias refit, becomes a small QP solved with cvxpy/Clarabel.
  - `ConstrainedSR3` l1 and the v1 dense fit: yes, through pysindy's existing inequality path. That path is a
    **single** cvxpy solve of the l1-penalized problem, with no prune-and-refit loop and no unbias step.
  - `ConstrainedSR3` l0 and Equiv-c: equalities only, so in `eq_mono` runs they get the equilibria only.
- Only constrained methods are rerun; nothing changes for the unconstrained ones.
- 7 of several thousand `ConstrainedSTLSQ` fits failed because the Clarabel QP didn't converge (on a few systems,
  including RayleighBenard, RikitakeDynamo, ShimizuMorioka and SprottA) and were dropped.

### 7.2 Results

Median coefficient error, best of the 4 thresholds (by coefficient error), at 1% / 5% noise:

| Method | Symmetry only | + equilibria | + equilibria + monotonicity |
|---|---|---|---|
| ConstrainedSTLSQ α=0.05 (old default) | 14% / 26.7% | 13.5% / 26.7% | **9.0% / 15.5%** |
| ConstrainedSTLSQ α=0 | 33.2% / 59.1% | 32.7% / 57.9% | **12.0% / 23.3%** |
| **ConstrainedSTLSQ α=0, scaled** | 1.23% / 2.18% | 1.00% / 2.19% | **0.96% / 1.63%** |
| v1 dense ConstrainedSR3 | 27.8% / 62.3% | 27.5% / 60.4% | **17.0% / 19.0%** |
| Equiv-c* | 33.2% / 59.1% | 32.6% / 57.2% | (equilibria only) |
| ConstrainedSR3 l0* | 33.1% / 58.9% | 32.7% / 59.2% | (equilibria only) |
| ConstrainedSR3 l0, scaled* | 1.92% / 4.16% | 1.84% / 2.39% | (equilibria only) |
| ConstrainedSR3 l1 (inequality path for + mono) | 6.49% / 7.15% | 6.28% / 7.21% | 4.72% / 5.58% |
| ConstrainedSR3 l1, scaled (inequality path for + mono) | 3.6% / 6.02% | 2.68% / 6.16% | 12.2% / 15.3% |
| SINDy-SI (sum-of-squares monotonicity) | 1.23% / 2.18% | 1.00% / 2.19% | 0.96% / 1.63% |

\* Equality constraints only. Equiv-c's symmetry-only entry is ConstrainedSTLSQ α=0, which it matches to every digit.

Median trajectory RMSE against the clean data (same fits), at 1% / 5% noise:

| Method | Symmetry only | + equilibria | + equilibria + monotonicity |
|---|---|---|---|
| ConstrainedSTLSQ α=0.05 | 27.6% / 69.7% | 13.9% / 26.3% | 19.9% / 28.3% |
| ConstrainedSTLSQ α=0 | 18.9% / 34.3% | 13.9% / 25.6% | 17.3% / 25.5% |
| **ConstrainedSTLSQ α=0, scaled** | 10.3% / 35.1% | 10.0% / 30.3% | **8.37% / 15.6%** |
| v1 dense ConstrainedSR3 | 10.0% / 15.6% | 10.6% / 14.4% | 14.1% / 17.6% |
| Equiv-c* | 18.9% / 34.3% | 17.1% / 28.7% | (equilibria only) |
| ConstrainedSR3 l0* | 17.4% / 36.1% | 22.2% / 19.2% | (equilibria only) |
| ConstrainedSR3 l0, scaled* | 20.8% / 28.7% | 11.8% / 27.8% | (equilibria only) |
| ConstrainedSR3 l1 | 25.6% / 42.6% | 21.5% / 28.1% | 12.8% / 16.1% |
| ConstrainedSR3 l1, scaled | 14.2% / 25.4% | 16.3% / 27.9% | 39.4% / 38.9% |
| SINDy-SI | 10.3% / 35.1% | 10.0% / 30.3% | 8.37% / 15.6% |

Other metrics at 5% noise, symmetry only → + equilibria → + equilibria + monotonicity:

| Method | Exact terms | Vector-field error | Forecast (Lyapunov times) |
|---|---|---|---|
| ConstrainedSTLSQ α=0.05 | 0.14 → 0.17 → 0.19 | 6.9% → 2.5% → 2.2% | 0.17 → 0.74 → 0.77 |
| ConstrainedSTLSQ α=0 | 0.19 → 0.19 → 0.24 | 3.4% → 2.6% → 2.3% | 0.65 → 0.95 → 0.79 |
| ConstrainedSTLSQ α=0, scaled | 0.52 → 0.52 → 0.64 | 1.9% → 1.5% → 1.2% | 0.58 → 0.61 → 0.90 |
| v1 dense ConstrainedSR3 | 0 → 0 → 0 | 2.1% → 2.2% → 1.9% | 1.55 → 1.35 → 1.31 |
| SINDy-SI | 0.52 → 0.52 → 0.64 | 1.9% → 1.5% → 1.2% | 0.58 → 0.61 → 0.99 |

**Paired effect** of the added side information vs the same method with symmetry only (better / same / worse
systems, geometric-mean ratio), coefficient error and trajectory RMSE (clean):

| Method | Coef. + eq, 1% | 5% | Coef. + eq + mono, 1% | 5% | Traj + eq, 1% | 5% | Traj + eq + mono, 1% | 5% |
|---|---|---|---|---|---|---|---|---|
| ConstrainedSTLSQ α=0.05 | 5/25/12, ×1.11 | 6/25/11, ×1.13 | 20/14/8, ×0.81 | 19/15/8, ×0.75 | ×0.71 | ×0.62 | ×0.91 | ×0.62 |
| ConstrainedSTLSQ α=0 | 5/33/4, ×1.04 | 3/35/4, ×1.01 | 25/15/2, ×0.47 | 30/10/2, ×0.33 | ×0.87 | ×0.69 | ×0.92 | ×0.75 |
| ConstrainedSTLSQ α=0, scaled | 9/31/2, ×0.93 | 6/33/3, ×0.98 | 15/25/2, ×0.68 | 20/20/2, ×0.32 | ×0.91 | ×0.85 | ×0.83 | ×0.73 |
| v1 dense ConstrainedSR3 | 3/39/0, ×0.96 | 4/38/0, ×0.95 | 32/5/5, ×0.38 | 40/1/1, ×0.19 | ×1.02 | ×1.05 | ×1.50 | ×1.24 |
| ConstrainedSR3 l0, scaled | 8/31/3, ×1.00 | 6/34/2, ×0.93 | (eq only) | (eq only) | ×0.85 | ×0.89 | (eq only) | (eq only) |
| ConstrainedSR3 l1 | 10/26/6, ×0.95 | 10/23/9, ×1.00 | 21/7/14, ×0.89 | 21/6/15, ×0.87 | ×0.88 | ×0.80 | ×0.76 | ×0.66 |
| ConstrainedSR3 l1, scaled | 10/30/2, ×0.93 | 13/28/1, ×0.89 | 4/5/33, ×3.61 | 6/7/29, ×2.29 | ×1.00 | ×1.00 | ×2.79 | ×1.71 |
| SINDy-SI | 9/31/2, ×0.93 | 6/33/3, ×0.98 | 15/25/2, ×0.68 | 20/20/2, ×0.35 | ×0.91 | ×0.85 | ×0.82 | ×0.69 |

**Scaled ConstrainedSTLSQ with all side information vs SINDy-SI:** coefficient error the same on all 42 systems at
1% noise; 3 better, 37 same, 2 worse (×0.90) at 5%. Trajectory RMSE: 0/40/1 (×1.00) and 1/36/2 (×1.03).

### 7.3 Findings

1. **SINDy-SI's advantage was its side information, not its solver.** Column-scaled ConstrainedSTLSQ (α = 0) with
   the same equilibria and a 1000-point sampled monotonicity constraint reproduces SINDy-SI, on coefficients and on
   trajectories. The sum-of-squares certificate and the null-space solver add nothing over ordinary pysindy
   machinery here.
2. **Monotonicity helps every method that enforces it in an exact least-squares or QP solve,** by ×0.19–0.81 in
   coefficient error with noise, better on 19–40 of 42 systems at 5% noise. It helps the dense v1 fit's
   coefficients most (×0.19 at 5% noise), but makes its trajectories *worse* (×1.50 / ×1.24): the sign constraints
   pull the many wrong terms toward the right signs without making the vector field more accurate.
3. **Equilibria alone barely change coefficients** (×0.89–1.13), but they improve the trajectories of most methods
   (×0.62–0.91), most for the ridge fits: the old default at 5% noise goes from 70% to 26% trajectory RMSE and from
   0.17 to 0.74 Lyapunov times of forecast.
4. **Side information doesn't make up for raw columns.** Raw ConstrainedSTLSQ with all of it is still 3–9× worse
   than SINDy-SI on coefficients (×9.4 / ×5.9 with α = 0, ×5.2 / ×3.3 with α = 0.05), and Equiv-c (equilibria only) 17–20× worse. Column scaling remains the biggest
   single factor for coefficients.
5. **ConstrainedSR3's inequality path hurts with column scaling** (×3.61 / ×2.29 on coefficients, ×2.79 / ×1.71 on
   trajectories) and helps a little without it (×0.89 / ×0.87). One l1 solve with no prune-and-refit loop keeps the
   shrinkage and almost never finds exactly the right terms (0–5% of systems with noise). The v1 dense fit via the
   same cvxpy path also loses precision without noise (0.65% → 6.2% at 0% noise), from solver tolerance rather than
   the constraints.

## 8. Bugs fixed and features added in the pysindy fork

All of these are in the `songk42/pysindy` fork (commit `40a6403`, installed at `~/pysindy` on klone). Items 1–4 are bug fixes; item 5 is a new feature.

1. **`ConstrainedSTLSQ` returned ridge-biased coefficients.** It had `unbias=False` hard-coded. Fixed earlier with
   a constrained refit on the final term set.
2. **`ConstrainedSTLSQ(normalize_columns=True)` imposed C·w = d on the scaled coefficients.** Fixed by rescaling C
   by the column norms (helper `constraint_lhs_for_reduce`).
3. **`ConstrainedSR3` had the same scaling bug, and worse.** On the unit test the constraint was violated by 92
   and the error was about 3000%. Same fix.
4. **`ConstrainedSR3` refused `unbias=True` with equality constraints.** l1 results kept their shrinkage, and the
   thresholded `coef_` could violate C·w = d (only `coef_full_` satisfies it). Now `unbias=True` does a constrained
   refit, shared with `ConstrainedSTLSQ` (`constrained_lstsq`). It is still refused with inequality constraints and
   in `TrappingSR3`/`StableLinearSR3`.
5. **`ConstrainedSTLSQ` now accepts inequality constraints** A·w ≤ b (`inequality_lhs`, `inequality_rhs`, same
   ordering and column-norm rescaling as the equalities). With inequalities each refit is a small cvxpy QP
   (Clarabel when installed); without them the equality-only solve is unchanged. Used for the sampled monotonicity
   in section 7.

Tests: 6 new tests for items 1–4. The fork's `test_optimizers.py` gives 249 passed before the changes and 255 after; other test
files are unchanged. This was run in an isolated environment:
- cvxpy was kept from picking Gurobi, which rejects pysindy's solver options;
- the jax-based `SBR` optimizer was skipped because it crashes Python in this environment;
- the notebook tests were skipped because they need extra dependencies.

For item 5, a further test (an inequality that is active at the optimum, with and without column scaling) passes,
along with all the constrained-optimizer tests above. That was run with the project venv only; the full fork suite
was not rerun after item 5.

**Known precision limit.** The constrained least-squares solve forms XᵀX, giving about 1e-10 to 1e-13 accuracy
instead of 1e-15 with exact derivatives. This is irrelevant at any realistic noise level. The SINDy-SI null-space
solver avoids it.

## 9. Caveats

- **Thresholds are chosen using the true answer** in every table except section 5.2. Absolute numbers there are
  optimistic; comparisons are fair. Section 5.2 picks thresholds by trajectory RMSE on the training data instead,
  which needs no ground truth but is in-sample (the forecasts start from the same data the model was fit on).
- **One trajectory per system**, about 100 samples per period. Sections 1–5 and 7 use one noise draw per system;
  section 6 uses 5 windows and noise draws per data amount.
- **Several systems fail for every method,** mainly LuChen, LorenzStenflo, VallisElNino, Finance, Qi and the Hyper*
  systems. That is a derivative-accuracy problem, not a symmetry one; all of them are fine with exact derivatives.
- **The derivative estimate limits accuracy at 1–5% noise.** `SmoothedFiniteDifference` shrinks derivatives, so the
  best fits' coefficients come out uniformly 0.5–4% too small. A better differentiator would lower every method's
  error with noise.
- **Trajectory RMSE details.** One Lyapunov time from 32 starts; ∞ on blow-up, so only medians and counts are
  reported. Against noisy data its floor (9.6% / 38% at 1% / 5% noise) is close to what the best methods reach at 5%
  noise.
- **Side information comes from the true equations,** for the symmetries as well as the equilibria and monotonicity.
- **VPT is noisy per system** (8 starting points) and is censored on systems with long Lyapunov times.
- **The detection only looks for signed permutations;** continuous symmetries are out of scope.
- **Data-efficiency details.**
  - The 4-of-5 rule and the 10% / 5% / 10% cutoffs are arbitrary and were not varied. The three criteria already rank
    the dense v1 fit very differently (worst on coefficients, best on vector-field error at 5% noise).
  - Short windows still use true coefficients, equilibria and the monotonicity ball taken from the full trajectory.
  - Data amounts are on a doubling grid, so N_min is only resolved to a factor of 2.
- **Side-information details (section 7).**
  - Monotonicity is only checked at 1000 sampled points, so a fit can violate it between them. SINDy-SI's
    sum-of-squares version covers the whole ball.
  - The data-efficiency sweep (section 6) still gives equilibria and monotonicity only to SINDy-SI.

## 10. Reproduction

On klone, everything runs on `ckpt-all` (account `stf`) through `run_dysts_klone.slurm`, which takes the script name
and optional system names and writes to `logs/dysts_traj/`. `./submit_dysts_traj.sh` submits the whole set with the
right dependencies (SINDy-SI first, since the side-info runs and the data-efficiency sweep read its
`side_info.csv`). The engaging versions (`run_*_engaging.slurm`, `run_sindy_si_dysts.slurm`,
`run_data_efficiency_dysts.slurm`) take the same environment variables.

| What | Command | Output (under `logs/dysts_traj/`) |
|---|---|---|
| STLSQ family + v1 | `sbatch --export=ALL,SINDY_ALPHA={0,0.05},SINDY_NORMALIZE={0,1} run_dysts_klone.slurm symmetric_sindy_dysts.py` | `symmetric_sindy_dysts/alpha{a}[_norm]/` |
| Sparse SR3 | `sbatch --export=ALL,SINDY_METHODS=sr3,SINDY_NORMALIZE={0,1} run_dysts_klone.slurm symmetric_sindy_dysts.py` | `symmetric_sindy_dysts/sr3[_norm]/` |
| SINDy-SI + Equiv-c | `sbatch run_dysts_klone.slurm sindy_si_dysts.py` | `sindy_si_dysts/{fits,side_info}.csv` |
| Side info for other methods | the two commands above with `SINDY_SIDE={eq,eq_mono}` added (needs `side_info.csv`) | `symmetric_sindy_dysts/<run>_{eq,eq_mono}/` |
| Data-efficiency sweep | `sbatch --array=0-41 -c 2 --mem=4G --time=4:00:00 run_dysts_klone.slurm data_efficiency_dysts.py` (needs `side_info.csv`) | `data_efficiency/fits_<system>.csv` |
| …slow systems | DequanLi, NoseHoover, SprottA and Tsucs2 need ~3 h per noise level (long Lyapunov times make the trajectory-RMSE forecasts long): `sbatch --time=10:00:00 --export=ALL,DE_NOISE=<level> run_dysts_klone.slurm data_efficiency_dysts.py <system>` | `data_efficiency/fits_<system>_noise<level>.csv` |
| Weak form x orbits (section 11) | `./submit_weak_orbit.sh` (two 168-task arrays, one task per system x noise; needs `alpha0_norm` and `data_efficiency` for the pairing checks) | `weak_orbit_{dysts,data_eff}/m25_p8/`; tables: `SELECT=... python notebooks/summarize_weak_orbit.py` → `weak_orbit_summary_<select>.txt` |
| Summary tables | `[SELECT=traj_nrmse_noisy] python notebooks/compare_sindy_si.py`, `compare_side_info.py`, `summarize_data_efficiency.py` (with `LOG_DIR=logs/dysts_traj`) | `compare_sindy_si_<select>.txt`, `compare_side_info_<select>.txt`, `data_efficiency_summary_<select>.txt` |
| Diagrams | `python notebooks/plot_constrained_stlsq_dysts.py` (with `PYTHONPATH=$HOME/dysts`) | `figures/constrained_stlsq/` |

Each full-data run takes 2–4 hours with 4 CPUs, mostly spent on the forecasts. Passing system names as arguments
restricts a run (for example `... symmetric_sindy_dysts.py Lorenz ChenLee`). The pre-rerun version of this note is
saved as `logs/dysts_traj/checks/symmetric_sindy_dysts.before_traj.md`.

| File | Contents |
|---|---|
| `notebooks/symmetric_sindy_dysts.py` | symmetry detection, M_g and C construction, STLSQ / SR3 families |
| `notebooks/sindy_si_dysts.py` | SINDy-SI and Equiv-c harness, side-information extraction |
| `notebooks/sindy_si.py` | SINDy-SI estimator (null-space and sum-of-squares solvers) |
| `notebooks/equiv_c.py` | Equiv-c estimator |
| `notebooks/fit_metrics.py` | all metrics, including trajectory RMSE |
| `notebooks/compare_sindy_si.py` | aggregation and tables |
| `notebooks/dysts_common.py` | shared system loading and symmetry construction (used by the data-efficiency sweep) |
| `notebooks/data_efficiency_dysts.py` | data-efficiency sweep |
| `notebooks/summarize_data_efficiency.py` | data-efficiency aggregation and tables |
| `notebooks/compare_side_info.py` | side-info (section 7) aggregation and tables |
| `notebooks/plot_constrained_stlsq_dysts.py` | per-system diagrams (section 5.5) |
| `notebooks/weak_orbit_dysts.py` | weak form x orbit thresholding, full data (section 11) |
| `notebooks/weak_orbit_data_eff.py` | weak form x orbit thresholding, data efficiency (section 11) |
| `notebooks/summarize_weak_orbit.py` | section 11 tables (`SELECT` as above) |

## 11. Weak form and orbit thresholding

Two changes were tested on top of the recommended setup (column-scaled, α = 0, symmetry constraints). The weak form
targets the error that symmetry can't remove. Orbit thresholding was meant to use the symmetry more fully.

- **Weak (integral) form:** there are no derivatives of the noisy data. For test functions φₖ(t) = (1 − s²)⁸ on
  windows of 2m+1 samples (m = 25, about half a period; centres every m/4 samples), integration by parts turns
  ∫φₖẋ dt = ∫φₖΘ(x) dt Ξ into −V′X = (VΘ(X)) Ξ. Short windows shrink m to (N−1)/4. m = 25 was fixed before a
  small check on 5 of the 42 systems (m ∈ {8, 16, 25, 50}). 25 was the best or within noise of the best there;
  50 fails at 5–10% noise.
- **Orbit thresholding:** the symmetry is built into the basis instead of imposed as constraints. With vec(Ξ) = Bv,
  each column of B is the signed indicator of one orbit of the group acting on (term, equation) pairs: the
  group-average of a unit coefficient. B spans exactly null(C), which is checked for every system. STLSQ on
  (I ⊗ Θ)B then keeps or drops whole orbits by their column-scaled contribution.

Both are crossed with {no symmetry, coefficient-wise constraints (ConstrainedSTLSQ), orbits}. All runs use the
noise draws (full data) and windows (data efficiency) of the earlier runs, plus a new 10% noise level. Pairing
check: the strong-form STLSQ/ConstrainedSTLSQ rows reproduce `alpha0_norm` and `data_efficiency` exactly (all 252
data-efficiency checks; 124 of 126 full-data checks). The exception is LorenzStenflo, whose regression is so
ill-conditioned that floating-point differences change the mid-threshold supports; its best-threshold values agree.

### 11.1 Full data (42 systems, median at the oracle threshold)

| | coef err 1% | 5% | 10% | traj RMSE 1% | 5% | 10% | exact terms 1% | 5% | 10% |
|---|---|---|---|---|---|---|---|---|---|
| strong, no symmetry | 1.23% | 2.14% | 9.65% | 11.1% | 46.1% | 59.4% | 0.64 | 0.55 | 0.40 |
| strong, ConstrainedSTLSQ | 1.23% | 2.14% | 4.99% | 11.1% | 56.2% | 55.6% | 0.67 | 0.55 | 0.45 |
| strong, orbits | 1.23% | 2.18% | 3.91% | 11.1% | 56.2% | 55.6% | 0.64 | 0.52 | 0.45 |
| weak, no symmetry | 0.032% | 0.23% | 0.97% | 0.35% | 5.1% | 17.2% | 0.86 | 0.67 | 0.64 |
| **weak, ConstrainedSTLSQ** | **0.030%** | **0.23%** | **0.85%** | **0.26%** | **4.6%** | **16.3%** | **0.88** | **0.71** | **0.67** |
| weak, orbits | 0.030% | 0.23% | 0.96% | 0.31% | 4.6% | 17.3% | 0.86 | 0.71 | 0.64 |

- **The weak form is the big win.** Paired against the strong form with the same symmetry handling, coefficient
  error drops ×0.02–0.03 at 1% noise (better on 39 of 42 systems), ×0.20 at 5% and ×0.39 at 10%. Trajectory RMSE
  over one Lyapunov time drops from 11% to 0.26% at 1% noise and from 56% to 4.6% at 5%. Clean data gains too
  (×0.02): the trapezoid rule on smooth test functions beats 4th-order finite differences.
- **Where it loses:** at 10% noise it is worse on 8–10 systems: HyperBao, HyperPang, HyperYangChen, LuChen,
  RayleighBenard, QiChen, LorenzStenflo and HyperCai. These are systems where every method fails. The strong form
  scores better there only because its best threshold is the zero model (100%), while the weak form returns a wrong
  model. Weak fits blow up more often at the oracle threshold (10% of systems at 5% noise, 17% at 10%, against
  7% and 0%). Under trajectory selection, blow-ups fall to 0–5%.
- **Orbits make no difference on full data:** they give the same result as coefficient-wise constraints on 39–41
  of 42 systems at every noise level. A full trajectory covers the attractor symmetrically, so tied terms already
  have equal column norms and coefficient-wise thresholding drops them together.
- **Symmetry counts for more with realistic threshold selection.** Picking thresholds by trajectory RMSE on the
  noisy data (no ground truth), median coefficient error at 5% noise is:
  - weak, no symmetry: 16.8%
  - weak, ConstrainedSTLSQ: 7.8%
  - weak, orbits: 2.9%
  - strong: 37% with or without symmetry

  At 10% noise the weak-form numbers are 26.8%, 7.2% and 8.2%. Trajectory RMSE under this selection is 0.31% /
  3.0% / 9.7% for weak ConstrainedSTLSQ, against 8.9% / 12.8% / 22% for the strong form (1/5/10% noise).

### 11.2 Data efficiency (same windows as section 6; oracle threshold)

Systems that become reliable (≥ 4 of 5 repeats succeed at N and above), and median N_min over those systems:

| | structure 1% | 5% | 10% | trajectory 1% | 5% | 10% |
|---|---|---|---|---|---|---|
| strong, no symmetry | 25 / 2400 | 21 / >max | 16 / >max | 20 / >max | 11 / >max | 8 / >max |
| strong, ConstrainedSTLSQ | 25 / 1200 | 21 / >max | 17 / >max | 19 / >max | 11 / >max | 9 / >max |
| strong, orbits | 24 / 1600 | 20 / >max | 17 / >max | 19 / >max | 10 / >max | 9 / >max |
| weak, no symmetry | 35 / 800 | 28 / 1600 | 24 / 8200 | 33 / 800 | 22 / 10000 | 15 / >max |
| **weak, ConstrainedSTLSQ** | **36 / 400** | **28 / 1200** | **28 / 4800** | **34 / 800** | **22 / 10000** | **17 / >max** |
| weak, orbits | 35 / 600 | 28 / 1600 | 28 / 3200 | 33 / 800 | 22 / 10000 | 17 / >max |

- **The weak form mostly raises the ceiling.** More systems become reliable at all: structure at 1% noise goes
  from 25 to 36 of 42, and trajectory goes from 19 to 34 at 1% and from 11 to 22 at 5%. Where both forms
  eventually succeed, the median N_min is about the same (×1.00).
- **On clean, very short windows the weak form is worse.** At N ≤ 100 the shrunken test functions leave too few
  equations. Without symmetry it needs 2× the data of the strong form for the dynamics criterion at noise 0.
  Symmetry fixes this: weak ConstrainedSTLSQ is at 2.3% (N = 50) and 0.04% (N = 100), where weak STLSQ is at 100%
  and 95%.
- **Symmetry still halves the data needed with the weak form:** ×0.5 median paired N_min against weak STLSQ under
  most criteria and noise levels. For accuracy that's 26/15/1 systems at noise 0, 21/10/1 at 5% and 19/9/1 at 10%.
- **Orbits help only on clean, very short windows.** With the strong form at noise 0 they need half the data of
  coefficient-wise constraints (33 of 42 systems need less; median coefficient error at N = 50 is 0.07% against
  100%). There a 50-sample window covers the attractor unevenly, so tied terms get unequal column norms and
  coefficient-wise thresholding drops one and keeps its partner. With any noise, or with the weak form, orbits are
  within noise of coefficient-wise constraints.
- **The same picture holds under trajectory-based threshold selection.** For structure at 1% noise, weak
  ConstrainedSTLSQ makes 30 systems reliable (N_min 3200), against 12 for the strong form.

### 11.3 Recommendation

Use **weak-form ConstrainedSTLSQ, column-scaled** as the default. It is the best or tied-best entry in almost every
table above and costs nothing extra: the weak matrices are smaller than the strong ones. Orbit thresholding isn't
worth making the default. It only matters for clean data on windows shorter than about a period, though it's a
cheap change to keep for that regime. The weak form widens the set of systems that can be identified, but not
the per-system data requirement. The remaining failures (Hyper*, LuChen, RayleighBenard, Qi at ≥5% noise) look
like library ill-conditioning, not noise. Ensembling or trajectory-loss refinement (the next items on the list)
are better aimed at those.
