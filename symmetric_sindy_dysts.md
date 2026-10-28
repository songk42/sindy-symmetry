# Symmetry-aware SINDy on the dysts benchmark

Do discrete symmetries help sparse identification of chaotic ODEs, and which way of using them works best?
This note covers every setup run so far (Oct 2026): symmetry constraints and augmentation (v1), SINDy-SI
(Machado & Jones, arXiv:2310.04227), Equiv-c (Yang et al., arXiv:2405.16756), column scaling, and sparse
ConstrainedSR3. All methods see the same systems, the same noise draws and the same derivative estimates, so
results pair system by system. Section 6 adds a data-efficiency sweep: how much training data each method needs
before its fits become reliable. Section 7 gives SINDy-SI's extra side information to the other constrained methods.

**Short answer.** The symmetry is cheap insurance. It rarely hurts, gives about a 10–20% lower typical error with
noise, and gives very large gains on the few systems whose trajectory covers the attractor unevenly (ChenLee,
LuChenCheng, Qi). Its biggest benefit is data efficiency: with column scaling it reaches a reliable fit with about
half the data (0.57–0.59× at 1–5% noise). Two other choices matter more:

- **Column scaling with no ridge** cuts the error 5–30× with noise, and the data needed to about 0.3×.
- **Extra physical side information** (known equilibria, sign of ∂fᵢ/∂xᵢ) gives another 2–3× lower error at 5%
  noise, and needs about 0.6× the data. Almost all of that comes from the sign constraints. Given to column-scaled
  ConstrainedSTLSQ (equilibria as equalities, signs as sampled inequalities), it matches SINDy-SI: identical on all
  42 systems at 1% noise, within 10% on 37 of 42 at 5%.

The recommended default is `ConstrainedSTLSQ(normalize_columns=True, alpha=0)` with the symmetry constraints and no
augmentation, plus equilibria and sampled sign constraints when they're known.

---

## 1. Data and ground truth

- **Source:** the `fine/` split of the HuggingFace dataset `williamgilpin/dysts`, stored locally at
  `/home/songk/data-songk/dysts_hf/fine`. Each system is one noise-free trajectory of 10,000 samples, at about 100
  samples per period of the main oscillation.
- **Time step:** the CSV time column is a sample index. The physical step `dt` is recovered by regressing the
  finite-difference derivative on the true dysts right-hand side; systems with a fit R² < 0.99 are skipped.
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

The VPT forecast stops at min(10 Lyapunov times, 2000 samples); the median cap is 10 and the smallest is 0.93
(systems with long Lyapunov times). Divergent models score the time at which they blow up.

**How well the metrics agree.** Spearman rank correlations over all fits at 1% and 5% noise:
- coefficient error vs support F1: −0.87
- coefficient error vs vector-field error: 0.77
- coefficient error vs VPT: −0.44
- vector-field error vs VPT: −0.75

Coefficient-level and dynamics-level correctness can therefore disagree. Dense fits with many wrong terms can match
the vector field on the attractor closely.

## 5. Results

### 5.1 Main table

Medians over the 42 systems, best of the 4 thresholds. "Exact" uses the relative cutoff.

| Method | Coef. err 0% | 1% | 5% | Exact 1% | 5% | F1 1% | 5% | VF err 1% | 5% | VPT 1% | 5% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| STLSQ α=0.05 (old default) | 0.89% | 18.4% | 27.1% | 0.19 | 0.17 | 0.63 | 0.65 | 2.5% | 11.6% | 0.68 | 0.18 |
| ConstrainedSTLSQ α=0.05 | 0.89% | 13.9% | 26.5% | 0.19 | 0.14 | 0.67 | 0.62 | 3.1% | 6.5% | 0.82 | 0.19 |
| STLSQ α=0 | 0.58% | 40.1% | 76% | 0.19 | 0.19 | 0.50 | 0.54 | 1.9% | 3.5% | 1.14 | 0.73 |
| ConstrainedSTLSQ α=0 | 0.57% | 32.7% | 59.6% | 0.19 | 0.19 | 0.60 | 0.58 | 2.1% | 3.2% | 1.17 | 0.66 |
| Equiv-c | 0.57% | 32.7% | 59.6% | 0.19 | 0.19 | 0.60 | 0.58 | 2.1% | 3.0% | 0.97 | 0.66 |
| v1 dense ConstrainedSR3 | 1.43% | 28.1% | 62.5% | 0 | 0 | 0.36 | 0.34 | 1.5% | 2.0% | 1.85 | 1.56 |
| STLSQ α=0.05, scaled | 0.95% | 12.2% | 17.3% | 0.10 | 0.07 | 0.55 | 0.57 | 1.7% | 2.0% | 1.15 | 0.92 |
| ConstrainedSTLSQ α=0.05, scaled | 0.70% | 9.5% | 15.0% | 0.10 | 0.07 | 0.57 | 0.57 | 1.4% | 1.8% | 1.25 | 0.91 |
| **STLSQ α=0, scaled** | 0.49% | 0.62% | 1.97% | 0.64 | 0.50 | 1.00 | 0.98 | 0.53% | 1.5% | 1.65 | 0.72 |
| **ConstrainedSTLSQ α=0, scaled** | 0.49% | 0.61% | 1.81% | 0.67 | 0.52 | 1.00 | 1.00 | 0.53% | 1.1% | 1.65 | 0.72 |
| SR3 l0 | 0.63% | 34.7% | 73.1% | 0.17 | 0.07 | 0.46 | 0.40 | 2.1% | 3.5% | 1.28 | 0.68 |
| ConstrainedSR3 l0 | 0.59% | 32.6% | 59.3% | 0.19 | 0.12 | 0.56 | 0.49 | 2.1% | 3.5% | 1.31 | 0.51 |
| ConstrainedSR3 l1 | 0.58% | 6.3% | 6.9% | 0.24 | 0.19 | 0.88 | 0.81 | 1.5% | 1.9% | 0.93 | 0.48 |
| SR3 l0, scaled | 0.51% | 1.70% | 4.06% | 0.48 | 0.43 | 0.95 | 0.91 | 0.97% | 1.4% | 1.02 | 0.80 |
| ConstrainedSR3 l0, scaled | 0.49% | 1.44% | 3.02% | 0.52 | 0.45 | 1.00 | 0.94 | 0.95% | 1.1% | 1.15 | 0.80 |
| ConstrainedSR3 l1, scaled | 0.65% | 3.27% | 5.78% | 0.26 | 0.19 | 0.80 | 0.79 | 0.82% | 1.3% | 1.43 | 0.75 |
| SINDy-SI, no side info | 0.51% | 0.62% | 1.97% | 0.64 | 0.50 | 1.00 | 0.98 | 0.53% | 1.5% | 1.65 | 0.72 |
| SINDy-SI, sym | 0.51% | 0.61% | 1.81% | 0.67 | 0.52 | 1.00 | 1.00 | 0.53% | 1.1% | 1.65 | 0.72 |
| SINDy-SI, eq + mono (paper) | 0.51% | 0.53% | 1.16% | 0.76 | 0.62 | 1.00 | 1.00 | 0.43% | 0.81% | 2.08 | 1.12 |
| **SINDy-SI, sym + eq + mono** | 0.51% | 0.53% | 1.09% | 0.79 | 0.64 | 1.00 | 1.00 | 0.43% | 0.81% | 2.08 | 1.12 |
| SI (all side info, no pruning) | 1.24% | 22.1% | 24.1% | 0 | 0 | 0.39 | 0.38 | 1.2% | 1.6% | 1.77 | 1.43 |

With exact derivatives every method recovers Ξ_true to between 1e-15 and 1e-12 (about 1e-6 for SI), and nearly all forecast
to the full cap.

### 5.2 Paired effects

Format: better / same / worse system counts, where "same" means the errors are within 10% of each other, followed by
the geometric-mean error ratio, at 1% and 5% noise.

| Comparison | 1% | 5% |
|---|---|---|
| **Symmetry constraints**, STLSQ α=0.05 | 5/36/1, ×0.90 | 3/36/3, ×0.98 |
| Symmetry constraints, STLSQ α=0 | 7/34/1, ×0.89 | 7/34/1, ×0.87 |
| Symmetry constraints, STLSQ α=0 scaled | 2/40/0, ×0.91 | 2/39/1, ×0.84 |
| Symmetry constraints, SR3 l0 | 8/34/0, ×0.86 | 9/32/1, ×0.77 |
| Symmetry constraints, SR3 l0 scaled | 3/39/0, ×0.84 | 3/38/1, ×0.81 |
| Symmetry added to SINDy-SI eq + mono | 2/40/0, ×0.91 | 2/40/0, ×0.91 |
| **Column scaling + α=0** vs old default STLSQ | 29/10/3, ×0.16 | 20/8/14, ×0.62 |
| Same, 5-point grid | 30/12/0, ×0.14 | 24/10/8, ×0.35 |
| **eq + mono side info** vs SINDy-SI with none | 16/24/2, ×0.61 | 21/19/2, ×0.30 |
| Augmentation vs constraints, scaled | 4/31/7, ×1.31 | 7/25/10, ×1.05 |
| ConstrainedSR3 l0 scaled vs ConstrainedSTLSQ scaled | 3/31/8, ×1.64 | 4/30/8, ×1.15 |
| ConstrainedSR3 l1 vs l0, unscaled | 32/8/2, ×0.16 | 33/8/1, ×0.12 |

**Where symmetry matters most.** These systems' trajectories cover the attractor unevenly:

| System | Without symmetry | With symmetry | Setting |
|---|---|---|---|
| ChenLee | 272% | 0.9% | 5% noise, SINDy-SI |
| ChenLee | 19% | 3% | STLSQ |
| LuChenCheng | 42% | 4% | STLSQ, 1% noise, α=0 |
| LuChenCheng | 7.9% | 0.6% | ConstrainedSR3 scaled, 1% noise |
| Qi | 1328% | 100% | 5% noise, SINDy-SI |

Exception: PanXuZhou at 5% noise is worse with symmetry alone (154% → 348%), and becomes 0.4% once monotonicity is
added.

### 5.3 Findings

1. **Symmetry: constraints and augmentation are interchangeable without a ridge or scaling.**
   - With α = 0 and raw columns, constraints only, augmentation only, and both give identical fits. Because g is
     orthogonal, the augmented least-squares loss restricted to symmetric models is |G| × the plain loss.
   - With a ridge or with column scaling they differ, and constraints without augmentation are better: the ridge
     penalty doesn't grow with the extra |G|× data, and the column norms change under augmentation.
2. **The ridge must be off when columns are scaled.** With unit-norm columns, α = 0.05 is a strong penalty. It
   distorts which terms survive the threshold, and the final refit can't repair the term set.
3. **Column scaling explains most of SINDy-SI's gain.** SINDy-SI with no side information equals scaled STLSQ with
   α = 0 to every digit. Thresholding scaled coefficients removes spurious cubic terms whose raw coefficients are
   large but whose contribution to the fit is small.
4. **Sign and equilibrium constraints are the strongest extra information,** about 2–3× at 5% noise. This is near
   oracle knowledge, though: it is read off the true equations.
5. **Equiv-c on finite groups reduces to v1 constrained STLSQ with α = 0.** On the same Θ, Ẋ and threshold it
   matches `ConstrainedSTLSQ` to every digit, and freeing the constant terms changes nothing on these systems.
6. **ConstrainedSR3 doesn't beat constrained STLSQ.**
   - The best SR3 option is l0 with column scaling and `unbias=True`.
   - Without scaling, l1 is far better than l0.
   - ν = 1 was not tuned.
7. **Dense fits forecast surprisingly well.** v1 dense SR3 and SI have 25–60% coefficient error at noise but the
   longest forecasts among methods without sign/equilibrium information. Fitting every term matches the vector
   field on the attractor even when individual coefficients are wrong.

## 6. How much data is needed

Script: `notebooks/data_efficiency_dysts.py`; summary: `notebooks/summarize_data_efficiency.py`, saved to
`logs/data_efficiency_summary.txt`.

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
  per repeat, using the true answer, as before.
- **Metrics:** forecasts are skipped for cost.
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
- *dynamics*: vector-field error on the attractor < 5%.

**Reliable.** A method is reliable for a system at N if at least 4 of 5 repeats succeed at N **and at every larger
N**. N_min is the smallest such N; it is infinite if the method never gets there.

One SINDy-SI fit out of about 100,000 failed (solver didn't converge) and was dropped.

### 6.2 Data needed for a reliable fit

Each cell gives the number of systems (of 42) that ever become reliable, then the median N_min over all 42
systems. "More than 10,000" means fewer than half the systems ever get there. Periods are in parentheses.

**Accuracy criterion** (coefficient error < 10%):

| Method | 0% noise | 1% noise | 5% noise |
|---|---|---|---|
| STLSQ α=0.05 (old default) | 37, 200 (2) | 15, more than 10,000 | 12, more than 10,000 |
| ConstrainedSTLSQ α=0.05 | 38, 200 (2) | 17, more than 10,000 | 13, more than 10,000 |
| STLSQ, scaled | 42, 200 (2) | 34, 400 (4) | 28, 1600 (16) |
| **ConstrainedSTLSQ, scaled** | 42, **100 (1)** | 35, 400 (4) | 29, **800 (8)** |
| ConstrainedSR3 l0, scaled | 42, 100 (1) | 35, 800 (8) | 27, 1600 (16) |
| v1 dense ConstrainedSR3 | 36, 200 (2) | 11, more than 10,000 | 7, more than 10,000 |
| **SINDy-SI, all side info** | 42, 100 (1) | 36, **200 (2)** | 31, **400 (4)** |

**Structure criterion** (exactly the right terms, and error < 10%):

| Method | 0% noise | 1% noise | 5% noise |
|---|---|---|---|
| STLSQ α=0.05 | 15, more than 10,000 | 8, more than 10,000 | 7, more than 10,000 |
| ConstrainedSTLSQ α=0.05 | 17, more than 10,000 | 8, more than 10,000 | 7, more than 10,000 |
| STLSQ, scaled | 41, 200 | 25, 2400 (24) | 21, more than 10,000 |
| **ConstrainedSTLSQ, scaled** | 41, 100 | 25, 1200 (12) | 21, more than 10,000 |
| ConstrainedSR3 l0, scaled | 39, 200 | 25, 3200 | 18, more than 10,000 |
| v1 dense ConstrainedSR3 | 10, more than 10,000 | 0 | 0 |
| **SINDy-SI, all side info** | 41, 100 | 33, **400 (4)** | 26, **3200 (32)** |

**Dynamics criterion** (vector-field error < 5%):

| Method | 0% noise | 1% noise | 5% noise |
|---|---|---|---|
| STLSQ α=0.05 | 39, 200 | 24, 4800 | 19, more than 10,000 |
| ConstrainedSTLSQ α=0.05 | 40, 200 | 24, 6400 | 19, more than 10,000 |
| STLSQ, scaled | 42, 200 | 31, 800 | 26, 4800 |
| ConstrainedSTLSQ, scaled | 42, 100 | 31, 400 | 25, 3200 |
| ConstrainedSR3 l0, scaled | 41, 100 | 32, 1600 | 26, 6400 |
| **v1 dense ConstrainedSR3** | 42, 200 | **37**, 800 | **37**, 1200 |
| SINDy-SI, all side info | 42, 100 | 36, **200** | 29, **800** |

### 6.3 Success rate and error as data grows

Success rate on the accuracy criterion over all systems and repeats, at 1% noise:

| Method | N=50 | 100 | 200 | 400 | 800 | 1600 | 3200 | 6400 | 10000 |
|---|---|---|---|---|---|---|---|---|---|
| STLSQ α=0.05 | 0.00 | 0.01 | 0.13 | 0.26 | 0.29 | 0.32 | 0.39 | 0.40 | 0.37 |
| ConstrainedSTLSQ α=0.05 | 0.01 | 0.12 | 0.21 | 0.33 | 0.33 | 0.36 | 0.43 | 0.41 | 0.41 |
| STLSQ, scaled | 0.00 | 0.02 | 0.30 | 0.57 | 0.70 | 0.75 | 0.82 | 0.81 | 0.81 |
| ConstrainedSTLSQ, scaled | 0.03 | 0.29 | 0.56 | 0.68 | 0.78 | 0.81 | 0.86 | 0.85 | 0.84 |
| ConstrainedSR3 l0, scaled | 0.00 | 0.10 | 0.32 | 0.54 | 0.67 | 0.74 | 0.80 | 0.82 | 0.84 |
| v1 dense ConstrainedSR3 | 0.00 | 0.00 | 0.02 | 0.10 | 0.18 | 0.22 | 0.24 | 0.28 | 0.29 |
| SINDy-SI, all side info | 0.08 | 0.46 | 0.65 | 0.73 | 0.80 | 0.85 | 0.87 | 0.89 | 0.86 |

Median coefficient error at 1% / 5% noise:

| Method | N=200 | 400 | 800 | 1600 | 10000 |
|---|---|---|---|---|---|
| STLSQ α=0.05 | 53% / 82% | 31% / 45% | 27% / 33% | 25% / 27% | 18% / 27% |
| ConstrainedSTLSQ α=0.05 | 43% / 64% | 28% / 37% | 25% / 31% | 23% / 23% | 14% / 23% |
| STLSQ, scaled | 100% / 100% | 5.0% / 60% | 1.4% / 7.8% | 1.3% / 4.1% | 0.63% / 1.4% |
| ConstrainedSTLSQ, scaled | 5.1% / 67% | 1.5% / 7.9% | 1.1% / 3.9% | 1.0% / 1.7% | 0.63% / 1.4% |
| ConstrainedSR3 l0, scaled | 64% / 100% | 7.0% / 30% | 1.7% / 9.6% | 1.3% / 4.0% | 0.85% / 1.6% |
| v1 dense ConstrainedSR3 | 532% / 1890% | 119% / 329% | 65% / 175% | 45% / 87% | 34% / 57% |
| SINDy-SI, all side info | 1.6% / 11% | 1.0% / 2.9% | 0.76% / 1.6% | 0.59% / 1.2% | 0.54% / 0.99% |

A median of 100% means more than half the windows' best fit was the all-zero model.

### 6.4 Paired effect on the data needed

Geometric-mean ratio of N_min, on systems where both methods become reliable (n = number of such systems). The
less / same / more counts include systems where only one of the two becomes reliable.

| Comparison | Criterion | 0% noise | 1% noise | 5% noise |
|---|---|---|---|---|
| **Symmetry**, scaled STLSQ | accuracy | ×0.76 (n=42), 14/28/0 | ×0.57 (n=34), 27/8/0 | ×0.59 (n=28), 16/13/0 |
| | structure | ×0.66 (n=41) | ×0.55 (n=24) | ×0.55 (n=21) |
| Symmetry, STLSQ α=0.05 | accuracy | ×0.71 (n=37) | ×0.76 (n=15) | ×0.84 (n=12) |
| Scaled constrained vs old default STLSQ | accuracy | ×0.31 (n=37) | ×0.28 (n=15) | ×0.40 (n=12) |
| | structure | ×0.10 (n=15) | ×0.17 (n=8) | ×0.26 (n=7) |
| SINDy-SI vs scaled constrained STLSQ | accuracy | ×1.02 (n=42) | ×0.87 (n=35), 13/20/3 | ×0.63 (n=28), 18/12/2 |
| ConstrainedSR3 l0 vs ConstrainedSTLSQ, both scaled | accuracy | 0/34/8 | 1/9/26 | 1/11/18 |

### 6.5 Findings

1. **Symmetry pays off most when data is short.**
   - With column scaling it cuts the data needed for a reliable fit to about 0.55–0.6× at 1–5% noise, and never
     increases it on the accuracy criterion.
   - At 200 samples and 1% noise it raises the success rate from 0.30 to 0.56. With the full trajectory the gain is
     only 0.81 → 0.84.
   - This matches section 5: a short window covers the attractor unevenly, and that is exactly when the symmetry
     adds information.
2. **Column scaling with α = 0 is the biggest single factor here too.** About 0.3× the data of the old default, or
   0.1–0.26× when exactly the right terms are required. The old default never becomes reliable at 1% noise on 27 of
   42 systems.
3. **Side information adds another ×0.63 at 5% noise.** SINDy-SI reaches 10% coefficient error with about 2 periods
   at 1% noise and 4 periods at 5%. It is the only method that gets exactly the right terms on most 4-D (Hyper*)
   systems.
4. **ConstrainedSR3 needs more data than constrained STLSQ** on most systems (26 of 42 at 1% noise).
5. **The dense v1 fit can't recover coefficients with noise, but it does recover the dynamics.** On the dynamics
   criterion it becomes reliable on 37 of 42 systems at both 1% and 5% noise, more than any sparse method, after about
   800–1200 samples. Sparse methods do better on the dynamics criterion with less data than that.
6. **Some systems never become reliable with any method.** On the accuracy criterion at 1% noise: Finance,
   LorenzStenflo, LuChen, Qi, QiChen and Sakarya; SINDy-SI reaches HyperYangChen, but only at 1600 samples.
   On the structure criterion, Lorenz never gets exactly the right terms at 1% noise even with the full trajectory.
   Its error is below 10% from 200–400 samples, but small leftover terms remain.

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

Median coefficient error, best of the 4 thresholds, at 1% / 5% noise:

| Method | Symmetry only | + equilibria | + equilibria + monotonicity |
|---|---|---|---|
| ConstrainedSTLSQ α=0.05 (old default) | 13.9% / 26.5% | 13.6% / 26.8% | **9.1% / 15.3%** |
| ConstrainedSTLSQ α=0 | 32.7% / 59.6% | 32.2% / 58.3% | **12.1% / 23.4%** |
| **ConstrainedSTLSQ α=0, scaled** | 0.61% / 1.81% | 0.57% / 1.81% | **0.53% / 1.09%** |
| v1 dense ConstrainedSR3 | 28.1% / 62.5% | 27.8% / 60.6% | **17.1% / 19.1%** |
| Equiv-c* | 32.7% / 59.6% | 32.0% / 57.5% | (equilibria only) |
| ConstrainedSR3 l0* | 32.6% / 59.3% | 32.2% / 59.6% | (equilibria only) |
| ConstrainedSR3 l0, scaled* | 1.44% / 3.02% | 1.02% / 1.97% | (equilibria only) |
| ConstrainedSR3 l1 (inequality path for + mono) | 6.3% / 6.9% | 5.9% / 7.2% | 4.4% / 5.0% |
| ConstrainedSR3 l1, scaled (inequality path for + mono) | 3.3% / 5.8% | 2.2% / 5.9% | 12.1% / 15.3% |
| SINDy-SI (sum-of-squares monotonicity) | 0.61% / 1.81% | 0.57% / 1.81% | 0.53% / 1.09% |

\* Equality constraints only. Equiv-c's symmetry-only entry is ConstrainedSTLSQ α=0, which it matches to every digit.

Other metrics at 5% noise, symmetry only → + equilibria → + equilibria + monotonicity:

| Method | Exact terms | Vector-field error | Forecast (Lyapunov times) |
|---|---|---|---|
| ConstrainedSTLSQ α=0.05 | 0.14 → 0.17 → 0.19 | 6.5% → 2.1% → 1.9% | 0.19 → 0.63 → 0.76 |
| ConstrainedSTLSQ α=0 | 0.19 → 0.19 → 0.24 | 3.2% → 2.2% → 2.0% | 0.66 → 0.71 → 0.61 |
| ConstrainedSTLSQ α=0, scaled | 0.52 → 0.52 → 0.64 | 1.1% → 1.0% → 0.82% | 0.72 → 0.80 → 1.12 |
| v1 dense ConstrainedSR3 | 0 → 0 → 0 | 2.0% → 2.1% → 1.7% | 1.56 → 1.54 → 1.37 |
| SINDy-SI | 0.52 → 0.52 → 0.64 | 1.1% → 1.0% → 0.81% | 0.72 → 0.80 → 1.12 |

**Paired effect** of the added side information vs the same method with symmetry only (better / same / worse
systems, geometric-mean error ratio):

| Method | + equilibria, 1% | 5% | + eq + mono, 1% | 5% |
|---|---|---|---|---|
| ConstrainedSTLSQ α=0.05 | 5/25/12, ×1.07 | 6/25/11, ×1.10 | 20/14/8, ×0.78 | 18/16/8, ×0.72 |
| ConstrainedSTLSQ α=0 | 5/33/4, ×1.00 | 3/35/4, ×0.99 | 25/15/2, ×0.45 | 30/10/2, ×0.32 |
| ConstrainedSTLSQ α=0, scaled | 10/30/2, ×0.89 | 6/33/3, ×0.96 | 16/24/2, ×0.61 | 20/20/2, ×0.29 |
| v1 dense ConstrainedSR3 | 3/39/0, ×0.96 | 4/38/0, ×0.95 | 32/5/5, ×0.37 | 40/1/1, ×0.19 |
| ConstrainedSR3 l0, scaled | 9/30/3, ×0.96 | 6/34/2, ×0.89 | (eq only) | (eq only) |
| ConstrainedSR3 l1, scaled | 11/28/3, ×0.91 | 14/26/2, ×0.87 | 4/5/33, ×4.25 | 6/7/29, ×2.62 |
| SINDy-SI | 10/30/2, ×0.89 | 6/33/3, ×0.96 | 16/24/2, ×0.61 | 20/20/2, ×0.32 |

**Scaled ConstrainedSTLSQ with all side information vs SINDy-SI:** same on all 42 systems at 1% noise; 3 better,
37 same, 2 worse (×0.90) at 5%.

### 7.3 Findings

1. **SINDy-SI's advantage was its side information, not its solver.** Column-scaled ConstrainedSTLSQ (α = 0) with
   the same equilibria and a 1000-point sampled monotonicity constraint reproduces SINDy-SI. The sum-of-squares
   certificate and the null-space solver add nothing over ordinary pysindy machinery here.
2. **Monotonicity helps every method that enforces it in an exact least-squares or QP solve,** by ×0.19–0.78 with
   noise (equilibria + monotonicity vs symmetry only), better on 18–40 of 42 systems at 5% noise. It helps the
   dense v1 fit most (×0.19 at 5% noise): with no pruning, the sign constraints are what keep its many wrong-signed
   terms in check.
3. **Equilibria alone barely change coefficients** (×0.9–1.1), but they improve the dynamics of the ridge fits: for
   the old default at 5% noise, vector-field error 6.5% → 2.1% and forecast 0.19 → 0.63 Lyapunov times.
4. **Side information doesn't make up for raw columns.** Raw ConstrainedSTLSQ with all of it is still 7–14× worse
   than SINDy-SI, and Equiv-c (equilibria only) 23–32× worse. Column scaling remains the biggest single factor.
5. **ConstrainedSR3's inequality path hurts with column scaling** (×4.25 at 1% noise, ×2.62 at 5%) and is mixed
   without it (×0.97 / ×0.90). One l1 solve with no prune-and-refit loop keeps the shrinkage and almost never finds
   exactly the right terms (0–5% of systems with noise). The v1 dense fit via the same cvxpy path
   also loses precision without noise (1.4% → 6.2% at 0% noise), from solver tolerance rather than the constraints.

## 8. Bugs fixed and features added in the pysindy fork

All of these are uncommitted in `pysindy/`. Items 1–4 are bug fixes; item 5 is a new feature.

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

- **Thresholds are chosen using the true answer** for every method. Absolute numbers are therefore optimistic;
  comparisons are fair. Picking thresholds by cross-validation would give honest absolute numbers.
- **One trajectory per system**, about 100 samples per period. Sections 1–5 and 7 use one noise draw per system;
  section 6 uses 5 windows and noise draws per data amount.
- **Several systems fail for every method,** mainly LuChen, LorenzStenflo, VallisElNino, Finance and the Hyper*
  systems. That is a derivative-accuracy problem, not a symmetry one; LuChen, LorenzStenflo and VallisElNino are fine
  with exact derivatives.
- **Side information comes from the true equations,** for the symmetries as well as the equilibria and monotonicity.
- **VPT is noisy per system** (8 starting points) and is censored on systems with long Lyapunov times.
- **The detection only looks for signed permutations;** continuous symmetries are out of scope.
- **Data-efficiency details.**
  - The 4-of-5 rule and the 10% / 5% cutoffs are arbitrary and were not varied. The three criteria already rank
    the dense v1 fit very differently (worst on coefficients, best on vector-field error at 5% noise).
  - Short windows still use true coefficients, equilibria and the monotonicity ball taken from the full trajectory.
  - Data amounts are on a doubling grid, so N_min is only resolved to a factor of 2.
- **Side-information details (section 7).**
  - Monotonicity is only checked at 1000 sampled points, so a fit can violate it between them. SINDy-SI's
    sum-of-squares version covers the whole ball.
  - The data-efficiency sweep (section 6) still gives equilibria and monotonicity only to SINDy-SI.

## 10. Reproduction

| What | Command (Slurm, `mit_preemptable,mit_normal`) | Output |
|---|---|---|
| STLSQ family + v1 | `sbatch --export=ALL,SINDY_ALPHA={0,0.05},SINDY_NORMALIZE={0,1} run_symmetric_sindy_dysts_engaging.slurm` | `logs/symmetric_sindy_dysts/alpha{a}[_norm]/` |
| Sparse SR3 | `sbatch --export=ALL,SINDY_METHODS=sr3,SINDY_NORMALIZE={0,1} run_symmetric_sindy_dysts_engaging.slurm` | `logs/symmetric_sindy_dysts/sr3[_norm]/` |
| SINDy-SI + Equiv-c | `sbatch run_sindy_si_dysts.slurm` | `logs/sindy_si_dysts/{fits,side_info}.csv` |
| Summary tables | `python notebooks/compare_sindy_si.py > logs/compare_sindy_si.txt` | `logs/compare_sindy_si.txt` |
| Data-efficiency sweep | `sbatch run_data_efficiency_dysts.slurm` (array of 42 tasks, 2 CPUs each, a few minutes per system); needs `logs/sindy_si_dysts/side_info.csv` | `logs/data_efficiency/fits_<system>.csv` |
| Data-efficiency summary | `python notebooks/summarize_data_efficiency.py > logs/data_efficiency_summary.txt` | `logs/data_efficiency_summary.txt` |
| Side info for other methods | the STLSQ-family and sparse-SR3 commands above with `SINDY_SIDE={eq,eq_mono}` added (only constrained methods are fitted; needs `logs/sindy_si_dysts/side_info.csv`) | `logs/symmetric_sindy_dysts/<run>_{eq,eq_mono}/` |
| Side-info summary | `python notebooks/compare_side_info.py > logs/compare_side_info.txt` | `logs/compare_side_info.txt` |

Each of the full-data runs takes 45–120 minutes with 4 CPUs, mostly spent on the forecasts. Passing system names as arguments restricts a
run (for example `... Lorenz ChenLee`).

| File | Contents |
|---|---|
| `notebooks/symmetric_sindy_dysts.py` | symmetry detection, M_g and C construction, STLSQ / SR3 families |
| `notebooks/sindy_si_dysts.py` | SINDy-SI and Equiv-c harness, side-information extraction |
| `notebooks/sindy_si.py` | SINDy-SI estimator (null-space and sum-of-squares solvers) |
| `notebooks/equiv_c.py` | Equiv-c estimator |
| `notebooks/fit_metrics.py` | all metrics |
| `notebooks/compare_sindy_si.py` | aggregation and tables |
| `notebooks/dysts_common.py` | shared system loading and symmetry construction (used by the data-efficiency sweep) |
| `notebooks/data_efficiency_dysts.py` | data-efficiency sweep |
| `notebooks/summarize_data_efficiency.py` | data-efficiency aggregation and tables |
| `notebooks/compare_side_info.py` | side-info (section 7) aggregation and tables |
