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
     error. This is finding 7 below, now measured directly against the data.
3. **Picking the threshold by trajectory RMSE on the noisy data works for forecasting, without ground truth.**
   Scaled ConstrainedSTLSQ then gives 8.9% / 12.8% trajectory RMSE with no blow-ups, better than the dense fit and
   than any oracle-coefficient choice. It costs coefficient accuracy (10% / 37%) because it prefers denser models.
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

At 1% noise most attractors are reproduced closely. The failures are the systems that fail on every metric:
HyperXu and HyperYangChen blow up, and Finance, LorenzStenflo, LuChen, Qi and Tsucs2 settle onto the wrong
attractor or a fixed point. Several with large trajectory RMSE (Lorenz 65%, MooreSpiegel 72%, ChenLee 72%) still
have the right attractor: their forecasts split from the data near a lobe switch within the Lyapunov time.

