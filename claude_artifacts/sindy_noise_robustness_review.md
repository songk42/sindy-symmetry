# Noise robustness in SINDy: what the literature has done, and which of it applies to your EHD pipeline

Compiled from arXiv metadata retrieved 2026-09-24 (178 records across 29 queries); arXiv ids
are given inline so each claim is checkable. Numbers attributed to "your notebook" are read
from `symmetric_sindy_v1.ipynb` as attached.

---

## 0. Read this first

The SINDy noise-robustness literature is overwhelmingly about **measurement noise on a system
that is closed in the given coordinates and library**. Your dominant error source is almost
certainly not that. Three numbers from your own notebook say so:

1. **Your fit has 8 free parameters.** Rebuilding your constraint matrix numerically
   (symmetry `G` + the energy-preserving row, degree-3 library, r=3) gives rank 52 of 60, so
   `null(C)` has dimension **8**; the printed Xi from your notebook satisfies `C vec(Xi) = 0`
   to 1.7e-15, confirming it is the same constraint set. With n = 2x10^4 samples that is a
   2500:1 sample-to-parameter ratio. Estimator *variance* — which is what nearly every
   robustness method in this review reduces — is not your binding constraint.
2. **Coarsening dt 8x (0.05 -> 0.4) barely moves the fit** (R^2(a1') 0.767-0.772 in your
   frequency sweep). Pointwise-derivative noise is the thing that dt sensitivity exposes. You
   are not seeing it. Consistent with that, weak SINDy — the canonical fix for derivative
   noise — was *not* an improvement for you (it traded a1 tracking for worse a2/a3, and became
   duration-sensitive: xi2/xi6 = -35.9 at duration 800, -75 at 400, +110 at 1200).
3. **R^2(a2'/a3') rises from 0.29-0.39 at r=3 to 0.63-0.69 at r=6, and keeps rising with r.**
   That is the signature of **projection/closure error**, not measurement noise: a meaningful
   fraction of the a2, a3 tendency is not a function of the retained state at all.

So the honest framing of your problem is: *the r=3..6 POD system is not dynamically closed, and
the unclosed part is what sets the switching statistic you are scoring against.* That reframes
which literature is relevant — Sections 3 and 8 below, not Sections 1 and 2 — and it is
consistent with every one of your negative results, including the ones that look
paradoxical (more data per window -> *fewer* switches; more modes -> no change in xi2/xi6;
the same windows fail at every r).

One more quantitative note on the r sweep: your constraints cut 60 -> 8 parameters at r=3, but
504 -> 119 at r=6 (using `G6 = blockdiag(G, I3)`). Going to r=6 buys real R^2 but costs a
**15-fold increase in free parameters**, so the variance-robustness of the fit genuinely
degrades as r grows even while the fit improves. That trade is worth tracking explicitly.

---

## 1. Avoid differentiating pointwise: integral and weak forms

The single largest robustness gain reported in the literature, for measurement noise.

| method | ref | note |
|---|---|---|
| Integral / sparse model selection via integral terms | Schaeffer & McCalla, PRE 96 (2017) | first "don't differentiate" formulation |
| WSINDy (ODEs) | arXiv:2005.04339 | replaces pointwise derivative approximations with linear transformations and a variance reduction technique, reported to improve on standard SINDy by orders of magnitude |
| WSINDy (PDEs) | arXiv:2007.02848 | several groups independently found the weak formulation gives orders-of-magnitude better noise robustness |
| Asymptotic consistency of WSINDy | arXiv:2211.16000 | proves the WSINDy estimator is unconditionally asymptotically consistent for a wide model class including Navier-Stokes, giving a rigorous explanation for the observed noise robustness — and shows it is only *conditionally* consistent in general |
| Physically constrained weak-form symbolic regression | arXiv:2102.12006; Reinbold et al. PRE 101 (2020) | weak form + physical constraints on noisy, incomplete experimental data |
| WENDy (weak-form parameter estimation) | arXiv:2302.13271 | no ODE solver in the loop; robust to large measurement noise, and for stiff/higher-dimensional systems often orders of magnitude faster than forward-solver nonlinear least squares |
| WENDy, nonlinear-in-parameters | arXiv:2502.08881 | extends WENDy beyond linear-in-parameters models via an approximate MLE |
| WENDy-IRLS bias/coverage | arXiv:2510.03365 | when the weak-form estimator's intervals are actually calibrated |
| Fourier Weak SINDy | arXiv:2604.20141 | **directly relevant to your WSINDy trouble**: replaces randomly placed compact test functions with orthogonal sinusoidal test functions chosen by multitaper spectral estimation — removes the random-subdomain-placement lottery entirely |
| Laplace-enhanced (LES-SINDy) | arXiv:2411.01719 | integral transform for high-order derivatives / discontinuities |
| Weak-form modified SINDy | arXiv:2410.17838 | weak form + explicit noise modelling |

**For you.** Your three WSINDy findings — RNG-dependent subdomain placement with no seed
parameter, `n_subdomains` not matched to window length (7/8 failures at duration 200), and
duration-dependent xi2/xi6 — are all *test-function selection* problems, not weak-form problems.
Fourier-WSINDy (arXiv:2604.20141) is the targeted fix, and Messenger & Bortz's own test-function
support selection (in arXiv:2005.04339 / 2007.02848) is the other. But given point 2 in Section 0,
I would not expect a large payoff: the weak form buys derivative-noise immunity you do not need.

---

## 2. Better derivatives / denoise first

| method | ref |
|---|---|
| Multi-objective framework for choosing a differentiation method + its hyperparameters (`pynumdiff`) | arXiv:2009.01911 |
| Comparative study of a priori denoising strategies for SINDy | arXiv:2201.12683 |
| DSINDy (derivative-based, joint state/derivative recovery) | arXiv:2211.05918 |
| Implicit neural representation + Runge-Kutta + total-variation denoising | arXiv:2509.14219 |
| Differentiation method as a *systematic uncertainty source* in equation discovery | arXiv:2512.12666 |
| Hamiltonian flow splines for derivative estimation | arXiv:2609.20399 |

**For you.** Low priority, and you have already bounded it: your stride sweep *is* a
differentiation-sensitivity study, and it came back flat. Worth one line in a writeup citing
arXiv:2512.12666 to say the differentiation choice was checked rather than assumed.

---

## 3. Treat the state itself as unknown: joint denoising / state estimation / data assimilation

This is the family that starts to be about *your* problem, because "unobserved state" and
"truncated state" are mathematically the same difficulty.

| method | ref | note |
|---|---|---|
| Automatic differentiation SINDy: simultaneously denoise, learn the noise distribution, and identify the model | arXiv:2009.08810 | the "modified SINDy" line |
| Neural implicit representation + integral form | arXiv:2309.07193 | surrogate state + integral residual |
| DeepMoD / sparsely constrained networks | arXiv:1904.09406, arXiv:2011.04336 | NN surrogate supplies the library |
| Physics-informed discovery from scarce, noisy data | arXiv:2005.03448 | |
| Robust SINDy autoencoder | arXiv:2604.04829 | denoising + coordinate discovery jointly |
| EKF-SINDy | arXiv:2404.07536 | SINDy model *inside* an extended Kalman filter |
| SINDy + ensemble Kalman filter, partial observability | arXiv:2607.27137 | |
| **Model selection with hidden variables via sparse data assimilation** | arXiv:2105.10068 | model selection is substantially harder when only a subset of variables is experimentally accessible — Ribera, Shirman, Nguyen & Mangan score candidate models on their ability to assimilate the *observed* variables, for chaotic systems specifically |

**For you.** arXiv:2105.10068 is the closest published analogue of your situation: chaotic
system, only part of the state retained, model selection must be scored on something other
than pointwise derivative residual. Its scoring idea (assimilate, then score) is a
drop-in replacement for your `trajectory_fit_quality`, and unlike NRMSE-at-50-time-units it
does not collapse to "no better than the mean" the moment the horizon exceeds the Lyapunov time.

---

## 4. More robust sparse regression itself

| method | ref | note |
|---|---|---|
| SR3 / constrained SR3 | arXiv:1807.05411, arXiv:1906.10612 | solutions of the relaxed problem are superior in error, false positives, **and conditioning**, and relaxation makes hard constraints tractable — the optimizer you already use |
| Reweighted l1 | arXiv:2005.13232 | |
| CINDy (conditional-gradient) | arXiv:2101.02630 | noise-robust recovery with better sparsity/accuracy trade-off than STLSQ/LASSO |
| MIOSR (mixed-integer, provably optimal sparsity) | arXiv:2206.00176 | an exact mixed-integer formulation solved to provable optimality in seconds, with dramatic improvement over heuristic sparse regression |
| Trimmed LASSO | arXiv:2308.01891 | exact identification where heuristic estimators return only sparse *approximations* |
| ILTS-SINDy (iterative least trimmed squares, then STLS) | arXiv:2606.28584 | filters unreliable observations first, then sparsifies; holds up to ~20% corrupted observations. Notably, it reports that E-SINDy can attain high mean support-recovery accuracy while its *exact* identification rate stays low — accurate trajectory reconstruction does not imply successful model discovery |
| Exact recovery from highly corrupted data | arXiv:1607.01067 | outliers/dropouts rather than Gaussian noise |
| High-noise toolkit (progressive library culling) | arXiv:2111.04870 | |
| Data-scale-independent regulariser | arXiv:2603.05201 | **directly relevant**: magnitude-based thresholding interacts with data normalisation, so a scaling choice can make the recovered model dense and physically wrong |
| Ill-conditioning in dictionary-based learning | arXiv:2603.11330 | **directly relevant**: strongly correlated candidate functions + poor sampling give multicollinearity where measurement noise leads to widely different recovered models, obscuring the true underlying dynamics |

**For you.** Your own notebook comment is a textbook statement of the arXiv:2603.11330 /
arXiv:2603.05201 problem: *"a1\*a_i and a1^2\*a_i are highly collinear in this data, so
ConstrainedSR3(reg_weight_lam=0) ... is free to trade weight between two nearly-tied solutions,
and the column scaling alone is enough to tip which one it lands on"* — with the two candidate
solutions differing in whether the a1-axis fixed points are sources or saddles. That is not a
footnote about normalisation; it is the same fragility your switching metric is measuring, and
it is the one place where a *statistical* robustness method should genuinely help you:

- Quantify it: condition number of the **reduced** (null-space) design matrix, pairwise
  library correlation, VIF — `library_conditioning()` in the attached script.
- Fix it with sparsity rather than scaling: `reg_weight_lam > 0`, or `MIOSR` under the same
  equality constraints, which chooses support by optimality rather than by magnitude threshold
  on scaled columns. You tested five sparsity levels at r=6 for one window; a systematic
  constrained-MIOSR-vs-SR3 comparison across your existing 32 windows is the missing experiment.

---

## 5. Ensembling and uncertainty quantification

This family does not make your fit better — it tells you **how much of your window-to-window
variation is estimation noise**, which is exactly the question your sweeps keep bumping into.

| method | ref | note |
|---|---|---|
| E-SINDy (bagging, inclusion probabilities) | arXiv:2111.10992 | aggregate statistics over an ensemble fit to subsets of limited, noisy data yield inclusion probabilities for candidate functions, enabling UQ and probabilistic forecasts |
| Convergence of ensemble/Bayesian uncertainty estimates | arXiv:2301.12649 | theory for bootstrapped STLSQ: provably correct variable selection with exponential convergence, addressing false-discovery control in the low-data/high-noise limit |
| Sparsifying priors for Bayesian UQ | arXiv:2107.02107 | posteriors over coefficients instead of point estimates |
| Threshold sparse Bayesian regression with error bars | Zhang & Lin, Proc. R. Soc. A (2018) | |
| Bayesian-SINDy | Fung, Fasel & Juniper, Proc. R. Soc. A 481 (2025) | sparse Bayesian learning + SINDy, aimed at noise robustness and data efficiency relative to STLS |
| BINDy (reversible-jump MCMC) | arXiv:2408.08062 | full joint posterior over model *structure* |
| Bayesian model averaging | arXiv:2604.10854 | when several supports explain the data comparably — point estimates alone can be misleading |
| Conformal prediction for SINDy | arXiv:2507.11739 | distribution-free coverage under exchangeability |
| HyperSINDy | arXiv:2310.04832 | deep generative *stochastic* governing equations |
| Multi-step-prediction + ensemble + elite selection | arXiv:2503.05154 | scores candidates on multi-step forecast, not one-step residual |

**For you — highest-value cheap experiment.** You already have the window machinery
(`run_multi_slice`, 32 windows). Turning it into an E-SINDy-style ensemble costs almost
nothing and converts statements like *"2 of 11 slices break the a2<->a3 symmetry"* and
*"t_start=1600 tracks well (does not replicate)"* into inclusion probabilities and coefficient
IQRs. Given only 8 free parameters, I would predict tight coefficient distributions — which
would be a *positive* result: it would prove the switching failure is structural rather than
a sampling artifact, and let you stop re-litigating window choice. Note the ILTS-SINDy caveat
(arXiv:2606.28584) that good mean recovery in an ensemble does not imply exact support recovery.

---

## 6. Constraints and structure — the family you are already in

Robustness via shrinking the hypothesis space rather than via better statistics.

| method | ref | note |
|---|---|---|
| Constrained sparse Galerkin regression | arXiv:1611.03271 | extends SINDy to enforce physical constraints, specifically energy-preserving quadratic nonlinearities — your energy row |
| **Trapping SINDy (global stability)** | arXiv:2105.01843 | incorporates the Schlegel & Noack trapping theorem for systems with energy-preserving quadratic nonlinearities directly into the model-discovery process, so long-time boundedness is promoted rather than hoped for |
| Local trapping extension | arXiv:2403.00324 | local stability guarantees where global trapping is unattainable |
| Side information via sum-of-squares (SINDy-SI) | arXiv:2310.04227 | guarantees learned models satisfy symmetries/physical laws; SOS certificates |
| Symmetry-informed term filtering | arXiv:2604.01592 | automates symmetry-based library restriction (cf. your `build_Mk`) |
| Dimensional consistency / Buckingham Pi | arXiv:2202.04643 | |
| Port-Hamiltonian structure (SIPHy) | arXiv:2609.20399 | structure-preserving identification from noisy data |
| Symmetry-constrained POD-SINDy for chaotic electroconvection | arXiv:2009.11862 | Guan, Brunton & Novosselov — the paper you are reproducing |

**For you.** Two concrete items:

- **Trapping SINDy is the natural next constraint**, and it is the one that targets the object
  your fidelity metric measures. Your current constraints pin *coefficient relations* but say
  nothing about long-time behaviour; xi2/xi6 flipping the fit between "everything decays to
  the origin" and "bounded saddle-connected switching" is precisely an unconstrained-stability
  symptom. Caveat: the trapping theorem is for **quadratic** energy-preserving nonlinearities,
  and your library is degree 3. Which raises the sharper question below.
- **Is the cubic part physical or is it absorbing closure error?** A Galerkin projection of the
  governing equations onto POD modes produces constant + linear + quadratic terms — not cubic.
  Your cubic self-terms (`a1^3`, `a1 a2^2`, ...) are doing the work of setting fixed-point
  location and stability, i.e. exactly the fragile part. Fitting `degree=2` with the
  energy-preserving constraint plus trapping, and scoring switching fidelity against it, is a
  cheap and highly diagnostic experiment: if a quadratic trapping model switches at a
  physically right rate where your cubic model does not, the cubic terms were closure error in
  disguise.

---

## 7. Model selection and honest metrics

| method | ref | note |
|---|---|---|
| AIC/BIC over the SINDy Pareto front | arXiv:1701.01773 | |
| Benchmarking sparse system ID on low-dimensional chaos (`dysts`) | arXiv:2302.10787 | the reference systematic comparison of SINDy variants across many chaotic systems |
| Sparsistency / irrepresentability of the design matrix | arXiv:2106.11936 | when sparse regression *cannot* recover the true support even with clean data |
| PySINDy (implementations of most of the above) | arXiv:2111.08481 | |

**For you.** arXiv:2106.11936 is worth reading against your collinearity note: it shows the
design matrix can violate the LASSO irrepresentability condition even for analytically exact
data. If your constrained design violates it, no amount of denoising or extra windows will
stabilise the support, and only the structural changes in Section 6 or the stochastic route in
Section 8 will move the needle.

Your switching-fidelity metric is the right instinct (a long-time statistic compared against
the POD amplitude data rather than the raw field). Two ways to make it fairer:
- Report the **data's own** switching-rate spread across windows as the tolerance band (your
  `logs/trajectory_data_summary.md` already has CV 0.06-0.16), rather than a 2x rule.
- Report the **Lyapunov time of the amplitude series**. If it is ~10-30 time units, then
  NRMSE ~ 1 at a 50-time-unit horizon is arithmetically expected and is not evidence of a bad
  fit, which resolves the (a)-vs-(b) ambiguity you flagged. `predictability_time()` in the
  attached script estimates it (validated: Lorenz-63 lambda_max = 0.87 vs reference 0.906).

---

## 8. When the "noise" *is* the truncation: stochastic closure

This is the family I would put first, because it is the only one whose predictions match all
of your observations simultaneously.

| method | ref | note |
|---|---|---|
| **Langevin regression** | arXiv:2009.01006 | Callaham, Loiseau, Rigas & Brunton: fit a *stochastic* model (drift + diffusion) to realistically-sampled time series by matching Kramers-Moyal averages and the stationary distribution — built for exactly the case where unresolved degrees of freedom act as random fluctuations in a fluid ROM |
| Langevin regression with finite-size effects | arXiv:2103.16791 | coarse-grained dynamics where the noise amplitude is set by the truncation |
| Nonlinear correlations in reduced-order modelling | arXiv:2106.02409 | Callaham, Brunton & Loiseau: POD converges slowly for advection-dominated flows and tends to overestimate the number of dynamically relevant variables — nonlinear correlations among POD coefficients mean the dynamically relevant manifold is lower-dimensional than the linear-subspace rank suggests |
| HyperSINDy | arXiv:2310.04832 | stochastic governing equations via deep generative modelling |

**The hypothesis.** Saddle-to-saddle switching in a system whose retained state explains only
~65% of the a2/a3 tendency is a **noise-activated escape** problem: the deterministic vector
field sets the fixed-point structure (which your fit recovers robustly — xi2/xi6 = -92 to -106
with std as low as +/-2.2), while the *rate* of switching is set by the amplitude of the
unresolved forcing (which a deterministic ROM has no representation of). This predicts, in order:

- the saddle ratio is robust while the switching rate is not — **observed**;
- more POD modes do not fix switching, because they move the closure boundary without removing
  it — **observed** (identical per-window pattern at r=3,4,5, xi2/xi6 essentially r-invariant);
- *more* data per window gives *more damped*, non-switching models, because least squares
  averages the fluctuating forcing toward zero — **observed** (duration=1200: both windows
  exactly 0 sign changes);
- the same windows fail and the same window survives at every r, since what varies is the
  realisation of the unresolved forcing in that window, not the information content —
  **observed**.

**The test** (cheap, and it is a decisive one). Take your best deterministic fit, compute the
one-step Kramers-Moyal residual `r_k = a_{k+1} - a_k - f(a_k) dt`, and ask three questions:
is it white (correlation time ~ dt)? how large is `sigma_eff = sqrt(var(r)/dt)`? and does a
noise-driven excursion over one dominant period reach the attractor's own scale? If yes,
add the fitted diffusion term and simulate the SDE — switching should reappear at a rate set by
`sigma_eff`, with no change to the deterministic coefficients you have already validated.
`closure_noise()` in the attached script does this (validated on a double-well SDE: recovers
sigma = 0.3505 against a true 0.35, correlation time 1.00 dt, and returns 2e-16 on a
noise-free control).

---

## 9. Suggested order of work

1. **Kramers-Moyal residual analysis** of your existing best fit (Section 8). Minutes of
   compute, and it either promotes or kills the stochastic-closure hypothesis outright.
2. **Lyapunov time of the amplitude series** (Section 7). Tells you what forecast horizon is
   even meaningful, and closes the (a)/(b) question in your job-40129880 notes.
3. **Conditioning report** on the reduced (null-space) design (Section 4) + a constrained
   `MIOSR`-vs-`ConstrainedSR3(lam>0)` comparison across your existing 32 windows.
4. **Ensemble inclusion probabilities** over the same 32 windows (Section 5) — converts
   window-to-window anecdotes into error bars.
5. **Degree-2 library + energy constraint + trapping** (Section 6) scored on switching fidelity;
   tests whether the cubic terms are physics or closure error.
6. If 1 comes back positive: **Langevin regression** on the POD amplitudes as the actual model
   class, and score the switching-time distribution rather than a single trajectory.

Items likely *not* worth more effort, given Section 0: further weak-form tuning, denoising or
derivative-estimation work, and larger r.

---

## Appendix: running the attached diagnostics

`robustness_diagnostics.py` is self-contained (numpy + scipy only, no pysindy) and operates on
the amplitude series, not the 7 GB field. Validation figures quoted above were obtained on
systems with known answers: Lorenz-63 (lambda_max = 0.868 vs reference 0.906, fit R^2 = 0.996),
Rossler (0.055 vs 0.071), a double-well SDE (sigma_eff = 0.3505 vs true 0.350, residual
correlation time 1.00 dt) with a noise-free control returning sigma_eff = 2e-16, and your own
constraint set (reproduces `C vec(Xi) = 0` for the notebook's printed Xi to 1.7e-15).

```python
import numpy as np, robustness_diagnostics as rd
from sindy_pipeline import compute_pod, slice_amplitudes   # your module

t, X = ...          # amplitude window the model was fit on, X shape (n, r)
model = ...         # fitted ps.SINDy

# 1. is the switching statistic reachable at all at this horizon?
ly = rd.predictability_time(X, dt=t[1]-t[0])
print(ly["lambda_max"], ly["lyapunov_time"], ly["fit_r2"])     # inspect log_divergence curve

# 2. is the unresolved tendency white and large enough to drive switching?
cn = rd.closure_noise(t, X, f_model=model.predict, Xdot=Xdot_estimate,
                      timescale=ly["lyapunov_time"])
for v in cn["per_variable"]:
    print(v["var"], v["unresolved_frac"], v["corr_time_over_dt"],
          v["sigma_eff"], v["excursion_ratio"], v["state_dependent"])

# 3. how ill-posed is the regression you are actually solving?
Theta = model.feature_library.transform(X)
print(rd.library_conditioning(Theta, constraint_lhs=constraints_lhs, n_targets=X.shape[1]))

# 4. constraints for any r / any linear group, built numerically (no sympy, no per-r rewiring)
out = rd.symmetry_constraint_matrix(G, n_vars=3, degree=3,
                                    energy_pairs=[((0, (0,2,0)), (1, (1,1,0)))])
print(out["rank"], out["n_free_params"], out["residual"])   # 52, 8, ~1e-13 for your r=3 case

# 5. error bars instead of one window's point estimate
res = rd.inclusion_probabilities(lambda w: run_slice(**w)["Xi"], your_32_windows)
print(res["inclusion_prob"], res["sign_consistency"], res["iqr"])
```

Interpretation guide for (2), which is the decisive one:

| `corr_time_over_dt` | `excursion_ratio` | reading |
|---|---|---|
| ~1 (white) | O(0.1-1) | unresolved forcing is diffusive and large: switching is plausibly noise-activated, and a Langevin/SDE closure (arXiv:2009.01006) should restore it without touching the deterministic coefficients |
| ~1 (white) | << 0.1 | forcing is white but too weak to matter: switching must be deterministic, so look at the drift terms (Section 6) |
| >> 1 (coloured) | any | the residual is structured, i.e. a *missing deterministic term* or a missing slow variable — extend the library or the state, or use a hidden-variable method (arXiv:2105.10068) |

`state_dependent` (|corr(|r|, |x|)|) distinguishes additive from multiplicative forcing, which
matters if you fit an SDE: state-dependent diffusion changes the escape rate's scaling.
