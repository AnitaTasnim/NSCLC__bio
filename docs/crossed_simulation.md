# Simulation study E1–E4 of the crossed pick-freeze estimator: analysis plan

Written and committed **before any simulation result was produced**. The thresholds, the primary interval,
the statistics evaluated and the GO/NO-GO rule below are not changed after results are seen. Any deviation is
listed in the results report with its reason.

## 1. Estimand

For a simulated training set T, a held-out patient x and the predicted log-odds z, the components
σ²_S, σ²_Q, σ²_SQ, σ²_E (Hoeffding components of the untied pipeline) and γ = V_R − D_both (coupling) are
defined in `docs/crossed_estimator_derivation.md`. The cohort-level estimands are the shares

    Π_c(T) = E_x[σ²_c(x)] / E_x[V_R(x)],  c ∈ {S, Q, SQ, E, γ},  and  Π_Q + Π_γ,

conditional on T and averaged over the patient population. Eq. 3 of the manuscript (fixed classifier seed,
frozen panel selected on all of T) is evaluated against its own estimand and against Π_Q + Π_γ.

## 2. Reference truth

For each T: R = 200 independent draws (subsample + selection seed), a sparse cyclic crossed design
(cells (r, r + d mod 200), d = 1..10 for logistic regression and d = 1..5 for XGBoost), the 200 tied fits,
M = 5 seeds on 40 cells, the two Eq. 3 arms, and 1,000 fresh patients from the generator. The Monte Carlo
error of the truth is estimated by a grouped jackknife over draws (10 groups) and reported.

## 3. Data generator and design

Two Gaussian classes (prevalence 0.6); p features in blocks of 10 with within-block correlation ρ; the first
20 features (two blocks) shifted by δ in class 1, with δ set so that the Bayes-optimal linear AUC equals the
target AUC. Panel size k = 27. Selectors: Mann–Whitney top-k, L1-logistic top-k (strongest penalty keeping at
least k features), random panels. Classifiers: L2-logistic regression (deterministic) and XGBoost with the
manuscript's settings (300 trees, depth 4, learning rate 0.03, row and column subsampling 0.8; a unique seed
per fit). Held-out patients per estimator replicate: n (fresh draws).

The full factorial of the requested factors has 3 × 2 × 3 × 3 × 4 × 4 × 3 × 2 = 5,184 cells. On the available
machine (2 CPU cores, no GPU; one XGBoost fit ≈ 0.2 s) it is not feasible. The design is one factor at a
time around a base point:

| Factor | Base | Variations |
| --- | --- | --- |
| n (training set size) | 300 | 100, 600 |
| p | 1,000 | 10,000 (logistic only; 2 training sets; R = 10 only) |
| ρ | 0.5 | 0, 0.9 |
| AUC | 0.90 | 0.75, 0.97 |
| f | 0.8 | 0.5, 0.632, 0.9 |
| R | 10 | 5, 20, 50 at the base point only (R = 50 logistic only) |

Every configuration is run for all three selectors with logistic regression (4 training sets; 50 replicates at
R = 10). XGBoost is run at the base point only (2 training sets; 15, 40 and 4 replicates at R = 5, 10, 20).

## 4. Intervals

Percentile intervals from B = 400 resamples. **Primary: pigeonhole** (draws and patients resampled
independently; weighted U-statistics so that a resampled draw never pairs with itself). Secondary: draws only;
patients only (the interval published for ψ).

## 5. Success criterion (from the audit, unchanged)

At R = 10, for every evaluated cell with n ≥ 300 (configuration × selector × classifier) and for each of
Π_S, Π_Q, Π_SQ, Π_E, Π_γ and Π_Q + Π_γ:

* |bias| ≤ 0.02, bias = mean over replicates and training sets of (estimate − truth);
* coverage of the pigeonhole 95% interval between 0.92 and 0.97.

Point estimates decide pass or fail. Wilson 95% intervals of the coverage and the standard error of the bias
(replicate variability plus the truth's Monte Carlo error) are reported; a cell whose interval crosses a
threshold is additionally flagged "Monte Carlo uncertain", but its pass/fail status is not changed.
For the deterministic logistic classifier Π_E is identically 0 in truth and estimate; it is reported and not
counted.

## 6. GO/NO-GO rule for applying the estimator to the NSCLC data

* **GO**: the criterion holds for all counted statistics in all evaluated cells with n ≥ 300.
* **CONDITIONAL GO**: it holds for Π_Q + Π_γ (the quantity that would replace Eq. 3) and Π_S in all such
  cells, but fails for one or more of Π_Q, Π_SQ, Π_E, Π_γ separately; only the passing statistics may then be
  reported for real data, and the failures are reported.
* **NO-GO**: Π_Q + Π_γ or Π_S fails in any such cell.

Results for n = 100 and for R ≠ 10 are descriptive.

## 7. Also reported (descriptive)

RMSE, interval width, frequency of negative component estimates, Eq. 3 bias against its own truth and its
difference from Π_Q + Π_γ, runtime and CPU time per unit, and fits per unit.
