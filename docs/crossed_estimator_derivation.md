# Crossed pick-freeze estimator: derivation and validation (Stage 0)

This document re-derives the estimator proposed in the audit of 7 Nov 2026 from first principles, states
where the audit was wrong or incomplete, and lists the checks that were run. Code: `pmtrd/crossed/estimator.py`.
Tests: `tests/test_crossed_estimator.py`, `tests/test_crossed_pipeline.py`.

## 1. Setting

Fix an outer training set T and a held-out patient x. One re-derivation draw r consists of a stratified
subsample S_r of T (fraction f) and a selection seed ξ_r; draws are independent and identically distributed
given T. A classifier fit additionally has a seed m. Write the predicted log-odds as

    h(U, V, W) = g(U, A(V), W),   U = training subsample, V = (selection subsample, ξ), W = classifier seed.

The real pipeline uses the same subsample twice: z_real = h(S_r, (S_r, ξ_r), m). The crossed design unties
the inputs: cell (r, q), r ≠ q, is h(S_r, (S_q, ξ_q), m_rq), with a seed m_rq unique to the fit. For r ≠ q
the three inputs are independent given T.

## 2. Hoeffding decomposition (exact, no model assumed)

For independent inputs and E h² < ∞,

    h = μ + h_U + h_V + h_W + h_UV + h_UW + h_VW + h_UVW,

with mutually uncorrelated, zero-mean terms (Efron & Stein 1981). Hence
Var h = σ²_U + σ²_V + σ²_W + σ²_UV + σ²_UW + σ²_VW + σ²_UVW. We name

    σ²_S = σ²_U,  σ²_Q = σ²_V,  σ²_SQ = σ²_UV,  σ²_E = σ²_W + σ²_UW + σ²_VW + σ²_UVW.

**Correction to the audit.** The audit called σ²_E "model-fitting randomness" as if it were one effect. It is
the sum of every term that involves the seed, including the seed × sample and seed × panel interactions.

## 3. Pick-freeze moments

For two evaluations that share the inputs in a set A and redraw the others independently,
Cov = Σ_{u ⊆ A} σ²_u (the closed index of A). Therefore ½ E (h − h')² = Var h − Σ_{u ⊆ A} σ²_u, and

| Moment | Shared inputs A | ½ E (z − z')² |
| --- | --- | --- |
| D_col: cells (r, q), (r', q), r ≠ r' | V | σ²_S + σ²_SQ + σ²_E |
| D_row: cells (r, q), (r, q'), q ≠ q' | U | σ²_Q + σ²_SQ + σ²_E |
| D_both: cells sharing no draw | none | σ²_S + σ²_Q + σ²_SQ + σ²_E |
| D_seed: one cell, two seeds | U, V | σ²_E |

These are the audit's definitions, and they are correct **provided** the following hold. The audit did not
state them.

1. Every fit has its own seed. If one seed value is reused across cells, W becomes a shared input of those
   pairs and the seed terms are allocated to the wrong moments.
2. The two cells of a D_both pair must share no draw in any role. Cell (r, q) and cell (r', q') must satisfy
   r ≠ r', q ≠ q', **r ≠ q' and q ≠ r'**. The cross-role coincidences (the training subsample of one cell is
   the selection subsample of the other) also create dependence.
3. Seeds within a cell must not be averaged before forming D_col, D_row or D_both. Averaging M seeds replaces
   σ²_E by σ²_E / M in those moments and biases σ²_SQ.

### 3.1 Unbiased estimators

Each moment is estimated by the average of ½ (z − z')² over all pairs of the required kind. Each pair is an
unbiased kernel, so any average of them is unbiased conditional on T and x, for complete and incomplete
designs alike. D_col and D_row reduce to pooled within-column and within-row sample variances.

### 3.2 D_both by inclusion–exclusion

Write D_both = mean(z²) − mean over disjoint ordered pairs of z z' (both terms unbiased). With cell weights
ω_rq (1, 0 for a missing cell, or bootstrap multiplicities v_r v_q) and u = ω z, the sum over disjoint ordered
pairs is

    (Σ u)² − Σ_r (row_r)² − Σ_q (col_q)² − 2 Σ_j row_j col_j + Σ u² + Σ_rq u_rq u_qr.

This follows because the four coincidence events (r = r', q = q', r = q', q = r') intersect only in c = c'
(the first two) and in the transposed cell (the last two). Every other intersection forces a diagonal cell,
which is not in the off-diagonal design. With z ≡ 1 the formula gives R(R−1)(R−2)(R−3), the number of
ordered pairs of four distinct draws (unit test). D_both requires R ≥ 4.

## 4. Component estimators

    σ̂²_E = D_seed,  σ̂²_S = D_both − D_row,  σ̂²_Q = D_both − D_col,  σ̂²_SQ = D_row + D_col − D_both − D_seed.

Substituting section 3 gives exactly σ²_S, σ²_Q, σ²_SQ and σ²_E. Each is unbiased, as a linear combination of
unbiased moments. Each can be **negative** in a finite design. No truncation is applied, because truncation
would bias them.

## 5. Coupling γ = V_R − D_both

V_R, the variance of the tied pipeline, is estimated from pairs of diagonal cells (distinct draws, distinct
seeds), unbiasedly. γ = V_R − D_both is therefore an unbiased estimator of Var(tied) − Var(untied).

**Correction to the audit.** γ is a contrast, not a Hoeffding component. The audit's line
"V_R = σ²_S + σ²_Q + σ²_SQ + σ²_E + γ" is true by definition, but it is not a decomposition of the real
pipeline's variance into independent sources. σ²_S, σ²_Q, σ²_SQ and σ²_E are components of the *untied*
pipeline, expressed relative to V_R. Tying changes two things:

- it correlates the sample and panel effects;
- it changes the distribution of the interaction on the diagonal.

γ can therefore be negative. In the analytic model of `pmtrd/crossed/synthetic.py` (panel latent
φ = a s + b ξ, interaction λ s φ) the exact value is γ = 2αβa + λ²a², which is negative whenever αβa < −λ²a²/2.
The audit's statement that the shares "add to 1 by construction" is correct only because γ is defined as the
remainder.

## 6. Relation to Eq. 3 (corrected)

Eq. 3 uses one fixed seed m0 in both arms and a frozen panel P0 = A(T) selected on all of T. Its estimand is
1 − E_x Var_U h(U, P0, m0) / E_x Var_S h(S, (S, ξ), m0). The audit claimed ψ ≈ Π_Q + Π_γ "under homogeneity".
That needs three conditions, not one:

1. Var_U h(U, P0, m0) equals the average over panels drawn from A(S) (homogeneity). P0 is selected on the
   whole of T, so it is not a draw from A(S), and the condition is a real assumption.
2. The seed terms are negligible. With m0 fixed, σ²_W is absent from both arms, while σ²_UW at m0 stays in
   both arms.
3. No P0-specific interaction.

Under these conditions Eq. 3 = (σ²_Q + γ) / V_R. The simulation measures the gap.

## 7. Identifiability with incomplete designs and several seeds

- σ²_S and σ²_Q need D_both, so at least one disjoint pair (R ≥ 4).
- D_col needs a column with two present cells; D_row needs a row with two present cells.
- σ²_E needs at least one cell with M ≥ 2 seeds. Otherwise only σ²_SQ + σ²_E is identified, and the code
  returns σ²_E = NaN with σ²_SQ holding the sum.
- Cells missing completely at random keep every moment unbiased (unit test with 30% missing). Missingness
  that depends on the cell's value (for example fits that fail only for some panels) biases the estimators.
- The sparse cyclic design used for the truth (cells (r, r + d)) satisfies all four conditions.

## 8. Finite-sample behaviour and dependence

- All held-out patients are scored by the same draws, so the per-patient moments are dependent. Averaging
  over patients does not reduce draw-level noise.
- The cohort share Π_c = mean_x σ̂²_c / mean_x V̂_R is a ratio of estimators that share the same R draws. It
  is consistent as R → ∞ but biased at finite R, with a bias of order 1/R. Its standard error is driven by
  the number of draws, not of patients.
- D_both is a U-statistic of degree 4 in the draws. At R = 10 it averages 5,040 ordered disjoint pairs, but
  they come from only 10 independent draws.

## 9. Validity of the pigeonhole bootstrap

The data form a crossed array: patients (independent and identically distributed from the population) by
draws (independent and identically distributed given T). The pigeonhole bootstrap (Owen 2007) resamples
both margins independently. For a crossed random-effects mean it is approximately unbiased for the variance
and slightly conservative. Two adaptations are needed here:

- **Ties.** A draw resampled twice must not pair with itself. Using the copies as distinct draws would add
  zero-difference pairs and shrink D_col and D_row. The implementation uses multiplicity weights over pairs of
  distinct original draws (weighted U-statistics).
- **Target.** Resampling patients targets the population of patients. If the estimand is defined for the given
  patients, only draws should be resampled ("draws" mode).

Bootstrap consistency for non-degenerate U-statistics is an asymptotic statement (R → ∞). At R = 10, with a
skewed ratio statistic and about 6.5 distinct draws per resample, nothing guarantees nominal coverage. The
validity of the interval at R = 10 is therefore not established by theory; it is tested empirically in E2.

## 10. Checks executed

`pytest tests/test_crossed_estimator.py tests/test_crossed_pipeline.py`: 22 tests, all passed on
8 Nov 2026 (log in `runs/crossed_sim/test_log.txt`). They cover:

- unbiasedness of σ̂²_S, σ̂²_Q, σ̂²_SQ, σ̂²_E, γ̂ and V̂_R for 7 analytic cases (additive, interaction, random
  panel, fixed panel, deterministic classifier, seed variability, coupling), within 4 Monte Carlo SEs over
  400 designs;
- unbiasedness with 30% of cells missing;
- exact zeros: D_seed = 0 for a deterministic classifier, both analytic and logistic regression, and for
  XGBoost without subsampling;
- the disjoint-pair count;
- equality of the vectorised bootstrap with the one-at-a-time computation;
- pipeline controls: fixed panel (Q, SQ, γ ≈ 0), random panel (γ ≈ 0, Q > 0), XGBoost seed variance > 0.
