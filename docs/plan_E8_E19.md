# Analysis plan for the independent revision experiments E8–E11 and E15–E19

Written and committed on 8 Nov 2026, before any of these experiments was run. Source: the experiment table of
the revision audit (D3). The experiments that use the crossed estimator (E5, E6, E7, E12, E13, E14) are not
part of this plan: the pre-registered E1–E4 simulation returned NO-GO, and their use waits for scientific
review.

Rules for every experiment:

- No threshold, metric or test below is changed after a result is seen. Every result is reported, whatever its
  direction.
- Seeds are fixed in the code; every output is written under `runs/revision/E*/`, with a log and metadata.
- Nothing in the manuscript is changed.
- The machine has 2 CPU cores. The six-stage selection A4 takes about 41 s per run on GSE89843 and 5–9 s on the
  other data sets (saved run logs). Where the audit design is not feasible on this machine, the reduction is
  stated here, before the run, and labelled as a deviation.

Data sets: GSE89843 (discovery, 633 samples, GREIN counts), GSE68086 pan-cancer (generic loader, as in the saved
generalisation run), Gordon lung and Singh prostate (microarray), and the 94 independent GSE68086 samples for the
external test. The four data sets of E16, E18 and E19 are those of the saved replicate studies.

## E19 — ambiguity depends on R (saved scores)

- Units: 4 data sets × 6 selectors (six_stage, mw_topk, l1_topk, stability, mrmr_ifs_svm, random) plus the fixed
  arm; R = 10 replicates per outer fold; saved held-out scores and training-derived thresholds.
- Quantities per (data set, method), averaged over folds:
  1. pairwise disagreement D: mean over replicate pairs of the fraction of samples labelled differently (its
     expectation does not depend on R) — **primary metric**;
  2. ambiguity A(m) for m = 2, …, 10 replicates: mean over all subsets of size m (at most 200 random subsets
     per m, seed 19);
  3. common-threshold disagreement D_c: every replicate labelled at the fold's median threshold; the share of
     D due to threshold variation is 1 − D_c / D (may be negative);
  4. ensemble disagreement: the 10 replicates split into two disjoint ensembles of 5 (mean score, mean
     threshold); fraction of samples labelled differently, averaged over all 126 splits.
- Baseline: the published ambiguity at R = 10.
- Interval: pigeonhole bootstrap (B = 1,000, seed 19) for D, D_c and the threshold share: patients resampled
  within folds, replicate indices resampled within folds with multiplicity weights, pairs formed only between
  distinct replicates. Ensemble disagreement and A(m) curves: descriptive.
- Amendment (8 Nov 2026, before E19 was run): A(10) gets a patients-only interval. Resampling replicates
  would leave about 6.5 distinct replicates per resample, and ambiguity depends on that number, which is the
  point of E19.
- Output: `runs/export/E19.csv` (copy) and `runs/revision/E19/`.

## E18 — correlation-aware stability (saved panels plus expression)

- Units: 4 data sets × 6 selectors; the 10 replicate panels of every outer fold, averaged over folds as Φ is.
- Correlations: Spearman correlation between genes on the training samples of each outer fold, on the scale the
  selectors use (log2-CPM for counts, log2 for microarray intensities).
- Measures:
  1. Φ_C (Sechidis et al. 2019): 1 − tr(C S) / tr(C Σ0), with S the unbiased covariance of the selection
     indicators, Σ0 their covariance under random selection of panels with the observed sizes, and
     C_fg = |ρ_fg| if |ρ_fg| ≥ θ (f ≠ g), C_ff = 1. θ = 0.5 primary; 0.3 and 0.7 reported as sensitivity.
     With C = I it equals Nogueira's Φ (unit test).
  2. Matched overlap at θ: for two panels, the fraction of genes of one with a partner (same gene or |ρ| ≥ θ)
     in the other, averaged both ways and over pairs. This is a descriptive measure defined here.
- The audit also named Hamer & Dupont (2021). Its definition was not verified in this session, so it is not
  implemented or cited.
- Interval: leave-one-replicate-out jackknife over folds, as for Φ.
- Output: `runs/export/E18.csv`.

## E16 — unit of inference (saved runs; exploratory)

- Units: 4 data sets × 5 XGBoost selectors (six_stage, mw_topk, l1_topk, stability, random); mRMR-IFS-SVM has
  no ψ and is excluded. 20 points, 5 per data set.
- Outcome: the bagging gain as defined in the saved runs (bagged AUC − mean single-replicate AUC).
- Predictors: Eq. 3 ψ, V_R (mean across-replicate variance of the predicted log-odds, re-selected arm), Φ,
  Φ_C (E18, θ = 0.5). Π_Q and Π_S are not available (crossed estimator NO-GO); E13 classifiers were not run, so
  only XGBoost enters. Both are deviations from the audit.
- Primary statistic: mean over data sets of the within-data-set Spearman correlation of predictor and gain.
  p-value: stratified permutation (predictor values permuted among selectors within each data set),
  20,000 permutations, seed 16, two-sided on the absolute mean correlation.
- Secondary: linear mixed model gain ~ x + (1 | data set) (statsmodels MixedLM, REML); leave-one-selector-out
  prediction within each data set (linear fit on 4 selectors, predict the 5th), mean absolute error against
  predicting the data-set mean.
- With five points per data set this is exploratory and is labelled so.
- Output: `runs/export/E16.csv`.

## E8 — matched benchmark (GSE89843)

- Split: the paper's five outer folds (`R8_outer_folds.csv`); selection and training on the whole outer
  training fold; held-out fold untouched.
- Arms: mw_topk, l1_topk, stability, mrmr (IFS size forced to 27) at k = 27, and the same four at k = the
  mRMR-IFS size chosen in that fold; each with the paper's XGBoost and with the mRMR-IFS linear SVM. Reference:
  A4 (six_stage) with XGBoost and with the SVM; the original mRMR-IFS-SVM (own size, SVM).
- Metric: pooled out-of-fold AUC with DeLong 95% CI.
- Primary comparisons (5, Holm-adjusted, paired DeLong): A4-XGBoost against mw-27-XGBoost, l1-27-XGBoost,
  stability-27-XGBoost, mrmr-27-XGBoost and the original mRMR-IFS-SVM. Every other comparison is descriptive.
- Φ and ψ are reported only where the saved replicate study matches the arm (k = 27, XGBoost); a replicate
  study for every new arm is not run (deviation: cost).
- Output: `runs/revision/E8/`.

## E10 — external transfer, 18 versus 27 genes

- Training: all 633 GSE89843 samples; test: the 94 independent GSE68086 samples; log2-CPM with CPM computed over
  the genes quantified in both cohorts (as in the paper); XGBoost with the paper's settings, no tuning.
- "Measurable" gene: non-zero count in at least 10% of the 94 external samples (no labels used).
- Models: (a1) locked 27-gene model as published (undetected genes enter as measured, i.e. zero); (a2) the same
  model with every non-measurable panel gene imputed by its discovery median log2-CPM; (b) 18-gene model
  re-trained on the measurable panel genes; (c) A4 re-derived on all 633 discovery samples restricted to
  measurable genes, then trained on all 633.
- Null: 1,000 random panels of the size of model (b), drawn uniformly from the discovery Tier-2 pool restricted
  to measurable genes (seed 10); empirical p = (1 + #{null AUC ≥ observed}) / 1,001 for (b) and (c).
- Metrics: external AUC (stratified bootstrap 95% CI, 2,000 resamples, seed 10), calibration intercept and
  slope, Brier score; paired bootstrap of the AUC difference for (b) − (a1), (c) − (a1) and (a2) − (a1).
- Check: (a1) must reproduce the published external AUC of the 27-gene model; if it does not, that is reported
  first.
- Amendment (8 Nov 2026, before E10 was run): the plan defined "measurable" as non-zero in at least 10% of the
  external samples but also called model (b) the 18-gene model. The paper's 18 genes are those with any read in
  GSE68086; under the 10% rule USP27X and NTNG2 (reads in at most three samples) drop out. Model (b) therefore
  uses the paper's 18 genes (any read), so that it tests the published claim, and (b') re-trains on the
  panel genes measurable under the 10% rule as a sensitivity analysis. Models (a2) and (c) and the null use
  the 10% rule as planned; the null panels have the size of (b).
- Output: `runs/external/E10.csv`.

## E11 — normalisation (GSE89843)

- Three scales: (a) GREIN counts as used now; (b) log2-CPM (co-primary); (c) log2 TMM-CPM, with TMM factors
  computed against a reference sample chosen within the training fold (upper-quartile closest to the mean, as
  in edgeR) and applied unchanged to held-out samples. Per-sample transforms only; nothing fitted on held-out
  samples.
- For each scale: nested A4 on the paper's folds (pooled out-of-fold AUC, DeLong CI), and the replicate design
  of the paper (R = 10, f = 0.8, same subsample seeds) for Φ and ψ (fixed arm: the scale's own A4 fold panel).
- No scale is chosen by AUC; all three are reported side by side. Paired DeLong (descriptive) of (b) and (c)
  against (a).
- Output: `runs/revision/E11/`.

## E15 — selection reproducibility (GSE89843)

- A4 on the full cohort with 20 seeds (1–20) in two environments: E-A (current settings, 2 threads) and E-B
  (one thread for XGBoost and BLAS). A second container or library version is not available here (deviation).
- Nested A4 on the paper's folds for seeds 1–5 (environment E-A).
- Metrics: Jaccard of each panel with P27 (the historical panel) and between runs; panel sizes; identical
  panels across environments for the same seed; nested AUC over the 5 seeds. Descriptive.
- Amendment (8 Nov 2026, before E15 was run): in the code, A4's `seed` argument sets only the Stage-2 CV split;
  Stage-5 resample b always uses RandomState(b). A "seed" in E15 therefore sets both: Stage-2 seed s and
  Stage-5 resample b drawn with RandomState(1000 s + b). This is done by wrapping the resampling function in
  the E15 module; the engine file is not edited. Seed 0 with offset 0 is the paper's setting and is run as a
  reference in both environments.
- Output: `runs/seeds/E15.csv`.

## E17 — pathway circularity (Vumc)

- 25 class-stratified random splits of the Vumc samples into halves (seeds 1–25).
- Per half: Mann–Whitney test per gene (log2-CPM) and per gene set (single-sample set scores of
  `pmtrd.pathway`, Hallmark and Reactome, as in the pathway model).
- Replication at matched list size m ∈ {27, 50}: the m features with the smallest p in half A are tested in half
  B; replication rate = fraction with BH q < 0.05 (over the m tested) and the same direction; both directions
  (A→B, B→A) averaged. Gene-level and set-level rates compared with a Wilcoxon signed-rank test over splits.
- A4 re-derived in each half; chance-corrected overlap of the two A4 panels (Kuncheva, at the smaller size)
  against that of the top-k set lists of the pathway model (k as in the pathway model). Wilcoxon signed-rank
  over splits.
- Baseline: the current split-half analysis (reported in the paper).
- Output: `results/pathway/E17.csv`.

## E9 — bagging (reduced design)

- Audit design: 5 repeats of 5-fold outer CV, B ∈ {10, 25, 50}, 4 data sets. At about 41 s per A4 run on
  GSE89843 that is about 14 hours for A4 on one data set, which is not feasible here.
- Reduced design (fixed now): A4 only (the selector the bagging claim is about); 2 partitions (the paper's
  seed-42 folds, whose 10 saved replicates are reused, and a new partition, seed 43); B ∈ {10, 25}; 25
  subsample selections per outer fold (replicates 0–24, the paper's seed scheme).
- Arms per outer fold: bagged ensembles of the first 10 and of all 25 replicate models (mean of scores); single
  replicate models; the frozen full-fold panel trained on the whole training fold.
- Gain (one definition throughout): AUC of the bagged ensemble on the held-out fold minus the mean AUC of its
  member models, per fold, averaged over folds. Also AUC(bagged) − AUC(frozen full-fold).
- Calibration of the bagged and single models: intercept, slope, Brier on held-out folds. Disagreement between
  two independent 10-model ensembles (replicates 0–9 and 10–19).
- Test: corrected resampled t-test (Nadeau & Bengio 2003) over the 10 fold-level differences, correction
  1/J + n_test/n_train.
- External: an ensemble of 25 A4 selections on stratified 80% subsamples of all 633 samples, applied to the 94
  GSE68086 samples (log2-CPM, no alignment); paired bootstrap against the locked 27-gene model.
- Output: `runs/bagging/E9_*.csv`.
