# Revision experiments E8–E19: results, deviations and claim audit

Nov 8, 2026 · @proma

## Summary

All 13 runnable experiments are finished (E1–E4, E8–E11, E15–E19). The six that need the crossed estimator (E5–E7, E12–E14) were not run, because its pre-registered check returned NO-GO. No threshold or test was changed after a result was seen, every deviation is listed below, and the manuscript was not changed.

- **Reproduction.** Every published number re-checked here reproduced exactly: nested AUC 0.9325, external 0.8481 with its calibration, bagged 0.9630 and +2.83, the benchmark AUCs, Φ and the ambiguity.
- **Discrimination.** With size and classifier matched, mRMR (+3.6 pts) and stability selection (+2.0) beat A4 (E8). The paper's seed gives the highest nested AUC of six seeds (0.921–0.933; E15), and TMM normalisation lowers it by 3.8 pts (E11).
- **External transfer.** Random panels from the same pool transfer about as well as the locked panel (0.848; p = 0.38; E10), so the transfer is not evidence for these genes. A bagged A4 ensemble transfers better (0.915; p = 0.002; E9).
- **Gene stability.** A4's genes are unstable even when correlated genes count as matches (Φ\_C 0.12; E18). A new seed gives a panel with Jaccard 0.62 to the 27 genes (E15), but the data drive most of the instability, not the seed.
- **Patient level.** The "40%" is the ambiguity at R = 10 and grows with R; the R-free disagreement is 14.9%, against 6.0% with the panel fixed (E19). Bagging cuts it to 4.0% and raises the AUC by 3 pts (E9).
- **ψ.** It tracks the bagging gain, but so does the simpler V\_R, which predicts it better (E16); the published ψ interval under-covers (E1–E4).
- **Genes versus pathways.** At matched list size, genes and gene sets replicate and are stable to the same degree; the gap the paper reports is between A4 and a simple ranking (E17).
- **Multiple testing.** Five results lose significance under at least one correction (stability selection vs A4, TMM, model b', Singh's gain, one GSE68086 contrast); the main findings hold under all of them.

**Verdict:** major revision of the claims. A4 has no demonstrated advantage in discrimination, stability, external transfer or pathway coherence, and the headline "pathway-level biology is stable while individual genes are not" must go. The evidence supports a reframed methods paper with an honest case study (Publication-readiness audit, below); whether to take that route is for the authors to decide.

## Status of every experiment

Thirteen of the 19 audit experiments have been run; six were not run because they depend on the crossed estimator, whose pre-registered simulation (E1–E4) returned NO-GO. No experiment was dropped because of its result.

| ID | Question (audit row) | Status | Output |
| --- | --- | --- | --- |
| E1–E4 | ψ identification, bias and CI; correlated genes; design dependence | Done. Pre-registered verdict NO-GO (34 of 168 cells passed) | runs/crossed\_sim/v1/ |
| E5 | Seed randomness; choice of control | Not run: needs the crossed estimator | — |
| E6 | Selection vs fitting variance | Not run: needs the crossed estimator | — |
| E7 | Decomposition-based deployment rule | Not run: needs Π components | — |
| E8 | Matched benchmark | Done | runs/revision/E8/ |
| E9 | Bagging gain, internal and external | Done | runs/bagging/ |
| E10 | External transfer, 18 vs 27 genes | Done | runs/external/ |
| E11 | Normalisation | Done | runs/revision/E11/ |
| E12 | Site confounding within Vumc | Not run: needs the crossed estimator and E7 | — |
| E13 | Classifier dependence of ψ | Not run: planned with Π components | — |
| E14 | Choice of R and f | Not run: needs the crossed estimator | — |
| E15 | Selection reproducibility over seeds | Done (descriptive) | runs/seeds/ |
| E16 | Unit of inference: which quantity predicts the bagging gain | Done (exploratory) | runs/export/E16.csv |
| E17 | Pathway circularity at matched list size | Done | results/pathway/ |
| E18 | Correlation-aware stability | Done | runs/export/E18.csv |
| E19 | Ambiguity depends on R | Done | runs/export/E19.csv |

E13 needs only classifier fits on saved panels and could be run without the crossed estimator if ψ alone (Eq. 3) is wanted; that would be a new decision, not part of the current plan.

## Results per experiment

Every number below comes from the saved output file named in each subsection. Intervals are 95%; "pts" means AUC percentage points.

### E8 — matched benchmark (GSE89843, paper's outer folds)

With the classifier and the panel size matched, mRMR and stability selection discriminated better than A4, and the univariate filter and L1-logistic selection matched it. The gap to mRMR is therefore a property of the selector, not of the SVM or of mRMR's own panel size.

| Arm (k = 27, XGBoost unless stated) | Nested AUC (95% CI) | A4 − arm, pts (95% CI) | Holm p |
| --- | --- | --- | --- |
| A4 (reference) | 0.9325 (0.913–0.952) | — | — |
| mRMR, forced to 27 genes | 0.9683 (0.957–0.980) | −3.58 (−5.31 to −1.84) | 0.0003 |
| mRMR–IFS–SVM, own size (as published) | 0.9630 (0.951–0.975) | −3.05 (−4.89 to −1.20) | 0.0048 |
| Stability selection | 0.9527 (0.936–0.969) | −2.02 (−3.48 to −0.55) | 0.021 |
| L1-logistic | 0.9460 (0.929–0.963) | −1.35 (−2.75 to +0.06) | 0.12 |
| Mann–Whitney filter | 0.9353 (0.917–0.954) | −0.29 (−2.43 to +1.86) | 0.79 |

- The four published arms reproduce the published AUCs exactly (0.9353, 0.9460, 0.9527, 0.9630). The new arm is mRMR at 27 genes with XGBoost.
- Descriptive arms: A4 genes with the linear SVM 0.8923; the highest AUC of all 18 arms was the Mann–Whitney filter at mRMR's per-fold size with the SVM (0.9702).
- Source: runs/revision/E8/E8.csv.

### E9 — selection bagging (A4; 2 partitions × 5 outer folds; 4 data sets; external ensemble)

Bagging A4's selections helps where re-selection moves predictions: the per-fold gain is +3.0 pts in the discovery cohort and +5.2 in GSE68086, about +1 in Singh and nothing in Gordon. The published pooled numbers reproduce exactly (0.9630, +2.83) and replicate on a new partition. Post hoc, bagged A4 is level with mRMR, not better. In the independent cohort, a 25-member ensemble transfers better than the locked panel (AUC 0.915 against 0.848; p = 0.002).

| Data set | Gain, B = 10, pts (95% CI; p) | Gain, B = 25 | Bagged (B = 25) − frozen panel | Two 10-model ensembles disagree | Single models disagree (D, E19) |
| --- | --- | --- | --- | --- | --- |
| GSE89843 | +3.03 (+2.09 to +3.96; < 0.0001) | +3.20 (+2.40 to +4.01; < 0.0001) | +3.55 (+1.87 to +5.24; 0.001) | 4.0% | 14.9% |
| GSE68086 | +5.15 (+2.61 to +7.70; 0.001) | +5.16 (+3.06 to +7.26; 0.0004) | +6.16 (+2.87 to +9.45; 0.002) | 5.4% | 14.0% |
| Singh | +1.05 (+0.03 to +2.08; 0.045) | +1.19 (−0.41 to +2.79; 0.13) | +2.45 (−4.45 to +9.36; 0.44) | 0.0% | 9.0% |
| Gordon | +0.37 (−0.46 to +1.20; 0.34) | +0.33 (−0.41 to +1.08; 0.34) | +0.10 (−0.19 to +0.40; 0.44) | 2.5% | 4.4% |

Gain = AUC of the bagged ensemble on the held-out fold minus the mean AUC of its members, averaged over the 10 folds; intervals and p from the corrected resampled t-test (Nadeau & Bengio), as planned. Bagged (B = 10) − frozen: +3.36 (p = 0.001), +6.29 (p = 0.004), +2.46 (p = 0.47), +0.11 (p = 0.61) in the same order.

| Pooled over folds, GSE89843 (published definition) | Paper's folds (seed 42) | New partition (seed 43) |
| --- | --- | --- |
| Bagged AUC, B = 10 | 0.9630 (published 0.9630) | 0.9609 |
| Gain, B = 10 (bagged − mean of the 50 single units) | +2.83 (published +2.83) | +2.91 |
| Bagged AUC, B = 25 | 0.9646 | 0.9655 |
| Gain, B = 25 | +2.97 | +3.35 |

Post hoc, not in the plan (D7): on the paper's folds, bagged A4 (B = 25, 0.9646) against mRMR at 27 genes with XGBoost (0.9683): −0.36 pts (−1.46 to +0.74; p = 0.52); against mRMR–IFS–SVM (0.9630): +0.17 (−1.19 to +1.53; p = 0.81); against A4's single frozen panel (0.9325): +3.22 (+2.13 to +4.30). With B = 10: −0.52 (p = 0.35), +0.01 (p = 0.99), +3.06. Paired DeLong; descriptive.

- Calibration: bagging lowers the Brier score in every data set (GSE89843, seed 42: 0.074 against 0.096 for single and frozen models) but makes the predictions under-confident: calibration slopes of the bagged models are 1.09–1.81 across data sets, against 0.69–1.54 for single models. A bagged model needs recalibration.
- B = 25 adds little to B = 10: the mean per-fold gain changes by −0.04 to +0.17 pts.
- "Matched the strongest comparator" compares a bagged selector with single ones; the comparators were not bagged.
- On the paper's folds, replicates 0–9 are the saved ones; replicates 10–24 and the whole of partition 43 are new.
- Sources: runs/bagging/E9\_folds.csv, E9\_calibration.csv, E9\_tests.csv, E9\_posthoc.csv.

**External part** (25 A4 selections on 80% subsamples of all 633 discovery samples, applied to the 94 GSE68086 samples; amendment of 14:38):

| Model | External AUC (95% CI) | Calibration intercept / slope | Brier |
| --- | --- | --- | --- |
| 25-member bagged A4 ensemble | 0.915 (0.856–0.962) | 1.28 / 1.98 | 0.174 |
| Its members, mean (range) | 0.871 (0.777–0.931) | — | — |
| Locked 27-gene model (E10 a1) | 0.848 | 2.80 / 0.91 | 0.236 |

- Pre-registered test: ensemble − locked = +6.71 pts (+2.31 to +11.99; paired bootstrap p = 0.002). Gain over its own members: +4.45 pts (+2.62 to +6.35).
- Each member selected a median of 27 genes, of which a median of 6 are not in the matrix shared with GSE68086 and were dropped, as the amendment specified.
- The ensemble is closer to calibrated in the new cohort than the locked model (intercept 1.28 against 2.80) but under-confident (slope 1.98); it still needs recalibration.
- The comparison is between a re-selecting ensemble and one locked panel; no random-panel or bagged-comparator ensemble was run, so this shows that bagging A4 helps transfer, not that A4's genes are special. The members alone already average 0.871, above the locked model and above the median random 18-gene panel of E10 (0.815).
- Sources: runs/bagging/E9\_external.csv, E9\_external\_members.csv.

### E10 — external transfer, 18 versus 27 genes (GSE89843 → 94 GSE68086 samples)

The panel transfers (AUC 0.848), but random panels from the same informative pool transfer about as well, so the transfer is not evidence for these particular genes. The pre-registered check passed: the locked model reproduced the published AUC 0.8481 and calibration (intercept 2.80, slope 0.91, Brier 0.236).

| Model | Genes | External AUC (95% CI) | Model − locked, pts (95% CI; p) | Calibration intercept / slope |
| --- | --- | --- | --- | --- |
| (a1) locked 27-gene, as published | 27 | 0.848 (0.761–0.923) | — | 2.80 / 0.91 |
| (a2) locked, unmeasurable genes set to discovery median | 27 | 0.853 (0.769–0.925) | +0.51 (−1.06 to +1.94; 0.46) | 2.23 / 0.83 |
| (b) re-trained on the paper's 18 genes | 18 | 0.844 (0.752–0.924) | −0.37 (−2.69 to +1.90; 0.76) | 2.57 / 0.96 |
| (b') re-trained on genes measurable in ≥10% of samples | 16 | 0.826 (0.732–0.909) | −2.22 (−4.77 to −0.14; 0.036) | 1.93 / 0.84 |
| (c) A4 re-derived on measurable genes | 29 | 0.884 (0.812–0.947) | +3.61 (−0.79 to +9.45; 0.13) | 2.37 / 1.01 |

Random-panel nulls (18 genes per panel):

| Null pool | Panels | Median AUC | p for (b) 0.844 | p for (c) 0.884 |
| --- | --- | --- | --- | --- |
| Pre-registered: discovery Tier-2 genes measurable in GSE68086 | 1,000 | 0.815 | 0.38 | 0.19 |
| Published (post hoc check): discovery mean log2-CPM > 1, panel genes excluded | 200 | 0.765 | 0.17 | 0.020 |

- The second null row is computed post hoc on the published file R3\_random\_panel\_control.csv; it is shown because the manuscript uses that null ("a baseline of about 0.75"). Against it the locked 27-gene model has p = 0.12.
- Every model under-predicts NSCLC in GSE68086 (intercepts 1.9–2.8), as the paper reports.
- The null panels have 18 genes; model (c) has 29, so its p-values compare unequal sizes, as the plan specified.
- Sources: runs/external/E10.csv, E10\_null.csv, E10\_published\_null\_posthoc.csv.

### E11 — normalisation (GSE89843, paper's outer folds)

Nested discrimination is unchanged under CPM but falls by 3.8 pts under TMM-CPM, so "robust in sensitivity analyses" holds for library-size scaling and not for composition-based normalisation. No scale was chosen by AUC; all three are reported.

| Input scale given to A4 and XGBoost | Nested AUC (95% CI) | Scale − counts, pts (95% CI; p) | Φ (replicates) | ψ | Fold panel sizes |
| --- | --- | --- | --- | --- | --- |
| (a) GREIN counts (published) | 0.9325 (0.913–0.952) | — | 0.309 | 0.844 | 26, 27, 27, 29, 24 |
| (b) CPM, linear | 0.9301 (0.911–0.949) | −0.23 (−2.16 to +1.69; 0.81) | 0.351 | 0.823 | 29, 30, 29, 29, 28 |
| (c) TMM-CPM, linear, training-fold reference | 0.8948 (0.870–0.920) | −3.77 (−6.33 to −1.20; 0.004) | 0.288 | 0.871 | 29, 29, 28, 28, 26 |

- The paired DeLong tests are descriptive, as planned.
- TMM factors in this cohort are extreme: 0.10 to 7.0 within each training fold (5th–95th percentile 0.30–4.8), and correlated −0.84 with library size. An independent implementation (conorm 1.2.0) gives the same pattern (range 0.08–7.6; correlation of log factors 0.996). R and edgeR are not available here, so the factors are not checked against edgeR itself.
- The ψ intervals are patients-only bootstrap intervals, which E1–E4 showed to under-cover; they are in E11.csv but not interpreted.
- Sources: runs/revision/E11/E11.csv, E11\_tmm\_factors.csv, E11\_tmm\_check.txt.

### E15 — selection reproducibility over seeds (GSE89843)

The seeded procedure is deterministic and its nested AUC moves by about one point across seeds, but the panel it selects depends on the seed: with a new seed, the panel's Jaccard with the published 27 genes averages 0.62, and 16 genes recur in every seed. Seed variation is still much smaller than variation across training subsamples.

| Quantity (fold-recomputable A4, without the PPI and prior terms) | Value |
| --- | --- |
| Same seed, 2 threads (A) vs 1 thread (B): identical panels | 21 of 21 seeds |
| Seed 0, the paper's setting: panel size; Jaccard with the 27-gene panel | 26 genes; 0.710 (published fold-recomputable value 0.710) |
| Seeds 1–20: panel size | 25–27 genes |
| Seeds 1–20: Jaccard with the 27-gene panel, mean (range) | 0.620 (0.472–0.767) |
| Seeds 1–20: mean pairwise Jaccard between seeds; Nogueira Φ | 0.749; 0.851 |
| Genes selected in all 20 seeds / in at least 10 / in any | 16 / 27 / 38 |
| Nested AUC, seeds 1–5, paper's folds (95% CI) | 0.921 (0.899–0.942), 0.930 (0.910–0.949), 0.924 (0.904–0.945), 0.928 (0.908–0.948), 0.924 (0.903–0.946) |
| Nested AUC, seed 0 (published, reproduced in E8) | 0.9325 |
| Fold panels across seeds 1–5: mean pairwise Jaccard, per outer fold | 0.81, 0.78, 0.70, 0.79, 0.78 |

- All of E15 is descriptive, as planned. Environment B differs from A only in thread count (D12), so the identical panels show determinism given the seed, not portability across machines or library versions.
- The paper's seed gives the highest of the six nested AUCs; five extra seeds are too few to say more. All six (0.921–0.933) are below E8's mRMR arm at 27 genes with XGBoost (0.968), so the E8 deficit is not an artefact of the paper's seed.
- For scale: between seeds on the full cohort, panels have Jaccard 0.749; between the 80% training subsamples of the replicate study, A4 panels have Jaccard 0.185 (Φ 0.31, E18). The data, not the seed, drive most of A4's instability.
- Post hoc, not in the plan: of the 27 published genes, 16 are selected in all 20 seeds, 4 more in at least 10, and 3 (DUOXA1, ITGB3BP, NTNG2) in none. Of the manuscript's 7-gene "recurrent core", 5 are selected in all 20 seeds, TNFRSF17 in 19 and DDAH1 in 9.
- Scope: the published panel came from an unseeded development version, and E15 does not re-run that version or the seeded re-implementation with the PPI and prior terms (ReproJaccard 0.929). The seed dependence shown here is that of the fold-recomputable procedure the paper recommends locking.
- Sources: runs/seeds/E15.csv, E15\_panels.csv, metadata.json.

### E16 — which quantity predicts the bagging gain (4 data sets × 5 XGBoost selectors; exploratory)

ψ tracks the bagging gain, but V\_R, the plain across-replicate variance that needs no panel-fixed arm, tracks it equally well and predicts a left-out selector's gain better; ψ predicts it no better than the data-set mean. Stability (Φ, Φ\_C) tracks it less closely.

| Predictor | Mean within-data-set Spearman | Permutation p | Mixed-model slope per SD, pts (95% CI) | Leave-one-selector-out MAE / baseline MAE |
| --- | --- | --- | --- | --- |
| Eq. 3 ψ | 0.90 | < 0.0001 | +1.25 (0.89 to 1.62); did not converge | 1.03 |
| V\_R | 0.90 | < 0.0001 | +1.44 (1.09 to 1.80) | 0.51 |
| Φ | −0.73 | 0.002 | −1.01 (−1.39 to −0.63) | 1.01 |
| Φ\_C (θ = 0.5) | −0.75 | 0.001 | −1.16 (−1.46 to −0.85) | 0.72 |

- Per data set (ψ): GSE68086 0.7, GSE89843 1.0, Gordon 0.9, Singh 1.0; V\_R has the same values.
- With five points per data set, each within-data-set correlation is coarse, and the difference between predictors was not tested (it was not pre-specified).
- The mixed model for ψ reported a convergence warning, so its interval is not reliable.
- Not done as audited: Π\_Q and Π\_S (crossed estimator NO-GO) and the E13 classifiers. Gains are those of the saved 10-replicate runs, not of E9.
- Sources: runs/export/E16.csv, runs/revision/E16/E16\_data.csv.

### E17 — genes versus pathways at matched list size (Vumc, 25 split halves of 188 samples)

At matched list size, gene-level associations replicate exactly as well as pathway-level ones, and simple top-k gene lists are as stable as top-k set lists. The large stability gap the manuscript reports is a gap between A4's selection and a simple ranking, not between genes and pathways.

| Comparison (pre-registered) | Genes or A4 | Gene sets | Wilcoxon p |
| --- | --- | --- | --- |
| Replication rate, top 27 (BH q < 0.05, same direction) | 1.000 | 1.000 | no difference in any split |
| Replication rate, top 50 | 1.000 | 0.999 | 0.16 |
| Kuncheva overlap of the two halves: A4 panel vs pathway model's top-27 sets | 0.083 | 0.582 | < 0.0001 |

| Post hoc (not pre-registered): same rule, smallest Mann–Whitney p | Top-k genes | Top-k sets | Wilcoxon p |
| --- | --- | --- | --- |
| Kuncheva overlap, k = 27 | 0.592 | 0.582 | 0.82 |
| Kuncheva overlap, k = 50 | 0.639 | 0.664 | 0.054 |
| Jaccard, k = 27 | 0.421 | 0.421 | — |

- The pre-registered replication rate hit its ceiling: in every split, every one of the top 27 genes and top 27 sets of one half was significant in the same direction in the other. It cannot separate the two levels in these data.
- The post-hoc rows were added because the pre-registered Kuncheva comparison confounds two things, the level (genes vs sets) and the selector (A4 vs a top-k ranking). They are labelled post hoc and were not tested for multiplicity.
- Universe: the 21,500 genes shared with GSE68086 and 1,064 Hallmark and Reactome sets, as in the pathway model. The published split-half analysis used GSEA, so its numbers (top-27 gene Jaccard 0.38; set-list Jaccard 0.59) are not directly comparable.
- Sources: results/pathway/E17.csv, E17\_splits.csv, E17\_posthoc\_toplists.csv.

### E18 — correlation-aware stability (saved replicate panels, 4 data sets)

Allowing correlated genes to count as partial matches does not rescue A4's stability: in the discovery cohort its Φ\_C is 0.12 against Φ = 0.31, and the univariate filter stays far ahead. This confirms the manuscript's statement that gene identity is unstable.

| Data set | A4 Φ | A4 Φ\_C, θ = 0.5 (95% CI) | A4 matched overlap, θ = 0.5 | Mann–Whitney Φ\_C | Random panels Φ\_C / matched overlap |
| --- | --- | --- | --- | --- | --- |
| GSE89843 | 0.309 | 0.119 (0.060–0.179) | 0.65 | 0.938 | −0.72 / 0.82 |
| Gordon | 0.438 | 0.476 (0.451–0.500) | 0.65 | 0.717 | −0.01 / 0.10 |
| GSE68086 | 0.328 | 0.327 (0.303–0.351) | 0.46 | 0.528 | 0.02 / 0.36 |
| Singh | 0.305 | 0.282 (0.218–0.347) | 0.69 | 0.395 | −0.24 / 0.84 |

- Matched overlap is not chance-corrected. Random panels drawn from the correlated Tier-2 pool reach 0.82 in GSE89843, so a high matched overlap is not evidence of stable selection.
- Φ\_C becomes strongly negative for mRMR in GSE68086 (−2.00) and Singh (−0.56), and for random panels in GSE89843. Φ\_C is not bounded below, and why it falls so far for these selectors was not investigated. It should be reported with Φ, not instead of it.
- Sensitivity: A4 Φ\_C in GSE89843 is 0.098 at θ = 0.3 and 0.202 at θ = 0.7.
- Not done as audited: the Hamer & Dupont (2021) index (definition not verified here).
- Source: runs/export/E18.csv.

### E19 — ambiguity depends on R (saved scores and thresholds, 4 data sets)

The published "40% of patients received conflicting classifications" is the ambiguity at R = 10 replicates and grows with R; the R-free quantity, the disagreement between two re-derived models, is 14.9% for A4 in the discovery cohort against 6.0% with the panel fixed. Re-selection still more than doubles disagreement, but the headline number depends on an arbitrary R.

| Data set | A4 disagreement D (95% CI) | A4 ambiguity A(10) (95% CI) | Panel fixed: D / A(10) | Two 5-model ensembles disagree (A4) |
| --- | --- | --- | --- | --- |
| GSE89843 | 0.149 (0.128–0.170) | 0.401 (0.365–0.439) | 0.060 / 0.164 | 0.083 |
| GSE68086 | 0.140 (0.112–0.167) | 0.386 (0.329–0.443) | 0.087 / 0.246 | 0.065 |
| Singh | 0.090 (0.050–0.131) | 0.286 (0.198–0.374) | 0.078 / 0.206 | 0.020 |
| Gordon | 0.044 (0.025–0.066) | 0.115 (0.072–0.160) | 0.034 / 0.093 | 0.027 |

- A(m) rises steadily with m for A4 in GSE89843: 0.149, 0.223, 0.271, 0.307, 0.333, 0.355, 0.373, 0.388, 0.401 for m = 2–10. A(2) equals D by construction, and A(10) reproduces every published ambiguity.
- Threshold variation explains little of D for A4 (6.3%, −0.2% to 12.7%); most disagreement comes from the scores.
- Ensembles of five halve the disagreement of single models (0.083 against 0.149), consistent with bagging reducing the patient-level cost.
- D, D\_c and the threshold share use the pigeonhole bootstrap; A(10) uses a patients-only interval (amendment), which ignores replicate uncertainty.
- Source: runs/export/E19.csv.

## Plan versus implementation, and every deviation

The analysis plan (docs/plan\_E8\_E19.md) was committed at 12:38 on 8 November, before any of E8–E19 was run; each of its five amendments was committed before the experiment it changes was started. Deviations are of three kinds: reductions stated in the plan before running (P), amendments committed before the run (A), and departures found or made after the run started (R).

| Exp. | Audit design | What was run | Deviations |
| --- | --- | --- | --- |
| E1–E4 | Full factorial simulation | One-factor-at-a-time design, 7,266 units | D1 (P), D2 (R) |
| E8 | All selectors × 2 sizes × 2 classifiers; Holm over 5; Φ, ψ | As audited, 18 arms | D3 (P), D4 (R) |
| E9 | 5 × 5-fold CV, B ∈ {10, 25, 50}, 4 data sets; external ensemble | 2 partitions, B ∈ {10, 25}, A4 only, 4 data sets; external ensemble | D5 (P), D6 (A), D7 (R) |
| E10 | Locked, 18-gene, A4 on measurable genes; random 18-gene null | As audited, plus (a1) as published and (b') | D8 (A), D9 (R) |
| E11 | A4 on counts, log2-CPM, log2 TMM-CPM | Counts, linear CPM, linear TMM-CPM | D10 (A), D11 (R) |
| E15 | 20 seeds × 2 environments incl. container; nested for 5 seeds | 21 seeds × 2 thread settings; nested for 5 seeds | D12 (P), D13 (A), D19 (R) |
| E16 | Predictors ψ, V\_R, Φ, Φ\_C, Π\_Q, Π\_S; E13 classifiers | ψ, V\_R, Φ, Φ\_C; XGBoost only | D14 (P), D15 (R) |
| E17 | Matched-size replication, 25 Vumc splits; A4 per half | As audited | D16 (P) |
| E18 | Sechidis Φ\_C; Hamer & Dupont | Φ\_C at 3 thresholds; matched overlap | D17 (P) |
| E19 | D, A(m), D\_c, ensembles; pigeonhole CI | As audited | D18 (A) |

1. **D1 (E1–E4, P).** The 5,184-cell factorial was not feasible on 2 cores; a one-factor-at-a-time design around a base point was fixed in the plan before any run.
2. **D2 (E1–E4, R).** A post-hoc jackknife pilot first shared a panel cache across training sets, which biased it; that run was deleted, fixed and rerun. The pilot is outside the verdict and labelled post hoc in the crossed-estimator report.
3. **D3 (E8, P).** Φ and ψ are reported only for the arms that match the saved replicate study (k = 27, XGBoost); no new replicate study was run.
4. **D4 (E8, R).** An early version took the k = 27 lists of L1 and stability selection by truncating a longer selection. It was replaced by separate selections at each size, as the plan specifies; the committed output is a fresh run of the corrected code (12:47–12:50). mRMR–IFS was re-implemented in the revision module; it reproduces the published 0.9630, and A4 reproduces the saved nested scores to 1e-16.
5. **D5 (E9, P).** The audit design (about 14 h for A4 on GSE89843 alone) was reduced before running: 2 partitions (seed 42 with the 10 saved replicates reused, and seed 43), 25 replicates per fold, B ∈ {10, 25}, A4 only.
6. **D6 (E9, A).** The external part left the subsample seeds, the handling of genes absent from GSE68086 and the classifier's input open; these were fixed in an amendment committed before that part was started (14:38).
7. **D7 (E9, R).** The audit's baseline "best non-bagged selector" was not carried into the plan. Any comparison of bagged A4 with mRMR is therefore post hoc and descriptive. The gain uses one per-fold definition, which differs from the published pooled gain (bagged pooled AUC minus mean single-unit AUC).
8. **D8 (E10, A).** The plan's 10% rule would have dropped two of the paper's 18 genes; model (b) uses the paper's 18 genes, and (b') applies the 10% rule.
9. **D9 (E10, R).** The pre-registered null pool (Tier-2 ∩ measurable) differs from the paper's (expressed genes); the paper's null is shown as a labelled post-hoc comparison. The shared gene set has 15,830 genes against the published 15,833; the locked AUC still reproduces exactly.
10. **D10 (E11, A).** Scales (b) and (c) are linear, not log2, because Stage 1 computes log2 fold changes from the input means; fed log values it would remove almost every gene.
11. **D11 (E11, R).** TMM factors are not validated against edgeR (no R here); they agree with conorm 1.2.0 to a log-factor correlation of 0.996. The output folder is runs/revision/E11, not runs/normalisation.
12. **D12 (E15, P).** No second container or lockfile; environment B differs only in using one thread for XGBoost and BLAS.
13. **D13 (E15, A).** A4's seed sets only the Stage-2 CV split; Stage-5 resample b always uses RandomState(b). An E15 seed s therefore also offsets Stage 5 to RandomState(1000 s + b), through a wrapper in the E15 module; the engine is not edited. Seed 0 (the paper's setting) is added as a reference.
14. **D14 (E16, P).** Π\_Q and Π\_S are unavailable (NO-GO) and E13 was not run, so only Eq. 3 ψ, V\_R, Φ and Φ\_C enter, with XGBoost only.
15. **D15 (E16, R).** The mixed model for ψ did not converge; its interval is reported but not relied on.
16. **D16 (E17, P).** The set-level test is a Mann–Whitney test on single-sample set scores (the pathway model's representation), not the GSEA of the published split-half analysis, so the two are not directly comparable.
17. **D17 (E18, P).** The Hamer & Dupont (2021) index was not implemented because its definition was not verified here.
18. **D18 (E19, A).** A(10) has a patients-only interval, since resampling replicates would change the number of distinct replicates, which is what A(m) depends on.
19. D19 (E15, R). How often each published gene, and each gene of the manuscript's 7-gene core, recurs across the 20 seeds was added after the run as a labelled post-hoc description; the plan specified Jaccard, sizes, identity across environments and nested AUC only.

Common to all runs: the engine's data loader needs about 4 GB, which two concurrent jobs could not afford, so a lean loader reads the same files; it was checked to give the same 633 × 27,949 matrix, sample order, labels and folds. Interrupted runs resume from per-unit caches with unchanged seeds; E9 and E15 were resumed after the 2-hour limit on background jobs. E17's first run ran out of memory because it read the whole GSE68086 table only to define the gene universe; it now reads the table's column names alone (the same 33,526 columns), so the universe is unchanged.

## Multiple-testing check

Only E8 had a pre-specified correction (Holm over its five primary comparisons). E9, E10 and E17 specified none, and E11 and E16 were declared descriptive or exploratory. As sensitivity analyses, Holm was applied within each family and, most conservatively, over all 36 distinct tests of all experiments at once. Five tests lose significance at 0.05 under at least one correction; the table shows them with the three key tests that hold. Every other test keeps its conclusion.

| Test | Raw p | Holm within family | Holm over all 36 tests | Conclusion |
| --- | --- | --- | --- | --- |
| E8 mRMR at 27 genes vs A4 | 0.00005 | 0.0003 (planned) | 0.002 | Holds under every correction |
| E8 mRMR–IFS–SVM vs A4 | 0.001 | 0.005 (planned) | 0.032 | Holds under every correction |
| E8 stability selection vs A4 | 0.007 | 0.021 (planned) | 0.13 | Holds under the planned correction only |
| E10 (b') − locked | 0.036 | 0.14 | 0.65 | Lost under any correction |
| E11 TMM-CPM − counts | 0.004 | 0.008 | 0.084 | Lost over all tests |
| E9 Singh gain, B = 10 | 0.045 | 0.36 | 0.77 | Lost under any correction |
| E9 GSE68086 bagged (B = 10) − frozen | 0.004 | 0.037 | 0.084 | Lost over all tests |
| E9 external ensemble − locked | 0.002 | 0.002 (single test) | 0.047 | Holds, at the margin |

- Hold under every correction: E9's gains in GSE89843 and GSE68086 (both B), bagged − frozen in GSE89843 (both B) and in GSE68086 for B = 25; the E16 permutation tests of all four predictors (exploratory); the E17 Kuncheva comparison of A4 with the pathway lists.
- Not significant before any correction: E10 (a2), (b) and (c) against the locked model and both random-panel tests; E11 CPM; E9 in Gordon and the other Singh tests; E8 L1 and Mann–Whitney; E17 replication rates.
- Timing: the within-family check was scripted at 14:47, after E8, E10, E11 and E16 had been seen; the all-tests column was added at 18:05, before E9's tests were computed. Neither changes a planned analysis.
- Post-hoc comparisons (E9 bagged A4 vs mRMR, E10 against the published null, E17 top-k lists) are descriptive and are not in the table.
- Source: runs/revision/multiple\_testing.csv (every test, with raw, within-family and all-tests p).

## Claim-by-claim evidence audit

Every empirical claim of the abstract, the contribution list (Introduction) and the Conclusion, checked against E1–E4 and E8–E19; claims that recur are merged. Verdicts: **Supported** (reproduced or confirmed), **Weakened** (holds only in part, or needs a qualification the text lacks), **Contradicted** (the experiment shows otherwise under matched conditions), **Not tested** (no experiment here bears on it). Of the 22 claims, 9 are supported (2 more strongly than stated), 7 weakened, 1 contradicted and 5 not tested.

| # | Claim (where) | Evidence | Verdict |
| --- | --- | --- | --- |
| 1 | Nested AUC 0.9325 (abstract, conclusion) | E8 reproduces 0.9325 exactly. E15: seeds 1–5 give 0.921–0.930; the paper's seed is the highest of six. | Supported; add the seed range |
| 2 | Five-term nested AUC 0.9370; robust to the network term, composite weights and tuning (conclusion) | Not re-run in E8–E19 | Not tested |
| 3 | "Robust in sensitivity analyses" (abstract) | E11: CPM −0.23 pts (p = 0.81); TMM-CPM −3.77 pts (p = 0.004, descriptive; 0.08 after Holm over all tests). E15: about 1 pt across seeds. | Weakened: holds for library-size scaling, not for TMM |
| 4 | A univariate filter matched it with far more stable genes (abstract) | E8: Mann–Whitney −0.29 pts vs A4 (Holm p = 0.79). E18: Φ\_C 0.938 vs A4 0.119 (GSE89843). | Supported |
| 5 | Simpler established selectors discriminated as well or better on identical folds (conclusion; contribution 3) | E8, classifier and size matched: mRMR +3.58 pts (Holm p = 0.0003), mRMR–IFS–SVM +3.05 (0.0048), stability selection +2.02 (0.021; 0.13 after Holm over all tests) over A4; L1 and Mann–Whitney not different. | Supported, and stronger than stated |
| 6 | No selector or model was best on every dimension (abstract) | The best-discriminating new arm (mRMR, 27 genes, XGBoost) has no replicate study (D3), so its Φ, ψ and D are unknown. | Not tested |
| 7 | Re-selection caused ψ = 0.844 of the prediction variance (abstract, contribution 2, conclusion) | 0.844 is the saved replicate study's value; it was not recomputed here. E1–E4: Eq. 3's point estimate is nearly unbiased for its own estimand (bias within ±0.022) and close to Π\_Q + Π\_γ (re-selection plus coupling; differences up to 0.13), but the published patients-only 95% interval covered the truth in only 16–55% of simulated cohorts at R = 10. | Weakened: the estimate stands with its estimand stated; its interval does not |
| 8 | 40% of held-out patients received conflicting classifications (abstract, conclusion twice) | E19: this is A(10), which rises with R (0.149 at R = 2 to 0.401 at R = 10). The R-free disagreement is D = 14.9% for A4, 6.0% with the panel fixed. | Weakened: true only at R = 10 |
| 9 | ψ tracked the bagging gain in all four data sets (Spearman 0.86 overall), at least as closely as stability (abstract, contribution 2, conclusion) | E16 (exploratory): mean within-data-set Spearman 0.90 for ψ vs −0.73 for Φ and −0.75 for Φ\_C; but V\_R also gives 0.90 and predicts a left-out selector's gain better (MAE ratio 0.51 vs 1.03; ψ is no better than the data-set mean). | Weakened: the tracking holds; the specific value of ψ is not shown |
| 10 | Bagging raised the panel's pooled AUC to 0.9630, +2.83 pts, and matched the strongest comparator (abstract; discussion) | E9 reproduces 0.9630 and +2.83 exactly on the paper's folds; a new partition gives 0.9609 and +2.91. Per-fold gain +3.03 pts (p < 0.0001). Post hoc: bagged A4 vs mRMR–IFS–SVM +0.01 pts (p = 0.99), vs mRMR at 27 genes −0.52 (p = 0.35); the comparators were not bagged. | Supported; "matched" is post hoc and compares a bagged selector with single ones |
| 11 | The patient-level cost was largely removed by selection bagging (conclusion) | E9: two disjoint 10-model ensembles disagree on 4.0% of discovery patients, against 14.9% for single models (E19); 0–5.4% against 4.4–14.0% in the other data sets. | Supported |
| 12 | PMTRD and ψ indicate when an unstable selector should be deployed as an ensemble (abstract, contribution 2) | E9: the gain is large where single models disagree most (GSE89843 +3.0, GSE68086 +5.2 pts) and absent where they agree (Gordon +0.4, p = 0.34). E16: V\_R indicates this as well as ψ. | Weakened: re-selection variance indicates it; ψ is not needed |
| 13 | Pathway-level biology replicated where genes did not; "pathway-level biology is stable while individual genes are not" (abstract, contributions, conclusion) | E17, matched list size: top-27 genes and top-27 sets replicate equally (100% each). Post hoc, same selection rule: top-k gene lists are as stable as top-k set lists (Kuncheva 0.592 vs 0.582, p = 0.82). The unstable list is A4's (0.083). | Contradicted as a genes-versus-pathways statement |
| 14 | Platelet-activation and haemostasis pathways replicated in every split (conclusion) | E17 tested replication by rank; it did not re-run GSEA or check which sets (D16) | Not tested |
| 15 | The pathway model transferred better externally but failed site transfer (abstract, contribution 4, conclusion) | Not re-run; E10's random-panel null was not applied to it. Its premise (built where molecular reliability was highest) is undercut by row 13. | Not tested |
| 16 | External AUC 0.8481; calibration did not transfer; locked-model validation with gene availability (conclusion, contributions) | E10 reproduces 0.8481, intercept 2.80, slope 0.91. The paper's 18 genes give 0.844 (−0.37 pts, p = 0.76). Random 18-gene panels from the measurable Tier-2 pool: median 0.815, p = 0.38 for the 18-gene model; against the paper's own null the locked model has p = 0.12 (post hoc). A bagged A4 ensemble transfers better than the locked panel (row 22). | Supported as numbers; Weakened as evidence for these genes |
| 17 | Gene identity is unstable (conclusion; limitations) | E18: Φ\_C 0.12 against Φ 0.31 (GSE89843). E15: a new seed gives a panel with Jaccard 0.62 to the 27 genes; across training subsamples, panels share Jaccard 0.185. | Supported, and stronger than stated |
| 18 | The procedure is reproducible, the gene list is not (conclusion) | E15: identical panels for the same seed under both thread settings (21 of 21); nested AUC 0.921–0.933 over six seeds; the panel changes with the seed. | Supported |
| 19 | A seeded code base reproduces every number (contributions) | Every published number re-checked here reproduced (0.9325; 0.9353, 0.9460, 0.9527, 0.9630; 0.8481 with calibration; Φ and A(10) recomputed from the saved panels and scores; fold-recomputable Jaccard 0.710; bagged 0.9630 and +2.83). Most numbers were not re-checked. | Supported for the numbers checked |
| 20 | In three further data sets re-selection again moved patient-level predictions and ψ again tracked the bagging gain (conclusion) | E19: A4's D exceeds the panel-fixed D in all three (0.140 vs 0.087; 0.090 vs 0.078; 0.044 vs 0.034). E16: ψ's per-data-set Spearman 0.7–1.0, with V\_R identical. | Supported for predictions; Weakened for ψ (row 9) |
| 21 | Site confounding audit (within-Vumc nested AUC 0.9399; site-free transfer); contamination-prone genes not needed; harmonisation creates no signal (abstract, contributions, conclusion) | Not covered by E8–E19; E12 was not run (needs the crossed estimator) | Not tested |
| 22 | Recommendation: a locked, library-size-normalised model or a bagged ensemble with recalibration (conclusion) | E9 external (pre-registered): a 25-member A4 ensemble reaches 0.915 against 0.848 for the locked panel (+6.71 pts, p = 0.002; 0.047 after Holm over all tests). Bagged models are under-confident internally (slopes 1.09–1.81) and externally (1.98). E11: CPM keeps the AUC, TMM lowers it. | Supported, including the need to recalibrate |

## Publication-readiness audit

Not ready to submit as written: one headline claim is contradicted, seven are weakened and five are untested. No new experiment is needed to make it submittable, provided the claims below are revised or dropped; new experiments are needed only to keep a claim at its present strength.

The supervisor's question was whether A4 has a distinct advantage that would carry a reframed paper: in external transfer, gene-level reproducibility or pathway coherence. The answer is no on all three. The locked panel transfers no better than random panels from the same pool (E10); its genes are less stable than a univariate filter's and change with the data and the seed (E15, E18); and pathway lists are no more stable than gene lists at matched size (E17). The one strong positive result concerns a deployment strategy, not A4 itself: bagging A4's selections raises discrimination internally (+3.0 pts) and in the independent cohort (+6.7 pts) and cuts patient-level disagreement in the discovery cohort from 14.9% to 4.0% (E9).

| Decision | Claim | What to do |
| --- | --- | --- |
| Keep | Nested AUC 0.9325 and the selection-optimism gap | Add the seed range (0.921–0.933; E15) |
| Keep | Locked external AUC 0.8481 and the failed calibration | Add that random panels from the measurable informative pool transfer about as well (E10), so the transfer is not evidence for these genes |
| Keep, strengthen | Simpler selectors discriminate as well or better | State the matched result: mRMR is better than A4 at the same size and classifier under every correction; stability selection under the planned one (E8) |
| Keep, strengthen | Gene identity is unstable; the procedure, not the list, is reproducible | Add Φ\_C and the seed results (E15, E18): the data drive most of the instability |
| Keep | Bagging raises the pooled AUC to 0.9630 (+2.83) and removes most patient-level disagreement | Reproduced exactly and replicated on a new partition; report the per-fold test, B = 25, and the under-confident calibration (E9) |
| Keep, add | A bagged ensemble transfers better than the locked panel | New, pre-registered: 0.915 vs 0.848 (p = 0.002; E9). State that it was not compared with bagged comparators or random-panel ensembles |
| Keep | The seeded code reproduces the published numbers | Limit the claim to what was re-checked, or re-check the rest |
| Revise | ψ = 0.844 | Keep the point estimate with its estimand stated (approximately Π\_Q + Π\_γ); remove the patients-only interval, which under-covers (E1–E4) |
| Revise | "40% of patients received conflicting classifications" | Lead with the R-free disagreement D (14.9% vs 6.0% with the panel fixed); give A(R) only with R stated (E19) |
| Revise | ψ tracks the bagging gain, at least as closely as stability | Report that V\_R, which needs no panel-fixed arm, tracks it equally and predicts it better; keep the result exploratory (E16) |
| Revise | Bagged A4 "matched the strongest comparator" | Say it is level with mRMR (post hoc, −0.5 to +0.2 pts) and that the comparators were not bagged |
| Revise | "Robust in sensitivity analyses" | Name the analyses; add the TMM drop of 3.8 pts (E11) |
| Drop | "Pathway-level biology is stable while individual genes are not" (abstract, contributions in bold, conclusion) | Contradicted at matched list size (E17). What survives: A4's panel is less stable than a simple top-k list of genes or of sets |
| Drop or rebuild | The pathway model as "built where molecular reliability was highest" | Its premise falls with the row above; keep its site-transfer failure as a cautionary result |
| Needs a new experiment | ψ as an interval-estimated share of variance | A valid interval: the crossed estimator failed (NO-GO), so E5–E7, E13 and E14 cannot be run as planned |
| Needs a new experiment | The pathway model transfers better externally | A random-gene-set null like E10's |
| Needs a new experiment | No selector is best on every dimension | A replicate study (Φ, ψ, D) for mRMR with 27 genes and XGBoost |
| Needs a new experiment | Bagging A4 beats bagging a simpler selector | Bag mRMR and the univariate filter on the same folds and in the external cohort |
| Not re-audited | Site confounding, contamination, harmonisation | E8–E19 do not touch these; E12 needs the crossed estimator. They stand on the original analyses |

### A reframing the evidence supports

Forcing A4 into a superiority claim would rest on nothing the experiments show. The evidence does support a methods paper about what a single AUC hides, with the TEP panel as a worked example:

1. **An audit protocol** (PMTRD): fold-recomputable selection, size- and classifier-matched benchmarks, random-panel nulls for external transfer, and patient-level disagreement with R stated, released as reproducible code.
2. **A patient-level finding across four data sets:** re-selecting features changes individual classifications more than a fixed panel does (D 4–15% against 3–9%), bagging the selections cuts that disagreement and raises discrimination where it is large (internally +3 to +5 pts; externally +6.7 pts), and the across-replicate prediction variance (V\_R, or equally ψ) tracks how much bagging helps. The last part is exploratory, with five selectors per data set.
3. **An honest case study:** a panel with nested AUC 0.93 and external AUC 0.85 that simpler selectors beat at matched size, whose genes change with the data and the seed, whose external transfer is matched by random panels, and whose apparent pathway-over-gene stability disappears at matched list size.
4. **A simulation check of the ψ estimator** that found its published interval under-covers, reported as a methodological caution.

This keeps every reproduced number, needs no superiority claim, and makes the negative results and the bagging result the contribution. Whether to take this route is the authors' decision.

## Raw outputs, logs and reproducibility

Everything is in the repository on branch `crossed-estimator`. Paths are relative to the repository root; logs are in runs/revision/logs/ and raw per-unit outputs, as gzip JSON lines, in runs/revision/raw/.

| Exp. | Summary outputs (each folder has a metadata.json with time, versions and seeds) | Raw unit outputs | Log |
| --- | --- | --- | --- |
| E1–E4 | runs/crossed\_sim/v1/summary/ (verdict.json, summary.csv) | runs/crossed\_sim/v1/units/ | runs/crossed\_sim/v1/log\_worker\*.txt |
| E8 | runs/revision/E8/E8.csv | E8\_units.jsonl.gz | queueB.out |
| E9 | runs/bagging/E9\_folds.csv, E9\_calibration.csv, E9\_tests.csv, E9\_posthoc.csv, E9\_external.csv, E9\_external\_members.csv | E9\_{Singh, Gordon, GSE68086, GSE89843, external}\_units.jsonl.gz | e9.out, queueC.out, queueF.out, e9\_summary.out |
| E10 | runs/external/E10.csv, E10\_null.csv, E10\_predictions.csv, E10\_published\_null\_posthoc.csv | the predictions and null files are the raw outputs | e10.out, queueB.out (first attempt) |
| E11 | runs/revision/E11/E11.csv, E11\_tmm\_factors.csv, E11\_tmm\_check.txt | E11\_cpm\_units.jsonl.gz, E11\_tmm\_units.jsonl.gz | e11.out |
| E15 | runs/seeds/E15.csv, E15\_panels.csv | E15\_units.jsonl.gz | queueB.out, queueD.out, queueE.out |
| E16, E18, E19 | runs/revision/E16/, E18/, E19/ (copies in runs/export/) | computed from the saved replicate runs, which are unchanged | e18.out; E16 and E19 ran in the foreground |
| E17 | results/pathway/E17.csv, E17\_splits.csv, E17\_posthoc\_toplists.csv | E17\_units.jsonl.gz | queueD.out (first attempt), queueE.out |
| Multiple testing | runs/revision/multiple\_testing.csv | — | e9\_summary.out |

- Failed or stopped runs, all kept in the logs: E10's first attempt ran out of memory (exit 137) and was rerun unchanged; E17's first attempt ran out of memory and was rerun after the fix described under the deviations; E15 (environment A) and E9 (GSE89843, at 182 of 260 units) stopped at the 2-hour limit and resumed from their caches. To finish within the limit, E9's external members ran in a second queue beside the last GSE89843 units, with the same code and seeds (queueF.sh). The queue scripts are in runs/revision/logs/.
- Unit tests of the revision helpers (tests/test\_revision.py: Φ\_C equals Nogueira's Φ when C = I; disagreement and ambiguity; TMM factor properties): 3 passed, run at 18:04.
- Commits: the plan at 12:38 and its five amendments before the runs they change; every result commit follows its run. The post-hoc E9 module was committed at 18:00, before the E9 summaries were computed (18:26). The branch has not been pushed: this session has no write access to the GitHub repository (HTTP 403). A git bundle of the whole branch goes with this report.
- Manuscript untouched: MD5 checksums of the eight source files (pkg\_extract/manuscript/src/\*.tex) taken at 14:39 and again at the end are identical, and the files were last modified on 5–7 November. Both checksum lists are in runs/revision/logs/manuscript\_md5\_before.txt and manuscript\_md5\_after.txt.
