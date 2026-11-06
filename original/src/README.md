# Code for the TEP reliability paper

Code for *Beyond Peak Accuracy: A Predictive–Molecular–Technical Reliability Decomposition of a Tumour-Educated
Platelet RNA Signature for Non-Small Cell Lung Cancer* (Tasnim, Islam, Mahmood, Alam and Morol). Every number,
table and figure in the manuscript was produced with these scripts from public GEO data, except the figures
marked as coming from the development notebooks in `docs/figure_table_map.md`.

## Files

| File | What it does |
|---|---|
| `tep_analysis.py` | Library: data loading, statistics (DeLong, TOST, bootstrap), and the analyses R0 to R10 and SHAP (step names used in the file and table names). |
| `selection_pipeline.py` | Seeded implementation of the six-stage selection pipeline (follows the development notebook `nsclc_phase_1.ipynb`). Used for every analysis that re-runs the selection. |
| `fold_recompute.py` | Fully fold-recomputable selection: the feature-importance screen (Stage 0) and the STRING term are re-derived inside every outer training fold; nested CV with this version (`fi_full`, `nested`, `mrmr_union`, `ppi`, `summarise`). |
| `get_string_edges.py` | Downloads the STRING interactions (combined score ≥ 0.4; the STRING version is printed) among the genes listed by `fold_recompute.py mrmr_union`. Needs internet access (e.g. Colab). |
| `sheng_sensitivity.py` | Sensitivity of the re-implemented mRMR–IFS–SVM comparator to its settings (SVM C and kernel, discretisation, mRMR scheme, IFS criterion and range), on the same outer folds. |
| `ext_reliability.py` | Locked-model reporting in GSE68086: calibration intercept and slope, Brier score, binned calibration, sensitivity and specificity at the discovery threshold (Wilson CIs), and the dependence of the external AUC on the number of available panel genes. |
| `make_fig01_pmtrd.py`, `make_fig09.py` | Figure 1 (PMTRD schematic) and the ROC/PR figure. |
| `run_steps.py` | Runs the steps one at a time. Each step saves its numbers (`out/macros/*.json`), tables and figures, so a crashed session can resume. `merge` writes `results_macros.tex`. |
| `prepare_gse68086.py` | Builds the external cohort from `GSE68086_TEP_data_matrix.txt.gz`: labels, removal of the 19 overlapping samples, Ensembl-75 → symbol mapping. |
| `external_analysis.py` | All GSE68086 analyses: the development configuration (22 genes, per-cohort standardisation), the 27-gene panel on log2-CPM under every alignment setting, operating characteristics, the Vumc-derived panel, and the Sheng et al. model. |
| `harm_sim.py` | Alignment experiment inside one site (Vumc), where there is no batch effect to remove. |
| `r5_external_table.py` | Adds the GSE68086 AUCs to the QC-panel table. |
| `r7_gsea.py` | Pathway analysis: within-Vumc pre-ranked GSEA (one gene-set library per process), the contamination-adjusted sensitivity analysis, and the over-representation test of the panel. |
| `pathway_stability.py` | Genes versus pathways: 25 split-half replications within Vumc (different patients, same site), rank-only GSEA, unadjusted and contamination-adjusted; also the rank-only GSEA on the full Vumc ranking. About 50 min on 2 cores. |
| `extra_analyses.py` | Smaller analyses: seeded reproduction (`repro`), library size (`libsize`), composite-weight sensitivity (`weights`) and others. |
| `dev_reruns.py` | Seeded re-runs of the ablation, feature-count curve, RFECV ceiling and stability selection. |
| `crosscheck_independent.py` | Compares the main estimates with the results folder of the second, independently written notebook (`TEP_Reviewer_Revision_Run.ipynb`). |
| `make_fig10.py`, `redraw_r1_effects.py`, `redraw_r3sim.py`, `redraw_r9.py`, `make_forest.py` | Final versions of the figures. |
| `config.json` | Paths and settings used for the paper (edit the paths). |
| `TEP_analysis_colab.ipynb` | Runs everything in Google Colab, in order. |

## Inputs

1. `dataset_NSCLC`: the GSE89843 matrix from GREIN (633 samples × 28,089 genes, columns `Sample_Names`, genes, `disease`).
2. `GSE89843_filtered_metadata.csv`: the sample metadata. The GEO title gives the collection site.
3. `DEFINITIVE_biomarker_panel.csv`: the 27-gene panel. Its PPI scores are used only for the full-cohort reproducibility check.
4. `GSE68086_TEP_data_matrix.txt.gz`: from GEO.
5. `dgeGenesEnsembl75.RData`: the Ensembl-75 annotation from the thromboSeq source code
   (<https://github.com/MyronBest/thromboSeq_source_code>, folder `bin/`).
6. The MSigDB v7.5.1 gene sets as `.gmt` files for the pathway analysis (included in `inputs/genesets/`; made
   from the data of the `msigdbr` R package, version 7.5.1).

## Order of the runs (as done for the paper)

```bash
python prepare_inputs.py dataset_NSCLC GSE89843_filtered_metadata.csv inputs/
python prepare_gse68086.py GSE68086_TEP_data_matrix.txt.gz dgeGenesEnsembl75.RData inputs/gse68086/
python extra_analyses.py config.json repro              # seeded full-cohort run of the selection (~20 min)
python run_steps.py config.json r0 r1t r1g nested nestedfig r8 r4 r1v r1n r2e r5 r6 shap
for s in libsize r0extra nestedextra subsvumc r5extra weights; do python extra_analyses.py config.json $s; done
python r7_gsea.py config.json rank                      # pathway analysis, one library per process
for g in inputs/genesets/*.gmt; do python r7_gsea.py config.json lib $g 1000 500; done
python r7_gsea.py config.json ora ; python r7_gsea.py config.json combine
python r7_gsea.py config.json rank_adj                  # contamination-adjusted sensitivity analysis
python pathway_stability.py config.json run 25 500 ; python pathway_stability.py config.json full 1000 ; python pathway_stability.py config.json summarise
for g in inputs/genesets/*.gmt; do python r7_gsea.py config.json lib $g 1000 500 adj; done
python r7_gsea.py config.json combine_adj
for s in ablation featcount rfecv; do python dev_reruns.py config.json $s; done   # ~35 min
python run_steps.py config_ext.json r3                  # external harmonisation audit (needs GSE68086)
python external_analysis.py config_ext.json
python ext_reliability.py config_ext.json            # locked model: calibration and gene availability
python fold_recompute.py config.json fi_full ; python fold_recompute.py config.json nested   # ~1.5 h
python fold_recompute.py config.json mrmr_union         # gene list for get_string_edges.py
cp out/tables/FR_mrmr_union_genes.txt FR_string_query_genes.txt
python get_string_edges.py                              # needs internet; writes string_edges.tsv
python fold_recompute.py config.json ppi string_edges.tsv
python sheng_sensitivity.py config.json run ; python sheng_sensitivity.py config.json summarise
python harm_sim.py config.json 30 50
python r5_external_table.py config_ext.json
python run_steps.py config_r9.json r9                   # 5,000 random + 5,000 matched panels (longest step)
python crosscheck_independent.py out PAPER_2865dafe4618 # optional: comparison with the second notebook
python run_steps.py config.json merge                   # -> out/results_macros.tex
python make_forest.py out ; python make_fig10.py out 0.8481 ; python redraw_r9.py out
```

Then copy `out/results_macros.tex`, `out/figures/` and `out/tables/` into `manuscript/results/` and
compile the manuscript.

## Reproducibility notes

* Every random step is seeded (seed 42; bootstrap resample *b* uses seed *b*). Samples are in GEO
  accession order, and fold assignments are saved (`R8_outer_folds.csv`).
* The classifier is XGBoost with 300 trees, learning rate 0.03, depth 4, and row and column subsampling 0.8.
* Two nested-CV versions are reported. The *four-term* version (`run_steps.py nested`) omits the STRING term
  and the G106 term inside folds (G106, 96 unique genes, was derived from all 633 samples); without them the
  full-cohort pipeline keeps 22 of the 27 panel genes. The *fully fold-recomputable* version
  (`fold_recompute.py`) re-derives the feature-importance screen that produced G106, and the STRING term, from
  each training fold, so that no step of the selection sees the held-out samples.
* The feature-importance screen is a re-implementation of the development notebook: eight classifiers, the
  five with feature importances ranked by inner 3-fold CV AUC, top 3,000 genes with non-zero importance per
  model, intersection of all five (at least four of five if fewer than 20 genes).
* Run times on 2 CPU cores: nested CV ≈ 20 min; Vumc nested ≈ 5 min; head-to-head ≈ 2 min;
  contamination step with re-selection ≈ 25 min; harmonisation audit ≈ 20 min; pathway analysis ≈ 20 min
  (peak memory ≈ 3.5 GB for GO BP); random panels ≈ 1 s per panel.
