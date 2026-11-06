# Which script produces which figure, table and number

Every number in the paper is a LaTeX macro (for example `\NestedAUC`). Each analysis step writes its macros
to `out/macros/<step>.json`; `python src/run_steps.py config/config.json merge` collects them into
`out/results_macros.tex`, which the manuscript reads. The macro files used for the paper are in
`results/macros/`, so any number can be traced by searching for its macro name there.

Commands are run from `src/`; `config.json` is the discovery configuration, `config_ext.json` adds GSE68086
and `config_r9.json` sets the null-panel analysis.

## Main text

| Item | Content | Produced by | Output file(s) | Macro file |
|---|---|---|---|---|
| Figure 1 | PMTRD framework (schematic) | `make_fig01_pmtrd.py` | `figs/fig01_pmtrd.pdf` | – |
| Figure 2 | Six-stage selection funnel (schematic; gene counts from the seeded run) | drawn; counts from `extra_analyses.py config.json repro` | `fig02_funnel.pdf` | `repro.json` |
| Algorithm 1 | Selection procedure | `selection_pipeline.py` (`phase1_select`) | – | – |
| Table 1 (`tab:landscape`) | Related work | literature | – | – |
| Tables 2–4 (`tab:discovery`, `tab:external`, `tab:funnel`) | Cohorts and selection stages | `prepare_inputs.py`, `prepare_gse68086.py`, `extra_analyses.py repro` | – | `repro.json` |
| Table 5 (`tab:panel`) | The 27-gene panel | `selection_pipeline.py` | `data/DEFINITIVE_biomarker_panel.csv` | `r0.json` |
| Figure 3 (`fig:roc`) | ROC and PR curves, fixed panel | `run_steps.py config.json r0`, `make_fig09.py` | `fig09_roc_pr_seeded.pdf` | `r0.json`, `r0extra.json`, `libsize.json` |
| Section 3.2.1 | Nested CV, four-term selection | `run_steps.py config.json nested nestedfig`, `extra_analyses.py nestedextra` | `R4_nested_folds.tex`, `R4_nested_oof.csv`, `R8_outer_folds.csv` | `nested.json`, `nestedextra.json` |
| Section 3.2.1, Table `tab:foldrec` | Nested CV, fully fold-recomputable selection (feature-importance screen and STRING term re-derived in every training fold) | `fold_recompute.py config.json fi_full`, `nested`, `mrmr_union`, `ppi <edges>`, `summarise` | `FR_nested_folds.csv`, `FR_nested_oof_*.csv`, `FR_full_panel*.csv` | `foldrec.json` |
| Table `tab:h2h` | Head-to-head with the re-implemented mRMR–IFS–SVM comparator | `run_steps.py config.json r8` | `R8_head_to_head.tex` | `r8.json` |
| Table `tab:shsens` | Comparator sensitivity to its implementation settings | `sheng_sensitivity.py config.json run`, `summarise` | `SH_comparator_sensitivity.tex/.csv` | `shsens.json` |
| Table `tab:externalres` | GSE68086, every alignment setting | `run_steps.py config_ext.json r3`, `external_analysis.py config_ext.json` | `R2_external_all_settings.tex` | `external.json`, `external_extra.json`, `r3.json` |
| Figure `fig:locked` | Locked model in GSE68086: calibration and gene availability | `ext_reliability.py config_ext.json` | `fig_locked_external.pdf`, `LK_*.csv` | `locked.json` |
| Figure `fig:nested`, Table `tab:nested` | Fold AUCs and Jaccard similarities | `make_fig10.py out 0.8481` | `fig10_nested_simplified.pdf`, `R4_nested_folds.tex` | `nested.json`, `r4.json` |
| Figure `fig:random` | Random and matched null panels | `run_steps.py config_r9.json r9`, `redraw_r9.py out` | `R9_random_panels.pdf` | `r9.json` |
| Table `tab:gsea` | Within-site pathway analysis | `r7_gsea.py` (modes `rank`, `lib`, `ora`, `combine`, `rank_adj`, `combine_adj`) | `R7_*` | `r7.json`, `r7adj.json` |
| Figure `fig:pathstab` | Genes versus pathways, 25 split-halves | `pathway_stability.py config.json run 25 500`, `full 1000`, `summarise` | `fig_pathway_stability.pdf`, `PS_*.csv` | `pathstab.json` |
| Figure `fig:pca` | PCA by site and diagnosis | development notebook `nsclc_phase_1.ipynb` (deterministic) | `fig04_pca.png` | – |
| Tables `tab:site`, `tab:siteaware` | Site–label association and site-aware estimates | `run_steps.py config.json r1t r1n r1v` | – | `r1t.json`, `r1n.json`, `r1v.json` |
| Table `tab:genesite`, Figures `fig:genesite`, `fig:sitescores` | Site versus disease effect per gene | `run_steps.py config.json r1g`, `redraw_r1_effects.py` | `R1_panel_site_vs_disease.tex`, `R1_*.pdf` | `r1g.json` |
| Table `tab:qcpanels` | Contamination-filtered panels and equivalence tests | `run_steps.py config.json r5`, `extra_analyses.py r5extra`, `r5_external_table.py config_ext.json` | `R5_contamination_panels.tex` | `r5.json`, `r5ext.json`, `r5extra.json` |
| Table `tab:harm`, Figure `fig:harm` | Harmonisation audit and within-site alignment experiment | `run_steps.py config_ext.json r3`, `harm_sim.py config.json 30 50`, `redraw_r3sim.py` | `R3sim_alignment_within_vumc.pdf` | `r3.json`, `r3sim.json` |
| Table `tab:pmtrd` | PMTRD reliability card (all of the above) | – | – | all |

## Supplement

| Item | Produced by | Output / macro file |
|---|---|---|
| Stability profiles, evidence heatmap (`fig:stabprof`, `fig:heatmap`) | development run that produced the panel; seeded re-run of the stability selection in `dev_reruns.py` | `fig06_*.jpg`, `fig07_*.jpg`; `rfecv.json` |
| Contamination check for three genes (`tab:contam`, `fig:contam`) | `run_steps.py config.json r5` | `r5.json` |
| Composite-weight sensitivity | `extra_analyses.py config.json weights` | `weights.json` |
| Out-of-fold SHAP (`fig:shap`) | `run_steps.py config.json shap` | `minor_shap_out_of_fold.pdf`; `shap.json` |
| Feature-count curve, ablation, RFECV (`tab:featcount`, `fig:featcount`, `fig:ablation`, `fig:rfecv`) | `dev_reruns.py config.json featcount ablation rfecv` | `DEV_*.pdf/.tex`; `featcount.json`, `ablation.json`, `rfecv.json` |
| Classifier benchmark (`fig:benchmark`, `tab:classifiers`) | exploratory, unseeded: development notebook `NSCLC_PHASE2.ipynb` | `fig03_benchmarking.pdf` |
| Tissue cohorts (`tab:tissue`) | `run_steps.py config.json r2e` | `r2e.json` |
| Covariates (inferred sex) | `run_steps.py config.json r6` | `r6.json` |
| Differential-expression landscape (`fig:de`) | development notebook `nsclc_phase_1.ipynb` | `fig05_de_landscape.png` |
| Forest plot of all AUCs (`fig:forest`), Table `tab:allauc` | `make_forest.py out` | `forest_all_aucs.pdf`, `ALL_AUCs_with_95CI.tex/.csv` |
| Independent cross-check (`tab:crosscheck`) | `crosscheck_independent.py out <folder>` | `S_crosscheck_independent.tex`; `crosscheck.json` |

## Fold assignments and seeds

* Global seed 42 (`config.json`, key `seed`); bootstrap resample *b* of Stage 5 uses seed *b*.
* The five outer folds used by every cross-validated estimate are saved in `results/tables/R8_outer_folds.csv`
  (sample accession, fold). Samples are in GEO accession order.
* The selection is deterministic for a given input and seed: ties in the composite score, the
  feature-importance ranking and the choice of models in the feature-importance screen are broken by name
  (gene symbol or model name), and the remaining ranking steps use the fixed gene order of the input matrix.
