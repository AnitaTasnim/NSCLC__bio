#!/bin/bash
# Re-runs every analysis of the paper in the order used for it. Run from the repository root.
# Edit the input paths in config/*.json first (see data/README.md).
set -e
cd src
python prepare_inputs.py ../data/dataset_NSCLC ../data/GSE89843_filtered_metadata.csv inputs/
python prepare_gse68086.py ../data/GSE68086_TEP_data_matrix.txt.gz ../data/dgeGenesEnsembl75.RData inputs/gse68086/
cp -r ../inputs/genesets inputs/
C=../config/config.json; CE=../config/config_ext.json
# discovery cohort: core estimates, site audit, four-term nested CV, comparator, stability, contamination
python extra_analyses.py $C repro
python run_steps.py $C r0 r1t r1g nested nestedfig r8 r4 r1v r1n r2e r5 r6 shap
for s in libsize r0extra nestedextra subsvumc r5extra weights; do python extra_analyses.py $C $s; done
for s in ablation featcount rfecv; do python dev_reruns.py $C $s; done
# fully fold-recomputable nested CV (feature-importance screen and STRING term inside every fold)
python fold_recompute.py $C fi_full
python fold_recompute.py $C nested
python fold_recompute.py $C mrmr_union        # writes the gene list for get_string_edges.py
cp out/tables/FR_mrmr_union_genes.txt FR_string_query_genes.txt
python get_string_edges.py                     # needs internet; writes string_edges.tsv
python fold_recompute.py $C ppi string_edges.tsv
# comparator sensitivity
python sheng_sensitivity.py $C run && python sheng_sensitivity.py $C summarise
# pathways within one site, and split-half replication of genes and gene sets
python r7_gsea.py $C rank
for g in inputs/genesets/*.gmt; do python r7_gsea.py $C lib $g 1000 500; done
python r7_gsea.py $C ora && python r7_gsea.py $C combine
python r7_gsea.py $C rank_adj
for g in inputs/genesets/*.gmt; do python r7_gsea.py $C lib $g 1000 500 adj; done
python r7_gsea.py $C combine_adj
python pathway_stability.py $C run 25 500 && python pathway_stability.py $C full 1000 && python pathway_stability.py $C summarise
# independent cohort GSE68086
python run_steps.py $CE r3
python external_analysis.py $CE
python ext_reliability.py $CE
python harm_sim.py $C 30 50
python r5_external_table.py $CE
# null panels (longest step)
python run_steps.py ../config/config_r9.json r9
# all numbers -> LaTeX macros
python run_steps.py $C merge
