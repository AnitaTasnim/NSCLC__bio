# Input data

All data are public. Put the files below in one folder and point `config/config.json` to it.

| File | Source | How to obtain |
|---|---|---|
| `dataset_NSCLC` | GSE89843 (Best et al., 2017), GREIN counts | GREIN (iLINCS) → GSE89843 → download the count matrix; keep NSCLC and healthy-donor samples (633), one column per gene, plus `Sample_Names` and `disease` (1 = NSCLC, 0 = healthy). |
| `GSE89843_filtered_metadata.csv` | GEO sample titles | included here (`data/`). The title gives the collection site. |
| `DEFINITIVE_biomarker_panel.csv` | this study | included here (`data/`): the 27-gene panel. |
| `GSE68086_TEP_data_matrix.txt.gz` | GEO GSE68086 (Best et al., 2015) | GEO → Supplementary file. |
| `dgeGenesEnsembl75.RData` | thromboSeq source code | https://github.com/MyronBest/thromboSeq_source_code (folder `bin/`). |
| MSigDB v7.5.1 `.gmt` files | msigdbr 7.5.1 | included in `inputs/genesets/`. |

`src/prepare_inputs.py` and `src/prepare_gse68086.py` convert these files into the inputs read by the analysis code.
The STRING edge table used for the fold-wise PPI term (`string_edges.tsv`) is produced by `src/get_string_edges.py`
(needs internet access) from the gene list written by `fold_recompute.py mrmr_union`.
