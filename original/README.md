# TEP-PMTRD

Code and results for

> **Beyond Peak Accuracy: A Predictive–Molecular–Technical Reliability Decomposition of a Tumour-Educated
> Platelet RNA Signature for Non-Small Cell Lung Cancer.**
> A. Tasnim, S. M. S. Islam, H. Mahmood, T. T. Alam, M. K. Morol.

Every number, table and figure in the paper and its supplement is produced by the scripts in `src/`, from
public GEO data, with fixed seeds. The `results/` folder contains the outputs used for the paper, so each
result can be checked without re-running anything.

## Contents

| Folder | |
|---|---|
| `src/` | analysis code (see the table below) |
| `config/` | settings used for the paper: `config.json` (discovery), `config_ext.json` (with GSE68086), `config_r9.json` (null panels) |
| `data/` | sample metadata, the 27-gene panel, and instructions for downloading the expression data (`data/README.md`) |
| `inputs/genesets/` | MSigDB v7.5.1 Hallmark, Reactome and GO BP gene sets |
| `notebooks/` | a Google Colab notebook that runs the main steps in order |
| `results/tables`, `results/figures` | every table and figure, as produced |
| `results/macros/` | every number quoted in the paper (JSON, one file per analysis step) |
| `docs/figure_table_map.md` | which script produces which figure, table and number |

## Installation

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt  (Python 3.13)
```

## Reproducing the paper

1. Download the inputs listed in `data/README.md` and edit the paths in `config/*.json`.
2. Run `reproduce.sh` (about 8 hours on 2 CPU cores; the longest steps are the 10,000 null panels, the
   nested cross-validation and the split-half replication). Each step writes its numbers to
   `out/macros/*.json`, its tables to `out/tables/` and its figures to `out/figures/`, so a crashed run can resume.
3. `python src/run_steps.py config/config.json merge` writes `out/results_macros.tex`, which the LaTeX
   manuscript reads.

## Main scripts

| Script | Analysis |
|---|---|
| `selection_pipeline.py` | seeded six-stage selection (Stages 1–6) |
| `fold_recompute.py` | fully fold-recomputable selection: feature-importance screen and STRING term re-derived in every training fold; nested CV |
| `get_string_edges.py` | downloads the STRING interactions for the genes listed by `fold_recompute.py mrmr_union` (needs internet) |
| `tep_analysis.py`, `run_steps.py` | library and step runner: core estimates, site audit, nested CV (four-term version), contamination, covariates, null panels, comparator |
| `sheng_sensitivity.py` | sensitivity of the re-implemented comparator to its implementation settings |
| `external_analysis.py`, `ext_reliability.py` | GSE68086: all alignment settings; locked-model calibration and gene-availability analysis |
| `harm_sim.py` | alignment within one site, where there is no batch effect |
| `r7_gsea.py`, `pathway_stability.py` | within-site pathway analysis; split-half replication of genes and gene sets |
| `dev_reruns.py`, `extra_analyses.py` | ablation, feature count, RFECV, stability selection, weight sensitivity and smaller analyses |
| `crosscheck_independent.py` | comparison with an independently written analysis |
| `make_*.py`, `redraw_*.py` | final figures |

## Licence

MIT (see `LICENSE`). The data are public and subject to the terms of GEO.
