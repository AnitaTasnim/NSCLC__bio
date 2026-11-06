# Figures

| Folder | Content |
|---|---|
| `main/` | The 11 main-text figures (Fig01–Fig11), as vector PDF, exactly as in the paper. |
| `supplement/` | The 14 supplementary figures (FigS01–FigS14), as vector PDF, exactly as in the supplement. `figure_macros.tex` holds the PCA numbers that the paper prints. |
| `code/` | The code that draws the supplementary figures. |
| `inputs/` | Inputs the figure code reads that are not elsewhere in the repository: the macro file with every number the paper prints, the development-run selection profile (Supplementary Table "panel") and the original image used for panel B of Figure S7. |

## Main-text figures

The main-text figures were designed, and their plotting code written, with the assistance of OpenAI Codex. The paper's declaration on generative AI says so.

Every number they show is a saved result in this repository:

- `original/results/tables/*.csv`;
- the run directories in `runs/`;
- `runs/export/` (for Figures 2 and 9–11).

The table below gives the source of each figure.

| Figure | Source of the numbers |
|---|---|
| 1 | framework schematic (counts as in the Methods) |
| 2 | `runs/replicates/<LATEST>/results/scores.csv`, `units.csv` |
| 3 | `original/results/tables/ALL_AUCs_with_95CI.csv` |
| 4 | `LK_calibration_bins.csv`, `LK_feature_availability_summary.csv` |
| 5 | `R4_nested_folds.csv` and the stability macros |
| 6 | `R9_random_panels.csv` |
| 7 | `PS_split_half_summary.csv`, `PS_panel_gene_replication.csv`, `PS_gene_set_replication.csv` |
| 8 | `R1_genomewide_site_vs_disease.csv`, `R1_panel_site_vs_disease.csv` |
| 9 | `runs/export/macros_v2.json` (`VxBm*`, `VxCd*`, `VxRp*`, `VxPw*`) |
| 10 | `runs/replicates/<LATEST>/results/`, `runs/pathway_model/<LATEST>/summary.json` |
| 11 | `runs/export/V2_generalisation.csv` |

## Re-drawing the supplementary figures

```bash
bash figures/make_figures.sh             # writes figures/build/FigS*.pdf
```

Most figures read only the saved tables. Four figures recompute from the discovery data through the analysis engine:

- S3, the contamination correlations;
- S7 panel A, the Stage 1 Mann–Whitney tiers;
- S9, the out-of-fold ROC, which is checked against the published AUC 0.9626;
- S10, the PCA.

These four figures need the prepared GSE89843 inputs in `data/inputs/` (see the main README).

Panel B of Figure S7 is the count-model (DESeq2) analysis of the development run. No code for that analysis is in this repository, so the panel is reproduced from the original image (`inputs/fig05_de_landscape.png`), as the caption states.

`python figures/code/make_all.py fig_` also draws an alternative, earlier version of the main-text figures from the same data. Those versions are not the ones in the paper.
