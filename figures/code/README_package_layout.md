# Figures (version 7)

Draws all 25 figures of the paper (11) and the supplement (14) in one style, as vector PDFs (plus PNG previews), into `manuscript/figures_v7/`.

    cd code/figures_v7
    python make_all.py              # all figures
    python make_all.py fig_link     # one figure (any part of a function name)

## Inputs

Every plotted number is read from a saved result:

- `results/tables/*.csv` of this package;
- the PMTRD run directories (`runs/<experiment>/LATEST`, set `PMTRD_V2` to the project root);
- `manuscript/results/results_macros.tex`, which holds the numbers the paper prints.

Three figures recompute from the discovery data through the analysis engine (`code/tep_analysis.py`), with the same loading, folds and seed as the paper:

- S3, the contamination correlations;
- S7 panel A, the Stage 1 Mann–Whitney tiers (457 / 3,557 / 7,750);
- S9, the out-of-fold ROC; the script checks that the AUC equals `\CoreFixedAUC`.

S10 (PCA of log2(x+1) values) also recomputes from the discovery data, and writes its numbers to `manuscript/figures_v7/figure_macros.tex`, which the paper reads.

S7 panel B, the DESeq2 analysis of the development run, has no code in this package. It is therefore reproduced as an image from the original figure; the caption says so.

## Style

All style settings are in `style.py`:

- **Type:** FreeSans, 6–8.5 pt at print size. The figures are drawn at the printed size (A4 page with 2.2 cm margins) and included at 100%.
- **Colours** have one meaning across all figures:
  - predictive, molecular and technical reliability (blue, teal and violet);
  - one fixed colour plus a marker shape per selector;
  - NSCLC and healthy.
- **Palette check:** the selector palette was checked for colour-vision deficiency, and the marker shapes act as a second cue.
