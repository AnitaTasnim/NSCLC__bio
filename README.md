# TEP-PMTRD

Code, results and figures for the paper:

> **Beyond Peak Accuracy: Decomposing the Reliability of Machine-Learning Biomarkers Reveals Patient-Level
> Instability in a Tumour-Educated Platelet RNA Signature for Non-Small Cell Lung Cancer**
> Anita Tasnim, Syed Mohammed Shamsul Islam, Hassan Mahmood, Tamzid Tanvi Alam, Md Kishor Morol,
> Md. Hasibur Rahman (submitted).

It contains:

- **PMTRD:** a Predictive–Molecular–Technical Reliability Decomposition of a machine-learning biomarker;
- **ψ:** a control-arm estimate of how much the re-selection of features moves the predictions for individual
  patients;
- every experiment of the paper, with its completed run (configuration, log, metadata with input and code hashes,
  and results).

| Folder | What it is |
|---|---|
| `pmtrd/`, `configs/`, `tests/` | The reliability experiments: replicate study and ψ, selector benchmark, pathway model, generalisation to three further data sets, and sensitivity analyses. |
| `engine/` | The verified analysis code that produced the published panel results, byte for byte. |
| `runs/` | Every completed run used in the paper (`LATEST` in each folder names it). `runs/export/` holds the numbers, tables and figure data that the manuscript reads. |
| `original/` | The original analysis: scripts, configuration, notebook, and all result tables (`original/results/tables/`). |
| `figures/` | All figures of the paper and the supplement as vector PDF, and the code that draws the supplementary figures (see `figures/README.md`). |
| `data/` | The metadata, the reference outputs the experiments build on, and the gene-set files. The expression data are public and are not redistributed; see Data below. |

Quick start:

```bash
pip install -r requirements.txt
python -m pytest -q tests          # 37 tests
python -m pmtrd list               # the experiments
bash figures/make_figures.sh       # re-draw the supplementary figures
```

---

# TEP-PMTRD v2: reliability experiments for the 27-gene NSCLC platelet signature

This project adds the reliability experiments of the paper to the original analysis, without changing it.
`engine/` holds the verified code that produced the published results (`tep_analysis.py`,
`selection_pipeline.py`, `fold_recompute.py`, `sheng_sensitivity.py`), byte for byte. Everything new is in
`pmtrd/`, which imports the engine and adds configuration, validation, experiment tracking, metrics and
the new experiments.

```
configs/            one JSON file per experiment (merged over configs/base.json)
engine/             original analysis code and its configuration (unchanged)
pmtrd/
  config.py         loading, schema validation, hashing
  tracking.py       versioned run directories, structured log, metadata (versions, input and code SHA-256)
  engine.py         data loading through the engine, input validation
  selectors.py      six-stage, Mann-Whitney top-k, L1 top-k, stability selection, mRMR-IFS-SVM, random, fixed
  metrics/          stability (Jaccard, Kuncheva, Nogueira phi, jackknife), predictive (calibration, Brier,
                    Wilson, Youden, bootstrap), agreement (Spearman, discrepancy, ambiguity, identity share psi)
  pathway.py        single-sample gene-set scores (within-sample ranks) for the pathway model
  datasets.py       loaders of the generalisation data sets (GSE68086 pan-cancer, Singh, Gordon)
  experiments/      replicates, benchmark, selector_cards, pathway_model, internal_calibration, tuning,
                    ppi_sensitivity, composite_ablation, column_order, replicates_generic
  export.py         collects the latest completed runs into manuscript macros, LaTeX tables and figures
tests/              unit tests (pytest; synthetic data, for the code only)
data/inputs/        prepared GSE89843 and GSE68086 inputs (see below; not distributed) and the MSigDB
                    v7.5.1 gene-set files (distributed, in data/inputs/genesets/)
data/reference/     outputs of the original analysis that the experiments build on (fold assignment,
                    nested-CV fold panels and predictions, five-term caches)
runs/               one directory per run: config.json, metadata.json, log.jsonl, cache/, results/, summary.json
```

## Installation and tests

```bash
pip install -r requirements.txt        # Python 3.13; versions pinned to those used for the paper
python -m pytest -q tests              # 37 tests, about 5 s
```

## Data

The inputs are public (GEO GSE89843 via GREIN, and GSE68086). Download them as described in
`original/DATA.md` and prepare them with the original scripts:

```bash
python original/src/prepare_inputs.py dataset_NSCLC data/GSE89843_filtered_metadata.csv data/inputs/
python original/src/prepare_gse68086.py GSE68086_TEP_data_matrix.txt.gz dgeGenesEnsembl75.RData data/inputs/gse68086/
```

The generalisation runs (`replicates_generic`) use three further public data sets, placed in `data/external/`:
`GSE68086_TEP_data_matrix.txt.gz` (GEO GSE68086 supplementary file), and `singh.RData` and `gordon.RData`
from the `datamicroarray` collection (https://github.com/ramhiser/datamicroarray, `data/` folder). The
loaders and their fixed, label-free preprocessing are documented in `pmtrd/datasets.py`.

No private or restricted data are used or distributed. `original/` holds the code, configuration, notebook
and outputs of the original analysis (see `original/README.md` and `original/docs/figure_table_map.md`).

## Running the experiments

```bash
python -m pmtrd list
python -m pmtrd column_order         --config configs/column_order.json          # ~1 min
python -m pmtrd internal_calibration --config configs/internal_calibration.json  # ~2 min
python -m pmtrd ppi_sensitivity      --config configs/ppi_sensitivity.json       # ~5 min
python -m pmtrd composite_ablation   --config configs/composite_ablation.json    # ~10 min
python -m pmtrd tuning               --config configs/tuning.json                # ~10 min
python -m pmtrd benchmark            --config configs/benchmark.json             # ~10 min
python -m pmtrd replicates           --config configs/replicates.json            # ~1 h on 2 cores
python -m pmtrd pathway_model        --config configs/pathway_model.json         # ~3 min
python -m pmtrd pathway_model        --config configs/pathway_model_gobp.json    # sensitivity: + GO BP sets
python -m pmtrd selector_cards       --config configs/selector_cards.json        # ~15 min
python -m pmtrd replicates_generic   --config configs/generic_singh_prostate.json      # ~15 min
python -m pmtrd replicates_generic   --config configs/generic_gordon_lung.json         # ~25 min
python -m pmtrd replicates_generic   --config configs/generic_gse68086_pancancer.json  # ~35 min
python -m pmtrd.export runs/export                                               # macros, tables, figure
```

Each command creates `runs/<experiment>/<UTC time>_<config hash>/`. An interrupted run continues with
`--resume <that directory name>` (completed units are cached; a changed configuration is refused).
`--mode summarise` recomputes the summary tables from the cache. `runs/<experiment>/LATEST` names the most
recent completed run. Times are on 2 CPU cores; do not run two experiments at once on 2 cores (XGBoost
threads then oversubscribe the CPU and every step slows down many-fold).

## Determinism

The global seed is 42 (`configs/base.json`). Outer folds are the paper's (`R8_outer_folds.csv`) and every
experiment verifies them before it starts. Subsample r of fold k uses seed 42*1000 + 100k + r. Random panels
use seeds 10,000 + 100k + r (replicates) and 20,000 + 100r + k (benchmark). Experiments that re-use the
paper's panels check that they reproduce them exactly (for example, the internal-calibration experiment
re-creates the paper's four-term out-of-fold predictions and stops if any differs by more than 1e-6).

## Crossed pick-freeze estimator (methods development, not yet applied to the NSCLC data)

New, separate code; nothing in `engine/`, `original/`, `pmtrd/experiments/` or `runs/` (other than
`runs/crossed_sim/`) was changed.

| Path | Content |
| --- | --- |
| `pmtrd/crossed/estimator.py` | Moments D_col, D_row, D_both, D_seed, V_R; components; Eq. 3; pigeonhole, draws and patients bootstraps |
| `pmtrd/crossed/synthetic.py` | Analytic models with known components (test suite) |
| `pmtrd/crossed/simulate.py` | Block-correlated Gaussian generator, selectors, classifiers, crossed designs, resumable runner |
| `pmtrd/crossed/report.py` | Summary tables, pre-registered criterion check, figures (reads saved outputs only) |
| `configs/crossed_sim.json` | Simulation design (fixed before any run) |
| `docs/crossed_estimator_derivation.md` | Derivation and validation (Stage 0) |
| `docs/crossed_simulation.md` | Pre-registered analysis plan for E1–E4 |
| `tests/test_crossed_estimator.py`, `tests/test_crossed_pipeline.py` | 22 tests |
| `pmtrd/crossed/diagnostics.py`, `tests/test_crossed_diagnostics.py` | POST-HOC interval diagnostic and candidate jackknife interval (3 tests); not part of the pre-registered verdict |

Exact commands (CPU; set `"device": "cuda"` for XGBoost in the config to use a GPU):

```bash
python -m pytest tests/test_crossed_estimator.py tests/test_crossed_pipeline.py -q
# two workers in parallel; each can be stopped and restarted (finished units are skipped)
OMP_NUM_THREADS=1 python -m pmtrd.crossed.simulate --config configs/crossed_sim.json --out runs/crossed_sim/v1 --worker 0 --workers 2 &
OMP_NUM_THREADS=1 python -m pmtrd.crossed.simulate --config configs/crossed_sim.json --out runs/crossed_sim/v1 --worker 1 --workers 2 &
wait
python -m pmtrd.crossed.report --run runs/crossed_sim/v1     # writes runs/crossed_sim/v1/summary/
# post hoc (written after the results were seen): analytic diagnostic, then the fresh-seed pipeline pilot
python -m pmtrd.crossed.diagnostics --out runs/crossed_sim/diagnostics_posthoc
python -m pmtrd.crossed.diagnostics --pilot logistic --n-T 4 --n-rep 25
python -c "from pmtrd.crossed.diagnostics import pilot_figure; pilot_figure('runs/crossed_sim/diagnostics_posthoc/pilot_logistic')"
```

Result of the pre-registered run (8 Nov 2026): **NO-GO** (`runs/crossed_sim/v1/summary/verdict.json`).
The share estimators met the bias criterion in 166 of 168 counted cells, but the pigeonhole interval met
the coverage criterion in only 34 of 168 (Pi_S in none).

Outputs: `runs/crossed_sim/v1/units/*.json` (one file per truth or replicate; raw estimates, intervals,
runtime), `panels/*.npz` (selected panels, reused across classifiers), `metadata.json` (configuration hash,
package versions, CPU), `log_worker*.txt`, and `summary/` (truth.csv, replicates_raw.csv, summary.csv,
criterion_cells.csv, runtime.csv, verdict.json, figures).
