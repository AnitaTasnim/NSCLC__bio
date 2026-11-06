#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_gse68086.py - build the external-cohort inputs for tep_analysis.py from the GEO file
GSE68086_TEP_data_matrix.txt.gz (57,736 Ensembl-75 genes x 285 samples, read counts).

    python prepare_gse68086.py GSE68086_TEP_data_matrix.txt.gz dgeGenesEnsembl75.RData outdir/

* labels from the sample names, exactly as in the development notebook (clean_phase3.ipynb):
  'Lung-...' or containing 'NSCLC' -> 1; 'HD-...' or 'Control-...' -> 0; other tumour types dropped
* the 19 samples that are also in GSE89843 are removed (list from the development notebook)
* Ensembl-75 gene ids -> HGNC symbols with the annotation shipped in the thromboSeq source code
  (Best et al., github.com/MyronBest/thromboSeq_source_code, bin/dgeGenesEnsembl75.RData);
  if an RData file is not available, a two-column CSV (ensembl_gene_id, hgnc_symbol) also works.
  Several ids with the same symbol are summed.
Writes  GSE68086_counts_symbols.parquet (samples x genes) and GSE68086_meta.csv (sample, label).
"""
import os
import sys

import numpy as np
import pandas as pd

KNOWN_OVERLAP = {
    "Vumc-NSCLC-49", "Vumc-NSCLC-57", "Vumc-NSCLC-59",
    "MGH-NSCLC-L01-TR458", "MGH-NSCLC-L03-TR488", "MGH-NSCLC-L07-TR459",
    "MGH-NSCLC-L11-TR477", "MGH-NSCLC-L12-TR478", "MGH-NSCLC-L18-TR479",
    "MGH-NSCLC-L19-TR461", "MGH-NSCLC-L20-TR500", "MGH-NSCLC-L22-TR462",
    "MGH-NSCLC-L23-TR524", "MGH-NSCLC-L25-TR480", "MGH-NSCLC-L39-TR519",
    "MGH-NSCLC-L40-TR520", "MGH-NSCLC-L51-TR521", "MGH-NSCLC-L58-TR525",
    "MGH-NSCLC-L65-TR523",
}


def label_of(name: str):
    if name.startswith("Lung-") or "NSCLC" in name:
        return 1
    if name.startswith("HD-") or name.startswith("Control-"):
        return 0
    return None


def load_annotation(path: str) -> pd.Series:
    if path.lower().endswith((".rdata", ".rda")):
        import pyreadr
        r = pyreadr.read_r(path)
        ann = next(iter(r.values()))
    else:
        ann = pd.read_csv(path)
    idc = [c for c in ann.columns if "ensembl" in c.lower()][0]
    syc = [c for c in ann.columns if "symbol" in c.lower()][0]
    ann = ann[[idc, syc]].dropna()
    ann = ann[ann[syc].astype(str).str.len() > 0]
    return ann.drop_duplicates(idc).set_index(idc)[syc].astype(str)


def main(matrix_path, ann_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    M = pd.read_csv(matrix_path, sep="\t", index_col=0, low_memory=False)
    print("matrix", M.shape)
    cols = [c for c in M.columns if c not in KNOWN_OVERLAP and label_of(c) is not None]
    excl = [c for c in M.columns if c in KNOWN_OVERLAP]
    y = pd.Series({c: label_of(c) for c in cols}, name="label")
    print(f"excluded (overlap with GSE89843): {len(excl)}; independent NSCLC {int(y.sum())}, healthy {int((1 - y).sum())}")
    sym = load_annotation(ann_path)
    C = M[cols].copy()
    C.index = C.index.astype(str)
    C["__sym"] = C.index.map(sym)
    print(f"ids mapped to a symbol: {C['__sym'].notna().sum()} / {len(C)}")
    C = C.dropna(subset=["__sym"]).groupby("__sym").sum()
    X = C.T
    X.index.name = "sample"
    X.to_parquet(os.path.join(outdir, "GSE68086_counts_symbols.parquet"))
    pd.DataFrame({"sample": y.index, "label": y.values}).to_csv(os.path.join(outdir, "GSE68086_meta.csv"), index=False)
    print("written:", X.shape)


if __name__ == "__main__":
    main(*sys.argv[1:4])
