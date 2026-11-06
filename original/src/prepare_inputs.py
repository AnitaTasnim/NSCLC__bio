#!/usr/bin/env python3
"""Convert the original GSE89843 files into the inputs used by the analysis code.

    python prepare_inputs.py dataset_NSCLC GSE89843_filtered_metadata.csv inputs/

dataset_NSCLC: CSV with Sample_Names, one column per gene, and 'disease' (1 = NSCLC, 0 = healthy)
metadata:      CSV with the GSM id (first column), 'characteristics' (the GEO title) and 'disease'
Writes inputs/GSE89843_expr.parquet (samples x genes, GEO accession order) and
inputs/GSE89843_meta.csv (sample, label, title, disease_txt)."""
import os
import sys

import pandas as pd


def main(expr_csv, meta_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(expr_csv).set_index("Sample_Names").sort_index()
    y = df.pop("disease").astype(int)
    meta = pd.read_csv(meta_csv)
    meta.columns = ["gsm", "title", "disease_txt"]
    m = meta.set_index("gsm").loc[df.index]
    df.index.name = "sample"
    df.astype("float64").to_parquet(os.path.join(outdir, "GSE89843_expr.parquet"))
    pd.DataFrame({"sample": df.index, "label": y.values, "title": m["title"].values,
                  "disease_txt": m["disease_txt"].values}).to_csv(os.path.join(outdir, "GSE89843_meta.csv"), index=False)
    print(f"{df.shape[0]} samples x {df.shape[1]} genes; NSCLC {int(y.sum())}, healthy {int((1 - y).sum())}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
