#!/usr/bin/env python3
"""Label and feature checks quoted in Paper 514 (revision plan Step 3 sanity checks).

    python scripts/check_label_leakage.py      # -> outputs/results/label_checks.csv

1. The five pre-engineered columns in Dataset.csv are recomputed from raw MAP and Resp alone. An exact match
   on every row shows they are not derived from SepsisLabel (and use only the current and earlier hours).
2. Label persistence: share of positive rows that are post-onset continuations, 1->0 reversions, and the
   row-level AUROC of a "model" that copies the previous hour's label.
3. Missing values in the raw clinical columns (0% means the file was imputed before it reached us).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
data = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Dataset.csv"
out = ROOT / "outputs" / "results" / "label_checks.csv"

df = pd.read_csv(data)
g = df.groupby("Patient_ID", sort=False)          # file order within a stay
rows = []

# 1. pre-engineered columns from raw vitals only
rebuilt = {
    "MAP_rolling_mean_6hr": g["MAP"].transform(lambda s: s.rolling(6, min_periods=1).mean()),
    "MAP_delta_1hr": g["MAP"].diff().fillna(0),
    "qSOFA_Resp": (df["Resp"] >= 22).astype(int),
    "qSOFA_MAP": (df["MAP"] < 65).astype(int),
}
rebuilt["qSOFA_Score_simple"] = rebuilt["qSOFA_Resp"] + rebuilt["qSOFA_MAP"]
for col, values in rebuilt.items():
    rows.append(("feature_rebuilt_from_raw_share", col, float(np.isclose(df[col], values, atol=1e-6).mean())))

# 2. label persistence
y = df["SepsisLabel"]
prev = g["SepsisLabel"].shift(1).fillna(0)
onset = (y == 1) & (prev == 0)
rows += [
    ("positive_rows", "SepsisLabel", int(y.sum())),
    ("onset_rows", "SepsisLabel", int(onset.sum())),
    ("post_onset_share_of_positive_rows", "SepsisLabel", float(1 - onset.sum() / y.sum())),
    ("label_reversions_1_to_0", "SepsisLabel", int(((prev == 1) & (y == 0)).sum())),
    ("copy_previous_label_row_auroc", "SepsisLabel", float(roc_auc_score(y, prev))),
]

# 3. missingness of raw clinical columns
raw = [c for c in df.columns if c not in {"Unnamed: 0", "Hour", "Patient_ID", "SepsisLabel", *rebuilt}]
rows.append(("raw_columns_missing_share_max", "all raw columns", float(df[raw].isna().mean().max())))

res = pd.DataFrame(rows, columns=["check", "column", "value"])
out.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(out, index=False)
print(res.to_string(index=False))
print(f"\nsaved {out}")

assert all(v == 1.0 for c, _, v in rows if c == "feature_rebuilt_from_raw_share"), "an engineered column is not a pure function of MAP/Resp"
