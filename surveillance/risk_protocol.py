"""Leakage-safe out-of-fold patient risk for the surveillance layer, from the project's own models.

Uses explainability_utils.protocol (the same labels, preprocessing, windows and architectures as
scripts/final_evaluation.py), so the risk signal comes from the digital-twin models and not a
separate classifier. Differences from risk.patient_risk_scores (kept as the default):
  * post-onset hours are never scored, so an alert cannot come from a patient who is already septic
    (89.5% of positive rows in Dataset.csv are such continuations);
  * the model can be any of the project's models (xgboost, grud, lstm, cnn_lstm, transformer, ...);
  * patient-grouped K-fold: every patient is scored by a model that never saw them.
Patients whose label is already positive on their first row have no pre-onset hours and are excluded.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from explainability_utils.protocol import (DL_CONFIGS, build_data, fit_baseline, make_samples, predict_dl,
                                           preprocess, subsample, train_dl, windows)

ROOT = Path(__file__).resolve().parent.parent


def patient_risk_scores_protocol(csv_path, source="grud", horizon=6, n_splits=5, seed=42,
                                 epochs=25, patience=5, neg_stride=4, cache_dir=None, verbose=True):
    """Return (patient_df, metrics). patient_df: Patient_ID, max_risk, ever_septic."""
    cache = None
    if cache_dir:
        cache = Path(cache_dir) / f"surveillance_risk_{source}_H{horizon}_k{n_splits}_s{seed}"
        if cache.with_suffix(".csv").exists() and cache.with_suffix(".json").exists():
            return pd.read_csv(cache.with_suffix(".csv")), json.load(open(cache.with_suffix(".json")))

    feats = json.load(open(ROOT / "outputs" / "config.json"))["feature_cols"]
    ids = pd.read_csv(csv_path, usecols=["Patient_ID"]).Patient_ID.unique()
    d = build_data(csv_path, feats, {"all": ids.tolist()})
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.sort(ids))
    fold_of = {p: i % n_splits for i, p in enumerate(order)}
    fold = pd.Series(d["pid"]).map(fold_of).to_numpy()
    device = None
    if source in DL_CONFIGS:
        from explainability_utils.training import get_device
        device = get_device()

    rows_all, p_all, y_all = [], [], []
    for k in range(n_splits):
        t0 = time.time()
        in_test = fold == k
        train_pats = np.unique(d["pid"][~in_test])
        val_pats = set(rng.choice(train_pats, size=int(0.15 * len(train_pats)), replace=False).tolist())
        split = np.where(in_test, "test", np.where(pd.Series(d["pid"]).isin(val_pats), "val", "train")).astype(object)
        d["split"] = split
        r_tr, y_tr = make_samples(d, horizon, "train")
        r_va, y_va = make_samples(d, horizon, "val")
        r_te, y_te = make_samples(d, horizon, "test")
        Xs, Mk = preprocess(d, r_tr)
        rs, ys = subsample(r_tr, y_tr, neg_stride, seed)
        if source in DL_CONFIGS:
            rv, yv = subsample(r_va, y_va, neg_stride, 0)
            ckpt = Path(cache_dir or ROOT / "outputs" / "final_run") / f"_oof_{source}_fold{k}.pt"
            net = train_dl(source, seed, windows(Xs, Mk, d["start"], rs), windows(Xs, Mk, d["start"], rv),
                           ys, yv, device, epochs, patience, ckpt)
            p = predict_dl(net, windows(Xs, Mk, d["start"], r_te), device)
            ckpt.unlink(missing_ok=True)
        else:
            m = fit_baseline(source, Xs[rs], ys, seed, time_col=feats.index("ICULOS"))
            p = m.predict_proba(Xs[r_te])[:, 1]
        rows_all.append(r_te); p_all.append(p); y_all.append(y_te)
        if verbose:
            print(f"  fold {k + 1}/{n_splits}: {len(r_te):,} held-out hours, AUROC {roc_auc_score(y_te, p):.3f}  [{time.time() - t0:.0f}s]")

    rows, p, y = np.concatenate(rows_all), np.concatenate(p_all), np.concatenate(y_all)
    df = pd.DataFrame({"Patient_ID": d["pid"][rows], "risk": p})
    pt = df.groupby("Patient_ID").risk.max().rename("max_risk").reset_index()
    ever = pd.Series(d["onset"] < np.iinfo(np.int32).max, index=d["pid"]).groupby(level=0).max()
    pt["ever_septic"] = pt.Patient_ID.map(ever).astype(int)
    n_all = len(ids)
    metrics = {
        "risk_source": source, "horizon_h": horizon,
        "row_level_auroc": float(roc_auc_score(y, p)), "row_level_auprc": float(average_precision_score(y, p)),
        "patient_level_auroc": float(roc_auc_score(pt.ever_septic, pt.max_risk)),
        "patient_level_auprc": float(average_precision_score(pt.ever_septic, pt.max_risk)),
        "n_patients": int(len(pt)), "n_excluded_no_pre_onset_hours": int(n_all - len(pt)),
        "patient_sepsis_rate": float(pt.ever_septic.mean()),
        "cv": f"{n_splits}-fold patient-grouped, out-of-fold, pre-onset hours only",
    }
    if cache:
        pt.to_csv(cache.with_suffix(".csv"), index=False)
        json.dump(metrics, open(cache.with_suffix(".json"), "w"), indent=2)
    return pt, metrics
