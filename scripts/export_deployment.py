#!/usr/bin/env python3
"""Package the models trained by scripts/final_evaluation.py into a deployable bundle.

    python scripts/final_evaluation.py            # trains/caches predictions in outputs/final_run/
    python scripts/export_deployment.py           # -> outputs/deployed/protocol_H6/

The bundle holds the checkpoints, the imputer/scaler fitted on the training patients, a Platt
calibrator per model and the validation operating points (80%/95% specificity) used for risk levels.
Nothing is re-tuned: calibrators and thresholds come from the cached validation predictions, and the
baselines are retrained with the evaluation's seed and checked to reproduce its predictions exactly.
Load it with explainability_utils.deploy.ProtocolBundle (or SEPSIS_MODEL_SET=protocol for the dashboard).
"""
import argparse, json, pickle, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

import final_evaluation as fe
from explainability_utils.protocol import DL_CONFIGS, WINDOW, build_data, fit_baseline, make_samples, preprocess, subsample

BASELINES_EXPORTED = ["logistic_regression", "xgboost", "random_forest"]


def platt_params(p, y):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def calibrated(p, coef, intercept):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1.0 / (1.0 + np.exp(-(coef * z + intercept)))


def describe(p_va, y_va, p_te, y_te):
    coef, icpt = platt_params(p_va, y_va)
    pc_va = calibrated(p_va, coef, icpt)
    neg = pc_va[y_va == 0]
    spec = dict(platt=dict(coef=coef, intercept=icpt),
                threshold_medium=float(np.quantile(neg, 0.80)),     # 80% specificity on validation
                threshold_high=float(np.quantile(neg, 0.95)),       # 95% specificity on validation
                threshold_f1=fe.best_f1_threshold(y_va, pc_va))
    m = fe.point_metrics(y_te, p_te, p_va, y_va)
    pc_te = calibrated(p_te, coef, icpt)
    spec["test"] = {k: float(m[k]) for k in ("auroc", "auprc", "sens_at_80_spec", "brier")}
    spec["test"]["share_flagged_medium"] = float((pc_te >= spec["threshold_medium"]).mean())
    spec["test"]["share_flagged_high"] = float((pc_te >= spec["threshold_high"]).mean())
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "Dataset.csv"))
    ap.add_argument("--workdir", default=str(ROOT / "outputs" / "final_run"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "deployed" / "protocol_H6"))
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--neg-stride", type=int, default=4)
    a = ap.parse_args()
    work, out, H = Path(a.workdir), Path(a.out), a.horizon
    (out / "models").mkdir(parents=True, exist_ok=True)

    feats = json.load(open(ROOT / "outputs" / "config.json"))["feature_cols"]
    splits = json.load(open(ROOT / "outputs" / "cache" / "patient_splits_seed42.json"))
    d = build_data(a.data, feats, splits)
    r_tr, y_tr = make_samples(d, H, "train")
    r_va, y_va = make_samples(d, H, "val")
    r_te, y_te = make_samples(d, H, "test")
    Xs, Mk, imp, sc = preprocess(d, r_tr, return_fitted=True)
    joblib.dump(dict(imputer=imp, scaler=sc, features=feats, window=WINDOW, horizon=H), out / "preprocessor.joblib")

    cal = dict(horizon=H, window=WINDOW, features=feats, models={},
               protocol="scripts/final_evaluation.py: patient split seed 42, onset-relative labels, pre-onset hours only",
               risk_levels="Medium >= 80%-specificity threshold, High >= 95%-specificity threshold (validation, calibrated)")
    cached = lambda name, seed: np.load(work / "preds" / f"{name}_H{H}_s{seed}.npz")

    for name in DL_CONFIGS:
        zs = [cached(name, s) for s in a.seeds]
        p_va, p_te = np.mean([z["val"] for z in zs], 0), np.mean([z["test"] for z in zs], 0)
        cal["models"][name] = dict(kind="dl", seeds=a.seeds, **describe(p_va, y_va, p_te, y_te))
        for s in a.seeds:
            shutil.copy(work / f"{name}_H{H}_s{s}.pt", out / "models" / f"{name}_s{s}.pt")

    seed = a.seeds[0]
    for name in BASELINES_EXPORTED:
        rs, ys = subsample(r_tr, y_tr, a.neg_stride, seed)
        m = fit_baseline(name, Xs[rs], ys, seed, time_col=feats.index("ICULOS"))
        p_va, p_te = m.predict_proba(Xs[r_va])[:, 1], m.predict_proba(Xs[r_te])[:, 1]
        z = cached(name, seed)
        same = bool(np.allclose(p_va, z["val"], atol=1e-5) and np.allclose(p_te, z["test"], atol=1e-5))
        print(f"  {name}: retrained model reproduces cached evaluation predictions: {same}")
        if not same:
            raise SystemExit(f"{name} does not reproduce the evaluation predictions; refusing to export")
        with open(out / "models" / f"{name}_s{seed}.pkl", "wb") as f:
            pickle.dump(m, f)
        cal["models"][name] = dict(kind="sklearn", file=f"{name}_s{seed}.pkl", **describe(p_va, y_va, p_te, y_te))

    members = list(DL_CONFIGS)
    p_va = np.mean([np.mean([cached(n, s)["val"] for s in a.seeds], 0) for n in members], 0)
    p_te = np.mean([np.mean([cached(n, s)["test"] for s in a.seeds], 0) for n in members], 0)
    cal["ensemble_dl"] = dict(members=members, **describe(p_va, y_va, p_te, y_te))

    json.dump(cal, open(out / "calibration.json", "w"), indent=2)
    print(f"\nbundle written to {out}")
    for k, v in {**cal["models"], "ensemble_dl": cal["ensemble_dl"]}.items():
        t = v["test"]
        print(f"  {k:22s} AUROC {t['auroc']:.3f} AUPRC {t['auprc']:.3f} sens@80spec {t['sens_at_80_spec']:.3f} | "
              f"flags {t['share_flagged_medium']*100:.0f}% Medium+, {t['share_flagged_high']*100:.0f}% High")


if __name__ == "__main__":
    main()
