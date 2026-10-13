#!/usr/bin/env python3
"""Verify that the deployment bundle reproduces the evaluation's predictions on real held-out patients.

    python scripts/check_deployment.py [--n 300]

For random test-set hours it rebuilds the raw patient history (last 24 rows), runs it through
explainability_utils.deploy.ProtocolBundle (the path the dashboard/CLI use) and compares the raw model
probabilities with the evaluation path's. Exits non-zero on a mismatch.

The evaluation-path probabilities come from the cache written by scripts/final_evaluation.py
(outputs/final_run/preds) when present. From a fresh clone there is no cache, so they are recomputed from the
bundle's own checkpoints and preprocessor, and the full-test-set AUROC/AUPRC are also compared with the
metrics recorded in calibration.json. This needs Dataset.csv but no training and no cache.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from explainability_utils.deploy import ProtocolBundle
from sklearn.metrics import average_precision_score, roc_auc_score
from explainability_utils.protocol import WINDOW, build_data, make_samples, windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--workdir", default=str(ROOT / "outputs" / "final_run"))
    a = ap.parse_args()
    b = ProtocolBundle()
    H = b.cal["horizon"]
    feats = b.features
    splits = json.load(open(ROOT / "outputs" / "cache" / "patient_splits_seed42.json"))
    d = build_data(str(ROOT / "Dataset.csv"), feats, splits)
    r_te, y_te = make_samples(d, H, "test")
    pick = np.random.default_rng(0).choice(len(r_te), a.n, replace=False)
    raw_df = d["df"]
    seeds = {n: spec["seeds"] for n, spec in b.cal["models"].items() if spec["kind"] == "dl"}
    cache_dir = Path(a.workdir) / "preds"
    use_cache = all((cache_dir / f"{n}_H{H}_s{s}.npz").exists() for n, ss in seeds.items() for s in ss)
    Xs = b.scaler.transform(b.imputer.transform(d["X"])).astype(np.float32)     # the bundle's own preprocessing
    Mk = ~np.isnan(d["X"])
    cached, cached_sk, full = {}, {}, True
    if use_cache:
        print("evaluation path: cached predictions from", cache_dir)
        cached = {n: np.mean([np.load(cache_dir / f"{n}_H{H}_s{s}.npz")["test"] for s in ss], 0) for n, ss in seeds.items()}
        cached_sk = {n: np.load(cache_dir / f"{n}_H{H}_s{spec['file'].split('_s')[-1].split('.')[0]}.npz")["test"]
                     for n, spec in b.cal["models"].items() if spec["kind"] == "sklearn" and n in b.sk}
    else:
        print("evaluation path: no cache found; recomputing from the bundle's checkpoints (all test hours)")
        import torch
        with torch.no_grad():
            for n, nets in b.nets.items():
                out = []
                for i in range(0, len(r_te), 4096):
                    f, m, dt = windows(Xs, Mk, d["start"], r_te[i:i + 4096])
                    out.append(np.mean([torch.sigmoid(net(f, m, dt)).flatten().numpy() for net in nets], 0))
                cached[n] = np.concatenate(out)
        cached_sk = {n: b.sk[n].predict_proba(Xs[r_te])[:, 1] for n in b.sk}
        # whole-test-set metrics must match what calibration.json recorded at export time
        print("full test set, recomputed vs recorded in calibration.json:")
        for n, p in {**cached, **cached_sk}.items():
            spec = b.cal["models"][n]
            au, ap = roc_auc_score(y_te, p), average_precision_score(y_te, p)
            ok = abs(au - spec["test"]["auroc"]) < a.tol and abs(ap - spec["test"]["auprc"]) < a.tol
            print(f"  {n:22s} AUROC {au:.4f} (recorded {spec['test']['auroc']:.4f})  AUPRC {ap:.4f} (recorded {spec['test']['auprc']:.4f})  {'OK' if ok else 'MISMATCH'}")
            full &= ok
    worst = {}
    for i in pick:
        r = r_te[i]
        hist = raw_df.iloc[max(d["start"][r], r - WINDOW + 1): r + 1][feats]
        got = b.raw_dl(hist)
        for n, p in got.items():
            worst[n] = max(worst.get(n, 0.0), abs(p - cached[n][i]))
        for n, res in b.predict_baselines(hist).items():
            worst[n] = max(worst.get(n, 0.0), abs(res["raw_score"] - cached_sk[n][i]))
    print(f"{a.n} random held-out hours, max |runtime - evaluation| raw probability per model:")
    bad = False
    for n, w in sorted(worst.items()):
        ok = w <= a.tol
        bad |= not ok
        print(f"  {n:22s} {w:.2e}  {'OK' if ok else 'MISMATCH'}")
    sys.exit(1 if (bad or not full) else 0)


if __name__ == "__main__":
    main()
