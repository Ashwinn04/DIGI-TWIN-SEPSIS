#!/usr/bin/env python3
"""Verify that the deployment bundle reproduces the evaluation's predictions on real held-out patients.

    python scripts/check_deployment.py [--n 300]

For random test-set hours it rebuilds the raw patient history (last 24 rows), runs it through
explainability_utils.deploy.ProtocolBundle (the path the dashboard/CLI use) and compares the raw model
probabilities with the ones cached by scripts/final_evaluation.py. Exits non-zero on a mismatch.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from explainability_utils.deploy import ProtocolBundle
from explainability_utils.protocol import WINDOW, build_data, make_samples


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
    cached = {n: np.mean([np.load(Path(a.workdir) / "preds" / f"{n}_H{H}_s{s}.npz")["test"] for s in ss], 0) for n, ss in seeds.items()}
    cached_sk = {n: np.load(Path(a.workdir) / "preds" / f"{n}_H{H}_s{spec['file'].split('_s')[-1].split('.')[0]}.npz")["test"]
                 for n, spec in b.cal["models"].items() if spec["kind"] == "sklearn"}
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
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
