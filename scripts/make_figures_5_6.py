#!/usr/bin/env python3
"""Real replacements for manuscript Figures 5 and 6 (the originals were dashboard placeholders).

    python scripts/make_figures_5_6.py

Figure 5  Hour-by-hour risk from the evaluated models for two held-out test patients chosen by a fixed rule
          (not by how good they look): the septic patient whose onset hour is closest to the median onset
          among test septics, and the non-septic patient whose stay length is closest to that one.
          Score = calibrated seed/model-averaged probability of onset within 6 h (explainability_utils.deploy).
Figure 6  Mean |TreeSHAP| of the XGBoost model (xgboost's own exact TreeSHAP via pred_contribs) over held-out
          test pre-onset hours: all positives plus a seeded sample of negatives. It explains ONE model
          (XGBoost), not "all models".
Writes outputs/figures/figure5_risk_trajectory.png, figure6_shap_xgboost.png and the CSVs behind them
(outputs/results/figure5_trajectory.csv, figure6_shap_mean_abs.csv).
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explainability_utils.deploy import ProtocolBundle
from explainability_utils.protocol import WINDOW, build_data, make_samples

OUT_FIG, OUT_RES = ROOT / "outputs" / "figures", ROOT / "outputs" / "results"


def pick_patients(d, test_pats):
    """Deterministic selection (see module docstring)."""
    pid, pos, onset = d["pid"], d["pos"], d["onset"]
    last_pos = pd.Series(pos).groupby(pid).max()
    onset_pat = pd.Series(onset).groupby(pid).min()
    big = np.iinfo(np.int32).max
    septic = onset_pat[(onset_pat < big) & onset_pat.index.isin(test_pats) & (onset_pat >= WINDOW)]
    target = float(np.median(septic.values))
    s_id = (septic - target).abs().sort_values(kind="stable").index[0]
    non = onset_pat[(onset_pat >= big) & onset_pat.index.isin(test_pats)].index
    length = last_pos + 1
    n_id = (length.loc[non] - (int(septic.loc[s_id]) + 1)).abs().sort_values(kind="stable").index[0]
    return int(s_id), int(n_id)


def trajectory(b, d, patient, last_hour):
    """Ensemble risk for hours last_hour-WINDOW+1 .. last_hour (history truncated at each hour)."""
    rows = np.flatnonzero(d["pid"] == patient)
    raw = d["df"].iloc[rows][b.features].reset_index(drop=True)
    out = []
    for t in range(max(0, last_hour - WINDOW + 1), last_hour + 1):
        ens = b.predict_ensemble(raw.iloc[max(0, t - WINDOW + 1): t + 1])
        out.append((t, ens["risk_score"]))
    return out


def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    b = ProtocolBundle()
    H = b.cal["horizon"]
    feats = b.features
    splits = json.load(open(ROOT / "outputs" / "cache" / "patient_splits_seed42.json"))
    d = build_data(str(ROOT / "Dataset.csv"), feats, splits)
    s_id, n_id = pick_patients(d, set(splits["test"]))
    onset_pos = int(pd.Series(d["onset"]).groupby(d["pid"]).min().loc[s_id])
    n_len = int((d["pid"] == n_id).sum())
    spec = b.cal["ensemble_dl"]

    # ---------------- Figure 5
    ts = trajectory(b, d, s_id, onset_pos - 1)                 # last pre-onset hour
    tn = trajectory(b, d, n_id, n_len - 1)
    rows = ([dict(patient=s_id, group="septic", hour=t, hours_to_onset=t - onset_pos, risk=r) for t, r in ts] +
            [dict(patient=n_id, group="non-septic", hour=t, hours_to_onset=np.nan, risk=r) for t, r in tn])
    pd.DataFrame(rows).to_csv(OUT_RES / "figure5_trajectory.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, data, title, xlab in [
            (ax[0], ts, f"Septic test patient {s_id}: 24 h before labelled onset", "hours relative to labelled onset"),
            (ax[1], tn, f"Non-septic test patient {n_id}: last 24 h of stay", "hour of stay")]:
        x = [t - onset_pos for t, _ in data] if data is ts else [t for t, _ in data]
        a.plot(x, [r for _, r in data], marker="o", ms=3, lw=1.5, color="#1f77b4")
        a.axhline(spec["threshold_high"], color="#d62728", ls="--", lw=1, label="High (95% specificity)")
        a.axhline(spec["threshold_medium"], color="#ff7f0e", ls="--", lw=1, label="Medium (80% specificity)")
        a.set(title=title, xlabel=xlab)
    ax[0].axvline(0, color="grey", ls=":", lw=1)
    ax[0].set_ylabel("calibrated P(onset within 6 h)")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Digital-twin risk trajectory (average of four deep models, 3 seeds each)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "figure5_risk_trajectory.png", dpi=170)
    plt.close(fig)

    # ---------------- Figure 6: exact TreeSHAP of XGBoost on held-out pre-onset hours
    import xgboost as xgb
    r_te, y_te = make_samples(d, H, "test")
    rng = np.random.default_rng(0)
    keep = np.flatnonzero(y_te == 1).tolist() + rng.choice(np.flatnonzero(y_te == 0), 4000, replace=False).tolist()
    rows_sel = r_te[np.array(keep)]
    X = b.scaler.transform(b.imputer.transform(d["X"][rows_sel])).astype(np.float32)
    booster = b.sk["xgboost"].get_booster()
    contrib = booster.predict(xgb.DMatrix(X, feature_names=booster.feature_names), pred_contribs=True)[:, :-1]
    mean_abs = np.abs(contrib).mean(0)
    mean_signed_pos = contrib[np.array([k < int(y_te.sum()) for k in range(len(keep))])].mean(0)
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs, "mean_shap_on_pre_onset_hours": mean_signed_pos}
                       ).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(OUT_RES / "figure6_shap_mean_abs.csv", index=False)
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(top.feature, top.mean_abs_shap, color="#1f77b4")
    ax.set(xlabel="mean |SHAP| (log-odds), held-out pre-onset hours",
           title="XGBoost: top 10 features by TreeSHAP importance")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "figure6_shap_xgboost.png", dpi=170)
    plt.close(fig)
    print(f"septic patient {s_id} (onset hour {onset_pos}), non-septic patient {n_id} (length {n_len})")
    print("trajectory (septic) risk first/last:", round(ts[0][1], 4), round(ts[-1][1], 4),
          "| max", round(max(r for _, r in ts), 4), "| non-septic max", round(max(r for _, r in tn), 4))
    print(imp.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
