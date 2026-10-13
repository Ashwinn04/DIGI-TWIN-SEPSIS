#!/usr/bin/env python3
"""Compute the manuscript numbers that are not stored in any other results file.

    python scripts/paper_numbers.py        # -> outputs/results/paper_numbers.csv

Every other number in the paper comes straight from final_results*.csv, sanity_checks.csv, label_checks.csv,
figure2_trajectory.csv, figure3_shap_mean_abs.csv or the surveillance JSON/CSV files (see paper/README.md).
This script covers the rest: dataset counts, how many test patients are positive, how the file was imputed,
a measurement-indicator control, the cross-machine reproduction statement and a few derived ratios.
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from explainability_utils.protocol import build_data, fit_baseline, make_samples, subsample

RES = ROOT / "outputs" / "results"
rows = []


def add(key, value, note=""):
    rows.append(dict(key=key, value=value if isinstance(value, str) else float(value), note=note))


def main():
    csv = ROOT / "Dataset.csv"
    df = pd.read_csv(csv)
    feats = json.load(open(ROOT / "outputs" / "config.json"))["feature_cols"]

    # ---- dataset
    n_pat, n_rows = df.Patient_ID.nunique(), len(df)
    add("dataset_icu_stays", n_pat, "distinct Patient_ID")
    add("dataset_hourly_records", n_rows)
    add("share_records_positive_label", df.SepsisLabel.mean(), "rows with SepsisLabel = 1")
    add("share_patients_ever_positive", df.groupby("Patient_ID").SepsisLabel.max().mean())

    # ---- how the file was imputed (global median, no carry-forward)
    labs = ["Lactate", "Creatinine", "Platelets", "Glucose", "WBC", "Hct", "Hgb", "BUN", "Potassium", "PTT", "Calcium",
            "Magnesium", "Phosphate", "Chloride", "pH", "PaCO2", "AST", "Bilirubin_total", "Fibrinogen", "SaO2",
            "HCO3", "BaseExcess", "FiO2"]
    d = df.sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)
    add("missing_cells_in_model_columns", int(d[feats].isna().sum().sum()), "44 model input columns")
    same_pat = d.Patient_ID == d.Patient_ID.shift(1)
    first_share, run_len, repeat = [], {}, {}
    for c in labs:
        med = d[c].median()
        first_share.append((d.groupby("Patient_ID")[c].first() == med).mean())
    add("labs_share_of_patients_first_row_equals_global_median_mean", np.mean(first_share),
        f"mean over {len(labs)} lab columns")
    for c in ["Platelets", "Creatinine", "Lactate"]:
        med = d[c].median()
        real = d[c] != med
        nxt = (d[c] == d[c].shift(-1)) & (d.Patient_ID == d.Patient_ID.shift(-1))
        add(f"{c}_P_next_hour_equal_given_real_value", nxt[real].mean(),
            "carry-forward would make this large; here it is small")
    add("platelets_share_of_hours_equal_median", (d["Platelets"] == d["Platelets"].median()).mean())

    # ---- splits, test set, positives
    splits = json.load(open(ROOT / "outputs" / "cache" / "patient_splits_seed42.json"))
    ds = build_data(str(csv), feats, splits)
    r_te, y_te = make_samples(ds, 6, "test")
    pats_te = ds["pid"][r_te]
    add("test_patients", len(np.unique(pats_te)))
    add("test_hours", len(r_te))
    add("test_positive_hours_h6", int(y_te.sum()))
    add("test_positive_patients_h6", len(np.unique(pats_te[y_te == 1])), "patients contributing >= 1 positive hour")
    add("test_prevalence_h6", y_te.mean())
    first_row_pos = (ds["onset"] == 0)
    add("patients_excluded_positive_from_first_hour",
        len(np.unique(ds["pid"][first_row_pos])), "no pre-onset hours to predict from")

    # ---- measurement-indicator control: value != train median means "measured this hour"
    r_tr, y_tr = make_samples(ds, 6, "train")
    X = ds["X"]
    med = np.median(X[r_tr], axis=0)
    keep = [i for i, f in enumerate(feats)
            if f not in ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS") and not f.startswith(("qSOFA", "MAP_"))]
    ind = (X != med).astype(np.float32)[:, keep]
    rs, ys = subsample(r_tr, y_tr, 4, 42)
    m = fit_baseline("xgboost", ind[rs], ys, 42)
    p = m.predict_proba(ind[r_te])[:, 1]
    add("indicator_only_xgboost_auroc_h6", roc_auc_score(y_te, p), f"{len(keep)} vital/lab measurement indicators only")
    add("indicator_only_xgboost_auprc_h6", average_precision_score(y_te, p))
    cnt = pd.DataFrame(ind).groupby(ds["pid"]).transform(lambda s: s.rolling(6, min_periods=1).sum()).to_numpy(np.float32)
    m = fit_baseline("xgboost", cnt[rs], ys, 42)
    p = m.predict_proba(cnt[r_te])[:, 1]
    add("measurement_count_6h_xgboost_auroc_h6", roc_auc_score(y_te, p))

    # ---- derived from the results tables
    S = pd.read_csv(RES / "final_results_summary.csv")
    R = pd.read_csv(RES / "final_results.csv")
    R["seed"] = R.seed.astype(str)
    s6 = S[(S.horizon == 6) & (S.model != "time_only")]
    add("models_auroc_min_h6", s6.auroc_mean.min(), "eleven models, mean over seeds")
    add("models_auroc_max_h6", s6.auroc_mean.max())
    prev = R[(R.horizon == 6)].n_test_pos.iloc[0] / R[(R.horizon == 6)].n_test.iloc[0]
    add("auprc_over_chance_min_h6", s6.auprc_mean.min() / prev, "AUPRC divided by prevalence")
    add("auprc_over_chance_max_h6", s6.auprc_mean.max() / prev)
    e6 = R[(R.horizon == 6) & (R.seed == "ensemble") & (R.model != "time_only")]
    add("precision_at_f1_threshold_min_h6", e6.precision.min())
    add("precision_at_f1_threshold_max_h6", e6.precision.max())
    base = prev * (1 - prev)
    skill = (base - R[(R.horizon == 6) & (R.seed == "ensemble")].brier) / base
    add("brier_skill_min_h6", skill.min(), "relative to predicting the base rate")
    add("brier_skill_max_h6", skill.max())

    # ---- cross-machine reproduction (Anjith's rerun vs ours), seed means vs seed SD
    B = pd.read_csv(RES / "reproduction" / "anjith_final_results.csv")
    B["seed"] = B.seed.astype(str)
    for model in ("transformer", "cnn_lstm", "ann", "lstm"):
        for h in (6, 12):
            a = R[(R.model == model) & (R.horizon == h) & (R.seed != "ensemble")].auroc
            b = B[(B.model == model) & (B.horizon == h) & (B.seed != "ensemble")].auroc
            add(f"repro_{model}_h{h}_abs_diff_of_seed_means", abs(a.mean() - b.mean()))
            add(f"repro_{model}_h{h}_seed_sd_ours", a.std())
    exact = ["time_only", "logistic_regression", "decision_tree", "knn", "random_forest", "xgboost", "hist_gradient_boosting", "grud"]
    mm = R.merge(B, on=["model", "horizon", "seed"], suffixes=("_o", "_h"))
    add("repro_max_abs_auroc_diff_exact_models", (mm[mm.model.isin(exact)].auroc_o - mm[mm.model.isin(exact)].auroc_h).abs().max(),
        "time-only, LR, DT, KNN, RF, XGBoost, HistGB, GRU-D, all horizons and seeds")

    # ---- Figure 2 facts
    T = pd.read_csv(RES / "figure2_trajectory.csv")
    thr = json.load(open(ROOT / "outputs" / "deployed" / "protocol_H6" / "calibration.json"))["ensemble_dl"]
    sep, non = T[T.group == "septic"].sort_values("hour"), T[T.group == "non-septic"]
    add("fig2_septic_first_risk", sep.risk.iloc[0])
    add("fig2_septic_last_risk", sep.risk.iloc[-1])
    add("fig2_high_threshold", thr["threshold_high"])
    above = sep[sep.risk >= thr["threshold_high"]]
    add("fig2_hours_before_onset_first_at_or_above_high", -above.hours_to_onset.iloc[0] if len(above) else np.nan,
        "first hour from which the septic patient's risk is at or above High (hours before labelled onset)")
    add("fig2_nonseptic_max_risk", non.risk.max())

    out = pd.DataFrame(rows)
    out.to_csv(RES / "paper_numbers.csv", index=False)
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.max_colwidth", 70):
        print(out.assign(value=out.value.map(lambda v: f"{v:.4g}" if not isinstance(v, str) else v)).to_string(index=False))


if __name__ == "__main__":
    main()
