#!/usr/bin/env python3
"""Step 8 cross-check: every table cell and quoted number in the manuscript against the result files.

    python scripts/verify_manuscript.py [paper/514_Manuscript_submission_copy.docx]

Exit status is non-zero if any check fails. Expected values are computed from the CSV/JSON files named in
paper/README.md, formatted the way the text prints them, and looked up in the manuscript.
"""
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"
SURV = ROOT / "outputs" / "surveillance" / "leakage_safe"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "514_Manuscript_submission_copy.docx"

root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
paras = ["".join((e.text or "") for e in p.iter() if e.tag == W + "t") for p in root.iter(W + "p")]
text = re.sub(r"\s+", " ", " ".join(paras))
tables = [[["".join((e.text or "") for e in c.iter() if e.tag == W + "t").strip() for c in r.iter(W + "tc")]
           for r in t.iter(W + "tr")] for t in root.iter(W + "tbl")]

results = []


def check(desc, ok, detail=""):
    results.append((bool(ok), desc, detail))


def has(desc, *needles):
    miss = [n for n in needles if n not in text]
    check(desc, not miss, f"missing: {miss}" if miss else "")


f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"

# ---------------------------------------------------------------- Table 2
S = pd.read_csv(RES / "final_results_summary.csv")
R = pd.read_csv(RES / "final_results.csv")
R["seed"] = R.seed.astype(str)
names = {"GRU-D": "grud", "XGBoost": "xgboost", "CNN–LSTM": "cnn_lstm", "HistGradientBoosting": "hist_gradient_boosting",
         "Random Forest": "random_forest", "Transformer": "transformer", "LSTM": "lstm", "Decision Tree": "decision_tree",
         "Logistic Regression": "logistic_regression", "Neural network (MLP)": "ann",
         "K-nearest neighbours": "knn", "ICU length of stay only": "time_only"}
t2 = next(t for t in tables if t and t[0][:3] == ["Model", "Seeds", "AUROC"])
for row in t2[1:]:
    m = names[row[0]]
    s = S[(S.model == m) & (S.horizon == 6)].iloc[0]
    single = int(s.n_seeds) == 1
    ci = R[(R.model == m) & (R.horizon == 6) & (R.seed == ("42" if single else "ensemble"))].iloc[0]
    auroc = f3(s.auroc_mean) if single else f"{s.auroc_mean:.3f} ± {s.auroc_std:.3f}"
    exp = [str(int(s.n_seeds)), auroc, f"{ci.auroc_ci_low:.3f}–{ci.auroc_ci_high:.3f}", f3(s.auprc_mean), f2(s.sens_at_80_spec_mean)]
    check(f"Table 2 row {row[0]}", row[1:6] == exp, f"table {row[1:6]} vs files {exp}")

# ---------------------------------------------------------------- Table 3
J = json.load(open(SURV / "grud_real_viral" / "surveillance_results.json"))
sig = {"Pooled high-risk alerts": "Sepsis-alert rate (pooled)", "Pooled viral cases": "Viral cases (pooled)",
       "Combined": "Combined", "Single hospital": "Single hospital"}
fmt = lambda m: f"{m['detection_rate']:.2f} / {m['median_delay_days']:g} / {m['false_alarms_per_100_region_days']:.2f}"
t3 = next(t for t in tables if t and t[0] == ["Signal", "EWMA", "CUSUM"])
for row in t3[1:]:
    d = J["detection"]
    check(f"Table 3 row {row[0]}", row[1:] == [fmt(d["EWMA"][sig[row[0]]]), fmt(d["CUSUM"][sig[row[0]]])],
          f"table {row[1:]}")

# ---------------------------------------------------------------- dataset, labels, test set
P = pd.read_csv(RES / "paper_numbers.csv").set_index("key").value
L = pd.read_csv(RES / "label_checks.csv").set_index(["check", "column"]).value
has("14,057 ICU stays / 546,123 records", f"{int(P['dataset_icu_stays']):,} ICU stays", f"{int(P['dataset_hourly_records']):,} hourly records")
has("2.2% records, 8.8% patients positive", f"{P['share_records_positive_label']*100:.1f}% of hourly records",
    f"{P['share_patients_ever_positive']*100:.1f}% of patients")
has("89.5% post-onset, copy-label AUROC 0.948", f"{L[('post_onset_share_of_positive_rows','SepsisLabel')]*100:.1f}%",
    f"{L[('copy_previous_label_row_auroc','SepsisLabel')]:.3f}")
r6 = R[(R.horizon == 6)].iloc[0]
has("test set counts", f"{int(r6.n_test_patients):,} patients", f"{int(r6.n_test):,} pre-onset hours",
    f"{int(r6.n_test_pos)} ({r6.n_test_pos / r6.n_test * 100:.1f}%)", f"{int(P['test_positive_patients_h6'])} patients contribute")
has("AUROC range of eleven models", f"{f3(P['models_auroc_min_h6'])} and {f3(P['models_auroc_max_h6'])}")
has("time-only AUROC", f3(S[(S.model == 'time_only') & (S.horizon == 6)].auroc_mean.iloc[0]))
has("AUPRC multiple of chance", "four to six times")
check("AUPRC multiple within 3.5-6.5", 3.5 <= P["auprc_over_chance_min_h6"] and P["auprc_over_chance_max_h6"] <= 6.5)
has("precision 9-11%", "9–11%")
check("precision range rounds to 9-11%", round(P["precision_at_f1_threshold_min_h6"] * 100) == 9 and round(P["precision_at_f1_threshold_max_h6"] * 100) == 11)
has("Brier 2-3%", "2–3% better")
check("Brier skill within 2-3%", 0.02 <= P["brier_skill_min_h6"] and P["brier_skill_max_h6"] <= 0.03)
sc = pd.read_csv(RES / "sanity_checks.csv")
s6 = sc[sc.horizon == 6].set_index("check").auroc
has("shuffled-label AUROCs", f"{s6['xgboost_shuffled_labels_plain']:.2f}–{s6['logistic_regression_shuffled_labels_plain']:.2f}",
    f"{s6['xgboost_shuffled_labels_within_6h_bins']:.2f} when labels")
has("indicator-only AUROC 0.58", f2(P["indicator_only_xgboost_auroc_h6"]))

# ---------------------------------------------------------------- horizons
g = lambda m, h: S[(S.model == m) & (S.horizon == h)].auroc_mean.iloc[0]
has("XGBoost / GRU-D at 3, 6, 12 h", f"XGBoost {f3(g('xgboost',3))}, {f3(g('xgboost',6))}, {f3(g('xgboost',12))}",
    f"GRU-D {f3(g('grud',3))}, {f3(g('grud',6))}, {f3(g('grud',12))}")
check("CNN-LSTM 0.796 at 3 h", re.search(rf"CNN[-–]LSTM {f3(g('cnn_lstm',3))}", text) is not None)
check("XGBoost 0.799 at 12 h (leader)", f"XGBoost {f3(g('xgboost',12))} at 12 hours" in text)

# ---------------------------------------------------------------- surveillance setup and sensitivity
rm = J["risk_model"]
has("13,920 patients, AUROC 0.743", f"{rm['n_patients']:,} patients", f"AUROC of {rm['patient_level_auroc']:.3f}")
on = sorted(J["config"]["detected_onsets"].values())
has("outbreak start days", f"days {on[0]} to {on[-1]}")
Jsim = json.load(open(SURV / "grud" / "surveillance_results.json"))["detection"]["EWMA"]
has("simulated-viral comparison",
    f"{Jsim['Combined']['median_delay_days']:g}-day delay",
    f"{Jsim['Single hospital']['detection_rate']*100:.0f}% detected, {Jsim['Single hospital']['median_delay_days']:g}-day delay")
Z = pd.read_csv(SURV / "grud_sweep" / "sensitivity.csv")
def z(p, v, sg="Combined", det="EWMA"):
    return Z[(Z.parameter == p) & (Z.value.astype(str) == str(v)) & (Z.signal == sg) & (Z.detector == det)].iloc[0]
has("eps 0.1: 60%, 17.5 days", f"{z('epsilon','0.1').detection_rate*100:.0f}% of outbreaks", f"{z('epsilon','0.1').median_delay_days:g}-day delay")
has("delay 8 -> 3 days", f"from {z('epsilon','0.5').median_delay_days:g} days to {z('epsilon','none').median_delay_days:g} days")
has("k=10 false alarms", f"from {z('k','3').false_alarms_per_100_region_days:.2f} to {z('k','10').false_alarms_per_100_region_days:.2f}")
has("k=10 mechanism (sparse reporting), normalised control",
    "suppresses two thirds of site-days", "no report on a third of days", "only one of its three sites",
    f"({P['normalised_viral_k10_false_alarms_combined']:.2f})")
check("k=10 suppression numbers back the wording",
      0.60 <= P["share_site_days_below_k10"] <= 0.75 and 0.28 <= P["region_days_with_no_report_k10"] <= 0.38
      and 0.8 <= P["mean_sites_reporting_per_region_day_k10"] <= 1.3 and P["share_site_days_below_k3"] < 0.05,
      f"below_k10={P['share_site_days_below_k10']:.2f}, no_report={P['region_days_with_no_report_k10']:.2f}, "
      f"sites={P['mean_sites_reporting_per_region_day_k10']:.2f}")
check("normalised viral count does not remove the k=10 jump",
      P["normalised_viral_k10_false_alarms_combined"] > 2 * z("k", "3").false_alarms_per_100_region_days)
has("intensity", f"{z('intensity','0.2').median_delay_days:g} days at intensity 0.2", f"{z('intensity','0.8').median_delay_days:g} days at 0.8")
vm = lambda v: z("viral_multiplier", v, "Viral cases (pooled)").detection_rate * 100
has("viral multiplier", f"{vm('1.5'):.0f}% detected at a 1.5-fold surge", f"{vm('5.0'):.0f}% at 5-fold")
ss = lambda v: z("sites_per_region", v, "Viral cases (pooled)").detection_rate * 100
has("sites per region", f"{ss('1'):.0f}% detected with one site per region", f"{ss('5'):.0f}% with five")
ctl = lambda det: Z[(Z.parameter == "control") & (Z.signal == "Combined") & (Z.detector == det)].iloc[0]
has("no-outbreak control", f"{ctl('EWMA').false_alarms_per_100_region_days:.2f} false alarms per 100 region-days",
    f"{ctl('CUSUM').false_alarms_per_100_region_days:.2f} with CUSUM", f"{ctl('EWMA').detection_rate*100:.0f}% (EWMA) to {ctl('CUSUM').detection_rate*100:.0f}% (CUSUM)")

# ---------------------------------------------------------------- figures 2 and 3
F2 = pd.read_csv(RES / "figure2_trajectory.csv")
sep = F2[F2.group == "septic"].sort_values("hour")
thr = json.load(open(ROOT / "outputs" / "deployed" / "protocol_H6" / "calibration.json"))["ensemble_dl"]["threshold_high"]
has("Figure 2 numbers", f"{sep.risk.iloc[0]:.3f} to {sep.risk.iloc[-1]:.3f}", f"({thr:.3f})",
    f"peaked at {F2[F2.group=='non-septic'].risk.max():.3f}")
F3 = pd.read_csv(RES / "figure3_shap_mean_abs.csv").sort_values("mean_abs_shap", ascending=False)
top = F3.set_index("feature").mean_abs_shap
has("Figure 3 SHAP values", f"mean |SHAP| {top['ICULOS']:.2f}", f"({top['HospAdmTime']:.2f})", f"({top['Unit1']:.2f})",
    f"age ({top['Age']:.2f})", f"({top['MAP_rolling_mean_6hr']:.2f})", f"({top['HR']:.2f})")
check("Figure 3 top features as listed", list(F3.feature.head(7)) == ["ICULOS", "HospAdmTime", "Unit1", "Age", "MAP_rolling_mean_6hr", "HR", "Resp"],
      str(list(F3.feature.head(7))))
check("lactate and WBC outside the SHAP top ten", not {"Lactate", "WBC"} & set(F3.feature.head(10)))

# ---------------------------------------------------------------- reproduction sentence
ok = all(P[f"repro_{m}_h6_abs_diff_of_seed_means"] < P[f"repro_{m}_h6_seed_sd_ours"] for m in ("transformer", "cnn_lstm", "ann"))
check("reproduction: Transformer/CNN-LSTM/MLP seed means within one SD at 6 h", ok)
check("reproduction: eight models exact (max diff < 1e-6)", P["repro_max_abs_auroc_diff_exact_models"] < 1e-6)

# ---------------------------------------------------------------- stale claims must be gone
# the persisted-label figures (0.948, "near 0.95") are legitimate and are removed before the scan
scan = re.sub(r"AUROC of 0\.948|AUROC near 0\.95", "", text)
for pat, why in [(r"0\.9[2-6]\d?\b", "old 0.92-0.96 results"), (r"placeholder", "placeholder text"), (r"\babout 100\b", "'about 100' positives"),
                 (r"formal\s+differential-privacy", "'formal DP' claim"), (r"explains risk in real time", "real-time explanation claim"),
                 (r"Real-time\s+EHR", "real-time EHR data"), (r"Heart rate, lactate, MAP trends", "fake SHAP caption")]:
    check(f"absent: {why}", not re.search(pat, scan), "")
fig_caps = re.findall(r"FIGURE\s*(\d+)\.", " ".join(paras))
check("absent: untested cause of the k=10 effect", "varies with the number of sites that pass suppression" not in text)
check("figure captions numbered 1..5 in order", fig_caps == ["1", "2", "3", "4", "5"], str(fig_caps))

bad = [r for r in results if not r[0]]
for ok, desc, detail in results:
    print(("PASS  " if ok else "FAIL  ") + desc + ("" if ok or not detail else f"   [{detail}]"))
print(f"\n{len(results) - len(bad)}/{len(results)} checks passed for {path.name}")
sys.exit(1 if bad else 0)
