# Paper 514: manuscript and where every number comes from

`514_Manuscript_revised_clean.docx` is the manuscript with all tracked changes accepted (13 Oct 2026). It still
contains the authors' review comments (11); delete them in the submission copy. Nothing in the text is typed from
memory: each number below is produced by a script in this repository and stored in a file.

Regenerate everything from a clean checkout (needs `Dataset.csv`, not tracked):

    python scripts/final_evaluation.py --horizons 3 6 12 --sanity      # final_results*.csv, sanity_checks.csv (~45 min)
    python scripts/check_label_leakage.py                              # label_checks.csv
    python scripts/export_deployment.py                                # outputs/deployed/protocol_H6 (needed by figures 2, 3)
    python -m surveillance.run_surveillance --data Dataset.csv --risk-source grud \
        --viral-csv data/viral/fluview_hhs_weekly.csv --out outputs/surveillance/leakage_safe/grud_real_viral
    python -m surveillance.sweep --data Dataset.csv --risk-source grud --out outputs/surveillance/leakage_safe/grud_sweep
    python scripts/make_figures_2_3.py ; python scripts/make_figure4.py
    python scripts/paper_numbers.py                                    # paper_numbers.csv (everything else)
    python scripts/check_deployment.py                                 # bundle reproduces the evaluation

## Tables and figures

| Manuscript | Source file(s) | Produced by |
|---|---|---|
| Table 1 (literature survey) | none (text) | — |
| Table 2 (6 h performance) | `outputs/results/final_results_summary.csv` (mean ± SD over seeds, AUPRC, sensitivity) and `final_results.csv` (95% CI of the seed-ensemble) | `scripts/final_evaluation.py` |
| Table 3 (outbreak detection) | `outputs/surveillance/leakage_safe/grud_real_viral/surveillance_results.json` | `surveillance.run_surveillance --risk-source grud --viral-csv ...` |
| Figure 1 (architecture) | conceptual drawing, no data | — |
| Figure 2 (risk trajectory) | `outputs/figures/figure2_risk_trajectory.png`, `outputs/results/figure2_trajectory.csv` | `scripts/make_figures_2_3.py` |
| Figure 3 (TreeSHAP, XGBoost) | `outputs/figures/figure3_shap_xgboost.png`, `outputs/results/figure3_shap_mean_abs.csv` | `scripts/make_figures_2_3.py` |
| Figure 4 (surveillance data flow) | `outputs/figures/figure4_dataflow.png` | `scripts/make_figure4.py` |
| Figure 5 (detection example) | `outputs/surveillance/leakage_safe/grud_real_viral/surveillance_ewma.png` | `surveillance.run_surveillance` (seed 0 replicate, East region) |

(The paper's old Figures 5 and 6 were dashboard placeholders and were replaced; Figures 2-4 of the original draft
were removed. The figure files were renamed to the final numbering on 13 Oct 2026.)

## Numbers quoted in the text

| Statement | Source |
|---|---|
| 14,057 stays, 546,123 hourly records, 2.2% of records and 8.8% of patients positive | `paper_numbers.csv` (`dataset_*`, `share_*`) |
| 89.5% of positive records post-onset; label never reverts; previous-label AUROC 0.948 | `label_checks.csv` |
| Test set: 2,092 patients, 80,098 hours, 860 positive (1.1%) from 155 patients; 137 stays excluded | `final_results.csv` (`n_test*`), `paper_numbers.csv` (`test_*`, `patients_excluded_*`) |
| Eleven models, AUROC 0.696-0.787 | `final_results_summary.csv` (`models_auroc_min/max_h6` in `paper_numbers.csv`) |
| AUPRC about four to six times chance; precision 9-11% at the F1 threshold; Brier 2-3% better than the base rate | `paper_numbers.csv` (`auprc_over_chance_*`, `precision_at_f1_*`, `brier_skill_*`) |
| Shuffled labels 0.40-0.41; within 6-hour bins 0.69; length of stay alone 0.673 | `sanity_checks.csv`, `final_results.csv` (`time_only`) |
| 3 h and 12 h results (XGBoost 0.780/0.799, GRU-D 0.793/0.793; CNN-LSTM 0.796 at 3 h) | `final_results_summary.csv` |
| Dataset was median-filled before we received it (no carry-forward) | `paper_numbers.csv` (`labs_share_*`, `*_P_next_hour_*`, `missing_cells_*`) |
| Measurement indicators alone: AUROC 0.58 | `paper_numbers.csv` (`indicator_only_*`) |
| Second-author rerun: exact for LR, DT, KNN, RF, XGBoost, HistGB, GRU-D; Transformer/CNN-LSTM/MLP seed means within one SD at 6 h | `outputs/results/reproduction/`, `paper_numbers.csv` (`repro_*`) |
| 13,920 patients, patient-level AUROC 0.743; outbreak start days 63-81; 30 replicates | `surveillance_results.json` (`risk_model`, `config`) |
| Sensitivity paragraph (ε, k, intensity, viral multiplier, sites, control 0.54, placebo detection 10-17%) | `outputs/surveillance/leakage_safe/grud_sweep/sensitivity.csv` |
| Septic patient risk 0.028 -> 0.069, High threshold 0.032 crossed about 20 h before onset; non-septic maximum 0.030 | `figure2_trajectory.csv`, `paper_numbers.csv` (`fig2_*`) |
| Top SHAP features (ICULOS 0.74, HospAdmTime 0.32, ...) | `figure3_shap_mean_abs.csv` |
| Viral series: CDC FluView ILINet via Delphi Epidata, accessed 9 Oct 2026, HHS regions 1/3/4/9 from epiweek 202436 | `data/viral/SOURCE.json` (local), `surveillance/README.md` |

Statements about design (Digital Twin architecture, tiered alerts, deployment) are not results and are flagged in
the text as intended or not evaluated.
