# Paper 514: manuscript and where every number comes from

`514_Manuscript_submission_copy.docx` is the final manuscript (the only version kept in the working tree). It has no
tracked changes, comments or placeholders and is the file `scripts/verify_manuscript.py` checks (71 checks).

Provenance: it was built by `scripts/make_submission_copy.py` from the authors' final-round tracked edits
(`514_Manuscript_authors_edit_2026-10-13.docx`, itself the accepted-changes `514_Manuscript_revised_clean.docx` plus
the authors' edits). Those two input files were removed from the working tree on 14 Oct 2026; they remain in git
history and in the release tag, e.g.
`git show paper514-final:paper/514_Manuscript_authors_edit_2026-10-13.docx > paper/514_Manuscript_authors_edit_2026-10-13.docx`.

The two placeholders in the Data and Code Availability section are filled by `make_submission_copy.py` (repository
link; data provenance and licence, evidence below).

## Data provenance (checked 13 Oct 2026 against physionet.org)

- **Source:** `Dataset.csv` is derived from the public PhysioNet/Computing in Cardiology Challenge 2019 training data.
  `Patient_ID` is the challenge's own patient number (training set A: p000001-p020336; set B numbers follow).
  13,851 of the 14,057 stays have IDs 1-20,336 (set A); 206 have IDs 20,337-20,643. IDs run 1-20,643; 68% of the
  numbers in that range are present.
- **Spot check:** 32 stays (20 from set A, 12 above 20,336) were compared with the public `.psv` records on rows,
  every vital/lab value that is not missing in the raw file, age and `SepsisLabel`. 31 matched exactly (all 20 from
  set A). One stay above 20,336 (ID 20621) differs from the file its number points to and was not located among the
  nearby set B files, so the number-to-file mapping for the 206 high IDs is unconfirmed.
- **Preprocessing visible in the file:** missing hourly values replaced by dataset-wide column medians (no
  carry-forward); five derived columns (6 h rolling mean of MAP, its 1 h change, qSOFA respiratory and MAP flags,
  simplified qSOFA score) that rebuild exactly from raw MAP and Resp.
- **Selection rule: not recoverable.** Retained and dropped patients (40 of each, set A) are alike in stay length,
  completeness and lab coverage (retained stays are somewhat more often septic, 8% vs 3% in this small sample), and
  the shortest retained stay is 8 hours. Whoever prepared the file should state the rule if it is known.
- **Licence:** the challenge project page lists Creative Commons Attribution 4.0 International
  (DOI 10.13026/v64v-d857). Nothing in the text is typed from
memory: each number below is produced by a script in this repository and stored in a file.

Regenerate everything from a clean checkout (needs `Dataset.csv`, not tracked):

    python scripts/final_evaluation.py --horizons 3 6 12 --sanity      # final_results*.csv, sanity_checks.csv (~45 min)
    python scripts/check_label_leakage.py                              # label_checks.csv
    python scripts/export_deployment.py                                # outputs/deployed/protocol_H6 (needed by figures 2, 3)
    python -m surveillance.run_surveillance --data Dataset.csv --risk-source grud \
        --viral-csv data/viral/fluview_hhs_weekly.csv --out outputs/surveillance/leakage_safe/grud_real_viral
    python -m surveillance.sweep --data Dataset.csv --risk-source grud --out outputs/surveillance/leakage_safe/grud_sweep
    python scripts/make_figures_2_3.py ; python scripts/make_figure4.py
    python -m surveillance.sweep --data Dataset.csv --risk-source grud --normalize-viral \
        --out outputs/surveillance/leakage_safe/grud_sweep_normalised   # control for the k = 10 explanation
    python scripts/paper_numbers.py                                    # paper_numbers.csv (everything else)
    python scripts/make_submission_copy.py                             # clean docx -> submission copy (no comments)
    python scripts/verify_manuscript.py                                # every table cell / quoted number vs the files
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
| k = 10: false alarms 0.61 -> 2.69; two thirds of site-days suppressed, a third of region-days without a report, one of three sites reporting; per-site normalisation leaves 2.84 | `sensitivity.csv`, `grud_sweep_normalised/sensitivity.csv`, `paper_numbers.csv` (`share_site_days_below_k10`, `region_days_with_no_report_k10`, `mean_sites_reporting_per_region_day_k10`, `normalised_*`) |
| Viral series: CDC FluView ILINet via Delphi Epidata, accessed 9 Oct 2026, HHS regions 1/3/4/9 from epiweek 202436 | `data/viral/SOURCE.json` (local), `surveillance/README.md` |

Statements about design (Digital Twin architecture, tiered alerts, deployment) are not results and are flagged in
the text as intended or not evaluated.
