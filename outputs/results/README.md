# outputs/results

Numbers for the paper come from these files only (written by `scripts/final_evaluation.py`):

| File | Content |
|---|---|
| `final_results.csv` | one row per model x horizon x seed: AUROC/AUPRC with 1000-resample patient-level bootstrap CIs, sens@80% spec, Brier, F1-point metrics |
| `final_results_summary.csv` | mean and std over seeds 42/43/44 |
| `sanity_checks.csv` | shuffled-label (should be ~0.5) and time-only controls |

Older files, different protocol, **do not quote**:

- `model_comparison.csv`, `test_set_model_comparison.csv`: 23-patient / old-label runs. Kept because the
  legacy dashboard path (`SEPSIS_MODEL_SET=legacy`) and `integration_real.py` ensemble weights read them.
- `literature_comparison.md`: published results from other papers, for the related-work section.
- `archive/`: `evaluation_summary.md` (23-patient run, chance-level AUROC) and the Oct 2025 GRU-D training logs.

`reproduction/`: Anjith's independent rerun of `final_evaluation.py` on another machine (plan Step 3
"second person reruns it"). Comparison in `docs/HANDOFF_ANJITH.md` section 7.
