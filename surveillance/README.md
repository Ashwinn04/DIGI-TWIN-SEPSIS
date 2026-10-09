# Multi-hospital surveillance layer (ICISML Paper 514 revision)

Adds the conference-requested extension: anonymized sepsis-risk and viral-infection signals
from multiple hospitals aggregated for **temporal and geographic** early outbreak warning.

## Run
    python -m surveillance.run_surveillance --data Dataset.csv --out outputs/surveillance
    # with the real viral series (fetch it first):
    python scripts/fetch_viral.py
    python -m surveillance.run_surveillance --data Dataset.csv --out outputs/surveillance/real_viral \
        --viral-csv data/viral/fluview_hhs_weekly.csv
    # one-at-a-time sensitivity analysis (30 replicates per setting, both detectors):
    python -m surveillance.sweep --data Dataset.csv --out outputs/surveillance
Needs: pandas, numpy, scikit-learn, matplotlib (already in requirements.txt).

## Pipeline
1. `risk.py`     out-of-fold patient risk scores (patient-grouped 3-fold CV, gradient boosting)
2. `simulate.py` 12 virtual hospitals / 4 regions, admission dates, injected outbreak, synthetic viral counts
   `viral_real.py` optional replacement for the synthetic viral counts: real CDC FluView series (see below)
3. `aggregate.py` sites share only daily counts: small-cell suppression (k) + Laplace noise (epsilon; `None` = no noise)
4. `detect.py`   EWMA and CUSUM on region-level z-scores (baseline = first 60 days)
5. `run_surveillance.py` 30 replicates; detection rate, delay, false alarms; figures
6. `sweep.py`    sensitivity analysis over epsilon, k, outbreak intensity, viral multiplier, sites per region,
   alert budget, plus a no-outbreak control (false alarms per 100 region-days)

## What is real vs simulated
- **Real:** the ICU data and its out-of-fold risk scores; with `--viral-csv`, also the viral series.
- **SIMULATED:** hospital/region assignment, admission dates and the sepsis-alert surge (the outbreak
  injected into the ICU admissions). Without `--viral-csv` the viral counts are simulated too.
- The real viral series is weekly CDC FluView ILINet weighted influenza-like-illness (wILI) for HHS
  regions hhs1/hhs4/hhs3/hhs9, used as North/South/East/West (labels are arbitrary; each region is a distinct
  real series). It is interpolated to daily, rescaled so the pre-surge baseline averages 0.8 viral cases per
  site per day, then drawn as Poisson counts per site. wILI is a percentage of visits, so only its **shape** is
  real; absolute counts are not. Window: 140 days from epiweek 202436 (Sep 2024). The outbreak
  start is detected from the data (first day after the baseline where the series stays >1.3x its baseline mean
  for 7 days): North 81, South 68, East 79, West 63.
- The ICU dataset has no dates or locations, so the sepsis surge and the real viral rise are aligned by
  construction, not observed together. Results evaluate the surveillance layer, not clinical accuracy.

### Viral data source
CDC FluView / ILINet, retrieved through the CMU Delphi Epidata API (`fluview` endpoint),
<https://cmu-delphi.github.io/delphi-epidata/api/fluview.html>. Request used:
`https://api.delphi.cmu.edu/epidata/fluview/?regions=hhs1,hhs2,hhs3,hhs4,hhs5,hhs6,hhs7,hhs8,hhs9,hhs10&epiweeks=201740-202540`. Accessed 2026-10-09. Licence: the endpoint documentation lists FluView as
"Publicly Accessible US Government" data. Raw downloads are not committed (`data/viral/` is gitignored); rerun
`scripts/fetch_viral.py`. Cite the CDC FluView ILINet system and the Delphi Epidata API.
If the API is unreachable, save the same fields (region, epiweek, week_start, wili, ili, num_ili, num_patients)
for HHS regions 1, 3, 4, 9 and epiweeks 202436 onward as `data/viral/fluview_hhs_weekly.csv`.

## Latest results (30 replicates, outputs/surveillance/surveillance_results.json)
Pooled sepsis-alert rate and combined signal detect 100% of simulated outbreaks with a median
delay of 4 days (EWMA); a single hospital alone: 97% detected, 7-day median delay, ~2x the
false-alarm rate. Out-of-fold risk model on the full dataset: patient-level AUROC 0.77.

### With the real viral series (outputs/surveillance/real_viral/)
detection rate / median delay (days) / false alarms per 100 region-days:

| Signal | EWMA | CUSUM |
|---|---|---|
| Sepsis-alert rate (pooled) | 1.00 / 4 / 1.76 | 1.00 / 3.5 / 1.37 |
| Viral cases (pooled) | 0.83 / 17 / 2.29 | 0.87 / 14.5 / 2.09 |
| Combined | 1.00 / 4 / 2.16 | 1.00 / 3.5 / 1.50 |
| Single hospital | 1.00 / 8.5 / 1.15 | 1.00 / 7 / 0.58 |

Caveats: real wILI drifts upward through autumn, so a 60-day baseline is not flat and the detectors
alarm on that drift (false alarms are higher than with simulated counts). In this mode the other
regions are scored for false alarms only before their own real onset (a few days to ~3 weeks after the
baseline), so false-alarm rates are **not comparable** with the simulated-viral run.

## Sensitivity analysis (outputs/surveillance/sensitivity.csv, one figure per parameter)
One parameter varied at a time (others at the defaults), 30 replicates, seeds 0-29 shared across settings.
Combined signal, detection rate / median delay (days) / false alarms per 100 region-days:

| Parameter | Value | EWMA | CUSUM |
|---|---|---|---|
| epsilon | 0.1 | 0.40 / 13 / 2.54 | 0.50 / 13 / 1.71 |
| epsilon | 0.5 | 1.00 / 7.5 / 0.74 | 1.00 / 6.5 / 0.70 |
| epsilon | 1.0 (default) | 1.00 / 4 / 0.49 | 1.00 / 4 / 0.47 |
| epsilon | 5.0 | 1.00 / 2.5 / 0.57 | 1.00 / 2 / 0.66 |
| epsilon | none | 1.00 / 2.5 / 0.49 | 1.00 / 2 / 0.65 |
| k | 1 | 1.00 / 4 / 0.41 | 1.00 / 4 / 0.51 |
| k | 3 (default) | 1.00 / 4 / 0.49 | 1.00 / 4 / 0.47 |
| k | 5 | 1.00 / 4 / 0.44 | 1.00 / 4 / 0.64 |
| k | 10 | 1.00 / 4 / 2.52 | 1.00 / 4 / 1.95 |
| intensity | 0.2 | 1.00 / 8.5 / 1.00 | 1.00 / 8 / 1.02 |
| intensity | 0.4 | 1.00 / 5 / 0.69 | 1.00 / 5 / 0.80 |
| intensity | 0.6 (default) | 1.00 / 4 / 0.49 | 1.00 / 4 / 0.47 |
| intensity | 0.8 | 1.00 / 4 / 0.65 | 1.00 / 3.5 / 0.71 |
| viral multiplier | 1.5 | 1.00 / 4 / 0.72 | 1.00 / 4 / 0.91 |
| viral multiplier | 2.0 | 1.00 / 4 / 0.66 | 1.00 / 3 / 0.77 |
| viral multiplier | 3.0 (default) | 1.00 / 4 / 0.49 | 1.00 / 4 / 0.47 |
| viral multiplier | 5.0 | 1.00 / 4 / 0.79 | 1.00 / 4 / 0.67 |
| sites/region | 1 | 1.00 / 4 / 0.85 | 1.00 / 3.5 / 0.92 |
| sites/region | 2 | 1.00 / 4 / 0.49 | 1.00 / 4 / 0.67 |
| sites/region | 3 (default) | 1.00 / 4 / 0.49 | 1.00 / 4 / 0.47 |
| sites/region | 5 | 1.00 / 5 / 0.71 | 1.00 / 4 / 0.81 |
| alert budget | 0.1 | 1.00 / 6 / 0.71 | 1.00 / 6 / 0.75 |
| alert budget | 0.25 (default) | 1.00 / 4 / 0.49 | 1.00 / 4 / 0.47 |
| alert budget | 0.4 | 1.00 / 4.5 / 0.81 | 1.00 / 4 / 0.82 |
| no-outbreak control | placebo | 0.17 / - / 0.82 | 0.27 / - / 0.89 |

Read with these caveats (each backed by `sensitivity_checks.csv`):
- The Combined detection rate is 1.00 in almost every row because the simulated sepsis surge dominates that signal; the
  viral-only and single-hospital rows in `sensitivity.csv` show where the settings actually matter.
- **Detection has a chance floor.** An alarm anywhere in the 35-day post-onset window counts as a detection, and the
  control (no outbreak, placebo window) already "detects" in the placebo row above. Weak settings (e.g. viral multiplier
  1.5, epsilon 0.1) sit close to that floor.
- **Small-cell suppression adds its own false alarms and leaks the sepsis surge into the viral signal.** The pooled viral
  count sums over whichever sites report, so it moves with the number of reporting sites (`k_no_outbreak`:
  false alarms rise at k=10 with no outbreak at all), and an admissions surge makes more sites pass k
  (`k_viral_only` vs the main sweep: viral detection at k=10 is lower without the sepsis surge).
- **More hospitals does not automatically reduce false alarms here.** The ICU patient pool is fixed, so more sites
  means fewer patients per site and more summed Laplace noise; the pooled sepsis series is the same patients
  regardless of site count (`sites_per_region_no_noise`). Only the viral-based signals gain from extra sites.
- Single-hospital false-alarm rates rest on few monitored days per run and are noisy and non-monotonic.
