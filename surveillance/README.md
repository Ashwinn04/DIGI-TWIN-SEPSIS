# Multi-hospital surveillance layer (ICISML Paper 514 revision)

Adds the conference-requested extension: anonymized sepsis-risk and viral-infection signals
from multiple hospitals aggregated for **temporal and geographic** early outbreak warning.

## Run
    python -m surveillance.run_surveillance --data Dataset.csv --out outputs/surveillance
Needs: pandas, numpy, scikit-learn, matplotlib (already in requirements.txt).

## Pipeline
1. `risk.py`     out-of-fold patient risk scores (patient-grouped 3-fold CV, gradient boosting)
2. `simulate.py` 12 virtual hospitals / 4 regions, admission dates, injected outbreak, synthetic viral counts
3. `aggregate.py` sites share only daily counts: small-cell suppression (k) + Laplace noise (epsilon)
4. `detect.py`   EWMA and CUSUM on region-level z-scores (baseline = first 60 days)
5. `run_surveillance.py` 30 replicates; detection rate, delay, false alarms; figures

## What is real vs simulated
Real: ICU data and the out-of-fold risk scores. SIMULATED: hospital/region assignment, admission
dates, the outbreak, and the viral-infection counts (the ICU dataset has no viral data).
Results evaluate the surveillance layer, not clinical accuracy. For real viral data, supply a
DataFrame with columns site, day, viral_cases in place of `simulate.viral_counts`.

## Latest results (30 replicates, outputs/surveillance/surveillance_results.json)
Pooled sepsis-alert rate and combined signal detect 100% of simulated outbreaks with a median
delay of 4 days (EWMA); a single hospital alone: 97% detected, 7-day median delay, ~2x the
false-alarm rate. Out-of-fold risk model on the full dataset: patient-level AUROC 0.77.
