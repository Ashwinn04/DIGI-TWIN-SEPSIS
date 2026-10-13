"""Privacy-preserving site -> central aggregation.

Each site shares ONLY daily counts (admissions, alerts, viral positives):
  * small-cell suppression (days with < k admissions are not reported)
  * Laplace noise (epsilon-DP on counts, sensitivity 1)
No patient-level data leaves a site.
"""
import numpy as np
import pandas as pd


def site_daily_reports(p, viral, n_days, k=3, epsilon=1.0, rng=None):
    g = p.groupby(["site", "region", "day"]).agg(
        admissions=("Patient_ID", "size"), alerts=("flag", "sum")).reset_index()
    full = (p[["site", "region"]].drop_duplicates().assign(key=1)
            .merge(pd.DataFrame({"day": range(n_days), "key": 1})).drop(columns="key"))
    g = full.merge(g, how="left", on=["site", "region", "day"]).fillna(0)
    g = g.merge(viral, on=["site", "day"], how="left")
    for c in ["admissions", "alerts", "viral_cases"]:
        noise = 0.0 if epsilon is None else rng.laplace(0, 1.0 / epsilon, len(g))   # None = no DP noise
        g[c] = np.clip(np.rint(g[c] + noise), 0, None)
    g["reported"] = g.admissions >= k
    return g


def central_region_series(reports, normalize_viral=False):
    """Sum the reported site-days per region and day.

    normalize_viral=False (default): viral_cases is the SUM over reporting sites, so it moves with the number of
    sites that pass small-cell suppression (this inflates false alarms at large k).
    normalize_viral=True: viral_cases is the MEAN per reporting site, which removes that dependence.
    """
    r = reports[reports.reported].groupby(["region", "day"]).agg(
        admissions=("admissions", "sum"), alerts=("alerts", "sum"),
        viral_cases=("viral_cases", "sum"), sites_reporting=("site", "nunique")).reset_index()
    if normalize_viral:
        r["viral_cases"] = r.viral_cases / r.sites_reporting.clip(lower=1)
    r["alert_rate"] = r.alerts / r.admissions.clip(lower=1)
    return r
