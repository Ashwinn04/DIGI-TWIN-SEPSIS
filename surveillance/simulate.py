"""Virtual hospital network + synthetic outbreak injection (all SIMULATED)."""
import numpy as np
import pandas as pd

REGIONS = ["North", "South", "East", "West"]


def build_network(n_regions=4, sites_per_region=3):
    sites = []
    for r in range(n_regions):
        for s in range(sites_per_region):
            sites.append({"site": f"{REGIONS[r][0]}{s + 1}", "region": REGIONS[r]})
    return pd.DataFrame(sites)


def assign_patients(patients, sites, n_days, rng):
    """Randomly assign each patient to a site and an admission day (simulated)."""
    p = patients.copy()
    idx = rng.integers(0, len(sites), len(p))
    p["site"] = sites.site.values[idx]
    p["region"] = sites.region.values[idx]
    p["day"] = rng.integers(0, n_days, len(p))
    return p


def inject_outbreak(p, region, start, duration, intensity, flag_col, rng):
    """Pull a share of the region's flagged (high-risk) admissions into the outbreak window
    with a linear ramp, mimicking an infection surge. Returns modified copy + true window."""
    q = p.copy()
    cand = q.index[(q.region == region) & q[flag_col]]
    move = rng.choice(cand, size=int(intensity * len(cand)), replace=False)
    ramp = np.sqrt(rng.random(len(move)))          # more cases late in the window
    q.loc[move, "day"] = start + np.floor(ramp * duration).astype(int)
    return q, (start, start + duration)


def viral_counts(p, region, window, n_days, sites, baseline=0.8, multiplier=3.0, rng=None):
    """Synthetic daily viral-infection (e.g. influenza-like) positives per site.
    Poisson baseline; multiplied inside the outbreak window for the outbreak region.
    Replace with real lab-confirmed viral counts when available (same schema)."""
    rows = []
    for _, s in sites.iterrows():
        lam = np.full(n_days, baseline)
        if s.region == region:
            a, b = window
            ramp = np.clip((np.arange(n_days) - a) / max(b - a, 1), 0, 1)
            lam = lam * np.where((np.arange(n_days) >= a) & (np.arange(n_days) < b),
                                 1 + (multiplier - 1) * ramp, 1.0)
        for d in range(n_days):
            rows.append({"site": s.site, "day": d, "viral_cases": rng.poisson(lam[d])})
    return pd.DataFrame(rows)
