"""Real weekly respiratory-virus series (CDC FluView ILINet by HHS region) -> daily per-site counts.

Fetch the CSV first with scripts/fetch_viral.py. Only the viral series is real; the hospital
network, admission dates and the sepsis-alert surge stay simulated.
"""
import numpy as np
import pandas as pd

# Our four simulated regions -> four HHS regions (labels are arbitrary; each is a distinct real series)
REGION_MAP = {"North": "hhs1", "South": "hhs4", "East": "hhs3", "West": "hhs9"}
DEFAULT_START_EPIWEEK = 202436      # Sep 2024: lowest baseline drift of the candidate starts, then the 2024-25 season


def daily_series(path, region_map, n_days, start_epiweek=DEFAULT_START_EPIWEEK):
    """Weekly wILI per region, linearly interpolated to daily (weekly value placed mid-week).
    Returns a DataFrame indexed by day (0..n_days-1) with one column per simulated region."""
    w = pd.read_csv(path, parse_dates=["week_start"])
    w = w[w.region.isin(region_map.values()) & (w.epiweek >= start_epiweek)]
    out = {}
    for reg, hhs in region_map.items():
        s = w[w.region == hhs].sort_values("epiweek").head(n_days // 7 + 2)
        if len(s) * 7 < n_days + 4:
            raise ValueError(f"{path} has too few weeks from epiweek {start_epiweek} for {n_days} days")
        x = np.arange(len(s)) * 7 + 3.0
        out[reg] = np.interp(np.arange(n_days), x, s.wili.values)
    return pd.DataFrame(out)


def detect_onsets(daily, base_days=60, rise=1.3, sustain=7):
    """First day after the baseline where the series stays > rise x its baseline mean for
    `sustain` consecutive days (the point the real series 'begins to rise'). Data-driven."""
    onsets = {}
    for reg in daily:
        v = daily[reg].values
        above = v > rise * v[:base_days].mean()
        run = np.convolve(above, np.ones(sustain, int), "valid") == sustain
        hit = np.where(run[base_days:])[0]
        onsets[reg] = int(hit[0]) + base_days if len(hit) else None
    return onsets


def load_viral(path, region_map, n_days, sites, baseline=0.8, *, rng, base_days=60,
               start_epiweek=DEFAULT_START_EPIWEEK):
    """Daily viral positives per site, same schema as simulate.viral_counts (site, day, viral_cases).
    Each region's real shape is rescaled so its pre-surge baseline averages `baseline` per site-day,
    then every site draws Poisson counts from that rate."""
    daily = daily_series(path, region_map, n_days, start_epiweek)
    rate = daily / daily.iloc[:base_days].mean() * baseline
    rows = []
    for _, s in sites.iterrows():
        counts = rng.poisson(rate[s.region].values)
        rows.append(pd.DataFrame({"site": s.site, "day": np.arange(n_days), "viral_cases": counts}))
    return pd.concat(rows, ignore_index=True)
