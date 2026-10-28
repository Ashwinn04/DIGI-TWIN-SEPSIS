"""Time-varying reproduction number per region from the real CDC FluView ILINet series.

    python scripts/fetch_viral.py
    python -m surveillance.rt --viral-csv data/viral/fluview_hhs_weekly.csv --out outputs/surveillance/rt

Method (weekly data, so a growth-rate estimator rather than a daily renewal model):
  growth rate  r_t from a quasi-Poisson regression of weekly ILI visits (num_ili) on time, with
               log(total visits, num_patients) as offset, over a TRAILING window of --window weeks, so each
               estimate uses only data available at the end of its week (ILINet reporting lag ignored).
  R_t          Wallinga & Lipsitch (2007): for a gamma generation interval with shape k and scale s,
               R = (1 + r s)^k. Influenza serial interval mean 3.6 d, sd 1.6 d (Cowling et al., 2009).
  95% CI       from the t-interval of the slope (dispersion estimated from the window), mapped through
               the same formula.
Caveats: ILI counts all influenza-like visits, not a single pathogen; R_t here measures growth of ILI
visits as a share of all visits, which tracks incidence growth only while care-seeking is stable.
Days are aligned with surveillance.viral_real.daily_series (week i is day 7i+3; an estimate for week i
becomes available on day 7i+6).
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from surveillance.viral_real import DEFAULT_START_EPIWEEK, REGION_MAP, daily_series, detect_onsets

SI_MEAN, SI_SD = 3.6, 1.6


def r_to_R(r_per_day, mean=SI_MEAN, sd=SI_SD):
    k, s = (mean / sd) ** 2, sd ** 2 / mean
    return np.power(np.maximum(1 + np.asarray(r_per_day) * s, 1e-9), k)


def quasi_poisson_slope(y, offset):
    """Poisson GLM log(mu) = a + b t + offset by IRLS; returns slope, its SE (quasi-Poisson) and df."""
    t = np.arange(len(y), dtype=float)
    X = np.c_[np.ones_like(t), t]
    beta = np.array([np.log(max(y.sum(), 1) / np.exp(offset).sum()), 0.0])
    for _ in range(50):
        mu = np.exp(X @ beta + offset)
        z = X @ beta + (y - mu) / mu
        Wt = mu
        new = np.linalg.solve(X.T @ (Wt[:, None] * X), X.T @ (Wt * z))
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    mu = np.exp(X @ beta + offset)
    df = len(y) - 2
    phi = max(float(((y - mu) ** 2 / mu).sum() / df), 1.0)
    cov = phi * np.linalg.inv(X.T @ (mu[:, None] * X))
    return float(beta[1]), float(np.sqrt(cov[1, 1])), df


def estimate(weekly, window=4):
    """weekly: rows of one region sorted by epiweek with num_ili, num_patients. -> per-week R_t table."""
    rows = []
    y_all, n_all = weekly.num_ili.to_numpy(float), weekly.num_patients.to_numpy(float)
    for i in range(window - 1, len(weekly)):
        sl = slice(i - window + 1, i + 1)
        b, se, df = quasi_poisson_slope(y_all[sl], np.log(n_all[sl]))
        tq = stats.t.ppf(0.975, df)
        r, lo, hi = b / 7, (b - tq * se) / 7, (b + tq * se) / 7
        rows.append(dict(epiweek=int(weekly.epiweek.iloc[i]), week_start=str(weekly.week_start.iloc[i])[:10],
                         ili_pct=float(100 * y_all[i] / n_all[i]), growth_per_week=b,
                         R=float(r_to_R(r)), R_low=float(r_to_R(lo)), R_high=float(r_to_R(hi))))
    return pd.DataFrame(rows)


def first_sustained(df, col, after_day=0, weeks=2):
    """Row of the first of `weeks` consecutive weeks with df[col] > 1, among weeks available on/after after_day."""
    ok = ((df[col] > 1) & (df.day_available >= after_day)).to_numpy()
    run = np.convolve(ok, np.ones(weeks, int), "valid") == weeks
    hit = np.flatnonzero(run)
    return None if not len(hit) else int(hit[0])


def selfcheck():
    """Known growth in, known growth out; zero growth gives R = 1."""
    rng = np.random.default_rng(0)
    n = np.full(4, 50000.0)
    y = rng.poisson(n * 0.01 * np.exp(0.2 * np.arange(4))).astype(float)
    b, se, _ = quasi_poisson_slope(y, np.log(n))
    assert abs(b - 0.2) < 3 * se, (b, se)
    assert abs(float(r_to_R(0.0)) - 1) < 1e-12
    assert abs(float(r_to_R(0.2 / 7)) - (1 + 0.2 / 7 * SI_MEAN)) < 0.01      # near 1 + r*Tg for small r
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true", help="run the estimator check and exit")
    ap.add_argument("--viral-csv", default="data/viral/fluview_hhs_weekly.csv")
    ap.add_argument("--out", default="outputs/surveillance/rt")
    ap.add_argument("--start-epiweek", type=int, default=DEFAULT_START_EPIWEEK)
    ap.add_argument("--days", type=int, default=140, help="surveillance window (same as run_surveillance)")
    ap.add_argument("--window", type=int, default=4, help="trailing weeks per growth-rate fit")
    ap.add_argument("--baseline-days", type=int, default=60, help="ignore weeks before this day (detector baseline)")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    w = pd.read_csv(a.viral_csv)
    onsets = detect_onsets(daily_series(a.viral_csv, REGION_MAP, a.days, a.start_epiweek))
    n_weeks = a.days // 7
    tables, summary = [], {}
    for reg, hhs in REGION_MAP.items():
        s = w[w.region == hhs].sort_values("epiweek").reset_index(drop=True)
        i0 = int(np.flatnonzero(s.epiweek.to_numpy() >= a.start_epiweek)[0])
        # include window-1 earlier weeks so the first surveillance week already has an estimate
        part = s.iloc[i0 - a.window + 1: i0 + n_weeks].reset_index(drop=True)
        est = estimate(part, a.window)
        est.insert(0, "region", reg)
        est.insert(1, "hhs_region", hhs)
        est["week_index"] = np.arange(len(est))
        est["day_mid"] = est.week_index * 7 + 3
        est["day_available"] = est.week_index * 7 + 6
        tables.append(est)
        k_pt = first_sustained(est, "R", a.baseline_days)
        k_lo = first_sustained(est, "R_low", a.baseline_days)
        pre = est[est.day_available < a.baseline_days]
        summary[reg] = dict(
            hhs_region=hhs, ili_onset_day=onsets[reg],
            R_above_1_day=None if k_pt is None else int(est.day_available[k_pt]),
            R_low_above_1_day=None if k_lo is None else int(est.day_available[k_lo]),
            R_baseline_range=[round(float(pre.R.min()), 2), round(float(pre.R.max()), 2)],
            R_peak=round(float(est.R.max()), 2), R_peak_week=str(est.week_start[est.R.idxmax()]),
            R_at_onset=None if onsets[reg] is None else round(float(np.interp(onsets[reg], est.day_mid, est.R)), 2))
        for key in ("R_above_1_day", "R_low_above_1_day"):
            v = summary[reg][key]
            summary[reg][key.replace("_day", "_lead_days")] = None if v is None or onsets[reg] is None else onsets[reg] - v
    tab = pd.concat(tables, ignore_index=True)
    tab.to_csv(out / "rt_by_region.csv", index=False)
    json.dump(dict(method="quasi-Poisson growth rate of weekly ILI visits (offset: all visits), trailing window; "
                          "R = (1 + r s)^k, gamma generation interval",
                   serial_interval=dict(mean_days=SI_MEAN, sd_days=SI_SD, source="Cowling et al., Epidemiology 2009"),
                   window_weeks=a.window, start_epiweek=a.start_epiweek, days=a.days,
                   onset_rule="surveillance.viral_real.detect_onsets (>1.3x baseline mean for 7 days)",
                   sustained_rule="first of 2 consecutive weeks above 1 after the baseline; day = when that week's estimate is available",
                   baseline_days=a.baseline_days,
                   regions=summary), open(out / "rt_summary.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.0), sharex=True, sharey=True)
    for ax, reg in zip(axes.flat, REGION_MAP):
        e = tab[tab.region == reg]
        ax.fill_between(e.day_available, e.R_low, e.R_high, color="#b9c9de", lw=0, label="95% CI")
        ax.plot(e.day_available, e.R, "-o", ms=2.5, lw=1, color="#3b6ea5", label="$R_t$")
        ax.axhline(1, color="k", lw=0.6)
        if onsets[reg] is not None:
            ax.axvline(onsets[reg], color="#b2262c", ls="--", lw=0.8, label="ILI rise (outbreak start)")
        ax.set_title(f"{reg} ({REGION_MAP[reg].upper()})")
        ax.set_ylim(0.6, 1.6)
    for ax in axes[1]:
        ax.set_xlabel("day of surveillance window")
    for ax in axes[:, 0]:
        ax.set_ylabel("reproduction number")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"rt_by_region.{ext}", dpi=300)
    plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
