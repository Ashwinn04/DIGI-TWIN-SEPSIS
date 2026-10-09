"""End-to-end multi-hospital surveillance experiment.

    python -m surveillance.run_surveillance --data Dataset.csv --out outputs/surveillance

Everything about the network, dates, viral counts and the outbreak is simulated; the
per-patient risk scores come from real out-of-fold model predictions on the ICU data.
Results therefore measure the SURVEILLANCE LAYER (detection delay / false alarms), not
clinical accuracy.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .risk import patient_risk_scores
from .simulate import build_network, assign_patients, inject_outbreak, viral_counts
from .aggregate import site_daily_reports, central_region_series
from .detect import zscore_baseline, ewma_alarm, cusum_alarm, combined_z
from .viral_real import REGION_MAP, DEFAULT_START_EPIWEEK, daily_series, detect_onsets, load_viral

N_DAYS, BASE_DAYS, DURATION = 140, 60, 21


def prepare_real_viral(csv_path, start_epiweek=DEFAULT_START_EPIWEEK):
    """Real viral series settings: file, season start and the onset day detected in each region."""
    onsets = detect_onsets(daily_series(csv_path, REGION_MAP, N_DAYS, start_epiweek), BASE_DAYS)
    return {"path": csv_path, "start_epiweek": start_epiweek, "onsets": onsets}


def run_replicate(pt, sites, seed, k, eps, intensity, multiplier, real=None):
    """real=None: simulated viral counts and random outbreak start (original behaviour).
    real=prepare_real_viral(...): real viral series; outbreak starts where it begins to rise."""
    rng = np.random.default_rng(seed)
    p = assign_patients(pt, sites, N_DAYS, rng)
    region = rng.choice(sorted(sites.region.unique()))
    start = int(rng.integers(75, 100)) if real is None else real["onsets"][region]
    p, window = inject_outbreak(p, region, start, DURATION, intensity, "flag", rng)
    if real is None:
        viral = viral_counts(p, region, window, N_DAYS, sites, multiplier=multiplier, rng=rng)
    else:
        viral = load_viral(real["path"], REGION_MAP, N_DAYS, sites, rng=rng, base_days=BASE_DAYS,
                           start_epiweek=real["start_epiweek"])
    reports = site_daily_reports(p, viral, N_DAYS, k=k, epsilon=eps, rng=rng)
    reg = central_region_series(reports)
    return reports, reg, region, window


def evaluate(reg, reports, region, window, detector, onsets=None, control=False):
    """onsets (real viral data): other regions are only scored for false alarms before their own
    real onset, since their real rise is genuine activity. control=True: no outbreak anywhere,
    every post-baseline day in every region counts toward false alarms."""
    fn = {"EWMA": ewma_alarm, "CUSUM": cusum_alarm}[detector]
    start, end = window
    out, series = {}, {}
    for r in sorted(reg.region.unique()):
        s = reg[reg.region == r].set_index("day").reindex(range(N_DAYS)).ffill().fillna(0)
        za = zscore_baseline(s.alert_rate.values, BASE_DAYS)
        zv = zscore_baseline(s.viral_cases.values, BASE_DAYS)
        series[r] = {"Sepsis-alert rate (pooled)": za, "Viral cases (pooled)": zv,
                     "Combined": zscore_baseline(combined_z(za, zv), BASE_DAYS)}  # re-standardised
    # single-hospital baseline: one site in the outbreak region, alert rate only, no pooling
    site = reports[reports.region == region].site.iloc[0]
    ss = reports[reports.site == site].set_index("day").reindex(range(N_DAYS)).fillna(0)
    zs = zscore_baseline((ss.alerts / ss.admissions.clip(lower=1)).values, BASE_DAYS)
    for name in ["Sepsis-alert rate (pooled)", "Viral cases (pooled)", "Combined", "Single hospital"]:
        det_day, fa, days = None, 0, 0
        for r, sig in series.items():
            z = zs if (name == "Single hospital" and r == region) else sig.get(name)
            if z is None:
                continue
            al = fn(z)
            if r == region:        # in the control this is a placebo window: detection by chance
                hit = np.where(al[start:end + 14])[0]
                det_day = int(hit[0]) if len(hit) else None
            if r == region and not control:
                fa += int(al[BASE_DAYS:start].sum()); days += start - BASE_DAYS
            else:
                quiet = N_DAYS if (control or onsets is None) else onsets.get(r) or N_DAYS
                fa += int(al[BASE_DAYS:quiet].sum()); days += quiet - BASE_DAYS
        out[name] = {"delay": det_day, "false_alarm_days": fa, "monitored_days": days}
    return out, series


def summarize(rows):
    res = {}
    for key in rows[0]:
        d = [r[key]["delay"] for r in rows]
        det = [x for x in d if x is not None]
        fa = sum(r[key]["false_alarm_days"] for r in rows)
        md = sum(r[key]["monitored_days"] for r in rows)
        res[key] = {"detection_rate": len(det) / len(d),
                    "median_delay_days": float(np.median(det)) if det else None,
                    "false_alarms_per_100_region_days": 100 * fa / md}
    return res


def figure(series, region, window, detector, path):
    fn = {"EWMA": ewma_alarm, "CUSUM": cusum_alarm}[detector]
    regions = sorted(series)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 2]})
    z = series[region]["Combined"]
    ax[0].plot(z, color="#1f77b4", lw=1.2, label=f"Combined signal ({region})")
    ax[0].axvspan(*window, color="#d62728", alpha=0.15, label="Simulated outbreak")
    ax[0].axvline(BASE_DAYS, color="grey", ls=":", label="Baseline ends")
    al = np.where(fn(z))[0]
    if len(al):
        ax[0].scatter(al[:1], z[al[:1]], color="#d62728", zorder=3, label=f"First {detector} alarm")
    ax[0].set(xlabel="Day", ylabel="z-score vs baseline", title="Temporal surveillance (pooled hospitals)")
    ax[0].legend(fontsize=8)
    mat = np.array([series[r]["Combined"] for r in regions])
    im = ax[1].imshow(mat, aspect="auto", cmap="coolwarm", vmin=-3, vmax=5)
    ax[1].set_yticks(range(len(regions)), regions)
    ax[1].axvline(window[0], color="k", lw=0.8)
    ax[1].set(xlabel="Day", title="Geographic x temporal heatmap")
    fig.colorbar(im, ax=ax[1], label="z")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Dataset.csv")
    ap.add_argument("--out", default="outputs/surveillance")
    ap.add_argument("--replicates", type=int, default=30)
    ap.add_argument("--k", type=int, default=3, help="min admissions for a site-day to be reported")
    ap.add_argument("--epsilon", type=float, default=1.0, help="Laplace noise budget per count")
    ap.add_argument("--intensity", type=float, default=0.6, help="share of region's alerts moved into outbreak")
    ap.add_argument("--viral-multiplier", type=float, default=3.0)
    ap.add_argument("--viral-csv", default=None,
                    help="real weekly viral series from scripts/fetch_viral.py (default: simulated counts)")
    ap.add_argument("--viral-start", type=int, default=DEFAULT_START_EPIWEEK,
                    help="epiweek (YYYYWW) where the 140-day window starts when --viral-csv is given")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pt, model_metrics = patient_risk_scores(a.data)
    pt["flag"] = pt.max_risk >= pt.max_risk.quantile(0.75)      # fixed 25% alert budget
    sites = build_network()
    real = prepare_real_viral(a.viral_csv, a.viral_start) if a.viral_csv else None
    if real:
        print("Real viral series: detected onset day per region", real["onsets"])
    print("Risk model (out-of-fold):", json.dumps(model_metrics, indent=1))

    results = {}
    for det in ["EWMA", "CUSUM"]:
        rows, fig_done = [], False
        for s in range(a.replicates):
            reports, reg, region, window = run_replicate(
                pt, sites, s, a.k, a.epsilon, a.intensity, a.viral_multiplier, real)
            ev, series = evaluate(reg, reports, region, window, det, real and real["onsets"])
            rows.append(ev)
            if s == 0:
                figure(series, region, window, det, os.path.join(a.out, f"surveillance_{det.lower()}.png"))
        results[det] = summarize(rows)

    out = {"risk_model": model_metrics,
           "config": {"days": N_DAYS, "baseline_days": BASE_DAYS, "outbreak_days": DURATION,
                      "sites": len(sites), "regions": sites.region.nunique(),
                      "replicates": a.replicates, "k_min_cell": a.k, "epsilon": a.epsilon,
                      "outbreak_intensity": a.intensity, "viral_multiplier": a.viral_multiplier,
                      **({"viral_source": "CDC FluView ILINet (real)", "viral_start_epiweek": a.viral_start,
                          "detected_onsets": real["onsets"]} if real else {})},
           "detection": results,
           "disclaimer": ("Hospital network, admission dates and the sepsis-alert surge are simulated; "
                          "the viral series is real CDC FluView data." if real else
                          "Hospital network, admission dates, viral counts and outbreak are simulated.")}
    json.dump(out, open(os.path.join(a.out, "surveillance_results.json"), "w"), indent=2)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
