"""One-at-a-time sensitivity analysis of the surveillance layer.

    python -m surveillance.sweep --data Dataset.csv --out outputs/surveillance --replicates 30

Risk scores are computed once and reused. Each parameter is varied on its own while the others stay
at the defaults below (the run_surveillance defaults). A no-outbreak control gives false alarms
per 100 region-days. All data are simulated except the out-of-fold risk scores.
Writes sensitivity.csv, sensitivity_config.json and one sensitivity_<parameter>.png per parameter.
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .risk import patient_risk_scores
from .simulate import build_network
from .run_surveillance import run_replicate, evaluate, summarize

DETECTORS = ["EWMA", "CUSUM"]
DEFAULTS = {"epsilon": 1.0, "k": 3, "intensity": 0.6, "viral_multiplier": 3.0,
            "sites_per_region": 3, "budget": 0.25}
GRID = {"epsilon": [0.1, 0.5, 1.0, 5.0, None],            # None = no Laplace noise
        "k": [1, 3, 5, 10],
        "intensity": [0.2, 0.4, 0.6, 0.8],
        "viral_multiplier": [1.5, 2.0, 3.0, 5.0],
        "sites_per_region": [1, 2, 3, 5],
        "budget": [0.10, 0.25, 0.40]}                     # share of patients flagged as high risk
LABELS = {"epsilon": "epsilon (Laplace noise budget)", "k": "k (min cell size)",
          "intensity": "outbreak intensity", "viral_multiplier": "viral multiplier",
          "sites_per_region": "sites per region", "budget": "alert budget (share flagged)"}
SIGNALS = ["Sepsis-alert rate (pooled)", "Viral cases (pooled)", "Combined", "Single hospital"]
COLORS = dict(zip(SIGNALS, ["#1f77b4", "#2ca02c", "#d62728", "#7f7f7f"]))

_PT = None


def _init(pt):
    global _PT
    _PT = pt


def run_cell(cfg, replicates, control=False):
    """All replicates for one configuration -> {detector: {signal: metrics}}."""
    pt = _PT.copy()
    pt["flag"] = pt.max_risk >= pt.max_risk.quantile(1 - cfg["budget"])
    sites = build_network(sites_per_region=cfg["sites_per_region"])
    rows = {d: [] for d in DETECTORS}
    for s in range(replicates):
        reports, reg, region, window = run_replicate(
            pt, sites, s, cfg["k"], cfg["epsilon"], cfg["intensity"], cfg["viral_multiplier"])
        for d in DETECTORS:
            rows[d].append(evaluate(reg, reports, region, window, d, control=control)[0])
    return {d: summarize(rows[d]) for d in DETECTORS}


def _tasks():
    tasks = [(p, v, {**DEFAULTS, p: v}, False) for p, vals in GRID.items() for v in vals]
    control = {**DEFAULTS, "intensity": 0.0, "viral_multiplier": 1.0}
    return tasks + [("control", "no outbreak", control, True)]


def _run(task, replicates):
    p, v, cfg, control = task
    return p, v, run_cell(cfg, replicates, control)


def to_frame(results):
    rows = []
    for p, v, res in results:
        for d, sig in res.items():
            for name, m in sig.items():
                rows.append({"parameter": p, "value": "none" if v is None else v, "detector": d,
                             "signal": name, "detection_rate": None if p == "control" else m["detection_rate"],
                             "median_delay_days": m["median_delay_days"],
                             "false_alarms_per_100_region_days": m["false_alarms_per_100_region_days"]})
    return pd.DataFrame(rows)


def figure(df, param, path):
    d = df[df.parameter == param]
    vals = list(dict.fromkeys(d.value))
    x = range(len(vals))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for a, col, title in zip(ax, ["detection_rate", "median_delay_days", "false_alarms_per_100_region_days"],
                             ["Detection rate", "Median delay (days)", "False alarms / 100 region-days"]):
        for sig in SIGNALS:
            for det, ls in zip(DETECTORS, ["-", "--"]):
                y = [d[(d.value == v) & (d.signal == sig) & (d.detector == det)][col].iloc[0] for v in vals]
                a.plot(x, y, ls, marker="o", ms=3, color=COLORS[sig], lw=1.2,
                       label=f"{sig} ({det})" if col == "detection_rate" else None)
        a.set_xticks(list(x), [str(v) for v in vals]); a.set(xlabel=LABELS[param], title=title)
    ax[0].set_ylim(-0.05, 1.05)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Dataset.csv")
    ap.add_argument("--out", default="outputs/surveillance")
    ap.add_argument("--replicates", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pt, metrics = patient_risk_scores(a.data)                  # computed once, reused by every cell
    print("Risk model (out-of-fold) patient AUROC:", round(metrics["patient_level_auroc"], 4))
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(pt,)) as ex:
        futs = [ex.submit(_run, t, a.replicates) for t in _tasks()]
        results = [f.result() for f in futs]
    df = to_frame(results)
    df.to_csv(os.path.join(a.out, "sensitivity.csv"), index=False)
    for p in GRID:
        figure(df, p, os.path.join(a.out, f"sensitivity_{p}.png"))
    json.dump({"replicates": a.replicates, "seeds": f"0..{a.replicates - 1}", "defaults": DEFAULTS,
               "grid": {k: ["none" if v is None else v for v in vs] for k, vs in GRID.items()},
               "control": "intensity=0, viral_multiplier=1; every post-baseline region-day counts toward false alarms",
               "risk_model": metrics}, open(os.path.join(a.out, "sensitivity_config.json"), "w"), indent=2)
    print(f"wrote {len(df)} rows to {os.path.join(a.out, 'sensitivity.csv')}")


if __name__ == "__main__":
    main()
