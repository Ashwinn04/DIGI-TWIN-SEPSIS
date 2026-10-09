"""Download the weekly CDC FluView ILINet series (by HHS region) into data/viral/.

    python scripts/fetch_viral.py [--start 201740] [--end 202540]

Source: CDC FluView / ILINet, served unmodified by the CMU Delphi Epidata API
(https://cmu-delphi.github.io/delphi-epidata/api/fluview.html, licence: publicly accessible US Government data).
Output columns: region, epiweek, week_start, wili, ili, num_ili, num_patients.
If the API is blocked, download the same fields manually (see README) and save to the output path.
"""
import argparse, csv, datetime as dt, json
import urllib.request
from pathlib import Path

API = "https://api.delphi.cmu.edu/epidata/fluview/"
REGIONS = ["hhs1", "hhs2", "hhs3", "hhs4", "hhs5", "hhs6", "hhs7", "hhs8", "hhs9", "hhs10"]
ROOT = Path(__file__).resolve().parent.parent


def mmwr_week_start(epiweek):
    """Sunday starting MMWR week `epiweek` (YYYYWW): week 1 is the first week with >= 4 days in the year."""
    year, week = divmod(int(epiweek), 100)
    jan1 = dt.date(year, 1, 1)
    first_sunday = jan1 - dt.timedelta(days=(jan1.weekday() + 1) % 7)
    if (jan1 - first_sunday).days >= 4:
        first_sunday += dt.timedelta(days=7)
    return first_sunday + dt.timedelta(weeks=week - 1)


def fetch(start, end):
    url = f"{API}?regions={','.join(REGIONS)}&epiweeks={start}-{end}"
    with urllib.request.urlopen(url, timeout=120) as r:
        payload = json.load(r)
    if payload.get("result") != 1:
        raise RuntimeError(f"API returned: {payload.get('message', payload)}")
    return url, payload["epidata"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="201740")
    ap.add_argument("--end", default="202540")
    ap.add_argument("--out", default=str(ROOT / "data" / "viral" / "fluview_hhs_weekly.csv"))
    a = ap.parse_args()
    url, rows = fetch(a.start, a.end)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "epiweek", "week_start", "wili", "ili", "num_ili", "num_patients"])
        for r in sorted(rows, key=lambda r: (r["region"], r["epiweek"])):
            w.writerow([r["region"], r["epiweek"], mmwr_week_start(r["epiweek"]),
                        r["wili"], r["ili"], r["num_ili"], r["num_patients"]])
    meta = {"url": url, "accessed": dt.date.today().isoformat(), "rows": len(rows),
            "source": "CDC FluView ILINet via CMU Delphi Epidata API"}
    (out.parent / "SOURCE.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(rows)} rows to {out}\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
