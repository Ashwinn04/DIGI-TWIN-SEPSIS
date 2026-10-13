#!/usr/bin/env python3
"""Check a fresh clone can reproduce the paper results: imports, library pins, data and split files.

    python scripts/test_setup.py        (exit code 0 = ready)
"""
import importlib, os, sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")      # xgboost/torch libomp clash on macOS (see CLAUDE.md)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
problems = []

for mod in ["xgboost", "torch", "numpy", "pandas", "sklearn", "joblib", "matplotlib",
            "models", "explainability_utils.protocol", "surveillance.run_surveillance", "surveillance.sweep"]:
    try:
        importlib.import_module(mod)
    except Exception as e:
        problems.append(f"import {mod}: {e}")

pins = {"sklearn": "1.7.2", "xgboost": "2."}       # pickles in outputs/ need these (see requirements.txt)
for mod, want in pins.items():
    try:
        have = importlib.import_module(mod).__version__
        if not have.startswith(want):
            problems.append(f"{mod} {have} installed, need {want}x")
    except ImportError:
        pass                                        # already reported above

for f, hint in [("Dataset.csv", "download it, see README 'Reproduce the paper results'"),
                ("outputs/config.json", "feature list used by final_evaluation.py"),
                ("outputs/cache/patient_splits_seed42.json", "shared patient split")]:
    if not (ROOT / f).exists():
        problems.append(f"missing {f} ({hint})")

if (ROOT / "Dataset.csv").exists():
    import pandas as pd
    df = pd.read_csv(ROOT / "Dataset.csv", usecols=["Patient_ID", "SepsisLabel"])
    if df.Patient_ID.nunique() != 14057 or len(df) != 546123:
        problems.append(f"Dataset.csv has {df.Patient_ID.nunique()} patients / {len(df)} rows, expected 14057 / 546123")

print("\n".join(f"FAIL {p}" for p in problems) or "OK: ready to run scripts/final_evaluation.py")
sys.exit(1 if problems else 0)
