"""Run the models trained under the final evaluation protocol on new patient data.

A bundle (built by scripts/export_deployment.py) holds, per model: the checkpoints, the fitted imputer and
scaler, a Platt calibrator and operating-point thresholds. This module reproduces the evaluation path
exactly (same features, median imputation, scaling, 24-hour zero-padded window with an observed-value
mask), so a prediction here equals the one in outputs/results/final_results.csv for the same input.

Risk levels are defined by the validation operating points, not by fixed 0.3/0.7 cut-offs: calibrated
probabilities here sit near the ~1% base rate, so 0.3/0.7 would never fire.
    Low     below the 80%-specificity threshold
    Medium  at or above it   (about 20% of pre-onset hours are flagged)
    High    at or above the 95%-specificity threshold   (about 5% of hours)
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from explainability_utils.protocol import DL_CONFIGS, WINDOW

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "outputs" / "deployed" / "protocol_H6"

KEY_MAP = {"heart_rate": "HR", "hr": "HR", "respiratory_rate": "Resp", "rr": "Resp", "temperature": "Temp",
           "temp": "Temp", "oxygen_saturation": "O2Sat", "spo2": "O2Sat", "map": "MAP", "sbp": "SBP", "dbp": "DBP",
           "lactate": "Lactate", "wbc": "WBC", "creatinine": "Creatinine", "bilirubin": "Bilirubin_total",
           "age": "Age", "gender": "Gender", "platelets": "Platelets", "iculos": "ICULOS"}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class ProtocolBundle:
    def __init__(self, root=DEFAULT_BUNDLE):
        self.root = Path(root)
        if not (self.root / "calibration.json").exists():
            raise FileNotFoundError(f"No deployment bundle at {self.root}. Build it with scripts/export_deployment.py")
        self.cal = json.load(open(self.root / "calibration.json"))
        pre = joblib.load(self.root / "preprocessor.joblib")
        self.imputer, self.scaler = pre["imputer"], pre["scaler"]
        self.features: List[str] = self.cal["features"]
        self.window: int = int(self.cal.get("window", WINDOW))
        self.nets: Dict[str, List[torch.nn.Module]] = {}
        self.sk: Dict[str, object] = {}
        import models as M
        for name, spec in self.cal["models"].items():
            if spec["kind"] == "dl":
                cls, kw = DL_CONFIGS[name]
                nets = []
                for seed in spec["seeds"]:
                    net = getattr(M, cls)(input_size=len(self.features), **kw)
                    net.load_state_dict(torch.load(self.root / "models" / f"{name}_s{seed}.pt", map_location="cpu"), strict=True)
                    nets.append(net.eval())
                self.nets[name] = nets
            else:
                path = self.root / "models" / spec["file"]
                if not path.exists():       # pickles are gitignored; a fresh clone may lack some
                    print(f"ℹ️ Baseline {name} not in the bundle ({spec['file']} missing); skipping it")
                    continue
                with open(path, "rb") as f:
                    self.sk[name] = pickle.load(f)

    # ------------------------------------------------------------------ features
    def _frame(self, patient_data) -> pd.DataFrame:
        df = pd.DataFrame([patient_data]) if isinstance(patient_data, dict) else pd.DataFrame(patient_data).copy()
        df = df.reset_index(drop=True)
        lower = {c.lower(): c for c in df.columns}
        for src, tgt in KEY_MAP.items():
            if src in lower and tgt not in df.columns:
                df[tgt] = df[lower[src]]
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0, "M": 1, "F": 0}).fillna(df["Gender"])
        for c in self.features:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # engineered columns exist in the training data; rebuild them from vitals when not supplied
        def fill(col, values):
            if col not in df.columns or df[col].isna().all():
                df[col] = values
        if "Resp" in df.columns:
            fill("qSOFA_Resp", (df["Resp"] >= 22).astype(float).where(df["Resp"].notna()))
        if "MAP" in df.columns:
            fill("qSOFA_MAP", (df["MAP"] <= 65).astype(float).where(df["MAP"].notna()))
            fill("MAP_rolling_mean_6hr", df["MAP"].rolling(6, min_periods=1).mean())
            fill("MAP_delta_1hr", df["MAP"].diff())
        if "qSOFA_Resp" in df.columns and "qSOFA_MAP" in df.columns:
            fill("qSOFA_Score_simple", df["qSOFA_Resp"].fillna(0) + df["qSOFA_MAP"].fillna(0))
        return df

    def featurize(self, patient_data):
        """-> (features [1, T, F] float32, observed mask [1, T, F] bool); T = window, left zero-padded."""
        df = self._frame(patient_data).tail(self.window)
        raw = np.full((len(df), len(self.features)), np.nan, dtype=np.float32)
        for j, c in enumerate(self.features):
            if c in df.columns:
                raw[:, j] = df[c].to_numpy(dtype=np.float32)
        obs = ~np.isnan(raw)
        x = self.scaler.transform(self.imputer.transform(raw)).astype(np.float32)
        f = np.zeros((self.window, len(self.features)), np.float32)
        m = np.zeros((self.window, len(self.features)), bool)
        f[-len(df):], m[-len(df):] = x, obs
        return torch.from_numpy(f)[None], torch.from_numpy(m)[None]

    def row_features(self, patient_data) -> np.ndarray:
        """Current-hour scaled features for the row-level baselines (last row)."""
        f, _ = self.featurize(patient_data)
        return f[0, -1].numpy()[None, :]

    # ------------------------------------------------------------------ prediction
    def _level(self, spec, p_cal: float) -> str:
        if p_cal >= spec["threshold_high"]:
            return "High"
        return "Medium" if p_cal >= spec["threshold_medium"] else "Low"

    def _calibrate(self, spec, p_raw: float) -> float:
        a, b = spec["platt"]["coef"], spec["platt"]["intercept"]
        return float(1.0 / (1.0 + np.exp(-(a * _logit(p_raw) + b))))

    def raw_dl(self, patient_data) -> Dict[str, float]:
        """Seed-averaged raw probability per deep model (what the evaluation calibrates)."""
        f, m = self.featurize(patient_data)
        dt = torch.ones(1, self.window, 1)
        dt[:, 0] = 0
        out = {}
        with torch.no_grad():
            for name, nets in self.nets.items():
                out[name] = float(np.mean([torch.sigmoid(n(f, m, dt)).item() for n in nets]))
        return out

    def predict_deep(self, patient_data) -> Dict[str, dict]:
        raw = self.raw_dl(patient_data)
        res = {}
        for name, p in raw.items():
            spec = self.cal["models"][name]
            pc = self._calibrate(spec, p)
            res[name] = dict(risk_score=pc, raw_score=p, risk_level=self._level(spec, pc),
                             confidence=float(np.clip(1.0 - abs(pc - spec["threshold_medium"]) / max(spec["threshold_high"], 1e-6), 0.0, 1.0)),
                             ensemble_members=len(self.nets[name]))
        return res

    def predict_baselines(self, patient_data) -> Dict[str, dict]:
        x = self.row_features(patient_data)
        res = {}
        for name, mdl in self.sk.items():
            spec = self.cal["models"][name]
            p = float(mdl.predict_proba(x)[:, 1][0])
            pc = self._calibrate(spec, p)
            res[name] = dict(risk_score=pc, raw_score=p, risk_level=self._level(spec, pc),
                             confidence=float(np.clip(1.0 - abs(pc - spec["threshold_medium"]) / max(spec["threshold_high"], 1e-6), 0.0, 1.0)))
        return res

    def predict_ensemble(self, patient_data, raw: Optional[Dict[str, float]] = None) -> dict:
        """Calibrated average of the deep models (the bundle's 'digital twin' score)."""
        spec = self.cal["ensemble_dl"]
        raw = raw or self.raw_dl(patient_data)
        members = [m for m in spec["members"] if m in raw]
        p = float(np.mean([raw[m] for m in members]))
        pc = self._calibrate(spec, p)
        member_cal = [self._calibrate(self.cal["models"][m], raw[m]) for m in members]
        return dict(risk_score=pc, raw_score=p, risk_level=self._level(spec, pc), members=members,
                    spread=float(np.std(member_cal)) if len(member_cal) > 1 else 0.0)
