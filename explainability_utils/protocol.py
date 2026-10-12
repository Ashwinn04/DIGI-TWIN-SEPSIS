"""Shared protocol code: leakage-safe labels, preprocessing, windows and the model zoo.

Used by scripts/final_evaluation.py (reported results) and surveillance/risk_protocol.py (out-of-fold
risk scores), so both rest on exactly the same data handling, labels and architectures.
See scripts/final_evaluation.py for the protocol description.
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")      # xgboost/torch libomp clash on macOS (see CLAUDE.md)
try:
    import xgboost  # noqa: F401  (must precede torch)
except ImportError:
    xgboost = None
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

WINDOW = 24
DL_CONFIGS = {   # architectures the repo's checkpoints use (models/ at 5f81942)
    "grud": ("GRUD", dict(hidden_size=128, num_layers=2, dropout=0.3)),
    "lstm": ("LSTM", dict(hidden_size=128, num_layers=2, dropout=0.3)),
    "cnn_lstm": ("CNNLSTM", dict(hidden_size=128, num_layers=1, dropout=0.3, cnn_filters=[64, 128], kernel_size=3)),
    "transformer": ("Transformer", dict(d_model=128, nhead=8, num_layers=4, dropout=0.1)),
}
BASELINES = ["time_only", "logistic_regression", "decision_tree", "knn", "random_forest", "xgboost",
             "hist_gradient_boosting", "ann"]
DETERMINISTIC = {"time_only", "logistic_regression", "decision_tree", "knn"}      # one seed is enough


# ----------------------------------------------------------------------------- data
def build_data(csv_path, feats, splits, sample_patients=None):
    df = pd.read_csv(csv_path)
    if sample_patients:
        keep = pd.Series(df.Patient_ID.unique()).sample(sample_patients, random_state=0)
        df = df[df.Patient_ID.isin(keep)]
    df = df.reset_index(drop=True)       # file order is ICULOS order within each stay (checked)
    pid = df.Patient_ID.to_numpy()
    first = np.r_[True, pid[1:] != pid[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(df)), 0))
    pos = np.arange(len(df)) - start
    onset = np.full(len(df), np.iinfo(np.int32).max, dtype=np.int64)       # position of first label 1
    lab = df.SepsisLabel.to_numpy()
    first_pos = pd.Series(np.where(lab == 1, pos, np.iinfo(np.int32).max)).groupby(pid).transform("min").to_numpy()
    onset[:] = first_pos
    return dict(df=df, X=df[feats].to_numpy(np.float32), pid=pid, start=start, pos=pos, onset=onset,
                split=pd.Series(pid).map({p: s for s, ids in splits.items() for p in ids}).to_numpy(object))


def make_samples(d, horizon, split):
    """Row indices and labels for one split, post-onset hours dropped."""
    ok = (d["split"] == split) & (d["pos"] < d["onset"])
    rows = np.flatnonzero(ok)
    y = ((d["onset"][rows] - d["pos"][rows]) <= horizon).astype(np.int8)
    return rows, y


def preprocess(d, train_rows, return_fitted=False):
    X, M = d["X"], ~np.isnan(d["X"])
    imp = SimpleImputer(strategy="median").fit(X[train_rows])
    Xi = imp.transform(X)
    sc = StandardScaler().fit(Xi[train_rows])
    Xs = sc.transform(Xi).astype(np.float32)
    return (Xs, M, imp, sc) if return_fitted else (Xs, M)


def windows(Xs, M, start, rows, window=WINDOW):
    idx = rows[:, None] + (np.arange(window) - (window - 1))[None, :]
    valid = idx >= start[rows][:, None]
    idx = np.where(valid, idx, 0)
    f = Xs[idx] * valid[..., None]
    m = M[idx] & valid[..., None]
    dt = np.ones((len(rows), window, 1), np.float32)
    dt[:, 0] = 0
    return torch.from_numpy(f.astype(np.float32)), torch.from_numpy(m), torch.from_numpy(dt)


def subsample(rows, y, stride, seed):
    if stride <= 1:
        return rows, y
    rng = np.random.default_rng(seed)
    keep = (y == 1) | (rng.random(len(y)) < 1.0 / stride)
    return rows[keep], y[keep]


# ----------------------------------------------------------------------------- models
class TimeOnly:
    """Logistic regression on hour-in-ICU alone: the reference every model has to beat.
    Positives sit late in a stay by construction, so time alone already ranks them above early hours."""
    def __init__(self, col):
        self.col = col

    def fit(self, X, y):
        self.m = LogisticRegression(max_iter=500, class_weight="balanced").fit(X[:, [self.col]], y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X[:, [self.col]])


def fit_baseline(name, Xtr, ytr, seed, time_col=None):
    pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    if name == "time_only":
        return TimeOnly(time_col).fit(Xtr, ytr)
    if name == "logistic_regression":
        m = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "decision_tree":
        m = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=seed)
    elif name == "knn":
        m = KNeighborsClassifier(n_neighbors=50, n_jobs=-1)
    elif name == "random_forest":
        m = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=pw, tree_method="hist", n_jobs=4, random_state=seed, verbosity=0)
    elif name == "hist_gradient_boosting":
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                           class_weight="balanced", random_state=seed)
    elif name == "ann":
        m = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=seed)
    else:
        raise ValueError(name)
    return m.fit(Xtr, ytr)


def train_dl(name, seed, tr, va, ytr, yva, device, epochs, patience, ckpt):
    import models as M
    from explainability_utils.training import ModelTrainer, set_random_seeds
    set_random_seeds(seed)
    cls, kw = DL_CONFIGS[name]
    net = getattr(M, cls)(input_size=tr[0].shape[2], **kw)
    trainer = ModelTrainer(net, device=device)
    mk = lambda t, y: torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(t[0], t[1], t[2], torch.from_numpy(y.astype(np.int64))[:, None]),
        batch_size=256, shuffle=(y is ytr))
    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    trainer.train(mk(tr, ytr), mk(va, yva), epochs=epochs,
                  optimizer_config={"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4},
                  scheduler_config={"name": "CosineAnnealing", "warmup_epochs": 1, "T_max": epochs - 1},
                  loss_config={"name": "BCE", "pos_weight": pw, "label_smoothing": 0.0, "focal_alpha": 0.25, "focal_gamma": 2.0},
                  early_stopping_config={"patience": patience, "metric": "auprc"}, grad_clip_value=1.0, save_path=str(ckpt))
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return net


@torch.no_grad()
def predict_dl(net, win, device, bs=2048):
    net.to(device).eval()
    out = []
    for i in range(0, len(win[0]), bs):
        f, m, dt = (t[i:i + bs].to(device) for t in win)
        out.append(torch.sigmoid(net(f, m, dt)).flatten().float().cpu().numpy())
    return np.concatenate(out)


