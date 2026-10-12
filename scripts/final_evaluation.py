#!/usr/bin/env python3
"""Final evaluation of every model under ONE protocol (revision plan, Step 3).

    python scripts/final_evaluation.py                       # H=6, all models, seeds 42-44
    python scripts/final_evaluation.py --horizons 3 6 12 --models xgboost lstm
    python scripts/final_evaluation.py --smoke               # tiny end-to-end check

Protocol (the same for the baselines and the deep models):
  split       70/15/15 by PATIENT from outputs/cache/patient_splits_seed42.json (created if missing).
  target      for hour t of a stay: y=1 if the first SepsisLabel==1 (the labelled onset, T1) is within the
              next H hours (0 < T1-t <= H). Rows at or after T1 are dropped, so no model ever sees
              post-onset hours and a persisted label cannot be learned. Non-septic stays are all y=0.
              Patients whose label is already 1 on their first row have no pre-onset rows and drop out.
  features    the 44 columns in outputs/config.json. Imputer (median) and scaler are fitted on the
              training patients' pre-onset rows only. Baselines see the current hour; deep models see
              the last 24 hours (zero-padded, with an observed-value mask).
  imbalance   class weights / pos_weight, plus keeping 1 in --neg-stride negatives in TRAINING only.
              No oversampling. Validation and test keep every pre-onset hour at the true prevalence.
  calibration Platt scaling fitted on validation; threshold = F1-optimal on validation; test is never
              used to choose anything.
  seeds       42, 43, 44 for every stochastic model (+ a seed-ensemble row).
  uncertainty 1000 patient-level bootstrap resamples of the test set (AUROC, AUPRC; 95% percentile CI).
Outputs: outputs/results/final_results.csv (+ _summary.csv, sanity_checks.csv); predictions and
checkpoints go to outputs/final_run/ (gitignored) and never overwrite outputs/models/.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")      # xgboost/torch libomp clash on macOS (see CLAUDE.md)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import xgboost  # noqa: F401  (must precede torch)
except ImportError:
    xgboost = None
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
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


def preprocess(d, train_rows):
    X, M = d["X"], ~np.isnan(d["X"])
    imp = SimpleImputer(strategy="median").fit(X[train_rows])
    Xi = imp.transform(X)
    sc = StandardScaler().fit(Xi[train_rows])
    return sc.transform(Xi).astype(np.float32), M


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


# ----------------------------------------------------------------------------- metrics
def platt(p_val, y_val):
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_val)[:, None], y_val)
    return lambda p: lr.predict_proba(logit(p)[:, None])[:, 1]


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[np.argmax(f1)])


def sens_at_spec(y, p, spec=0.8):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[np.searchsorted(fpr, 1 - spec, side="right") - 1])


def point_metrics(y_te, p_te, p_val, y_val):
    cal = platt(p_val, y_val)
    pc_val, pc_te = cal(p_val), cal(p_te)
    thr = best_f1_threshold(y_val, pc_val)
    pred = pc_te >= thr
    tp, fp = int((pred & (y_te == 1)).sum()), int((pred & (y_te == 0)).sum())
    fn = int((~pred & (y_te == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(auroc=roc_auc_score(y_te, p_te), auprc=average_precision_score(y_te, p_te),
                sens_at_80_spec=sens_at_spec(y_te, p_te), brier=brier_score_loss(y_te, pc_te),
                threshold=thr, precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12))


def bootstrap_ci(y, p, patients, n_boot=1000, seed=0):
    """Patient-level bootstrap using resample multiplicities as sample weights."""
    uniq, inv = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(seed)
    au, ap = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        au.append(roc_auc_score(y, p, sample_weight=w))
        ap.append(average_precision_score(y, p, sample_weight=w))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(au) + q(ap)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "Dataset.csv"))
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[6])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--neg-stride", type=int, default=4, help="keep 1 in N negative training hours")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workdir", default=str(ROOT / "outputs" / "final_run"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "results" / "final_results.csv"))
    ap.add_argument("--smoke", action="store_true", help="500 patients, 1 seed, 1 epoch, 20 bootstraps")
    ap.add_argument("--sanity", action="store_true", help="also run shuffled-label and time-only checks")
    ap.add_argument("--force", action="store_true", help="ignore cached predictions")
    a = ap.parse_args()
    a_out = a.out
    if a.smoke:
        a.seeds, a.epochs, a.n_boot, a.out = a.seeds[:1], 1, 20, str(Path(a.workdir) / "smoke_results.csv")
        a_out = a.out
        a.workdir = str(Path(a.workdir) / "smoke")
    work = Path(a.workdir)
    (work / "preds").mkdir(parents=True, exist_ok=True)

    feats = json.load(open(ROOT / "outputs" / "config.json"))["feature_cols"]
    split_path = ROOT / "outputs" / "cache" / "patient_splits_seed42.json"
    if split_path.exists():
        splits = json.load(open(split_path))
    else:
        from explainability_utils.data_loader import create_patient_splits, load_preprocessed_data
        splits = create_patient_splits(load_preprocessed_data(a.data)[0], seed=42)
    d = build_data(a.data, feats, splits, sample_patients=500 if a.smoke else None)
    print(f"{d['df'].Patient_ID.nunique():,} patients, {len(d['df']):,} rows; "
          f"split counts: { {s: int(len(set(d['pid'][d['split'] == s]))) for s in ('train', 'val', 'test')} }")

    models = (BASELINES + list(DL_CONFIGS)) if "all" in a.models else a.models
    from explainability_utils.training import get_device
    device = get_device()
    rows_out, ens_store = [], {}
    for H in a.horizons:
        r_tr, y_tr = make_samples(d, H, "train")
        r_va, y_va = make_samples(d, H, "val")
        r_te, y_te = make_samples(d, H, "test")
        Xs, Mk = preprocess(d, r_tr)
        pat_te = d["pid"][r_te]
        print(f"\n=== H={H}h | samples train/val/test: {len(r_tr):,}/{len(r_va):,}/{len(r_te):,} | "
              f"positives {int(y_tr.sum()):,}/{int(y_va.sum()):,}/{int(y_te.sum()):,} "
              f"({y_te.mean()*100:.2f}% of test) ===")
        stats = dict(n_test=len(r_te), n_test_pos=int(y_te.sum()), n_test_patients=len(np.unique(pat_te)))
        win_cache = {}
        for name in models:
            for seed in ([a.seeds[0]] if name in DETERMINISTIC else a.seeds):
                cache = work / "preds" / f"{name}_H{H}_s{seed}.npz"
                t0 = time.time()
                if cache.exists() and not a.force:
                    z = np.load(cache); p_va, p_te = z["val"], z["test"]
                else:
                    rs, ys = subsample(r_tr, y_tr, a.neg_stride, seed)
                    if name in DL_CONFIGS:
                        if "tr" not in win_cache:
                            rv, yv = subsample(r_va, y_va, a.neg_stride, 0)          # early stopping only
                            win_cache.update(va=windows(Xs, Mk, d["start"], rv), yva=yv,
                                             vaf=windows(Xs, Mk, d["start"], r_va), te=windows(Xs, Mk, d["start"], r_te))
                        tr = windows(Xs, Mk, d["start"], rs)
                        net = train_dl(name, seed, tr, win_cache["va"], ys, win_cache["yva"], device,
                                       a.epochs, a.patience, work / f"{name}_H{H}_s{seed}.pt")
                        p_va, p_te = predict_dl(net, win_cache["vaf"], device), predict_dl(net, win_cache["te"], device)
                    else:
                        if name == "knn" and len(rs) > 60000:
                            rs, ys = rs[:60000], ys[:60000]
                        m = fit_baseline(name, Xs[rs], ys, seed, time_col=feats.index("ICULOS"))
                        p_va, p_te = m.predict_proba(Xs[r_va])[:, 1], m.predict_proba(Xs[r_te])[:, 1]
                    np.savez_compressed(cache, val=p_va, test=p_te)
                met = point_metrics(y_te, p_te, p_va, y_va)
                rows_out.append(dict(model=name, horizon=H, seed=seed, **met, **stats))
                ens_store.setdefault((name, H), []).append((p_va, p_te))
                print(f"  {name:24s} seed {seed}: AUROC {met['auroc']:.3f} AUPRC {met['auprc']:.3f} "
                      f"sens@80spec {met['sens_at_80_spec']:.3f} Brier {met['brier']:.4f}  [{time.time()-t0:.0f}s]")
        # seed-ensemble rows (mean probability over seeds)
        for (name, h), lst in ens_store.items():
            if h != H or len(lst) < 2:
                continue
            pv, pt = np.mean([x[0] for x in lst], 0), np.mean([x[1] for x in lst], 0)
            rows_out.append(dict(model=name, horizon=H, seed="ensemble", **point_metrics(y_te, pt, pv, y_va), **stats))
        # bootstrap CIs for every row of this horizon, in parallel
        todo = [r for r in rows_out if r["horizon"] == H]
        def pred_for(r):
            if r["seed"] == "ensemble":
                return np.mean([x[1] for x in ens_store[(r["model"], H)]], 0)
            i = ([a.seeds[0]] if r["model"] in DETERMINISTIC else a.seeds).index(r["seed"])
            return ens_store[(r["model"], H)][i][1]
        cis = Parallel(n_jobs=max(1, (os.cpu_count() or 2) - 1))(
            delayed(bootstrap_ci)(y_te, pred_for(r), pat_te, a.n_boot) for r in todo)
        for r, ci in zip(todo, cis):
            r.update(auroc_ci_low=ci[0], auroc_ci_high=ci[1], auprc_ci_low=ci[2], auprc_ci_high=ci[3])

        if a.sanity:
            # Null models. Plain label shuffling is depressed below 0.5 because test positives sit late in a
            # stay; shuffling WITHIN hour-in-stay bins keeps that time prior and breaks the link to
            # physiology, so its AUROC should land near the time-only reference, not above it.
            rows, tcol = [], feats.index("ICULOS")
            bins = d["pos"][r_tr] // 6
            y_bin = y_tr.copy()
            rng = np.random.default_rng(0)
            for b_ in np.unique(bins):
                ix = np.flatnonzero(bins == b_)
                y_bin[ix] = rng.permutation(y_tr[ix])
            for label, ysh in [("shuffled_labels_plain", rng.permutation(y_tr)), ("shuffled_labels_within_6h_bins", y_bin)]:
                for mn in ("xgboost", "logistic_regression"):
                    m = fit_baseline(mn, Xs[r_tr], ysh, 42, time_col=tcol)
                    rows.append(dict(check=f"{mn}_{label}", horizon=H, auroc=roc_auc_score(y_te, m.predict_proba(Xs[r_te])[:, 1]),
                                     note="null model; compare with the time_only row, not with 0.5"))
            sp = Path(a_out).with_name("sanity_checks.csv")
            pd.DataFrame(rows).to_csv(sp, mode="a", header=not sp.exists(), index=False)
            print("sanity:", [(r["check"], round(r["auroc"], 3)) for r in rows])

    res = pd.DataFrame(rows_out)
    cols = ["model", "horizon", "seed", "auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high",
            "sens_at_80_spec", "brier", "threshold", "precision", "recall", "f1", "n_test", "n_test_pos", "n_test_patients"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res[cols].to_csv(a.out, index=False)
    per_seed = res[res.seed != "ensemble"]
    summ = per_seed.groupby(["model", "horizon"]).agg(
        n_seeds=("seed", "size"), auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"), sens_at_80_spec_mean=("sens_at_80_spec", "mean"),
        brier_mean=("brier", "mean")).reset_index()
    summ.to_csv(Path(a.out).with_name(Path(a.out).stem + "_summary.csv"), index=False)
    print(f"\nwrote {a.out}")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
