#!/usr/bin/env python3
"""GradientSHAP explanations for the leakage-safe GRU-D models (outputs/deployed/protocol_H6).

    python scripts/shap_grud.py                     # global importance + one patient trajectory
    python scripts/shap_grud.py --patient 12345     # trajectory for a chosen test patient
    python scripts/shap_grud.py --n-pos 100 --n-neg 100 --n-samples 20   # quicker run

Inputs are built exactly as in scripts/final_evaluation.py (same patient split, pre-onset hours only,
24-hour zero-padded window, imputer/scaler fitted on training patients). Attributions are computed with
Captum's GradientShap on each seed's output logit and averaged over the three seeds. Each window has two
inputs, the scaled values and the observed-value mask, so every feature gets a value part and a mask part;
a feature's attribution for one window is both parts summed over the 24 hours. Dataset.csv is already
imputed (no NaN), so the mask is 0 only on the zero-padded hours before a stay starts: the mask part
measures how much history the window has, not missed measurements.

Global importance: mean |attribution| over a stratified sample of test windows (all-positive cap --n-pos,
random negatives --n-neg), against a background of random training windows.

Trajectory: one septic test patient with at least --min-hours pre-onset hours, chosen by rule (median, among
eligible patients, of the highest calibrated GRU-D risk in the 6 hours before onset), so the example is
typical rather than the best case. Each hour is scored and explained with the window ending at that hour.

Outputs (outputs/shap/): grud_global_importance.csv, grud_shap_summary.png/.pdf,
grud_patient_<id>_trajectory.csv, grud_patient_trajectory.png/.pdf, shap_run.json.
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")      # xgboost/torch libomp clash on macOS (see CLAUDE.md)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import xgboost  # noqa: F401  (must precede torch)
except ImportError:
    pass
import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap

from explainability_utils.deploy import ProtocolBundle, _logit
from explainability_utils.protocol import build_data, make_samples, windows

MODEL = "grud"


class LogitOf(torch.nn.Module):
    """(values, mask) -> logit; delta_t defaults to regular hourly steps inside GRU-D."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, m):
        return self.net(x, m)


def attribute(nets, x, m, bx, bm, n_samples, chunk):
    """Seed-averaged GradientSHAP. Returns value and mask attributions [B, T, F] and mean |delta|."""
    ax, am, deltas = np.zeros(x.shape, np.float32), np.zeros(x.shape, np.float32), []
    for net in nets:
        gs = GradientShap(LogitOf(net.eval()))
        for i in range(0, len(x), chunk):
            torch.manual_seed(0)
            xi, mi = x[i:i + chunk].requires_grad_(), m[i:i + chunk].requires_grad_()
            (a_x, a_m), d = gs.attribute((xi, mi), baselines=(bx, bm), n_samples=n_samples,
                                         stdevs=0.0, return_convergence_delta=True)
            ax[i:i + chunk] += a_x.detach().numpy() / len(nets)
            am[i:i + chunk] += a_m.detach().numpy() / len(nets)
            deltas.append(d.abs().mean().item())
    return ax, am, float(np.mean(deltas))


@torch.no_grad()
def raw_prob(nets, x, m, bs=2048):
    out = []
    for i in range(0, len(x), bs):
        out.append(np.mean([torch.sigmoid(n(x[i:i + bs], m[i:i + bs])).flatten().numpy() for n in nets], 0))
    return np.concatenate(out)


def calibrate(p, spec):
    return 1.0 / (1.0 + np.exp(-(spec["platt"]["coef"] * _logit(p) + spec["platt"]["intercept"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "Dataset.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "shap"))
    ap.add_argument("--n-pos", type=int, default=500, help="test windows with y=1 (capped at all available)")
    ap.add_argument("--n-neg", type=int, default=500, help="random test windows with y=0")
    ap.add_argument("--n-background", type=int, default=200, help="random training windows")
    ap.add_argument("--n-samples", type=int, default=200, help="GradientSHAP samples per window")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--patient", type=int, default=None)
    ap.add_argument("--min-hours", type=int, default=24, help="pre-onset hours a trajectory patient needs")
    ap.add_argument("--hours", type=int, default=36, help="hours before onset shown in the trajectory")
    ap.add_argument("--workdir", nargs="+", default=[str(ROOT / "outputs" / "final_run"),
                    str(ROOT / "outputs" / "final_run" / "anjith_rerun")],
                    help="final_evaluation workdirs to look in for cached predictions (first complete one wins)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    b = ProtocolBundle()
    nets, spec, feats, H = b.nets[MODEL], b.cal["models"][MODEL], b.features, b.cal["horizon"]
    splits = json.load(open(ROOT / "outputs" / "cache" / "patient_splits_seed42.json"))
    d = build_data(a.data, feats, splits)
    Xs = b.scaler.transform(b.imputer.transform(d["X"])).astype(np.float32)
    Mk = ~np.isnan(d["X"])
    r_tr, _ = make_samples(d, H, "train")
    r_te, y_te = make_samples(d, H, "test")
    print(f"test: {len(r_te):,} pre-onset hours, {int(y_te.sum()):,} positive  [{time.time()-t0:.0f}s]")

    # the runtime inputs must reproduce the evaluation's cached test predictions
    cands = [[Path(w) / "preds" / f"{MODEL}_H{H}_s{s}.npz" for s in spec["seeds"]] for w in a.workdir]
    cache = next((c for c in cands if all(x.exists() for x in c)), None)
    if cache:
        chk = np.random.default_rng(1).choice(len(r_te), 500, replace=False)
        f, m, _ = windows(Xs, Mk, d["start"], r_te[chk])
        ref = np.mean([np.load(c)["test"][chk] for c in cache], 0)
        err = float(np.abs(raw_prob(nets, f, m.float()) - ref).max())
        print(f"reproduction check vs {cache[0].parent}: max |diff| {err:.2e}")
        if err > 2e-3:
            raise SystemExit("inputs do not reproduce the evaluation's predictions; refusing to explain them")
    else:
        err = None
        print("final_run cache missing; skipping reproduction check")

    rng = np.random.default_rng(0)
    bg = rng.choice(r_tr, a.n_background, replace=False)
    bx, bm, _ = windows(Xs, Mk, d["start"], bg)
    bm = bm.float()

    # ---------------------------------------------------------------- global importance
    pos, neg = np.flatnonzero(y_te == 1), np.flatnonzero(y_te == 0)
    pick = np.r_[rng.choice(pos, min(a.n_pos, len(pos)), replace=False), rng.choice(neg, a.n_neg, replace=False)]
    x, m, _ = windows(Xs, Mk, d["start"], r_te[pick])
    m = m.float()
    ax, am, delta = attribute(nets, x, m, bx, bm, a.n_samples, a.chunk)
    val, mis = ax.sum(1), am.sum(1)               # [N, F] summed over the 24 hours
    tot = val + mis
    # completeness: a window's attributions should add up to f(x) - E[f(background)] (seed-mean logits)
    with torch.no_grad():
        fx = np.mean([n(x, m).flatten().numpy() for n in nets], 0)
        fb = float(np.mean([n(bx, bm).flatten().numpy().mean() for n in nets]))
    gap = tot.sum(1) - (fx - fb)
    complete = dict(corr=float(np.corrcoef(tot.sum(1), fx - fb)[0, 1]), mean_abs_gap=float(np.abs(gap).mean()),
                    mean_abs_target=float(np.abs(fx - fb).mean()))
    last_val = x[:, -1].numpy()                    # last-hour scaled value (no NaN in Dataset.csv)
    g = pd.DataFrame(dict(feature=feats, mean_abs_shap=np.abs(tot).mean(0), mean_abs_value_part=np.abs(val).mean(0),
                          mean_abs_mask_part=np.abs(mis).mean(0),
                          mean_shap_positives=tot[y_te[pick] == 1].mean(0), mean_shap_negatives=tot[y_te[pick] == 0].mean(0),
                          ))
    g = g.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    g.to_csv(out / "grud_global_importance.csv", index=False)
    print(f"global: {len(pick)} windows ({int(y_te[pick].sum())} positive); completeness corr {complete['corr']:.3f}, "
          f"mean |gap| {complete['mean_abs_gap']:.3f} vs mean |f(x)-E f(bg)| {complete['mean_abs_target']:.3f}  [{time.time()-t0:.0f}s]")
    print(g.head(a.top).round(4).to_string(index=False))

    # ---------------------------------------------------------------- trajectory patient
    pid_te = d["pid"][r_te]
    if a.patient is None:
        p_all = calibrate(raw_prob(nets, *(lambda w: (w[0], w[1].float()))(windows(Xs, Mk, d["start"], r_te))), spec)
        to_onset = d["onset"][r_te] - d["pos"][r_te]
        cand = []
        for p in np.unique(pid_te[y_te == 1]):
            ix = np.flatnonzero(pid_te == p)
            if len(ix) >= a.min_hours:
                cand.append((p_all[ix][to_onset[ix] <= H].max(), p))
        cand.sort()
        patient = int(cand[len(cand) // 2][1])
        rule = (f"median (of {len(cand)} septic test patients with >= {a.min_hours} pre-onset hours) "
                f"of the highest calibrated risk in the {H} h before onset")
    else:
        patient, rule = a.patient, "chosen with --patient"
    ix = np.flatnonzero(pid_te == patient)
    if not len(ix):
        raise SystemExit(f"patient {patient} has no pre-onset test hours")
    ix = ix[-a.hours:]
    rows = r_te[ix]
    rel = (d["pos"][rows] - d["onset"][rows]).astype(int)     # hours relative to labelled onset (negative)
    tx, tm, _ = windows(Xs, Mk, d["start"], rows)
    tm = tm.float()
    risk = calibrate(raw_prob(nets, tx, tm), spec)
    pax, pam, pdelta = attribute(nets, tx, tm, bx, bm, a.n_samples, a.chunk)
    ptot = pax.sum(1) + pam.sum(1)                              # [hours, F]
    t_col = feats.index("ICULOS")                  # time in ICU: shown separately, it dwarfs the rest
    top_p = [j for j in np.argsort(-np.abs(ptot).sum(0)) if j != t_col][:6]
    traj = pd.DataFrame({"hour_rel_onset": rel, "label_within_6h": y_te[ix], "risk_calibrated": risk,
                         "shap_ICULOS": ptot[:, t_col], **{f"shap_{feats[j]}": ptot[:, j] for j in top_p}})
    traj.to_csv(out / f"grud_patient_{patient}_trajectory.csv", index=False)
    lead = rel[np.argmax(risk >= spec["threshold_high"])] if (risk >= spec["threshold_high"]).any() else None
    print(f"trajectory: patient {patient} ({rule}); {len(ix)} h; first High at {lead} h  [{time.time()-t0:.0f}s]")

    # ---------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8})

    top = g.head(a.top)
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(6.6, 3.6), gridspec_kw=dict(width_ratios=[1, 1.5]), sharey=True)
    yy = np.arange(len(top))[::-1]
    axb.barh(yy, top.mean_abs_value_part, color="#3b6ea5", label="value")
    axb.barh(yy, top.mean_abs_mask_part, left=top.mean_abs_value_part, color="#b9c9de", label="history mask")
    axb.set_yticks(yy, top.feature)
    axb.set_xlabel("mean |SHAP| (log-odds)")
    axb.legend(frameon=False, loc="lower right")
    order = [feats.index(f) for f in top.feature]
    jit = np.random.default_rng(0)
    for k, j in enumerate(order):
        yj = yy[k] + jit.uniform(-0.3, 0.3, len(last_val))
        sc = axs.scatter(tot[:, j], yj, s=3, c=np.clip(last_val[:, j], -2, 2), cmap="coolwarm", vmin=-2, vmax=2, lw=0)
    axs.axvline(0, color="k", lw=0.5)
    axs.set_xlabel("SHAP value per window (log-odds)")
    cb = fig.colorbar(sc, ax=axs, pad=0.01)
    cb.set_label("last-hour value (z-score)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"grud_shap_summary.{ext}", dpi=300)
    plt.close(fig)

    fig, (ar, at, ah) = plt.subplots(3, 1, figsize=(6.6, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 0.5, 1.1]))
    ar.axvspan(-H - 0.5, 0, color="#f3d6d6", lw=0, label=f"label window ({H} h before onset)")
    ar.plot(rel, risk, "-o", ms=2.5, lw=1, color="#3b6ea5", label="GRU-D risk (calibrated)")
    ar.axhline(spec["threshold_medium"], ls="--", lw=0.8, color="#d08a1e", label="Medium (80% specificity)")
    ar.axhline(spec["threshold_high"], ls="--", lw=0.8, color="#b2262c", label="High (95% specificity)")
    ar.set_ylabel("risk of onset\nwithin 6 h")
    ar.set_ylim(0, max(risk.max(), spec["threshold_high"]) * 1.45)
    ar.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)
    at.plot(rel, ptot[:, t_col], "-", lw=1, color="#555555")
    at.axhline(0, color="k", lw=0.4)
    at.set_ylabel("SHAP of\nICULOS")
    lim = np.abs(ptot[:, top_p]).max()
    im = ah.imshow(ptot[:, top_p].T, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim,
                   extent=[rel[0] - 0.5, rel[-1] + 0.5, len(top_p) - 0.5, -0.5], interpolation="nearest")
    ah.set_yticks(range(len(top_p)), [feats[j] for j in top_p])
    ah.set_xlabel("hours relative to labelled sepsis onset")
    fig.colorbar(im, ax=[ar, at, ah], pad=0.01, fraction=0.03, shrink=0.4, anchor=(0.0, 0.0)).set_label("SHAP (log-odds)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"grud_patient_trajectory.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    json.dump(dict(model=MODEL, bundle=str(b.root.relative_to(ROOT)), seeds=spec["seeds"], horizon=H,
                   method="captum GradientShap on each seed's logit, inputs (values, mask), averaged over seeds",
                   n_windows=int(len(pick)), n_positive=int(y_te[pick].sum()), n_background=a.n_background,
                   n_samples=a.n_samples, completeness=complete, per_draw_mean_abs_delta=delta, reproduction_max_diff=err,
                   reproduction_cache=None if not cache else str(cache[0].parent.relative_to(ROOT)),
                   note="Dataset.csv has no NaN; the mask only marks zero-padded hours before the stay starts",
                   trajectory=dict(patient=patient, rule=rule, hours=int(len(ix)), first_high_hour_rel_onset=None if lead is None else int(lead),
                                   per_draw_mean_abs_delta=pdelta, heatmap_features=[feats[j] for j in top_p],
                                   iculos_shap_first_last=[float(ptot[0, t_col]), float(ptot[-1, t_col])]),
                   thresholds=dict(medium=spec["threshold_medium"], high=spec["threshold_high"]),
                   top_global=top.feature.tolist(), seconds=round(time.time() - t0)),
              open(out / "shap_run.json", "w"), indent=2)
    print(f"wrote {out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
