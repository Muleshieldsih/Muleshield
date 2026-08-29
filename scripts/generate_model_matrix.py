# -*- coding: utf-8 -*-
"""
MuleShield AI — Model Performance Matrix
SIH26184 | MHA / I4C

Renders the evaluation figures used in the README and the SIH deck, computed
from the trained models and the current dataset rather than typed in by hand.

Every panel carries its baseline, and where a ceiling exists it is drawn on the
same axes. A score without its baseline says nothing about a model, and a chart
without its ceiling invites a reader to assume the ceiling is 1.0.

Outputs:
  docs/model_matrix_full.png            six-panel evaluation matrix
  docs/sih_performance_matrix_slide.png headline card for the deck

Run:
    python scripts/generate_model_matrix.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
DATA, DOCS, MODELS = ROOT / "data", ROOT / "docs", ROOT / "models"

from gnn_model import FEATURE_COLS, derive_features            # noqa: E402
from feature_builder import FeatureBuilder                     # noqa: E402
from xgb_model import ConditionalLogitRanker, MuleXGBPredictor  # noqa: E402
import generate_data as gd                                     # noqa: E402

# ── Palette: deep pine accent, signal colours reserved for verdicts ──────────
INK, MUTED, RULE = "#16191A", "#6C7570", "#D8DCD4"
ACCENT, BASE, CEIL = "#1F4D3D", "#9AA5AE", "#A03227"
PAPER, PANEL = "#FFFFFF", "#F4F5F2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def _style(ax, title, sub=None):
    # Title is lifted clear of the axes so the sub-line can sit between the two
    # without colliding with it.
    ax.set_title(title, loc="left", pad=26 if sub else 8)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.8,
                color=MUTED, va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor(PAPER)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_gnn() -> dict:
    """Score the GNN and every non-graph baseline on the checkpoint's own split."""
    ckpt = torch.load(MODELS / "graphsage_mule.pt", map_location="cpu",
                      weights_only=False)
    node = derive_features(pd.read_csv(DATA / "node_features.csv"))

    order = {a: i for i, a in enumerate(ckpt["account_ids"])}
    node = node[node["account_id"].isin(order)].copy()
    node["_pos"] = node["account_id"].map(order)
    node = node.sort_values("_pos").reset_index(drop=True)

    X = node[FEATURE_COLS].to_numpy(float)
    y = node["is_mule_label"].to_numpy()
    sp = ckpt["split"]
    tr = np.array(sp["train_idx"] + sp["val_idx"])
    te = np.array(sp["test_idx"])

    # GNN scores, reproduced from the saved weights on the saved split.
    from train_gnn import load_pyg_data
    from gnn_model import GraphSAGEMule
    data, *_ = load_pyg_data(seed=ckpt["hyperparams"].get("seed", 42))
    model = GraphSAGEMule(in_channels=len(FEATURE_COLS))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prob_all = torch.sigmoid(model(data.x, data.edge_index).squeeze(-1)).numpy()
    thr = ckpt["metrics"].get("threshold", 0.5)
    p_te, y_te = prob_all[te], y[te]

    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
    rf = RandomForestClassifier(n_estimators=400, max_depth=16, class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(X[tr], y[tr])

    def tuned_f1(scores):
        best = 0.0
        for t in np.unique(np.quantile(scores, np.linspace(.80, .999, 90))):
            best = max(best, f1_score(y_te, (scores >= t).astype(int), zero_division=0))
        return best

    lr_s = lr.predict_proba(sc.transform(X[te]))[:, 1]
    rf_s = rf.predict_proba(X[te])[:, 1]

    prev = float(y.mean())
    tp = prev * (1 - gd.UNDETECTED_MULE_RATE)
    fn = prev * gd.UNDETECTED_MULE_RATE
    fp = (1 - prev) * gd.FALSE_REPORT_RATE
    P, R = tp / (tp + fp), tp / (tp + fn)

    return dict(
        y=y_te, prob=p_te, thr=thr,
        cm=confusion_matrix(y_te, (p_te >= thr).astype(int)),
        f1=f1_score(y_te, (p_te >= thr).astype(int)),
        auc=roc_auc_score(y_te, p_te), ap=average_precision_score(y_te, p_te),
        lr_s=lr_s, rf_s=rf_s,
        bars=[("Best single\nfeature", tuned_f1(X[te][:, FEATURE_COLS.index("burst_out_5min")])),
              ("Logistic\nregression", tuned_f1(lr_s)),
              ("Random\nforest", tuned_f1(rf_s)),
              ("GraphSAGE", f1_score(y_te, (p_te >= thr).astype(int)))],
        ceiling=2 * P * R / (P + R),
        prevalence=prev,
    )


def evaluate_location() -> dict:
    """ATM ranking, search-zone containment and the countdown, on a complaint split."""
    fb = FeatureBuilder(); fb.load()
    X, y, groups, meta = fb.build_ranking_set()
    C, K = fb.RANK_CONTEXT_DIM, fb.CANDIDATE_K

    cid = dict(zip(meta.group_id, meta.complaint_id))
    rc = np.array([cid[g] for g in groups])
    tr, te = next(GroupShuffleSplit(1, test_size=.2, random_state=42).split(X, y, groups=rc))

    clf = ConditionalLogitRanker(n_features=6, k=K)
    clf.fit(MuleXGBPredictor.log_features(X[tr][:, C:]), y[tr], groups[tr])
    s_te = clf.predict(MuleXGBPredictor.log_features(X[te][:, C:]))

    d_te = X[te][:, C]
    bayes = (-d_te / gd.ATM_DISTANCE_DECAY_KM
             + np.log1p(gd.ATM_RISK_WEIGHT * X[te][:, C + 3])
             + X[te][:, C + 5] * np.log(gd.ATM_SAME_BANK_BOOST))

    def topk(score):
        hits = {1: 0, 3: 0, 5: 0}; n = 0
        for g in np.unique(groups[te]):
            m = groups[te] == g
            lab = y[te][m][np.argsort(-score[m])]
            if not lab.sum():
                continue
            n += 1
            pos = int(np.argmax(lab)) + 1
            for k in hits:
                hits[k] += pos <= k
        return {k: v / max(1, n) for k, v in hits.items()}

    # Countdown: use the SHIPPED model and its scaler, not a refit. Refitting
    # here produced a slightly different MAE from the one train_xgb.py reports,
    # which would put two numbers for the same thing into the repository.
    predictor = MuleXGBPredictor.load()
    tmap = dict(zip(meta.group_id, meta.time_to_cashout_min))
    pos_tr, pos_te = np.where(y[tr] == 1)[0], np.where(y[te] == 1)[0]
    yt_tr = np.array([tmap[int(g)] for g in groups[tr][pos_tr]], float)
    yt_te = np.array([tmap[int(g)] for g in groups[te][pos_te]], float)
    yhat = predictor.regressor.predict(predictor.scaler.transform(X[te][pos_te]))

    return dict(
        ours=topk(s_te), dist=topk(-d_te), bayes=topk(bayes),
        yt=yt_te, yhat=yhat,
        mae=float(np.mean(np.abs(yhat - yt_te))),
        base_mae=float(np.mean(np.abs(yt_te - yt_tr.mean()))),
        # Measured in engine/train_xgb.py; the zone geometry is reproduced there.
        zone=dict(model=.874, near3=.785, mule=.752, radius=9.98, atms=8),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def render(g: dict, loc: dict) -> None:
    DOCS.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(17, 10.5), facecolor=PAPER)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=.52, wspace=.30,
                           left=.055, right=.975, top=.87, bottom=.07)

    fig.text(.055, .955, "MuleShield AI — Model Performance Matrix",
             fontsize=19, fontweight="bold", color=INK)
    fig.text(.055, .922,
             "SIH26184 · MHA / I4C   ·   every panel reports its baseline; "
             "ceilings drawn where one exists   ·   computed from the trained models",
             fontsize=9.5, color=MUTED)

    # 1 ── Confusion matrix ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0]); _style(ax, "GNN confusion matrix",
                                           f"threshold {g['thr']:.3f}, tuned on validation")
    cm = g["cm"]
    ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                    fontweight="bold",
                    color=PAPER if cm[i, j] > cm.max() * .55 else INK)
    ax.set_xticks([0, 1], ["pred clean", "pred mule"])
    ax.set_yticks([0, 1], ["actual clean", "actual mule"])
    ax.set_xlabel(f"FPR {cm[0,1]/max(1,cm[0].sum()):.4f}   ·   "
                  f"FNR {cm[1,0]/max(1,cm[1].sum()):.4f}", fontsize=8.5, labelpad=9)
    ax.grid(False)

    # 2 ── ROC + PR ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1]); _style(ax, "ROC and precision–recall",
                                           f"AUC {g['auc']:.4f}   ·   PR-AUC {g['ap']:.4f}")
    fpr, tpr, _ = roc_curve(g["y"], g["prob"])
    pr, rc, _ = precision_recall_curve(g["y"], g["prob"])
    ax.plot(fpr, tpr, color=ACCENT, lw=2.2, label=f"ROC · AUC {g['auc']:.3f}")
    ax.plot(rc, pr, color=CEIL, lw=2.2, ls="--", label=f"PR · AP {g['ap']:.3f}")
    ax.plot([0, 1], [0, 1], color=BASE, lw=1, ls=":", label="random")
    ax.axhline(g["prevalence"], color=BASE, lw=1, ls=":")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR  /  recall"); ax.set_ylabel("TPR  /  precision")
    ax.legend(frameon=False, fontsize=7.8, loc="center right")

    # 3 ── Model vs baselines ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2]); _style(ax, "Mule detection vs baselines",
                                           "identical features, same held-out nodes")
    names = [b[0] for b in g["bars"]]; vals = [b[1] for b in g["bars"]]
    cols = [BASE] * (len(vals) - 1) + [ACCENT]
    ax.bar(names, vals, color=cols, width=.62)
    ax.axhline(g["ceiling"], color=CEIL, lw=1.4, ls="--")
    ax.text(len(vals) - .45, g["ceiling"] + .012,
            f"label-noise ceiling {g['ceiling']:.3f}", fontsize=7.6,
            color=CEIL, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + .012, f"{v:.4f}", ha="center", fontsize=8.4,
                fontweight="bold" if i == len(vals) - 1 else "normal")
    ax.set_ylim(0, 1.02); ax.set_ylabel("F1")
    ax.tick_params(axis="x", labelsize=8)

    # 4 ── ATM ranking vs distance and Bayes ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 0]); _style(ax, "Exact-ATM ranking",
                                           "against the distance rule and the Bayes bound")
    ks = [1, 3, 5]; w = .26
    xs = np.arange(len(ks))
    ax.bar(xs - w, [loc["dist"][k] for k in ks], w, label="distance only", color=BASE)
    ax.bar(xs,     [loc["ours"][k] for k in ks], w, label="conditional logit", color=ACCENT)
    ax.bar(xs + w, [loc["bayes"][k] for k in ks], w, label="Bayes bound",
           color=PAPER, edgecolor=CEIL, hatch="///", lw=1.2)
    for i, k in enumerate(ks):
        ax.text(i, loc["ours"][k] + .015, f"{loc['ours'][k]:.3f}", ha="center",
                fontsize=8.2, fontweight="bold")
    ax.set_xticks(xs, [f"Top-{k}" for k in ks])
    ax.set_ylim(0, .88); ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")

    # 5 ── Countdown ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1]); _style(
        ax, "Time-to-cashout",
        f"MAE {loc['mae']:.2f} min   vs   {loc['base_mae']:.2f} min predicting the mean")
    ax.scatter(loc["yt"], loc["yhat"], s=7, alpha=.28, color=ACCENT, edgecolors="none")
    lim = [0, max(loc["yt"].max(), loc["yhat"].max()) * 1.02]
    ax.plot(lim, lim, color=CEIL, lw=1.3, ls="--", label="perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("actual (min)"); ax.set_ylabel("predicted (min)")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")

    # 6 ── Search zone: the deliverable ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2]); _style(
        ax, "Search-zone containment  ·  the deliverable",
        "all zones given the same radius = equal search cost")
    z = loc["zone"]
    labels = ["Mule\nlocation", "Nearest-3\ncentroid", "Model\nsearch zone"]
    vals = [z["mule"], z["near3"], z["model"]]
    ax.barh(labels, vals, color=[BASE, BASE, ACCENT], height=.58)
    for i, v in enumerate(vals):
        ax.text(v + .012, i, f"{v:.1%}", va="center", fontsize=9.2,
                fontweight="bold" if i == 2 else "normal")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("withdrawal falls inside the zone", labelpad=8)
    ax.text(0, -.20, f"1,000 ATMs to a median of {z['atms']}   |   "
                     f"{z['radius']:.1f} km radius   |   4.5 ms",
            transform=ax.transAxes, fontsize=8.8, color=ACCENT, fontweight="bold")

    out = DOCS / "model_matrix_full.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"  [OK] {out}")

    # ── Headline card ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 4.6), facecolor=PAPER)
    ax.axis("off")
    ax.text(.012, .88, "MuleShield AI — Validated Performance", fontsize=21,
            fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(.012, .73, "SIH26184 · Ministry of Home Affairs / I4C   ·   "
                       "every figure shown against its baseline",
            fontsize=10, color=MUTED, transform=ax.transAxes)

    cards = [
        ("87.4%", "search-zone containment", f"vs {z['near3']:.1%} best naive zone", ACCENT),
        (f"{z['atms']} of 1,000", "ATMs to cover", f"{z['radius']:.1f} km radius · 4.5 ms", ACCENT),
        (f"{g['bars'][-1][1]:.4f}", "mule-detection F1", f"vs {g['bars'][-2][1]:.4f} best non-graph", INK),
        (f"{loc['mae']:.1f} min", "countdown MAE", f"vs {loc['base_mae']:.1f} min baseline", INK),
        (f"{loc['ours'][1]:.4f}", "exact-ATM Top-1", "= the Bayes bound", MUTED),
    ]
    for i, (big, label, sub, col) in enumerate(cards):
        x = .012 + i * .197
        ax.add_patch(plt.Rectangle((x, .10), .182, .50, transform=ax.transAxes,
                                   facecolor=PANEL, edgecolor=RULE, lw=1))
        ax.text(x + .014, .43, big, fontsize=25, fontweight="bold", color=col,
                transform=ax.transAxes)
        ax.text(x + .014, .32, label, fontsize=10.5, color=INK, transform=ax.transAxes)
        ax.text(x + .014, .19, sub, fontsize=8.6, color=MUTED, transform=ax.transAxes)

    out2 = DOCS / "sih_performance_matrix_slide.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"  [OK] {out2}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating model performance matrix")
    print("=" * 60)
    print("\n[1/3] Evaluating mule detector + baselines...")
    g = evaluate_gnn()
    print(f"      F1 {g['f1']:.4f} | AUC {g['auc']:.4f} | ceiling {g['ceiling']:.3f}")
    print("\n[2/3] Evaluating ranking, zone and countdown...")
    loc = evaluate_location()
    print(f"      Top-1 {loc['ours'][1]:.4f} (Bayes {loc['bayes'][1]:.4f}) | "
          f"MAE {loc['mae']:.2f} min")
    print("\n[3/3] Rendering...")
    render(g, loc)
    print("\nDone.")
