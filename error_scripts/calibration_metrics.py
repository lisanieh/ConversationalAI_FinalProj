"""
calibration_metrics.py — measures "estimate error": how far p_hat is from the true error rate (RQ2 / RQ4).

All functions take:  y in {0,1} (1 = slot is wrong),  p in [0,1] (predicted error probability p_hat)
Report the two qualities separately:
    calibration    : when it says 30%, are about 30% really wrong?   -> ECE, MCE, reliability diagram, Brier reliability term
    discrimination : do wrong slots get a higher p_hat than correct ones? -> AUROC, AUPRC, Brier resolution term
U2R needs calibration, because the formula multiplies p_hat by costs directly; good ranking alone (high AUROC) is not enough.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _bins(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:  # "quantile": equal-mass bins, more stable for skewed p
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -1e-12, 1 + 1e-12
    return np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)


def reliability_table(y, p, n_bins: int = 15, strategy: str = "uniform"):
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = _bins(p, n_bins, strategy)
    rows = []
    for k in np.unique(b):
        m = b == k
        rows.append({"bin": int(k), "n": int(m.sum()), "conf": float(p[m].mean()),
                     "acc": float(y[m].mean())})  # acc = observed error rate in the bin
    return rows


def ece(y, p, n_bins: int = 15, strategy: str = "uniform") -> float:
    """Expected Calibration Error = Σ_b (n_b/N) |mean(y_b) − mean(p_b)|。
    Always state n_bins and the binning; with few samples (<1000) 15-bin ECE has a clear upward bias, so report it with a bootstrap CI."""
    rows = reliability_table(y, p, n_bins, strategy)
    N = sum(r["n"] for r in rows)
    return float(sum(r["n"] / N * abs(r["acc"] - r["conf"]) for r in rows))


def mce(y, p, n_bins: int = 15, strategy: str = "uniform") -> float:
    return float(max(abs(r["acc"] - r["conf"]) for r in reliability_table(y, p, n_bins, strategy)))


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def brier_decomposition(y, p, n_bins: int = 15) -> dict:
    """Murphy decomposition (binned approximation): Brier ~ reliability - resolution + uncertainty.
    reliability: lower is better (calibration); resolution: higher is better (discrimination); uncertainty = ybar(1-ybar), model-independent."""
    y = np.asarray(y, float)
    rows = reliability_table(y, p, n_bins)
    N, ybar = len(y), y.mean()
    rel = sum(r["n"] * (r["conf"] - r["acc"]) ** 2 for r in rows) / N
    res = sum(r["n"] * (r["acc"] - ybar) ** 2 for r in rows) / N
    return {"brier": brier(y, p), "reliability": rel, "resolution": res, "uncertainty": ybar * (1 - ybar)}


def nll(y, p) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def discrimination(y, p) -> dict:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "base_rate": float(np.mean(y))}
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "base_rate": float(np.mean(y))}  # read AUPRC against base_rate


def full_report(y, p, n_bins: int = 15) -> dict:
    out = {"n": int(len(y)), "ece": ece(y, p, n_bins), "ece_quantile": ece(y, p, n_bins, "quantile"),
           "mce": mce(y, p, n_bins), "nll": nll(y, p)}
    out.update(brier_decomposition(y, p, n_bins))
    out.update(discrimination(y, p))
    return out


def per_error_type_auroc(y, p, error_type) -> dict:
    """RQ2 AUROC per error type: positives = one error type, negatives = all correct slots, other error types excluded.
    Shows whether the estimator is blind to e.g. deletions (expected to be weakest, cf. Peng et al. 2026)."""
    y, p, et = np.asarray(y), np.asarray(p), np.asarray(error_type)
    res = {}
    for t in sorted(set(et) - {"correct"}):
        m = (et == t) | (et == "correct")
        yy = (et[m] == t).astype(int)
        res[t] = {"n_pos": int(yy.sum()),
                  "auroc": float(roc_auc_score(yy, p[m])) if 0 < yy.sum() < len(yy) else float("nan")}
    return res


def plot_reliability(y, p, path: str, n_bins: int = 15, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = reliability_table(y, p, n_bins)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.5, 5.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    a1.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
    a1.plot([r["conf"] for r in rows], [r["acc"] for r in rows], "o-", c="#2a6fdb", label="estimator")
    a1.set_ylabel("observed error rate")
    a1.set_title(title or f"Reliability (ECE={ece(y, p, n_bins):.3f})")
    a1.legend(frameon=False, fontsize=8)
    a2.hist(p, bins=n_bins, range=(0, 1), color="#2a6fdb", alpha=.6)
    a2.set_xlabel("predicted p_hat (slot is wrong)")
    a2.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
