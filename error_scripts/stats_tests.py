"""
stats_tests.py — statistical tools for the proposal's Analysis section.

Important: the resampling unit is the dialogue (cluster), not the slot. Slots in one dialogue share user, noise clip and context,
so resampling slots makes confidence intervals too narrow.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def cluster_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], cluster: str = "dialogue_id",
                      B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Dialogue-level bootstrap 95% interval for any statistic (ECE, AUROC, wrong-commit rate, ...)."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    vals = []
    for _ in range(B):
        pick = rng.choice(keys, size=len(keys), replace=True)
        vals.append(stat(pd.concat([groups[k] for k in pick], ignore_index=True)))
    return stat(df), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, B: int = 10000, seed: int = 0):
    """a, b = per-dialogue metric of two policies on the same dialogues (aligned order). Returns mean difference and 95% CI."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_permutation(a: np.ndarray, b: np.ndarray, B: int = 10000, seed: int = 0) -> float:
    """Paired sign-flip permutation test (two-sided). H0: the two policies are exchangeable."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    obs = abs(d.mean())
    signs = rng.choice([-1, 1], size=(B, len(d)))
    null = np.abs((signs * d).mean(1))
    return float((1 + (null >= obs).sum()) / (B + 1))


def holm(pvals: dict[str, float], alpha: float = .05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values and reject decisions."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "reject": adj < alpha}
    return out
