"""Dependence-aware paired statistics on contingency-count arrays.

The statistical unit is the source cluster, not the item: paraphrases of one census cell
are not independent draws. Both procedures here resample or swap whole clusters, and both
carry every seed and both methods together on the same sampled clusters so the comparison
stays paired. They operate on (n_clusters x 8) count matrices produced by
common.counts_by_cluster, never on record lists, so ten thousand draws take seconds.

Bootstrap  -> interval for the mean paired difference in B (plan section 7.2)
Permutation-> p-value under cluster-wise exchangeability of the two methods' prediction
              blocks, Monte Carlo with (extreme + 1)/(draws + 1) (never an exact zero)
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

# column indices in a Counts.as_array() vector
N_DIFF, CORRECT_DIFF, N_EQUAL, CORRECT_EQUAL = 0, 1, 2, 3


def b_from_sums(s: np.ndarray) -> np.ndarray:
    """Vectorised harmonic B from summed counts, shape (..., 8) -> (...).
    NaN where a condition has no items; 0 where both accuracies are 0."""
    nd, cd, ne, ce = s[..., N_DIFF], s[..., CORRECT_DIFF], s[..., N_EQUAL], s[..., CORRECT_EQUAL]
    with np.errstate(divide="ignore", invalid="ignore"):
        ad = np.where(nd > 0, cd / np.maximum(nd, 1), np.nan)
        ae = np.where(ne > 0, ce / np.maximum(ne, 1), np.nan)
        denom = ad + ae
        b = np.where(denom > 0, 2 * ad * ae / np.where(denom > 0, denom, 1), 0.0)
    b = np.where(np.isnan(ad) | np.isnan(ae), np.nan, b)
    return b


def observed_difference(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    """B(a) - B(b) on the full sample, averaged over seeds if a 3-d (seed, cluster, 8) array."""
    a = np.asarray(counts_a); b = np.asarray(counts_b)
    if a.ndim == 2:
        a = a[None]; b = b[None]
    diffs = b_from_sums(a.sum(axis=1)) - b_from_sums(b.sum(axis=1))
    return float(np.nanmean(diffs))


def paired_cluster_bootstrap(counts_a: np.ndarray, counts_b: np.ndarray, draws: int, seed: int,
                             alpha: float = 0.05) -> Dict[str, float]:
    """Sample clusters with replacement; every seed and both methods use the same draw.

    counts_a / counts_b: (n_clusters, 8) or (n_seeds, n_clusters, 8), aligned by cluster.
    Multiplicity is preserved through bincount weights, so a cluster drawn twice counts twice.
    """
    a = np.asarray(counts_a, dtype=np.float64); b = np.asarray(counts_b, dtype=np.float64)
    if a.ndim == 2:
        a = a[None]; b = b[None]
    n_seeds, n_clusters, _ = a.shape
    if n_clusters < 2:
        return {"lower": float("nan"), "upper": float("nan"), "mean": float("nan"),
                "n_clusters": n_clusters, "draws": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(draws, n_clusters))
    # weights[d, c] = how many times cluster c was drawn in draw d
    weights = np.zeros((draws, n_clusters), dtype=np.float64)
    for d in range(draws):
        weights[d] = np.bincount(idx[d], minlength=n_clusters)
    # sums: (draws, n_seeds, 8) = weights (draws, C) @ counts (seed, C, 8)
    sums_a = np.einsum("dc,scm->dsm", weights, a)
    sums_b = np.einsum("dc,scm->dsm", weights, b)
    diff = np.nanmean(b_from_sums(sums_a) - b_from_sums(sums_b), axis=1)  # (draws,)
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return {"lower": float("nan"), "upper": float("nan"), "mean": float("nan"),
                "n_clusters": n_clusters, "draws": 0}
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lower": float(lo), "upper": float(hi), "mean": float(diff.mean()),
            "n_clusters": int(n_clusters), "draws": int(diff.size)}


def cluster_swap_permutation(counts_a: np.ndarray, counts_b: np.ndarray, draws: int, seed: int) -> Dict[str, float]:
    """Two-sided paired cluster-swap permutation test on |mean B(a) - B(b)|.

    Null: within each source cluster the two methods' complete prediction blocks are
    exchangeable. Each permutation draws one swap flag per cluster and applies the same
    flag to every seed. This is not a permutation of item identities. p uses
    (extreme + 1)/(draws + 1), so it is never reported as exactly zero.
    """
    a = np.asarray(counts_a, dtype=np.float64); b = np.asarray(counts_b, dtype=np.float64)
    if a.ndim == 2:
        a = a[None]; b = b[None]
    n_seeds, n_clusters, _ = a.shape
    obs = abs(observed_difference(a, b))
    # One cluster is degenerate but well defined: every swap reproduces |obs|, so p is 1.
    # Only an undefined statistic (a condition absent from the sample) yields NaN.
    if n_clusters < 1 or obs != obs:
        return {"observed_abs_difference": obs, "p_value": float("nan"), "draws": 0, "n_clusters": n_clusters}
    rng = np.random.default_rng(seed)
    flags = rng.integers(0, 2, size=(draws, n_clusters)).astype(np.float64)  # 1 = swap
    keep = 1.0 - flags
    # swapped A = keep*A + flag*B ; swapped B = keep*B + flag*A, summed over clusters
    sa = np.einsum("dc,scm->dsm", keep, a) + np.einsum("dc,scm->dsm", flags, b)
    sb = np.einsum("dc,scm->dsm", keep, b) + np.einsum("dc,scm->dsm", flags, a)
    stat = np.abs(np.nanmean(b_from_sums(sa) - b_from_sums(sb), axis=1))
    stat = stat[~np.isnan(stat)]
    extreme = int(np.sum(stat >= obs - 1e-12))
    return {"observed_abs_difference": float(obs), "p_value": (extreme + 1) / (stat.size + 1),
            "draws": int(stat.size), "n_clusters": int(n_clusters)}


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down, order preserved; NaN inputs stay NaN and do not count."""
    ps = np.asarray(p_values, dtype=np.float64)
    valid = ~np.isnan(ps)
    out = np.full_like(ps, np.nan)
    if valid.sum() == 0:
        return out.tolist()
    idx = np.where(valid)[0]
    order = idx[np.argsort(ps[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * ps[i])
        running = max(running, adj)
        out[i] = running
    return out.tolist()
