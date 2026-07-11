"""Hellinger-affinity registration-quality metric for the benchmark.

NCC assumes a linear intensity relationship (fine same-modality, degenerate cross-modal
e.g. fMRI->T1). This metric instead measures the statistical DEPENDENCE between
spatially-corresponding voxels — the Hellinger distance between the joint 2D intensity
histogram P(a,b) and the product of its marginals P(a)P(b) (independence). It is high
when the two aligned images are strongly dependent (well registered) and ~0 when they
are independent (misregistered), for ANY intensity relationship. Same family as
allineate's default Hellinger cost.

Range [0,1]; higher = better alignment.
"""
import numpy as np


def hellinger_distance(p, q):
    """Hellinger distance between two probability vectors (user-provided formula)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def hellinger_quality(a, b, mask=None, bins=64):
    """Registration quality = Hellinger distance of the joint histogram from independence.

    a, b: equal-shaped intensity arrays already on the same grid (registered output and
    template). mask: optional boolean array selecting voxels to score (e.g. brain mask).
    Returns a scalar in [0,1]; higher means better-aligned (more dependent).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if mask is not None:
        a = a[mask]; b = b[mask]
    else:
        a = a.ravel(); b = b.ravel()
    if a.size < 16:
        return 0.0
    H, _, _ = np.histogram2d(a, b, bins=bins)
    tot = H.sum()
    if tot <= 0:
        return 0.0
    P = H / tot                      # joint distribution
    pa = P.sum(axis=1)               # marginals
    pb = P.sum(axis=0)
    Q = np.outer(pa, pb)             # product of marginals = independence
    return hellinger_distance(P.ravel(), Q.ravel())
