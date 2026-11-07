"""Blend a known horizontal overlap between two aligned camera frames."""
import numpy as np


def blend(left, right, overlap):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim not in (2, 3) or right.ndim != left.ndim:
        raise ValueError('Expected two grayscale or channel-last images')
    if not left.size or not right.size or left.shape[0] != right.shape[0] or left.shape[2:] != right.shape[2:]:
        raise ValueError('Image heights and channels must match')
    if left.dtype.kind not in 'biuf' or right.dtype.kind not in 'biuf' or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Expected finite real pixels')
    if isinstance(overlap, bool) or not isinstance(overlap, (int, np.integer)) or not 1 <= overlap <= min(left.shape[1], right.shape[1]):
        raise ValueError('Overlap must be a positive integer within both widths')
    # Interior weights keep both cameras represented even for a one-pixel seam.
    weights = np.arange(1, overlap + 1, dtype=float) / (overlap + 1)
    weights = weights.reshape((1, overlap) + (1,) * (left.ndim - 2))
    seam = left[:, -overlap:] * (1 - weights) + right[:, :overlap] * weights
    return np.concatenate((left[:, :-overlap], seam, right[:, overlap:]), axis=1)
