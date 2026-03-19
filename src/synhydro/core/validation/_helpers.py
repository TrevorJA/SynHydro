"""
Shared helpers for ensemble validation metric computation.
"""

import numpy as np

_VALID_METRICS = frozenset(
    {
        "marginal",
        "temporal",
        "spatial",
        "drought",
        "spectral",
        "seasonal",
        "annual",
        "fdc",
        "lmoments",
        "extremes",
    }
)


def _skewness(x: np.ndarray) -> float:
    """
    Compute sample skewness (G1 formula).

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    float
        Sample skewness, or nan if n < 3 or variance is effectively zero.
    """
    n = len(x)
    if n < 3:
        return np.nan
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return 0.0
    return float(n / ((n - 1) * (n - 2)) * np.sum(((x - m) / s) ** 3))


def _extract_droughts(
    flows: np.ndarray, threshold: float
) -> tuple[list[int], list[float]]:
    """
    Extract drought events (consecutive below-threshold periods).

    Parameters
    ----------
    flows : np.ndarray
        Flow values.
    threshold : float
        Drought threshold.

    Returns
    -------
    durations : list of int
        Duration of each drought event in timesteps.
    severities : list of float
        Cumulative deficit of each drought event.
    """
    below = flows < threshold
    padded = np.concatenate(([False], below, [False]))
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0]

    durations = (ends - starts).tolist()
    severities = [float(np.sum(threshold - flows[s:e])) for s, e in zip(starts, ends)]
    return durations, severities


def _metric_entry(
    obs_val: float,
    syn_vals: list[float],
) -> dict | None:
    """
    Build a standardized metric comparison dict.

    Parameters
    ----------
    obs_val : float
        Observed statistic.
    syn_vals : list of float
        Per-realization synthetic values.

    Returns
    -------
    dict or None
        Keys: observed, synthetic_median, synthetic_p10, synthetic_p90,
        relative_error. Returns None if no finite synthetic values exist.
    """
    arr = np.array(syn_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    median = float(np.median(arr))
    rel_err = (median - obs_val) / abs(obs_val) if abs(obs_val) > 1e-10 else np.nan
    return {
        "observed": float(obs_val),
        "synthetic_median": median,
        "synthetic_p10": float(np.percentile(arr, 10)),
        "synthetic_p90": float(np.percentile(arr, 90)),
        "relative_error": float(rel_err),
    }
