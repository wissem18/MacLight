from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Dict
import numpy as np
import pandas as pd

# ------------------------- Windows -------------------------------------------

@dataclass(frozen=True)
class EvalWindows:
    """Evaluation windows for resilience analysis.

    Semantics:
        pre   = [pre_start, t0)
        shock = [t0, t1]
        post  = (t1, post_end]

    All times must be in the same units as the time index of your series
    (e.g., SUMO seconds or step indices).

    Fields:
        pre_start: float
            Start time of the pre-shock window.
        t0: float
            Shock start (also acts as pre_end).
        t1: float
            Shock end (also acts as post_start).
        post_end: float
            End time of the post-recovery evaluation window.
    """
    pre_start: float
    t0: float
    t1: float
    post_end: float

    @property
    def pre(self) -> Tuple[float, float]:
        """Tuple form of the pre window [pre_start, t0)."""
        return (self.pre_start, self.t0)

    @property
    def post(self) -> Tuple[float, float]:
        """Tuple form of the post window (t1, post_end]."""
        return (self.t1, self.post_end)


# ------------------------- Helpers -------------------------------------------

def _ensure_series(x: pd.Series) -> pd.Series:
    """Ensure a pandas Series with a monotonic increasing time index and float dtype.

    Raises:
        TypeError if x is not a Series.

    Returns:
        Series sorted by index and cast to float.
    """
    if not isinstance(x, pd.Series):
        raise TypeError("Expected a pandas Series with time index.")
    if not x.index.is_monotonic_increasing:
        x = x.sort_index()
    return x.astype(float)

def _slice_by_window(s: pd.Series, window: Tuple[float, float]) -> pd.Series:
    """Slice a series by a half-open window [a, b)."""
    a, b = window
    return s.loc[(s.index >= a) & (s.index < b)]

def _trapz(t: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal integral as a float."""
    return float(np.trapz(y, t))


# ------------------------- Normalization -> P(t) ------------------------------
# Hard rule:
#   - higher_is_better -> MINMAX (calibrated on pre window, may go outside [0,1] later)
#   - lower_is_better  -> INV1P  (bounded, smooth for heavy tails)

def to_performance(
    s: pd.Series,
    *,
    higher_is_better: bool,
    ref_window: Optional[Tuple[float, float]] = None,
    allow_clip: bool = False,
    eps: float = 1e-9,
) -> pd.Series:
    """Convert a raw metric series into a performance curve P(t) where higher is better.

    Rationale:
        Resilience metrics assume a "higher-is-better" performance signal P(t).
        This function standardizes any raw signal accordingly, using the pre-shock
        window for normalization so different runs are comparable.

    Method (fixed by direction):
        - If higher_is_better=True (e.g., average speed, throughput):
            Use pre-window MINMAX scaling:
                P(t) = (x(t) - min_pre) / (max_pre - min_pre)
            Note: outside the pre window, P(t) can be < 0 or > 1. This is informative
            (e.g., extreme shock or overshoot). Set allow_clip=True if you need [0,1].
        - If higher_is_better=False (e.g., queue, waiting time, delay):
            Use INV1P relative to pre-window mean:
                P(t) = 1 / (1 + x(t) / mean_pre)
            Always in (0, 1), robust to heavy tails.

    Args:
        s: Series
            Raw metric indexed by time (monotonic increasing).
        higher_is_better: bool
            Direction of desirability for the raw metric.
        ref_window: (float, float) or None
            Window used to compute reference stats (min/max or mean).
            Recommended: use the pre window [pre_start, t0).
        allow_clip: bool
            If True, clip P into [0,1]. Defaults to False (preserve excursions).
        eps: float
            Numerical safeguard against division by zero.

    Returns:
        Series P(t), aligned to s.index.

    Edge cases:
        - Flat pre-window range for higher-is-better: returns a constant 1.0 (best).
        - Empty ref_window: falls back to using whole series as reference.
    """
    s = _ensure_series(s)
    s_ref = _slice_by_window(s, ref_window) if ref_window else s
    if len(s_ref) == 0:
        s_ref = s

    if higher_is_better:
        # MINMAX on pre window
        lo = float(np.nanmin(s_ref.values))
        hi = float(np.nanmax(s_ref.values))
        rng = hi - lo
        if rng < eps:
            P = pd.Series(1.0, index=s.index)  # degenerate: flat best
        else:
            P = (s - lo) / rng
    else:
        # INV1P on pre mean (lower is better)
        scale = float(np.nanmean(s_ref.values)) + eps
        P = 1.0 / (1.0 + (s / scale))

    if allow_clip:
        P = P.clip(0.0, 1.0)
    return P


# ------------------------- Baseline (external only) ---------------------------

def build_baseline_external(
    baseline_raw: pd.Series,
    *,
    higher_is_better: bool,
    ref_window: Tuple[float, float],
    align_index: pd.Index,
    allow_clip: bool = False,
) -> pd.Series:
    """Build the baseline performance curve B(t) from an external (no-shock) run.

    Steps:
        1) Normalize the raw baseline series with the SAME rule used for P(t)
           (same higher_is_better, same ref_window, same allow_clip).
        2) Align to the evaluation index (typically P's index over [t0, post_end]),
           interpolating in time if needed.

    Args:
        baseline_raw: Series
            Raw metric from the control (no-perturbation) experiment.
        higher_is_better: bool
            Direction, must match the perturbed run's normalization.
        ref_window: (float, float)
            The pre-shock window (same as used for P(t)).
        align_index: pd.Index
            Target time index for alignment (e.g., P.index in [t0, post_end]).
        allow_clip: bool
            If True, clip B to [0,1] after normalization.

    Returns:
        Series B(t) aligned to align_index.
    """
    baseline_raw = _ensure_series(baseline_raw)
    B_full = to_performance(
        baseline_raw,
        higher_is_better=higher_is_better,
        ref_window=ref_window,
        allow_clip=allow_clip,
    )
    B = B_full.reindex(align_index, method=None)
    if B.isna().any():
        B = B_full.reindex(B_full.index.union(align_index)).interpolate(
            method="time"
        ).reindex(align_index)
    return B


# ------------------------- Atomic Resilience Metrics --------------------------

def _pre_mean(P: pd.Series, windows: EvalWindows) -> float:
    """Mean performance during the pre-shock window [pre_start, t0)."""
    pre = _slice_by_window(P, windows.pre)
    if len(pre) == 0:
        raise ValueError("Empty pre window for P(t).")
    return float(pre.mean())

def auc_ratio(P: pd.Series, B: pd.Series, windows: EvalWindows) -> float:
    """Area-Under-Curve resilience ratio over [t0, post_end].

    Definition:
        AUC_ratio = ∫ P(t) dt / ∫ B(t) dt, both integrated from t0 to post_end.

    Interpretation:
        1.0  -> no performance loss relative to baseline.
        <1.0 -> total loss accounting for depth and duration.
        >1.0 -> improved aggregate performance (rare, possible adaptation).

    Raises:
        ValueError if the evaluation window has fewer than 2 samples.
    """
    P = _ensure_series(P); B = _ensure_series(B)
    mask = (P.index >= windows.t0) & (P.index <= windows.post_end)
    t = P.index[mask].astype(float).to_numpy()
    if len(t) < 2:
        raise ValueError("Insufficient samples in [t0, post_end] for AUC.")
    p = P[mask].to_numpy()
    b = B.reindex(P.index[mask]).to_numpy()
    return _trapz(t, p) / (_trapz(t, b) + 1e-12)

def peak_loss(P: pd.Series, B: pd.Series, windows: EvalWindows) -> tuple[float, float]:
    """Worst instantaneous loss vs baseline and when it occurs.

    Definition:
        L(t) = B(t) - P(t)
        Returns (max L(t), t_at_max) for t in [t0, post_end].

    Interpretation:
        Larger peak_loss means a more severe worst-point degradation.
    """
    mask = (P.index >= windows.t0) & (P.index <= windows.post_end)
    t = P.index[mask].astype(float).to_numpy()
    p = P[mask].to_numpy()
    b = B.reindex(P.index[mask]).to_numpy()
    L = b - p
    i = int(np.argmax(L))
    return float(L[i]), float(t[i])

def recovery_time(P: pd.Series, windows: EvalWindows, target: float = 0.9) -> Optional[float]:
    """Time from t1 until P(t) reaches target * pre_mean (or None if never).

    Definition:
        pre_mean = mean P(t) over [pre_start, t0)
        RT_target = first time τ >= t1 s.t. P(τ) >= target * pre_mean, then RT = τ - t1

    Interpretation:
        Smaller is better (faster rebound).
        None indicates the system did not recover to the target level in the horizon.
    """
    pre_mu = _pre_mean(P, windows); thr = target * pre_mu
    mask = (P.index >= windows.t1) & (P.index <= windows.post_end)
    if not mask.any():
        return None
    P_after = P[mask]
    hit = P_after.index[P_after >= thr]
    return float(hit[0] - windows.t1) if len(hit) else None

def recovery_slope(P: pd.Series, windows: EvalWindows, target: float = 0.9) -> Optional[float]:
    """Average slope during early recovery until target is first hit.

    Definition:
        Let τ be the first time >= t1 where P(τ) >= target * pre_mean.
        Slope = (P(τ) - P(t1)) / (τ - t1).

    Interpretation:
        Larger slope indicates a steeper/faster initial recovery.
        None if the target is never reached in the horizon.
    """
    pre_mu = _pre_mean(P, windows); thr = target * pre_mu
    mask = (P.index >= windows.t1) & (P.index <= windows.post_end)
    if not mask.any():
        return None
    P_after = P[mask]
    idx_hit = P_after.index[P_after >= thr]
    if len(idx_hit) == 0:
        return None
    t1 = windows.t1; thit = float(idx_hit[0])
    p1 = float(P_after.iloc[0]); phit = float(P_after.loc[idx_hit[0]])
    dt = thit - t1
    return (phit - p1) / (dt + 1e-12)

def residual_delta(P: pd.Series, windows: EvalWindows) -> float:
    """Post steady-state minus pre-shock mean.

    Definition:
        residual_delta = mean P(t) over (t1, post_end]  -  mean P(t) over [pre_start, t0)

    Interpretation:
        Negative -> lasting damage; Positive -> beneficial adaptation; ~0 -> full recovery.
    """
    pre_mu = _pre_mean(P, windows)
    post = P.loc[(P.index > windows.t1) & (P.index <= windows.post_end)]
    if len(post) == 0:
        raise ValueError("Empty post window for residual delta.")
    return float(post.mean()) - pre_mu

def overshoot(P: pd.Series, windows: EvalWindows) -> float:
    """Relative overshoot during recovery w.r.t. pre-shock mean.

    Definition:
        overshoot = (max_{t>=t1} P(t) - pre_mean) / pre_mean

    Interpretation:
        >0 indicates the controller exceeded the pre-shock mean (possible oscillation or aggressive response).
        0 if no post value exceeds the pre-shock mean.
    """
    pre_mu = _pre_mean(P, windows)
    post = P.loc[(P.index >= windows.t1) & (P.index <= windows.post_end)]
    if len(post) == 0:
        return 0.0
    return (float(post.max()) - pre_mu) / (pre_mu + 1e-12)

def settling_time(P: pd.Series, windows: EvalWindows, band: float = 0.05) -> Optional[float]:
    """Settling time: when P(t) enters a ±band around pre-shock mean and stays there.

    Definition:
        Find the earliest τ >= t1 such that for all t >= τ,  P(t) ∈ [pre_mean*(1-band), pre_mean*(1+band)].
        Return τ - t1 (seconds/steps). If never satisfied, return None.

    Interpretation:
        Smaller indicates quicker stabilization (less oscillation).
    """
    pre_mu = _pre_mean(P, windows)
    post = P.loc[(P.index >= windows.t1) & (P.index <= windows.post_end)]
    if len(post) == 0:
        return None
    lo, hi = pre_mu * (1 - band), pre_mu * (1 + band)
    times = post.index.to_numpy(dtype=float)
    vals = post.to_numpy(dtype=float)
    for k in range(len(vals)):
        if np.all((vals[k:] >= lo) & (vals[k:] <= hi)):
            return float(times[k] - windows.t1)
    return None


# ------------------------- Risk / Extras --------------------------------------

def cvar(losses: Iterable[float], alpha: float = 0.95) -> float:
    """Conditional Value-at-Risk (CVaR) for a set of scalar losses (e.g., peak_loss) across runs.

    Definition:
        Given a list of losses, CVaR_alpha is the mean of the worst (1 - alpha) tail.
        E.g., alpha=0.95 averages the worst 5% outcomes.

    Interpretation:
        Higher CVaR means worse tail behavior (less resilient in worst cases).
    """
    arr = np.sort(np.asarray(list(losses), dtype=float))
    if len(arr) == 0:
        return np.nan
    k = int(np.ceil(alpha * len(arr))) - 1
    tail = arr[k:]
    return float(tail.mean()) if len(tail) > 0 else float(arr[-1])

def price_of_resilience(baseline_nominal: float, robust_nominal: float) -> float:
    """Price of resilience on no-shock days in the same performance units.

    Definition:
        PoR = baseline_nominal - robust_nominal
        Positive means the resilient controller trades some nominal performance for robustness.

    Use:
        Compare on experiments without any perturbation.
    """
    return float(baseline_nominal - robust_nominal)


# ------------------------- One-shot aggregator --------------------------------

def compute_all(
    P: pd.Series,
    windows: EvalWindows,
    *,
    baseline_raw: pd.Series,
    higher_is_better: bool,
    target: float = 0.9,
    band: float = 0.05,
    allow_clip: bool = False,
) -> Dict[str, float]:
    """Compute the full resilience suite for one run and one performance curve P(t).

    Pipeline:
        1) Build/align baseline B(t) from the no-perturbation run using the SAME
           normalization choice (minmax for higher-is-better, inv1p otherwise)
           and the SAME pre window.
        2) Compute:
            - auc_ratio
            - peak_loss and t_peak
            - recovery_time (to target * pre_mean)
            - recovery_slope (until target is first hit)
            - residual_delta (post mean - pre mean)
            - overshoot (relative to pre mean)
            - settling_time (±band around pre mean)

    Args:
        P: Series
            Performance curve of the perturbed run (already normalized by to_performance()).
        windows: EvalWindows
            Pre, shock, and post timings.
        baseline_raw: Series
            Raw metric from the control (no-perturbation) run.
        higher_is_better: bool
            Direction for normalization; must match what produced P.
        target: float
            Recovery fraction (e.g., 0.9 means 90% of pre-shock mean).
        band: float
            Settling band around pre-shock mean (e.g., 0.05 = ±5%).
        allow_clip: bool
            If True, baseline normalization is clipped to [0,1].

    Returns:
        Dict with keys:
            'auc_ratio', 'peak_loss', 't_peak', 'rt_target',
            'recovery_slope', 'residual_delta', 'overshoot', 'settling_time'
    """
    P = _ensure_series(P)
    # Align baseline on [t0, post_end] index of P
    idx = P.index[(P.index >= windows.t0) & (P.index <= windows.post_end)]
    B = build_baseline_external(
        baseline_raw,
        higher_is_better=higher_is_better,
        ref_window=windows.pre,
        align_index=idx,
        allow_clip=allow_clip,
    )
    # Metrics
    auc = auc_ratio(P, B, windows)
    Lmax, tpeak = peak_loss(P, B, windows)
    rt = recovery_time(P, windows, target=target)
    slope = recovery_slope(P, windows, target=target)
    rdelta = residual_delta(P, windows)
    os = overshoot(P, windows)
    st = settling_time(P, windows, band=band)
    return dict(
        auc_ratio=auc,
        peak_loss=Lmax,
        t_peak=tpeak,
        rt_target=rt,
        recovery_slope=slope,
        residual_delta=rdelta,
        overshoot=os,
        settling_time=st,
    )
