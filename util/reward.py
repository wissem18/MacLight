def _acc_wait_total(ts):
    # Same base metric as SUMO-RL's default (veh·s scaled by 1/100).
    return sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0

# ---------- COMPONENTS ----------
def relative_growth(w_t: float, w_tm1: float, eps: float = 1e-6) -> float:
    """
    g_t = (w_t - w_{t-1}) / (w_{t-1} + eps).
    Only keep the undesired (positive) part: max(g_t, 0).
    """
    g = (w_t - w_tm1) / (w_tm1 + eps)
    return g if g > 0.0 else 0.0

def size_penalty(g_t: float) -> float:
    """
    Map g_t >= 0 smoothly to [0,1): g / (1 + g).
    0 if flat; approaches 1 as growth gets large.
    """
    return g_t / (1.0 + g_t) if g_t > 0.0 else 0.0

def accel_penalty(g_t: float, g_tm1: float, eps: float = 1e-6) -> float:
    """
    Penalize only if growth accelerates (g_t > g_{t-1}).
    Bounded, scale-free: (g_t - g_{t-1}) / (|g_t| + |g_{t-1}| + eps).
    """
    if g_t <= g_tm1:
        return 0.0
    return (g_t - g_tm1) / (abs(g_t) + abs(g_tm1) + eps)  # in [0,1]

# ---------- WRAPPER ----------
def reward_neg_only_rel_growth(ts) -> float:
    """
    Final reward (negative-only, bounded in [-1, 0]):
      R_t = -0.5 * ( size_penalty(g_t) + accel_penalty(g_t, g_{t-1}) )
    where g_t is the positive relative growth of cumulative waiting.
    """
    # Snapshots
    w_t   = _acc_wait_total(ts)
    w_tm1 = getattr(ts, "_w_tm1", w_t)

    # Relative growth today & yesterday (warm start: equal -> neutral)
    g_t   = relative_growth(w_t, w_tm1)
    g_tm1 = getattr(ts, "_g_tm1", g_t)

    # Penalties
    p_size = size_penalty(g_t)               # ∈ [0,1)
    p_acc  = accel_penalty(g_t, g_tm1)       # ∈ [0,1]

    # Final negative-only reward
    R = -0.5 * (p_size + p_acc)              # ∈ [-1, 0]

    # Stash for next step
    ts._w_tm1 = w_t
    ts._g_tm1 = g_t
    return R