# rewards.py
# Negative-only, PPO-friendly reward in [-1, 0]
# Goal: squeeze the per-step growth of cumulative waiting (ΔW per vehicle) toward 0.

# ---------- basics ----------
def _acc_wait_total(ts):
    # Same metric SUMO-RL's default uses (veh·s scaled by 1/100)
    return sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0

def _veh_count(ts):
    # Vehicles currently on incoming lanes
    return sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in ts.lanes)

# ---------- guard components (each returns a penalty in [0,1]) ----------
def size_guard(d_t: float, m_t: float) -> float:
    """
    Absolute-size penalty: pushes the step height toward 0.
    d_t:   per-vehicle increment this step (ΔW_t / n_t, ≥ 0)
    m_t:   running max of d_t within the episode (≥ 0)
    Returns in [0,1]; 0 if flat, closer to 1 when d_t is large vs its running scale.
    """
    return 0.0 if (d_t + m_t) == 0.0 else d_t / (d_t + m_t)

def acc_guard(d_t: float, d_tm1: float) -> float:
    """
    Acceleration penalty: punishes only if this step grew vs last step.
    Returns in [0,1]; 0 if d_t <= d_tm1, increases as d_t surpasses d_tm1.
    """
    denom = d_t + d_tm1
    if denom == 0.0:
        return 0.0
    return max(0.0, d_t - d_tm1) / denom

# ---------- wrapper: combine guards into final reward ----------
def reward_neg_only_flatten(ts) -> float:
    """
    Final reward in [-1, 0]:
      R_t = -0.5 * ( size_guard + acc_guard )
    No positive rewards: improvements get 0, worsening yields a bounded penalty.
    """
    # Snapshots
    W_t   = _acc_wait_total(ts)
    W_tm1 = getattr(ts, "_W_tm1", W_t)

    # Per-vehicle increment this step (ΔW >= 0)
    n_t = _veh_count(ts)
    d_t = max(W_t - W_tm1, 0.0) / max(n_t, 1)

    # History (warm start = neutral)
    d_tm1 = getattr(ts, "_d_tm1", d_t)
    m_tm1 = getattr(ts, "_d_max", d_t)

    # Running max for self-normalization
    m_t = max(m_tm1, d_t)

    # Guard components
    g_size = size_guard(d_t, m_t)       # ∈ [0,1]
    g_acc  = acc_guard(d_t, d_tm1)      # ∈ [0,1]

    # Final negative-only reward (bounded)
    R = -0.5 * (g_size + g_acc)         # ∈ [-1, 0]

    # Stash for next step
    ts._W_tm1 = W_t
    ts._d_tm1 = d_t
    ts._d_max = m_t
    return R
