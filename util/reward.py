"""
Custom reward functions.
"""

def _acc_wait_total(ts):
    # same source metric as the default diff-wait reward (veh·s scaled)
    return sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0

def _veh_count(ts):
    # vehicles currently on incoming lanes
    return sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in ts.lanes)

def trend_reward(ts):
    """
    Negative-only anchor reward (no trend term), parameter-free and bounded in [-1, 0].
    Objective: shrink the step height ΔW towards 0 every control step.

    R_t = - d_t / (d_{t-1} + d_t)
    where d_t = max(ΔW_t, 0) normalized per-vehicle.
    """
    W_t   = _acc_wait_total(ts)
    W_tm1 = getattr(ts, "_W_tm1", W_t)

    # per-vehicle increment this step (ΔW >= 0)
    n_t = _veh_count(ts)
    d_t = max(W_t - W_tm1, 0.0) / max(n_t, 1)

    # previous step increment (warm start: equal -> R = -0.5 on first real change,
    # or exactly 0 if both are 0)
    d_tm1 = getattr(ts, "_d_tm1", d_t)

    denom = d_tm1 + d_t
    R = 0.0 if denom == 0.0 else - (d_t / denom)   # in [-1, 0]

    # stash for next call
    ts._W_tm1 = W_t
    ts._d_tm1 = d_t
    return R

def reward_neg_trend_increment(ts):
    # snapshots & per-vehicle increments
    W_t   = _acc_wait_total(ts)
    W_tm1 = getattr(ts, "_W_tm1", W_t)

    a = max(W_t - W_tm1, 0.0)                   # ΔW_t 
    b = getattr(ts, "_d_tm1", b)                # ΔW_{t-1} 

    ts._W_tm1 = W_t
    ts._d_tm1 = a

    # special cases
    if b == 0.0:
        return -1.0 if a > 0.0 else 0.0

    q = a / (b + 1e-6)                          # ratio of increments
    h = (1.0 - q) / (1.0 + q)                   # in [-1,1]
    return min(0.0, h)                          # only penalize worsening
