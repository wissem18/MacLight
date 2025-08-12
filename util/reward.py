"""
Custom reward functions.
"""
def trend_reward(ts):
    # total accumulated waiting time snapshot (veh·s), same source as default
    W_t = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0

    # previous snapshot
    W_prev = getattr(ts, "_W_prev", W_t)

    # increments (non-negative by definition)
    a = getattr(ts, "_d_prev", W_prev - getattr(ts, "_W_prevprev", W_prev))  # ΔW_{t-1}
    b = W_t - W_prev                                                         # ΔW_t

    # reward in [-1,1]
    denom = a + b
    R = 0.0 if denom == 0.0 else (a - b) / denom

    # stash for next call
    ts._W_prevprev = W_prev
    ts._W_prev = W_t
    ts._d_prev = b
    return R
