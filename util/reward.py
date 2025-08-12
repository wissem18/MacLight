"""
Custom reward functions.
"""
def mean_acc_wait(ts):
    # total accumulated waiting time over incoming lanes (veh·s), scaled like the lib
    total_acc = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0

    # vehicles currently on incoming lanes
    veh_count = sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in ts.lanes)

    if veh_count == 0:
        return 0.0
    return total_acc / veh_count

def trend_reward(ts):
    """
    trend reward based only on average waiting time.

    Args
    ----
    ts : sumo_rl.environment.traffic_signal.TrafficSignal
        The traffic-signal wrapper that called the reward.

    Returns
    -------
    float
        +ve if waiting-time dropped; -ve if it increased; 0 if unchanged.
    """
    # 1. current average waiting time
    cur = mean_acc_wait(ts)

    # 2. previous one stored as an ad-hoc attribute
    prev = getattr(ts, "_prev_wait", cur)   # fallback=cur on first call

    # 3. relative change  ——  eps=1e-6 avoids division by zero
    eps = 1e-6
    ratio = (cur + eps) / (prev + eps)

    # 4. remember for next step
    ts._prev_wait = cur

    # 5. trend formula
    return (1.0-ratio) / (1.0+ratio)