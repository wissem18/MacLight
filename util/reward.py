import math
def _acc_wait_total(ts):
    # Same base metric as SUMO-RL's default (veh·s scaled by 1/100).
    return sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0

# ---------- WRAPPER ----------
def simple_reward(ts) -> float:
    # Snapshots
    w_t   = _acc_wait_total(ts)
    w_tm1 = getattr(ts, "_w_tm1", w_t)  
    # Compute reward 
    r= 1 if w_t <= w_tm1 else -1 
    # Stash for next step
    ts._w_tm1 = w_t
    return r

def composite_reward(ts)-> float:
    # Snapshots
    w_t   = _acc_wait_total(ts)
    w_tm1 = getattr(ts, "_w_tm1", w_t)
    s_t = ts.get_average_speed()
    s_tm1 = getattr(ts, "_s_tm1", s_t)
    q_t   = ts.get_total_queued()
    q_tm1 = getattr(ts, "_q_tm1", q_t) 
    # Compute reward
    w = 1 if w_t<=w_tm1 else -1
    s = 1 if s_t<=s_tm1 else -1
    q = 1 if q_t<=q_tm1 else -1
    r = (w+s+q)*(1/3)
    # Stash for next step
    ts._w_tm1 = w_t
    ts._s_tm1 = s_t
    ts._q_tm1 = q_t
    return r

def exp_reward(ts)->float:
    # Snapshots
    w_t   = _acc_wait_total(ts)
    w_tm1 = getattr(ts, "_w_tm1", w_t)  
    # Compute reward
    d = w_t-w_tm1
    r=1-math.exp(d) if d<=0 else -1+math.exp(-d)
    # Stash for next step
    ts._w_tm1 = w_t
    return r