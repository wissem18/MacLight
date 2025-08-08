# env/wrap/weather_perturb.py
import traci

class WeatherPerturb:
    """
    Simple rain episode: applies (speedFactor=0.8, minGap=3.0 m)
    between `start` and `end` seconds.

    Parameters
    ----------
    env        : sumo_rl parallel_env
    seconds    : total simulation length
    start      : rain start time [s]
    end        : rain end time   [s]  (end<0 ⇒ until sim-end)
    """
    def __init__(self, env, *, seconds, start, end):
        self.env, self.end_time = env, seconds
        self.start, self.end = start, end
        self.speed = 0.8      # from Koenders et al. 2020
        self.min_gap = 3.0
        self.t = 0
        self.rain = False
        self.possible_agents = env.possible_agents
        self.step_len = 5       # env uses 5-s step

    # ----------------------------------------------------------
    def reset(self, seed=None):
        self.t, self.rain = 0, False
        return self.env.reset(seed=seed) if seed else self.env.reset()

    # ----------------------------------------------------------
    def _apply_rain(self):
        for vid in traci.vehicle.getIDList():
            traci.vehicle.setSpeedFactor(vid, self.speed)
            traci.vehicle.setMinGap(vid, self.min_gap)
            
    def _clear_rain(self):
        for vid in traci.vehicle.getIDList():
            traci.vehicle.setSpeedFactor(vid, 1.0)
            traci.vehicle.setMinGap(vid, 2.5)       

    # ----------------------------------------------------------
    def step(self, action):
        # advance sim first
        state, reward, done, trunc, info = self.env.step(action)
        self.t += self.step_len
        # decide whether to apply rainyweather
        in_rain = self.start <= self.t < (
            self.end if self.end > 0 else self.end_time)

        if in_rain:
            self._apply_rain();  self.rain = True
        else:
            self._clear_rain();  self.rain = False

        if self.t >= self.end_time:
            done = {a: True for a in self.possible_agents}

        return state, reward, done, trunc, info

    def close(self): self.env.close()
