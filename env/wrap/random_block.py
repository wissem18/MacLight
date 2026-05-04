import torch
import traci


class _BlockMixin:
    def _apply_blocked_edges(self):
        for edge_idx in self.rd_id:
            edge = self.blockable_edges[int(edge_idx)]
            self.original_speeds.setdefault(edge, traci.edge.getMaxSpeed(edge))
            traci.edge.setMaxSpeed(edge, 0.5)

    def _restore_blocked_edges(self):
        for edge_idx in self.rd_id:
            edge = self.blockable_edges[int(edge_idx)]
            traci.edge.setMaxSpeed(edge, self.original_speeds.get(edge, 13.89))

    @staticmethod
    def _reroute_vehicles():
        for vehicle_id in traci.vehicle.getIDList():
            traci.vehicle.rerouteTraveltime(vehicle_id)

    def _update_blocking(self):
        block_active = self.start_block <= self.time <= self.end_block
        if block_active:
            if (not self.was_blocking) or self.time % 300 == 0:
                if self.was_blocking:
                    self._restore_blocked_edges()
                self.rd_id = torch.randperm(len(self.blockable_edges))[:self.block_num]
                self._apply_blocked_edges()
                self._reroute_vehicles()
            self.was_blocking = True
        elif self.was_blocking:
            self._restore_blocked_edges()
            self._reroute_vehicles()
            self.was_blocking = False


class BlockStreet(_BlockMixin):
    def __init__(self, env, start=600, end=1800, network=None, block_num=8, seconds=3600) -> None:
        if network is None:
            block_num, seconds = start, end
            start, end, network = 600, 1800, 'ff'
        self.env = env
        self.time = 0
        self.network = network
        self.block_num = block_num
        self.start_block = start
        self.end_block = end
        self.end_time = seconds
        self.possible_agents = env.possible_agents

        if self.network == 'ff':
            self.blockable_edges = [
                'B2C2', 'B3C3', 'C1C2', 'C2B2', 'C2C1', 'C2C3',
                'C2D2', 'C3B3', 'C3C2', 'C3C4', 'C3D3', 'C4C3',
                'D1D2', 'D2C2', 'D2D1', 'D2D3', 'D2E2', 'D3C3',
                'D3D2', 'D3D4', 'D3E3', 'D4D3', 'E2D2', 'E3D3',
            ]
        else:
            self.blockable_edges = [
                "road_2_4_3", "road_2_3_1", "road_2_3_3", "road_2_2_1",
                "road_2_2_3", "road_2_1_1", "road_3_2_3", "road_3_1_1",
                "road_3_3_3", "road_3_2_1", "road_3_4_3", "road_3_3_1",
                "road_2_3_2", "road_1_3_0", "road_3_3_2", "road_2_3_0",
                "road_4_3_2", "road_3_3_0", "road_4_2_2", "road_3_2_0",
                "road_3_2_2", "road_2_2_0", "road_2_2_2", "road_1_2_0",
            ]

        self.rd_id = torch.randperm(len(self.blockable_edges))[:self.block_num]
        self.was_blocking = False
        self.original_speeds = {}

    def reset(self, seed=None):
        self.time = 0
        self.was_blocking = False
        self.original_speeds = {}
        return self.env.reset(seed=seed) if seed else self.env.reset()

    def step(self, action):
        self._update_blocking()
        self.time += 5
        next_state, reward, done, truncated, info = self.env.step(action)
        if self.time >= self.end_time:
            done = {agt: True for agt in self.possible_agents}
        return next_state, reward, done, truncated, info

    def close(self):
        self.env.close()


class SplitBlockStreet(_BlockMixin):
    def __init__(self, env, start, end, mode="train", block_num=4, seconds=3600) -> None:
        self.env = env
        self.time = 0
        self.block_num = block_num
        self.end_time = seconds
        self.start_block = start
        self.end_block = end
        self.possible_agents = env.possible_agents
        self.mode = mode

        self.train_edges = [
            "B2C2", "C2B2", "C1C2", "C2C1", "C2C3", "C3C2",
            "C2D2", "D2C2", "B3C3", "C3B3", "C3C4", "C4C3",
        ]
        self.test_edges = [
            "D3C3", "C3D3", "D3D2", "D2D3", "D3D4", "D4D3",
            "D3E3", "E3D3", "D1D2", "D2D1", "D2E2", "E2D2",
        ]
        self.blockable_edges = self.train_edges if self.mode == "train" else self.test_edges

        self.rd_id = torch.randperm(len(self.blockable_edges))[:self.block_num]
        self.was_blocking = False
        self.original_speeds = {}

    def reset(self, seed=None):
        self.time = 0
        self.was_blocking = False
        self.original_speeds = {}
        return self.env.reset(seed=seed) if seed else self.env.reset()

    def step(self, action):
        self._update_blocking()
        self.time += 5
        next_state, reward, done, truncated, info = self.env.step(action)
        if self.time >= self.end_time:
            done = {agt: True for agt in self.possible_agents}
        return next_state, reward, done, truncated, info

    def close(self):
        self.env.close()
