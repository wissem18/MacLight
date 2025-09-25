import gymnasium as gym
import torch
import traci


class BlockStreet:
    def __init__(self, env,start, end, block_num=8, seconds=3600) -> None:
        self.env = env
        self.time = 0
        # blockable edges can be refer to light yellow parts in `./doc.map_indicator.pdf`
        self.block_num = block_num
        self.start_block = start
        self.end_block = end
        self.end_time = seconds
        self.possible_agents = env.possible_agents

        self.blockable_edges = ['B2C2', 'B3C3', 'C1C2', 'C2B2', 'C2C1',
                                'C2C3', 'C2D2', 'C3B3', 'C3C2', 'C3C4',
                                'C3D3', 'C4C3', 'D1D2', 'D2C2', 'D2D1',
                                'D2D3', 'D2E2', 'D3C3', 'D3D2', 'D3D4',
                                'D3E3', 'D4D3', 'E2D2', 'E3D3',]
        self.rd_id = torch.randperm(len(self.blockable_edges))[:self.block_num]
        self.was_blocking = False # Track if we were in blocking scenario in previous step
    def reset(self, seed=None):
        self.time = 0
        self.was_blocking = False
        return self.env.reset(seed=seed) if seed else self.env.reset()


    def step(self, action):
        '''
        Randomly block 8 road sections every 300 seconds only inside the time window of start_block and end_block,
        unblock them at the end of 300 seconds and set new blocking targets
        '''
        block_active = (self.start_block <= self.time <= self.end_block)
        if block_active:
            if self.time % 300 != 0:
                for edge_id in self.rd_id:  # 阻塞通行
                    traci.edge.setMaxSpeed(self.blockable_edges[edge_id], 0.5)  # m/s
                vehicle_ids = traci.vehicle.getIDList()
                for vehicle_id in vehicle_ids:
                    traci.vehicle.rerouteTraveltime(vehicle_id)  # 车辆重新规划路径
            else:
                for edge_id in self.rd_id:  # 恢复通行
                    traci.edge.setMaxSpeed(
                        self.blockable_edges[edge_id], 13.89)  # m/s
                vehicle_ids = traci.vehicle.getIDList()
                for vehicle_id in vehicle_ids:
                    traci.vehicle.rerouteTraveltime(vehicle_id)
                self.rd_id = torch.randperm(len(self.blockable_edges))[:self.block_num]  # 重新抽n个车道
            self.was_blocking = True    
        else:
            if self.was_blocking:
                for edge_id in self.rd_id:
                    traci.edge.setMaxSpeed(self.blockable_edges[edge_id], 13.89)
                for vid in traci.vehicle.getIDList():
                    traci.vehicle.rerouteTraveltime(vid)
                self.was_blocking = False

        self.time += 5
        next_state, reward, done, truncated, info = self.env.step(action)
        if self.time >= self.end_time:
            done = {agt: True for agt in self.possible_agents}
        return next_state, reward, done, truncated, info

    def close(self):
        self.env.close()

class SplitBlockStreet:
    def __init__(self, env, mode="train", block_num=4, seconds=3600) -> None:
        self.env = env
        self.time = 0
        self.block_num = block_num
        self.end_time = seconds
        self.possible_agents = env.possible_agents
        self.mode = mode  # "train" or "test"

        # Fixed train/test sets
        self.train_edges = [
            # C2 hub (8)
            "B2C2","C2B2","C1C2","C2C1","C2C3","C3C2","C2D2","D2C2",
            # C3 corridor (4)
            "B3C3","C3B3","C3C4","C4C3",
        ]
        self.test_edges = [
            # D3 hub (8)
            "D3C3","C3D3","D3D2","D2D3","D3D4","D4D3","D3E3","E3D3",
            # D2 corridor (4)
            "D1D2","D2D1","D2E2","E2D2",
        ]
        # Choose edges depending on mode
        self.blockable_edges = self.train_edges if self.mode == "train" else self.test_edges

        # Initial random pick
        self.rd_id = torch.randperm(len(self.blockable_edges))[:self.block_num]

    def reset(self, seed=None):
        self.time = 0
        return self.env.reset(seed=seed) if seed else self.env.reset()

    def step(self, action):
        """
        Randomly block k road sections every 300 seconds,
        unblock them at the end of 300 seconds and set new blocking targets
        """
        if self.time % 300 != 0:
            for edge_id in self.rd_id:
                traci.edge.setMaxSpeed(self.blockable_edges[edge_id], 0.5)
            for vid in traci.vehicle.getIDList():
                traci.vehicle.rerouteTraveltime(vid)
        else:
            for edge_id in self.rd_id:
                traci.edge.setMaxSpeed(self.blockable_edges[edge_id], 13.89)
            for vid in traci.vehicle.getIDList():
                traci.vehicle.rerouteTraveltime(vid)
            self.rd_id = torch.randperm(len(self.blockable_edges))[:self.block_num]

        self.time += 5
        next_state, reward, done, truncated, info = self.env.step(action)
        if self.time >= self.end_time:
            done = {agt: True for agt in self.possible_agents}
        return next_state, reward, done, truncated, info

    def close(self):
        self.env.close()