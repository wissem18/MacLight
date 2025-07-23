import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import sumolib 
import scipy.sparse as sp 

def build_two_hop_adj_matrix(net_file: str, agent_ids) -> torch.BoolTensor:
    """
    Build a ≤ 2-hop adjacency matrix for the traffic-light agents.

    Parameters
    ----------
    net_file   : str
        Path to the SUMO .net.xml file.
    agent_ids  : iterable[str]
        IDs of traffic-light-controlled junctions (env.possible_agents).

    Returns
    -------
    adj_2hop : BoolTensor, shape (N, N)
        adj_2hop[i, j] == True  ⇔  there exists a path of length 1 or 2
        between agents i and j (undirected, self-loops cleared).
    """
    # 1-hop adjacency (re-use the existing helper)
    adj_1hop = build_adj_matrix(net_file, agent_ids)         # (N, N) bool

    # Compute 2-hop reachability:  A² = (A¹ @ A¹) > 0
    A1 = adj_1hop.to(torch.float32)                          # cast for matmul
    A2 = (A1 @ A1) > 0                                       # boolean matrix

    # Union of 1- and 2-hop edges, without self-loops
    adj_2hop = (adj_1hop | A2).clone()
    adj_2hop.fill_diagonal_(False)

    return adj_2hop

def build_adj_matrix(net_file: str, agent_ids) -> torch.BoolTensor:
    """
    Parameters
    ----------
    net_file : path to the SUMO network file
    agent_ids   : iterable[str] – env.possible_agents (agent ids)

    Returns
    -------
    adj : BoolTensor shape (N, N)
          adj[i, j] == True  ⇔  there exists an edge between agent i and agent j 
          (Undirected → matrix is symmetric; self-loops are False.)
    """
    # ---- read the network ------------------------------------------------
    net     = sumolib.net.readNet(net_file)
    id2idx  = {tid: i for i, tid in enumerate(agent_ids)}
    N       = len(agent_ids)
    adj     = torch.zeros((N, N), dtype=torch.bool)

    # ---- walk through every edge in the graph ---------------------------------
    for edge in net.getEdges():  
        u_id = edge.getFromNode().getID()   # junction id at edge origin
        v_id = edge.getToNode().getID()     # junction id at edge target

        # only keep edges where *both* ends are traffic-light agents
        if u_id in id2idx and v_id in id2idx and u_id != v_id:
            i, j       = id2idx[u_id], id2idx[v_id]
            adj[i, j]  = True
            adj[j, i]  = True               # undirected / symmetric
    
    return adj

def adj_to_edge_index(adj: torch.BoolTensor) -> torch.LongTensor:
    """
    adj : (N,N) BoolTensor – 1 ↔ edge exists (no self-loops)
    returns edge_index : (2,E) LongTensor for PyG GATConv
    """
    src, dst = adj.nonzero(as_tuple=True)        # indices of 1s
    return torch.stack([src, dst], dim=0)        # shape (2,E)

def get_action(time: int, flag: int, action_space: dict, action_dict: dict, agent_name: object):
    '''fixed-time 用，每 flag 秒切换一次所有相位'''
    if time % flag == 0 and time > 0:
        for agt in agent_name:
            if (action_dict[agt] + 1) in action_space[agt]:
                action_dict[agt] += 1
            else:
                action_dict[agt] = 0
            action_dict[agt] = action_space[agt][action_dict[agt]]
    else:
        pass
    return action_dict


def smooth(data: pd.Series, alpha: float = 0.05):
    if isinstance(data, list):
        data = pd.Series(data)
    smooth_data = pd.Series(data).ewm(alpha=alpha).mean()
    return smooth_data


def MARLWrap(type, alg, alg_args, policy, value, state_dim, hidden_dim, action_dim, **kwargs):
    '''
    - type: should be one of ['CTDE', 'I'] \n
    `CTDE`: CTDE agents like MAPPO \n
    `I`: independent agents like IPPO \n

    - alg: algorithm, should be a agent algorithm
    - alg_args: parameters of alg
    '''
    assert type in ['CTDE', 'I'], "type error!"

    if type == 'CTDE':
        assert isinstance(state_dim, list), "state dim should be a list"
        agents = alg(alg_args['agent_name'], policy, value, state_dim, hidden_dim, action_dim,
                     alg_args['actor_lr'], alg_args['critic_lr'], alg_args['gamma'], alg_args['lmbda'],
                     alg_args['epochs'], alg_args['eps'], alg_args['device'])
        return agents

    elif type == 'I':
        assert isinstance(state_dim, int), "state dim should be a integer"
        latent_dim = kwargs.get('latent_dim', 0)
        policy_net = policy(state_dim, hidden_dim, action_dim)
        critic_net = value(state_dim, hidden_dim, latent_dim)
        agents = {}
        for name in alg_args['agent_name']:
            agents[name] = alg(policy_net, critic_net, alg_args['actor_lr'],
                              alg_args['critic_lr'], alg_args['gamma'],
                              alg_args['lmbda'], alg_args['epochs'],
                              alg_args['eps'], alg_args['device'])
        return agents


class ReplayBuffer:
    """CEA."""

    def __init__(
        self,
        agent_num: int,
        states_dim: int,
        size: int,
        batch_size: int = 128,
    ):
        self.states_buf = torch.zeros([size, agent_num, states_dim], dtype=torch.float32)
        self.next_states_buf = torch.zeros([size, agent_num, states_dim], dtype=torch.float32)
        self.actions_buf = torch.zeros([size, agent_num], dtype=torch.float32)
        self.rewards_buf = torch.zeros([size, agent_num], dtype=torch.float32)
        self.dones_buf = torch.zeros([size, agent_num], dtype=torch.float32)
        self.truncated_buf = torch.zeros([size, agent_num], dtype=torch.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        self.capacity = size

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        truncated: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]:
        transition = (state, action, reward, next_state, done, truncated)
        self.states_buf[self.ptr] = state
        self.next_states_buf[self.ptr] = next_state
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.dones_buf[self.ptr] = done
        self.truncated_buf[self.ptr] = truncated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return transition

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        idxs = torch.randperm(self.size)[:self.batch_size]
        return dict(
            states=self.states_buf[idxs],
            next_states=self.next_states_buf[idxs],
            actions=self.actions_buf[idxs],
            rewards=self.rewards_buf[idxs],
            dones=self.dones_buf[idxs],
            truncated=self.truncated_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size
