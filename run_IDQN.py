import os
os.environ['LIBSUMO_AS_TRACI'] = '1'
import argparse
import torch
import gymnasium as gym
import sumo_rl
import random
import sys
import numpy as np
from train.Evaluator import Evaluator
from train.common_train import train_IDQN_agent
from agent.DQN_agent import DQN
from tqdm import trange
from net.net import Qnet
from env.wrap.random_block import BlockStreet
from util.tools import ReplayBuffer
import warnings
warnings.filterwarnings('ignore')

# * ---------------------- Parameters -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDQN mission')
    parser.add_argument('--model_name', default="IDQN", type=str, help='The name of the base algorithm')
    parser.add_argument('-t', '--task', default="block", type=str, help='task: regular / block')
    parser.add_argument('-r', '--representation', default=False, help='Whether or not to use VAE') # Cannot be set to True here
    parser.add_argument('-b', '--block_num', default=8, type=int, help='Number of blocked roads')
    parser.add_argument('-l', '--level', default='normal', type=str, help='Difficulty of the task: normal/hard')
    parser.add_argument('-n', '--network', default='ff', type=str,help='Scenario network key: ff / hangzhou') 
    parser.add_argument('-w', '--writer', default=0, type=int, help='Log mode, 0: no, 1: local')
    parser.add_argument('--seconds', default=3600, type=int, help='Simulation seconds')
    parser.add_argument('-e', '--episodes', default=80, type=int, help='Number of running episodes')
    parser.add_argument('-s', '--seed', nargs='+', default=[42, 46], type=int, help='Set a random seed range to run in sequence')
    args = parser.parse_args()
    
    # ENV
    NETWORK_TABLE = {
    # synthetic network shipped with MacLight
    "ff": {
        "net":  "env/map/ff.net.xml",
        "rou":  f'env/map/ff_{args.level}.rou.xml'
    },
    # Hangzhou real-world dataset (4-phase)
    "hangzhou": {
        "net":  "env/map/hangzhou_4x4_gudang_18041610_1h.net.xml",
        "rou":  "env/map/hangzhou_4x4_gudang_18041610_1h.rou.xml"
    }
}
    
    net_file=NETWORK_TABLE[args.network]["net"]
    route_file=NETWORK_TABLE[args.network]["rou"]
    # Environmental
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = sumo_rl.parallel_env(net_file=net_file,
                               route_file=route_file,
                               num_seconds=args.seconds,
                               use_gui=False,
                               sumo_warnings=False,
                               additional_sumo_cmd='--no-step-log')
    # Neural Networks
    agent_name = env.possible_agents
    state_dim = [env.observation_space(i).shape[0] for i in agent_name]
    hidden_dim = [env.observation_space(i).shape[0] * 2 for i in agent_name]
    action_dim = [env.action_space(i).n for i in agent_name]
    if len(set(state_dim)) == 1:
        state_dim = state_dim[0]
        action_dim = action_dim[0]
        hidden_dim = hidden_dim[0]
        if args.task == 'block' and args.network == 'ff':
            env = BlockStreet(env, args.block_num, args.seconds)
        else:
            args.block_num = None
        args.task = args.network + '_' + args.task + '_' + args.level

    # DQN
    alg_args = {}
    alg_args['q_lr'] = 1e-3
    alg_args['gamma'] = 0.98  # 时序差分学习率，也作为折算奖励的系数之一
    alg_args['device'] = device
    alg_args['epsilon'] = 0.8
    alg_args['targe_update'] = 5
    alg_args['agent_name'] = agent_name
    pool_size = 10000
    minimal_size = 1000
    batch_size = 20

    # Mission
    system_type = sys.platform  # operation system

    # * ------------------------ Train ----------------------------
    print(f'[ Start >>> task: {args.task} - {args.block_num} | model: {args.model_name} | repre: {args.representation} | device: {device} ]')
    for seed in trange(args.seed[0], args.seed[-1] + 1, mininterval=40, ncols=70):
        evaluator = Evaluator()
        CKP_PATH = f'ckpt/{args.task}/{args.model_name}'
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        magent = {}
        replay_buffer = ReplayBuffer(len(agent_name), state_dim, pool_size, batch_size)
        for idx, agt_name in enumerate(agent_name):
            q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
            magent[agt_name] = DQN(q_net, alg_args['q_lr'], alg_args['gamma'], alg_args['epsilon'],
                                   alg_args['targe_update'], alg_args['device'])
        return_list, train_time = train_IDQN_agent(env, magent, agent_name, replay_buffer, minimal_size, args.writer,
                                                   args.episodes, seed, CKP_PATH, evaluator,
                                                   )