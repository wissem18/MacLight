import os
os.environ['LIBSUMO_AS_TRACI'] = '1'
os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
import argparse
import torch
import gymnasium as gym
import sumo_rl
import random
import sys
import numpy as np
from train.Evaluator import Evaluator
from train.train_ours import train_ours_agent
from agent.Ours_agent import MacLight
from tqdm import trange
from net.net import PolicyNet, ValueNet, DynamicGNN
from env.wrap.random_block import BlockStreet
from util.tools import MARLWrap,adj_to_edge_index,build_two_hop_adj_matrix
import warnings
warnings.filterwarnings('ignore')
from transformers import get_cosine_schedule_with_warmup
# * ---------------------- Parameters -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ours mission')
    parser.add_argument('--model_name', default="RPPO", type=str, help='The name of the base algorithm') 
    parser.add_argument('-t', '--task', default="block", type=str, help='task: regular / block')
    parser.add_argument('-b', '--block_num', default=8, type=int, help='Number of blocked roads')
    parser.add_argument('-l', '--level', default='normal', type=str, help='Difficulty of the task: normal/hard')  # hard for Peak in paper
    parser.add_argument('-n', '--network', default='ff', type=str,help='Scenario network key: ff / hangzhou')    
    parser.add_argument('-w', '--writer', default=0, type=int, help='Log mode, 0: no, 1: local')
    parser.add_argument('--seconds', default=3600, type=int, help='Simulation seconds')
    parser.add_argument('-e', '--episodes', default=80, type=int, help='Number of running episodes')
    parser.add_argument('-s', '--seed', nargs='+', default=[42,46], type=int, help='Set a random seed range to run in sequence')
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = sumo_rl.parallel_env(net_file=net_file,
                               route_file=route_file,
                               num_seconds=args.seconds,
                               use_gui=False,
                               sumo_warnings=False,
                               additional_sumo_cmd='--no-step-log')
                               
    # Neural Networks
    agent_name = env.possible_agents
    global_emb_dim = 32
    state_dim = [env.observation_space(i).shape[0] for i in agent_name]
    hidden_dim = [(env.observation_space(i).shape[0]+global_emb_dim) * 2 for i in agent_name]
    action_dim = [env.action_space(i).n for i in agent_name]
    if len(set(state_dim)) == 1:
        state_dim = state_dim[0]
        action_dim = action_dim[0]
        hidden_dim = hidden_dim[0]

        if args.task == 'block' and args.network == 'ff':
            env = BlockStreet(env, args.block_num, args.seconds)
        else:
            args.block_num = None

        args.model_name = 'Ours_dynamic_adj_GATv2'
        args.task = args.network + '_' + args.task + '_' + args.level
    

    # PPO
    alg_args = {}
    alg_args['actor_lr'] = 1e-4
    alg_args['critic_lr'] = 1e-3
    alg_args['lmbda'] = 0.95
    alg_args['gamma'] = 0.99
    alg_args['device'] = device
    alg_args['epochs'] = 10
    alg_args['eps'] = 0.2
    alg_args['agent_name'] = agent_name
    
    system_type = sys.platform

    # ---------------------------- networks ------------------------------
    # 1. fixed ≤2-hop skeleton  
    adj2  = build_two_hop_adj_matrix(net_file=net_file, agent_ids=agent_name) 
    edge_candidates = adj_to_edge_index(adj2).to(device)
    # 2. dynamic GAT with learnable adjacency
    gat = DynamicGNN(obs_dim= state_dim,
                 hid_dim= 64,
                 out_dim= 32,
                 heads= 4,
                 edge_candidates= edge_candidates,
                 k= 4,
                 dropout= 0.1,
                 device= device).to(device)
    
    base_lr=1e-3
    warmup_frac  = 0.1                         # 10 % warm-up
    steps_per_ep = args.seconds//5              
    total_steps  = args.episodes
    warmup_steps = int(total_steps * warmup_frac)

    optimizer = torch.optim.Adam(gat.parameters(), lr=base_lr)

    scheduler = get_cosine_schedule_with_warmup(
                optimizer          = optimizer,
                num_warmup_steps   = warmup_steps,
                num_training_steps = total_steps,
                num_cycles         = 0.5)

    marl = MARLWrap('I', MacLight, alg_args,
                    PolicyNet, ValueNet,
                    state_dim, hidden_dim, action_dim,
                    latent_dim=global_emb_dim)        # critic will see +32 dims


    # * ------------------------ Train ----------------------------
    print(f'[ Start >>> task: {args.task} - {args.block_num} | model: {args.model_name} | device: {device} ]')
    for seed in trange(args.seed[0], args.seed[-1] + 1, mininterval=40, ncols=70):
        evaluator = Evaluator()
        CKP_PATH = f'ckpt/{args.task}/{args.model_name}'
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return_list, train_time = train_ours_agent(env, marl, agent_name, gat,optimizer,scheduler,  args.writer,
                                                   args.episodes, seed, CKP_PATH, evaluator,steps_per_ep)