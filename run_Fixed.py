import os
os.environ['LIBSUMO_AS_TRACI'] = '1'
import argparse
import gymnasium as gym
import sumo_rl
import random
import sys
import numpy as np
from train.Evaluator import Evaluator
from train.common_train import run_Fixed_agent
from agent.Fixed_agent import Fiexed
from tqdm import trange
from env.wrap.random_block import BlockStreet
import warnings
warnings.filterwarnings('ignore')

# * ---------------------- 参数 -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fixed agent 任务')
    parser.add_argument('--model_name', default="Fixed", type=str, help='基本算法名称')
    parser.add_argument('-t', '--task', default="block", type=str, help='任务名称 regular / block')
    parser.add_argument('-r', '--representation', default=False, help='是否采用表征学习')
    parser.add_argument('-b', '--block_num', default=8, type=int, help='封堵街道数')
    parser.add_argument('-l', '--level', default='normal', type=str, help='任务难度 normal/hard')
    parser.add_argument('-w', '--writer', default=0, type=int, help='存档等级, 0: 不存，1: 本地')
    parser.add_argument('--seconds', default=3600, type=int, help='运行时间')
    parser.add_argument('-e', '--episodes', default=80, type=int, help='运行回合数')
    parser.add_argument('-s', '--seed', nargs='+', default=[42, 46], type=int, help='起始种子')
    args = parser.parse_args()

    # 环境相关
    device = "cpu"
    env = sumo_rl.parallel_env(net_file='env/map/ff.net.xml',
                               route_file=f'env/map/ff_{args.level}.rou.xml',
                               num_seconds=args.seconds,
                               use_gui=False,
                               sumo_warnings=False,
                               additional_sumo_cmd='--no-step-log')
    # 神经网络相关
    agent_name = env.possible_agents
    action_dict = {agent: 0 for agent in agent_name}
    action_space = {agent: list(range(env.action_space(agent).n)) for agent in agent_name}

    state_dim = [env.observation_space(i).shape[0] for i in agent_name]
    hidden_dim = [env.observation_space(i).shape[0] * 2 for i in agent_name]
    action_dim = [env.action_space(i).n for i in agent_name]
    if len(set(state_dim)) == 1:
        state_dim = state_dim[0]
        action_dim = action_dim[0]
        hidden_dim = hidden_dim[0]
        if args.task == 'block':
            env = BlockStreet(env, args.block_num, args.seconds)
        else:
            args.block_num = None
        # 更新任务名
        args.task = args.task + '_' + args.level

    # Fixed 相关
    flag = 40

    # 任务相关
    system_type = sys.platform  # 操作系统

    # * ------------------------ 训练 ----------------------------
    print(f'[ Start >>> task: {args.task} - {args.block_num} | model: {args.model_name} | repre: {args.representation} | device: {device} ]')
    for seed in trange(args.seed[0], args.seed[-1] + 1, mininterval=40, ncols=70):
        evaluator = Evaluator()
        CKP_PATH = f'ckpt/{args.task}/{args.model_name}'
        random.seed(seed)
        np.random.seed(seed)
        agent = Fiexed(flag, action_space, action_dict, agent_name)
        evaluator = Evaluator()
        return_list, train_time = run_Fixed_agent(env, agent, agent_name, args.writer, args.episodes, seed, CKP_PATH, evaluator)