# test.py – Evaluate a trained MacLight agent in inference mode
# -----------------------------------------------------------
# • Loads actor weights from a checkpoint directory
# • Runs one or more SUMO episodes
# • Computes cumulative metrics (return,system waiting time, stopped queue length, mean speed)
# -----------------------------------------------------------

from env.wrap.weather_perturbation import WeatherPerturb
import os, argparse, time, torch, numpy as np, pandas as pd, sumo_rl, sys
from pathlib import Path
from net.net import PolicyNet, ValueNet
from agent.Ours_agent import MacLight
from util.tools import MARLWrap
from env.wrap.random_block import BlockStreet

def get_next_result_path(directory="test", basename="eval_results", ext="csv"):
    existing = [f for f in os.listdir(directory) if f.startswith(basename) and f.endswith('.'+ext)]
    nums = []
    for f in existing:
        s = f[len(basename)+1:-len(ext)-1] if f.startswith(basename+"_") else ''
        try:
            nums.append(int(s))
        except Exception:
            continue
    n = 1 if not nums else max(nums) + 1
    return os.path.join(directory, f"{basename}_{n}.{ext}"),n

def load_agents_from_pt(agent_names, pt_path, device):
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} not found")
    ckpt = torch.load(str(pt_path), map_location=device)
    if 'agent' not in ckpt:
        raise KeyError("No 'agent' key in checkpoint file")
    agent_dict = ckpt['agent']
    policies = {}
    for agent in agent_names:
        if agent not in agent_dict:
            raise KeyError(f"Agent {agent} not found in checkpoint")
        actor_state = agent_dict[agent].actor.state_dict()
        policies[agent] = actor_state
    return policies

def make_env(level, seconds, gui):
    env = sumo_rl.parallel_env(net_file='env/map/ff.net.xml',
                               route_file=f'env/map/ff_{level}.rou.xml',
                               num_seconds=seconds,
                               use_gui=gui,
                               sumo_warnings=False,
                               additional_sumo_cmd='--no-step-log')
    return env

def evaluate(args):
    perturbation_start = 600
    perturbation_end = 1800
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = make_env(args.level, args.seconds, bool(args.gui))

    names       = env.possible_agents
    state_dim   = env.observation_space(names[0]).shape[0]
    action_dim  = env.action_space(names[0]).n
    hidden_dim  = (state_dim+32) * 2

    if args.task == 'block':
        env = BlockStreet(env, perturbation_start, perturbation_end, args.block_num, args.seconds)

    if args.weather:
        env=WeatherPerturb(env,seconds=args.seconds,start=perturbation_start,end=perturbation_end)
    # Load all agents' actor policies
    actor_states = load_agents_from_pt(names, args.ckpt, device)
    policies = {}
    for agent in names:
        policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        policy.load_state_dict(actor_states[agent])
        policies[agent] = policy

    alg_args = dict(actor_lr=0, critic_lr=0, gamma=0.99, lmbda=0.95,
                    epochs=1, eps=0.2, pred_coef=0.01, device=device, agent_name=names)
    agents = MARLWrap('I', MacLight, alg_args, PolicyNet, ValueNet, state_dim, hidden_dim, action_dim)

    for n in names:
        agents[n].actor = policies[n]
        agents[n].device = device

    rows = []
    step_log={}
    for s in range(args.seed[0],args.seed[1]+1):
        for epi in range(1, args.episodes + 1):
            t0=time.time()
            state, done, truncated = env.reset(seed=s)[0], False, False
            cum_return = 0.0
            step_log[s][epi] = {"t": [], "ret": [],"wait": [],"queue": [],"speed": []}
            sim_time= 0
            while not done and not truncated:
                action = {n: agents[n].take_action(state[n]) for n in names}
                state, reward, done, truncated, info = env.step(action)
                cum_return += np.mean(list(reward.values()))
                done = all(done.values()); truncated = all(truncated.values())
                # --- record per-step metrics -----------------------------
                step_log[s][epi]["t"].append(sim_time)
                step_log[s][epi]["ret"].append(np.mean(list(reward.values())))
                step_log[s][epi]["wait"].append(info[names[0]]["system_total_waiting_time"])
                step_log[s][epi]["queue"].append(info[names[0]]["system_total_stopped"])
                step_log[s][epi]["speed"].append(info[names[0]]["system_mean_speed"])
                sim_time += 5

            infer_time = time.time() - t0  
            root = names[0]
            rows.append(dict(Episode=epi,
                            Return=cum_return,
                            Waiting=info[root]['system_total_waiting_time'],
                            Queue=info[root]['system_total_stopped'],
                            Speed=info[root]['system_mean_speed'],
                            Time=time.strftime('%m-%d %H:%M:%S'),
                            InferenceTime=infer_time,
                            Seed=s))
            print(f"Episode {epi}:\nreturn: {cum_return:.2f}")
            print(f"Total Waiting Time: {rows[-1]['Waiting']}")
            print(f"Total Queue: {rows[-1]['Queue']}")
            print(f"Average speed: {rows[-1]['Speed']}")
            print(f"Inference Time: {rows[-1]['InferenceTime']}")
        env.close()

        df = pd.DataFrame(rows)
        out_path,cur=get_next_result_path()
        df.to_csv(out_path, index=False)
        np.savez_compressed(f"test/{cur}_step_metrics.npz",step_log=step_log)

if __name__ == '__main__':
    p = argparse.ArgumentParser("MacLight evaluator")
    p.add_argument('--ckpt', required=True, help='Directory containing the .pt checkpoint')
    p.add_argument('-t','--task', default='block', choices=['block','regular'])
    p.add_argument('-l','--level', default='normal', choices=['normal','hard'])
    p.add_argument('-b','--block_num', type=int, default=8)
    p.add_argument('--weather', default=0, type=int, help='Whether or not to add the weather perturbation scenario')
    p.add_argument('-s','--seconds', type=int, default=3600)
    p.add_argument('-e','--episodes', type=int, default=1)
    p.add_argument('--seed',nargs='+', type=int, default=[42,46], help="Set a random seed range to run in sequence")
    p.add_argument('--gui', type=int, default=0)
    evaluate(p.parse_args())