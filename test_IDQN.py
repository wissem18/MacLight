import os, argparse, time, torch, numpy as np, pandas as pd, sumo_rl
from pathlib import Path
from net.net import Qnet
from env.wrap.random_block import BlockStreet

def get_next_result_path(directory="test", basename="eval_results", ext="csv"):
    os.makedirs(directory, exist_ok=True)
    existing = [f for f in os.listdir(directory) if f.startswith(basename) and f.endswith('.'+ext)]
    nums = []
    for f in existing:
        s = f[len(basename)+1:-len(ext)-1] if f.startswith(basename+"_") else ''
        try:
            nums.append(int(s))
        except Exception:
            continue
    n = 1 if not nums else max(nums) + 1
    return os.path.join(directory, f"{basename}_{n}.{ext}")

def make_env(level, seconds, network, gui):
    NETWORK_TABLE = {
        "ff": {
            "net":  "env/map/ff.net.xml",
            "rou":  f'env/map/ff_{level}.rou.xml'
        },
        "hangzhou": {
            "net":  "env/map/hangzhou_4x4_gudang_18041610_1h.net.xml",
            "rou":  "env/map/hangzhou_4x4_gudang_18041610_1h.rou.xml"
        }
    }
    net_file = NETWORK_TABLE[network]['net']
    rou_file = NETWORK_TABLE[network]['rou']
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=rou_file,
        num_seconds=seconds,
        use_gui=gui,
        sumo_warnings=False,
        additional_sumo_cmd='--no-step-log'
    )
    return env

def load_idqn_qnets(agent_names, pt_path, device):
   
    ckpt_path = Path(pt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if 'agent' not in ckpt:
        raise KeyError("Checkpoint missing 'agent' key")
    blob = ckpt['agent']
    agent_dict = blob if isinstance(blob, dict) else getattr(blob, 'agents', None)
    if agent_dict is None:
        raise KeyError("Checkpoint 'agent' does not provide a dict of per-agent models")

    q_states = {}
    for name in agent_names:
        if name not in agent_dict:
            raise KeyError(f"Agent '{name}' not found in checkpoint")
        q_states[name] = agent_dict[name].q_net.state_dict()
    return q_states

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = make_env(args.level, args.seconds, args.network, bool(args.gui))

    names      = env.possible_agents
    state_dim  = env.observation_space(names[0]).shape[0]
    action_dim = env.action_space(names[0]).n
    hidden_dim = state_dim * 2

    if args.task == 'block' and args.network == 'ff':
        env = BlockStreet(env, args.block_num, args.seconds)

    # Load Q-networks
    q_states = load_idqn_qnets(names, args.ckpt, device)

    # Build per-agent Q-nets
    qnets = {}
    for n in names:
        q = Qnet(state_dim, hidden_dim, action_dim).to(device)
        q.load_state_dict(q_states[n])
        q.eval()
        qnets[n] = q

    rows = []
    for epi in range(1, args.episodes + 1):
        for s in range(args.seed[0], args.seed[1] + 1):
            t0 = time.time()
            state, done, truncated = env.reset(seed=s)[0], False, False
            cum_return = 0.0
            while not done and not truncated:
                action = {}
                for n in names:
                    obs_t = torch.tensor(state[n], dtype=torch.float32, device=device).unsqueeze(0)
                    qvals = qnets[n](obs_t).squeeze(0)
                    action[n] = int(torch.argmax(qvals).item())
                state, reward, done, truncated, info = env.step(action)
                cum_return += np.mean(list(reward.values()))
                done = all(done.values()); truncated = all(truncated.values())

            infer_time = time.time() - t0
            root = names[0]
            rows.append(dict(
                Episode=epi,
                Return=cum_return,
                Waiting=info[root]['system_total_waiting_time'],
                Queue=info[root]['system_total_stopped'],
                Speed=info[root]['system_mean_speed'],
                Time=time.strftime('%m-%d %H:%M:%S'),
                InferenceTime=infer_time,
                Seed=s
            ))
            print(f"Episode {epi}:\nreturn: {cum_return:.2f}")
            print(f"Total Waiting Time: {rows[-1]['Waiting']}")
            print(f"Total Queue: {rows[-1]['Queue']}")
            print(f"Average speed: {rows[-1]['Speed']}")
            print(f"Inference Time: {rows[-1]['InferenceTime']}")
        env.close()

        df = pd.DataFrame(rows)
        out_path = get_next_result_path()
        df.to_csv(out_path, index=False)

if __name__ == '__main__':
    p = argparse.ArgumentParser("IDQN evaluator")
    p.add_argument('--ckpt', required=True, help='Path to IDQN checkpoint file')
    p.add_argument('-t','--task', default='block', choices=['block','regular'])
    p.add_argument('-l','--level', default='normal', choices=['normal','hard'])
    p.add_argument('-b','--block_num', type=int, default=8)
    p.add_argument('-n','--network', default='ff', type=str, help='Scenario network key: ff/hangzhou')
    p.add_argument('-s','--seconds', type=int, default=3600)
    p.add_argument('-e','--episodes', type=int, default=1)
    p.add_argument('--seed', nargs='+', default=[42,46], type=int, help='Set a random seed range')
    p.add_argument('--gui', type=int, default=0)
    evaluate(p.parse_args())
