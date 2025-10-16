import os, random, numpy as np, torch, json
from ray import tune, air
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import sumo_rl
from train.Evaluator import Evaluator
from train.train_ours import train_ours_agent
from agent.Ours_agent import MacLight
from util.tools import MARLWrap, build_adj_matrix, adj_to_edge_index
from net.net import PolicyNet, ValueNet, TemporalEncoder, DynamicPredictor, SATBlock
from transformers import get_cosine_schedule_with_warmup

def trainable(config):
    # seeds
    seed = int(config.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # ------------- ENV -------------
    seconds = 3600
    NETWORK_TABLE = {
        "ff": {
            "net":  "env/map/ff.net.xml",
            "rou":  "env/map/ff_normal.rou.xml"
        }
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = sumo_rl.parallel_env(
        net_file=NETWORK_TABLE["ff"]["net"],
        route_file=NETWORK_TABLE["ff"]["rou"],
        num_seconds=seconds,
        use_gui=False,
        sumo_warnings=False,
        additional_sumo_cmd='--no-step-log'
    )
    agent_name = env.possible_agents

    # ------------- Fixed dims & constants -------------
    embedding_dim = 32            # FIXED
    dropout = 0.10                # FIXED
    warmup_frac = 0.10            # FIXED

    state_dim = env.observation_space(agent_name[0]).shape[0]
    action_dim = env.action_space(agent_name[0]).n
    hidden_dim = (state_dim + embedding_dim) * 2

    # ------------- Graph -------------
    adj_mask  = build_adj_matrix(net_file=NETWORK_TABLE["ff"]["net"], agent_ids=agent_name)
    edge_index = adj_to_edge_index(adj_mask).to(device)

    # ------------- Blocks -------------
    sat = SATBlock(
        d_in=33,                    # keep your feature size
        d_out=embedding_dim,
        edge_index=edge_index,
        dropout=dropout,                            # FIXED
        k_hop=int(config["sat_k_hop"]),             # {1,2,3}
        gnn_type="gcn",
        num_layers=int(config["sat_layers"])        # {1,2,3}
    ).to(device)

    temp_enc = TemporalEncoder(
        d_model=embedding_dim,
        K=int(config["K_HISTORY"]),                 # {4,8,12}
        n_heads=4,                                  # fixed
        n_layers=2,
        dropout=dropout                              # FIXED
    ).to(device)

    predictor = DynamicPredictor(d_model=embedding_dim, action_dim=action_dim).to(device)

    # ------------- Optim & sched for SAT+Temp+Pred -------------
    params_st  = list(sat.parameters()) + list(temp_enc.parameters()) + list(predictor.parameters())
    optimizer  = torch.optim.Adam(params_st, lr=float(config["lr_st"]))

    total_steps  = int(config["episodes"])          # episodes used for warmup schedule steps
    warmup_steps = max(1, int(total_steps * warmup_frac))  # FIXED warmup_frac
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps, num_cycles=0.5
    )

    # ------------- PPO args -------------
    alg_args = {
        "actor_lr":  float(config["lr_actor"]),
        "critic_lr": float(config["lr_critic"]),
        "lmbda":     0.95,
        "gamma":     0.99,
        "device":    device,
        "epochs":    10,
        "eps":       0.2,
        "pred_coef": float(config["pred_coef"]),
        "agent_name": agent_name
    }

    marl = MARLWrap('I', MacLight, alg_args,
                    PolicyNet, ValueNet,
                    state_dim, hidden_dim, action_dim,
                    latent_dim=embedding_dim)

    steps_per_ep = seconds // 5
    evaluator = Evaluator()

    trial_dir = tune.get_trial_dir()
    ckpt_path = os.path.join(trial_dir, "ckpt")
    os.makedirs(ckpt_path, exist_ok=True)

    # Ray reporting hook
    def report_fn(metrics: dict):
        # Must include "episode" and "avg_return_tail" for ASHA + best_result
        tune.report(**metrics)

    # ------------- Train -------------
    return_list, _mins = train_ours_agent(
        env, marl, agent_name,
        sat, temp_enc, predictor,
        optimizer, scheduler,
        writer=0,
        total_episodes=int(config["episodes"]),
        seed=seed,
        ckpt_path=ckpt_path,
        evaluator=evaluator,
        steps_per_ep=steps_per_ep,
        report_fn=report_fn
    )

    # final report safeguard
    tail = 10
    sl = slice(max(0, len(return_list)-tail), None)
    tune.report(avg_return_tail=float(np.mean(return_list[sl])))

# --------- Search space (reduced; your choices fixed) ---------
search_space = {
    "K_HISTORY":     tune.choice([4, 8, 12]),

    # LRs (log-uniform in [1e-5, 1e-3])
    "lr_actor":  tune.loguniform(1e-5, 1e-3),
    "lr_critic": tune.loguniform(1e-5, 1e-3),
    "lr_st":     tune.loguniform(1e-5, 1e-3),

    # SAT
    "sat_k_hop":  tune.choice([1, 2, 3]),
    "sat_layers": tune.choice([1, 2, 3]),

    # Predictor coef
    "pred_coef":   tune.choice([0.0, 0.005, 0.01, 0.02]),

    # Budget & seed
    "episodes":    40,
    "seed":        42
}

def main():
    # Aggressive early-stopping to explore more configs
    scheduler = ASHAScheduler(
        time_attr="episode",
        metric="avg_return_tail",
        mode="max",
        grace_period=10,
        reduction_factor=3
    )
    search_alg = OptunaSearch(metric="avg_return_tail", mode="max")

    tuner = Tuner(
        # ↓ Use 2 CPUs per trial on CPU cluster to increase parallelism
        tune.with_resources(trainable, {"cpu": 2, "gpu": 0}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=256,                 # ↑ explore many configs
            scheduler=scheduler,
            search_alg=search_alg
        ),
        run_config=air.RunConfig(
            name="sat_tsc_raytune",
            local_dir="ray_results",
            checkpoint_config=air.CheckpointConfig(num_to_keep=1)
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="avg_return_tail", mode="max")
    print("Best config:", best.config)
    print("Best trial dir:", best.path)

    with open(os.path.join("ray_results", "best_config.json"), "w") as f:
        json.dump(best.config, f, indent=2)

if __name__ == "__main__":
    main()
