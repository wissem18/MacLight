import time
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from net.net import SATBlock, TemporalEncoder, DynamicPredictor

# declare the device
global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_ours_agent(
    env: object,
    agents: object,
    agent_name: list,
    sat: SATBlock,
    temp_enc: TemporalEncoder,
    predictor: DynamicPredictor,
    sat_optimizer,
    sat_scheduler,
    writer: int,
    total_episodes: int,
    seed: int,
    ckpt_path: str,
    evaluator: object,
    steps_per_ep: int,
    report_fn=None
):
    
    actor_loss_list = []
    critic_loss_list = []
    pred_loss_list = []
    attn_weights_list = []
    return_list = []
    waiting_list = []
    queue_list = []
    speed_list = []
    time_list = []
    seed_list = []
    start_time = time.time()
    best_score = -1e10
    actor_best_weight = {}
    critic_best_weight = {}
    step_log = {}

    
    for episode in range(total_episodes):
        epi_training = False
        transition_dict = {"states": {agt_name: 0 for agt_name in agent_name},
                           "actions": {agt_name: 0 for agt_name in agent_name},
                           "next_states": {agt_name: 0 for agt_name in agent_name},
                           "rewards": {agt_name: 0 for agt_name in agent_name},
                           "dones": {agt_name: 0 for agt_name in agent_name},
                           "global_emb": []}
        episode_return = 0
        ep_actor, ep_critic, ep_pred = [], [], []
        # * ---- execute simulation ----
        state, done, truncated = env.reset(seed=seed)[0], False, False
        # per-step containers
        step_log[episode] = {"t": [], "ret": [], "wait": [], "queue": [], "speed": []}
        sim_time = 0
        while not done | truncated:
            action = {}
            for agt_name in agent_name:
                action[agt_name] = agents[agt_name].take_action(state[agt_name])
            next_state, reward, done, truncated, info = env.step(action)

             # --- record per-step metrics -----------------------------
            step_log[episode]["t"].append(sim_time)
            step_log[episode]["ret"].append(np.mean(list(reward.values())))
            step_log[episode]["wait"].append(info[agent_name[0]]["system_total_waiting_time"])
            step_log[episode]["queue"].append(info[agent_name[0]]["system_total_stopped"])
            step_log[episode]["speed"].append(info[agent_name[0]]["system_mean_speed"])
            sim_time += 5

            transition_dict = update_transition(agent_name, epi_training, transition_dict, state,
                                                    done, action, next_state, reward)
            epi_training = True
            state = next_state
            episode_return += np.mean(list(reward.values()))
            done = all(list(done.values()))
            truncated = all(list(truncated.values()))

        # * ---- log to dict  ----
        return_list.append(episode_return)
        waiting_list.append(info[agent_name[0]]["system_total_waiting_time"])
        queue_list.append(info[agent_name[0]]["system_total_stopped"])
        speed_list.append(info[agent_name[0]]["system_mean_speed"])
        time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
        seed_list.append(seed)

        if report_fn is not None:
            tail = 10
            sl = slice(max(0, len(return_list)-tail), None)
            report_fn({
                "episode": episode + 1,
                "avg_return_tail": float(np.mean(return_list[sl])),
                "waiting_tail": float(np.mean(waiting_list[sl])),
                "queue_tail": float(np.mean(queue_list[sl])),
                "speed_tail": float(np.mean(speed_list[sl])),
            })
        
        # * ---- update agent and attention--- 
        sat_optimizer.zero_grad()          # clear shared grads
        for agt_name in agent_name:
            actor_loss, critic_loss,pred_loss = agents[agt_name].update(
                    transition_dict, agt_name,sat, temp_enc, predictor,
                    accumulate_attn_grad=True)   # ← grads accumulate
            ep_actor.append(actor_loss)
            ep_critic.append(critic_loss)
            ep_pred.append(pred_loss)

        sat_optimizer.step()               # single shared step
        sat_scheduler.step()
        
        # store per-episode means
        actor_loss_list.append(float(np.mean(ep_actor)))
        critic_loss_list.append(float(np.mean(ep_critic)))
        pred_loss_list.append(float(np.mean(ep_pred)))

        # attn_weights_list.append(
        #     agents[agent_name[0]].get_full_attention()
        # )

        # read best weights
        if episode_return > best_score:
            for agt_name in agent_name:
                actor_best_weight[agt_name] = agents[agt_name].actor.state_dict()
                critic_best_weight[agt_name] = agents[agt_name].critic.state_dict()
            best_score = episode_return

        # read best weight
        if epi_training:
            for agt_name in agent_name:
                agents[agt_name].actor.load_state_dict(actor_best_weight[agt_name])
                agents[agt_name].critic.load_state_dict(critic_best_weight[agt_name])

        # save log to file and report train status
        evaluator.evaluate_and_save(writer, return_list, waiting_list, queue_list, speed_list,
                                    time_list, seed_list, ckpt_path, episode, agents, seed,
                                    actor_loss_list, critic_loss_list,pred_loss_list, vae_loss_list=None, vae=None)
         
    #save attention weights for analysis
    if attn_weights_list: 
        save_path=f"{ckpt_path}/{seed}_gat_full_attn.fp16.npz"                                
        att_stack = torch.stack(attn_weights_list)        
        np.savez_compressed(save_path, att_stack.numpy())
        print(f"saved raw attention tensor → {os.path.getsize(save_path)/1e6:.1f} MB")  
                                    
    np.savez_compressed(f"{ckpt_path}/{seed}_step_metrics.npz",step_log=step_log)
    env.close()
    total_time = time.time() - start_time
    print(f"\033[32m[ Total time ]\033[0m {(total_time / 60):.2f} min")

    return return_list, total_time // 60

def update_transition(agent_name, epi_training, transition_dict, state, done, action, next_state, reward):
    '''
    Since the state and other data of each agent are stored separately in the form of dictionaries, they are merged here.
    '''
    for key, element in zip(['states', 'actions', 'next_states', 'rewards', 'dones'],
                            [state, action, next_state, reward, done]):
        for agt_name in agent_name:
            if not epi_training:
                transition_dict[key][agt_name] = torch.tensor(element[agt_name]).unsqueeze(0)
            else:
                transition_dict[key][agt_name] = torch.cat([transition_dict[key][agt_name],
                                                            torch.tensor([element[agt_name]])])
    return transition_dict
