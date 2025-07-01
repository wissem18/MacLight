import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from net.net import Attention
from transformers import get_cosine_schedule_with_warmup

# declare the device
global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_ours_agent(
    env: object,
    agents: object,
    agent_name: list,
    attention: Attention,
    attention_optimizer,
    attention_scheduler,
    writer: int,
    total_episodes: int,
    seed: int,
    ckpt_path: str,
    evaluator: object,
    steps_per_ep: int,
):
    
    actor_loss_list = []
    critic_loss_list = []
    attn_weights_list = []
    return_list = []
    waiting_list = []
    queue_list = []
    speed_list = []
    time_list = []
    seed_list = []
    lr_hist = []
    start_time = time.time()
    best_score = -1e10
    actor_best_weight = {}
    critic_best_weight = {}

    
    for episode in range(total_episodes):
        epi_training = False
        transition_dict = {"states": {agt_name: 0 for agt_name in agent_name},
                           "actions": {agt_name: 0 for agt_name in agent_name},
                           "next_states": {agt_name: 0 for agt_name in agent_name},
                           "rewards": {agt_name: 0 for agt_name in agent_name},
                           "dones": {agt_name: 0 for agt_name in agent_name},
                           "global_emb": []}
        episode_return = 0

        # * ---- execute simulation ----
        state, done, truncated = env.reset(seed=seed)[0], False, False
        while not done | truncated:
            action = {}
            for agt_name in agent_name:
                action[agt_name] = agents[agt_name].take_action(state[agt_name])
            next_state, reward, done, truncated, info = env.step(action)
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
        
        # * ---- update agent and attention--- 
        attention_optimizer.zero_grad()          # clear shared grads
        for agt_name in agent_name:
            actor_loss, critic_loss = agents[agt_name].update(
                    transition_dict, agt_name,attention,
                    accumulate_attn_grad=True)   # ← grads accumulate
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)

        attention_optimizer.step()               # single shared step
        attention_scheduler.step()
        
        lr_hist.append(attention_optimizer.param_groups[0]['lr'])
        # keep last A for logging
        attn_weights_list.append(
            agents[agent_name[0]].get_last_attention()
        )

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
                                    actor_loss_list, critic_loss_list, vae_loss_list=None, vae=None)
         
    #save attention weights and learning rate history for analysis
    if attn_weights_list:                                 
        ep_stack = torch.stack(attn_weights_list)         # (E,16,16)  E = episodes
        np.save(f"{ckpt_path}/{seed}_attn_per_episode.npy", ep_stack.numpy().astype(np.float32))
          
    if lr_hist:
        np.save(f"{ckpt_path}/{seed}_attn_lr_history.npy", np.array(lr_hist))
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
