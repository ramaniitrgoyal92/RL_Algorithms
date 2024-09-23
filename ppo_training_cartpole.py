import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import random
import time
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from stable_baselines3.common.buffers import ReplayBuffer
import csv
from noise_injector import OrnsteinUhlenbeckActionNoise

ENV_NAME = 'CartPole-v1'
csv_file = 'models/ppo/cartpole_output.csv' #csv file to store training progress
exp_name = 'cartpole_ep_30'
run_name = 'ppo'


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def reward_function(observation, action):
    diag_q = [1,10,1,1]; 
    r = 1
    #print("observation:", observation)
    #print("observation:", observation[0,1])
    cost = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
                diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2) +\
                r*(action**2)

    return -cost

def make_env(env_id, render_bool):

    if render_bool:
        env = gym.make(env_id,render_mode = "human")
    else:
        env = gym.make(env_id)
    # min_action = -20
    # max_action = 20
    # env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.reset()

    return env

def write_data_csv(data):

    # Write the data to a CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file is empty
        if file.tell() == 0:
            writer.writerow(['step', 'rewards', 'agent1_loss', 'observations', 'action'])
        
        # Write the data
        writer.writerow(data)

if __name__ == "__main__":

    given_seed = 1
    buffer_size = int(1e6)
    batch_size = 256
    total_timesteps = 500#000 #default = 1000000
    learning_starts = 25#000 #default = 25e3
    episode_length = 30
    exploration_noise = 0.001
    policy_frequency = 2
    tau = 0.005 # weight to update the target network
    gamma = 0.99 #discount factor
    learning_rate = 3e-5
    """the starting epsilon for exploration"""
    start_e = 1
    """the ending epsilon for exploration"""
    end_e = 0.05
    exploration_fraction = 0.5

    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True
    
    #reward function parameters
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}")

    env = make_env(ENV_NAME, render_bool = False)
    
    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(env).to(device)
    # load pretrained model.
    # checkpoint = torch.load(f"models/{run_name}/{exp_name}.pth",map_location=torch.device('cpu'))
    # agent.load_state_dict(checkpoint)

    #choose optimizer
    optimizer = optim.Adam(list(agent.parameters()), lr=learning_rate, eps=1e-5)

    #experience replay buffer
    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    episode_t = 0 
    cost = 0
    obs, _ = env.reset()
    # noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(np.prod(env.action_space.shape)))

    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array(env.action_space.sample())
        else:
            q_values = agent(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        rewards = reward_function(obs, actions)
        cost -= rewards 
        #print('step=', global_step, ' actions=', actions, ' rewards=', rewards,\
        #      ' obs=', next_obs, ' termination=', terminations, ' trunctions=', truncations)

        # save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        # if truncations:
        #     real_next_os = infos["final_observation"]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # reset observation
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > learning_starts:

            #sample experience from replay buffer
            data = rb.sample(batch_size)

            with torch.no_grad():
                target_max, _ = agent_target(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
            old_val = agent(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % policy_frequency == 0:
                # update the target network
                for param, target_param in zip(agent.parameters(), agent_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if global_step % 100 == 0:
                
                print("SPS:", int(global_step / (time.time() - start_time)))
                # Data to write
                write_data = [global_step, rewards, loss.item(), obs, actions]
                write_data_csv(write_data)

        episode_t = episode_t + 1
        if abs(next_obs[0])>= 10 or episode_t == episode_length:
            print('resetting')
            obs, _ = env.reset()
            episode_t = 0
            print(f'Cost = {cost}')
            cost = 0


    save_model = True
    if save_model:
        model_path = f"models/{run_name}/{exp_name}.pth"
        torch.save((agent.state_dict()), model_path)
        print(f"model saved to {model_path}")

    env.close()
