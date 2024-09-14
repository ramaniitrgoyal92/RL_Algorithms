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

from stable_baselines3.common.buffers import ReplayBuffer
import csv


ENV_NAME = 'InvertedPendulum-v4'
csv_file = 'sac_cartpole_output.csv' #csv file to store training progress
exp_name = 'sac_cartpole_ep_120'
run_name = 'sac'

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        #print(f"x_t = {x_t} y_t = {y_t} action = {action} log_prob = {log_prob} log_prob.shape = {log_prob.shape}")
        log_prob = log_prob.sum(0, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

def reward_function(observation, action):
    diag_q = [1,10,1,1]; 
    r = 1;
    #print("observation:", observation)
    #print("observation:", observation[0,1])
    cost = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
                diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2) +\
                r*(action**2)

    return -cost

def make_env(env_id, render_bool):

    if render_bool:

        env = gym.make('InvertedPendulum-v4',render_mode = "human")
    else:
        env = gym.make('InvertedPendulum-v4')

    min_action = -20
    max_action = 20
    env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.reset()

    return env

def write_data_csv(data):
    

    # Write the data to a CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file is empty
        if file.tell() == 0:
            writer.writerow(['step', 'rewards', 'qf1_loss', 'actor_loss', 'observations', 'action'])
        
        # Write the data
        writer.writerow(data)

if __name__ == "__main__":

    given_seed = 1
    buffer_size = int(1e6)
    batch_size = 256
    total_timesteps = 500000 #default = 1000000
    learning_starts = 25000 #default = 25e3
    episode_length = 120
    exploration_noise = 0.001
    policy_frequency = 2
    tau = 0.005 # weight to update the target network
    gamma = 0.99 #discount factor
    learning_rate = 3e-5
    alpha = 0.2 #Entropy regularization coefficient
    target_network_frequency = 1

    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True
    
    #reward function parameters
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}");

    env = make_env(ENV_NAME, render_bool = False)
    
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    #
    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)

    # load pretrained model.
    #checkpoint = torch.load(f"../runs/{run_name}/{exp_name}.pth")
    # actor.load_state_dict(checkpoint[0])
    # qf1.load_state_dict(checkpoint[1])
    # qf2.load_state_dict(checkpoint[2])

    #target network 
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)

    #initalizing target  with the same weights
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    #choose optimizer
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate)

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

    for global_step in range(total_timesteps):
        
        if global_step < learning_starts:
            actions = np.array(env.action_space.sample())
            
        else:
            with torch.no_grad():
                #print(f"observation = {obs}")
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)

        
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        rewards = reward_function(obs, actions)
        cost -=rewards 
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
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            
            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % policy_frequency == 0:

                for _ in range(policy_frequency):
                    # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                print(f'step= {global_step} rewards= {rewards} qf_loss = {qf_loss.item()} '
                        f'actor_loss = {actor_loss.item()} observations= {obs} action= {actions}')

            # update the target networks
            if global_step % target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if global_step % 100 == 0:
                
                print("SPS:", int(global_step / (time.time() - start_time)))
                # Data to write
                write_data = [global_step, rewards, qf1_loss.item(), actor_loss.item(), obs, actions]
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
        model_path = f"../runs/{run_name}/{exp_name}.pth"
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")

    env.close()
