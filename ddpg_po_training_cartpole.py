import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import random

import time


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from stable_baselines3.common.buffers import ReplayBuffer
import csv
from noise_injector import OrnsteinUhlenbeckActionNoise

ENV_NAME = 'InvertedPendulum-v4'
csv_file = 'models/ddpg_po/cartpole_po_output.csv' #csv file to store training progress
exp_name = 'carpole_test_po_q_10'
run_name = 'ddpg_po'

q = 10 # number of time history required. 
nx = 4 #dim of state
nu = 1 # dim of control
nz = 4 # dim of measurements
nZ = q*nz + (q-1)*nu #dim of information state


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() 
                             + np.prod(env.action_space.shape), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_mu = nn.Linear(512, np.prod(env.action_space.shape))
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
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

def reward_function(observation, action):
    diag_q = [10,10,10,10,1] #x(t),\theta(t), x(t-1), \theta(t-1), u(t-1)
    r = 1 #u(t)

    reward = diag_q[0]*(np.sum(observation[0:q*nz]**2)) + diag_q[4]*(np.sum(observation[q*nz:nZ]**2)) +\
                 r*(action**2)
    # reward = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
    #             diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2) +\
    #             diag_q[4]*(observation[4]**2) + r*(action**2)

    return -reward

def make_env(env_id, render_bool):

    if render_bool:

        env = gym.make(env_id,render_mode = "human")
    else:
        env = gym.make(env_id)

    min_action = -20
    max_action = 20
    env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.observation_space = gym.spaces.Box(-np.inf, np.inf, (nZ,), np.float64)
    env.reset()

    return env

def information_state(prev_info_state, next_obs, control):

    info_state = np.zeros((nZ))
    info_state[0:nz] = next_obs[0:nz]
    info_state[nz:q*nz] = prev_info_state[0:(q-1)*nz]

    info_state[q*nz:q*nz+nu] = control
    info_state[q*nz+nu:nZ] = prev_info_state[q*nz:q*nz + (q-2)*nu]
    
    return info_state

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
    total_timesteps = 200#000 #default = 1000000
    learning_starts = 25#000 #default = 25e3
    episode_length = 200
    exploration_noise = 0.001
    policy_frequency = 2
    tau = 0.005
    gamma = 0.99
    learning_rate = 3e-4
    
    
    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True

    #reward function parameters
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}")

    env = make_env(ENV_NAME, render_bool = False)
    
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    
    # load pretrained model.
    checkpoint = torch.load(f"models/{run_name}/{exp_name}.pth", map_location=torch.device('cpu'))
    actor.load_state_dict(checkpoint[0])
    qf1.load_state_dict(checkpoint[1])

    qf1_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate)

    env.observation_space.dtype = np.float32

    print('env.observation_space = ', env.observation_space)
    rb = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=given_seed)
    
    prev_actions = np.array([0])
    info_state = np.zeros((nZ))
    #initial transient
    for transient_step in range(q):
        actions = 0.001*np.array(env.action_space.sample())
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        info_state = information_state(info_state, next_obs, actions)

        #print('info_state = ', info_state, ' actions=', actions)
    
    episode_t = 0 
    cost = 0
    noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(np.prod(env.action_space.shape)))

    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here

        prev_info_state = info_state

        if global_step < learning_starts:
            actions = np.array(env.action_space.sample()) #choose random action
            
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(info_state).to(device))
                actions += torch.Tensor(noise()).to(device) #torch.normal(0, actor.action_scale * exploration_noise)
                actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        rewards = reward_function(info_state, actions)
        cost -=rewards 
        
        info_state = information_state(prev_info_state, next_obs, actions)

        
        #print('step=', global_step, ' actions=', actions, ' rewards=', rewards,\
        #      ' obs=', next_obs)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_info_state = info_state.copy()

        # if truncations:
        #     real_next_os = infos["final_observation"]
        rb.add(prev_info_state, real_info_state, actions, rewards, terminations, infos)

        
        obs = next_obs
        prev_actions = actions

        # ALGO LOGIC: training.
        if global_step > learning_starts:
            data = rb.sample(batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            
            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                print('step=', global_step, ' rewards=', rewards, ' qf1_loss = ', qf1_loss.item(), \
                      ' actor_loss = ', actor_loss.item(), ' observations=', obs, ' action=', actions)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if global_step % 100 == 0:
                
                print("SPS:", int(global_step / (time.time() - start_time)))
                # Data to write
                write_data = [global_step, rewards, qf1_loss.item(), actor_loss.item(), obs, actions]
                write_data_csv(write_data)

        episode_t = episode_t + 1
        if abs(next_obs[0])>= 10 or episode_t == episode_length: #reset environment
            print('resetting')
            obs, _ = env.reset()
            
            info_state = np.zeros((nZ))
            #initial transient
            for transient_step in range(q):
                actions = 0.001*np.array(env.action_space.sample())
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                info_state = information_state(info_state, next_obs, actions)
            episode_t = 0
            print(f'Cost = {cost}')
            cost = 0


    save_model = True
    if save_model:
        model_path = f"models/{run_name}/{exp_name}.pth"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

    env.close()
    


