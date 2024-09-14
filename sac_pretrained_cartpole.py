import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import random


import torch
import torch.nn as nn
import torch.nn.functional as F


ENV_NAME = 'InvertedPendulum-v4'
csv_file = 'sac_cartpole_output.csv' #csv file to store training progress
exp_name = 'sac_cartpole_ep_120'
run_name = 'sac'

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_mean = nn.Linear(512, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(512, np.prod(env.action_space.shape))
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

        print(f"log_std = {log_std.exp()}")
        std = log_std.exp() 
        normal = torch.distributions.Normal(mean, std)
        #x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        x_t = mean #makeing the action deterministic
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        #print(f"x_t = {x_t} y_t = {y_t} action = {action} log_prob = {log_prob} log_prob.shape = {log_prob.shape}")
        log_prob = log_prob.sum(0, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

def make_env(env_id, render_bool, record_video=False):

    if record_video:
        env = gym.make('InvertedPendulum-v4',render_mode = "rgb_array")
        env = gym.wrappers.RecordVideo(env, f"../videos/{run_name}")

    elif render_bool: 
        env = gym.make('InvertedPendulum-v4',render_mode = "human")

    else:
        env = gym.make('InvertedPendulum-v4')

    min_action = -20
    max_action = 20
    env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.reset()

    return env


if __name__ == "__main__":

    given_seed = 1
    total_timesteps = 120
    gamma = 0.99

    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True
    
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}")

    env = make_env(ENV_NAME, render_bool = True, record_video=True)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    checkpoint = torch.load(f"../runs/{run_name}/{exp_name}.pth")
    actor.load_state_dict(checkpoint[0])
    qf1.load_state_dict(checkpoint[1])

    actor.eval()
    qf1.eval() 

    obs, _ = env.reset(seed=1)
    # print(f'obs={obs}')
    # q_pos = np.array([-0.1,0.0])
    # q_vel = np.array([0,0])
    # env.set_state(q_pos, q_vel)
    # obs, rewards, terminations, truncations, infos = env.step(0.0)
    # print(f'obs={obs}')
    for global_step in range(total_timesteps):
        with torch.no_grad():
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            cost_to_go = -qf1(torch.Tensor(obs).to(device), actions).item()
            actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        if terminations:
            obs, _ = env.reset()
            # env.set_state(q_pos, q_vel)
            # obs, rewards, terminations, truncations, infos = env.step(0.0)
            
        #env.render()
        
        print("observation:", next_obs, " action:", actions, ' CTG=', cost_to_go)
        
        obs = next_obs

    env.close()
    