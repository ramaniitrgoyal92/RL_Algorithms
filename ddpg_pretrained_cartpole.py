import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import random


import torch
import torch.nn as nn
import torch.nn.functional as F


ENV_NAME = 'InvertedPendulum-v4'
exp_name = 'cartpole_ep_30'
run_name = 'ddpg'

def make_env(env_id, render_bool, record_video=False):

    if record_video:
        env = gym.make('InvertedPendulum-v4',render_mode = "rgb_array")
        env = gym.wrappers.RecordVideo(env, f"models/{run_name}/{exp_name}")

    elif render_bool: 
        env = gym.make('InvertedPendulum-v4',render_mode = "human")

    else:
        env = gym.make('InvertedPendulum-v4')

    min_action = -20
    max_action = 20
    env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.reset()

    return env

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() 
                             + np.prod(env.action_space.shape), 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mu = nn.Linear(300, np.prod(env.action_space.shape))
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

if __name__ == "__main__":

    given_seed = 1
    total_timesteps = 1000
    gamma = 0.99

    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True
    
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}")

    env = make_env(ENV_NAME, render_bool = True, record_video=False)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    checkpoint = torch.load(f"models/{run_name}/{exp_name}.pth")
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
            actions = actor(torch.Tensor(obs).to(device))
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
    