"""
Inverse Reinforcement Learning (IRL) Example for Autonomous Cars

This script demonstrates a basic IRL pipeline using expert demonstrations
to recover a reward function for autonomous driving in a simulated environment.
It uses a simplified grid world as a placeholder for a car environment.
Replace the environment and feature extraction with your own car simulation as needed.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# --- Simple GridWorld Environment as a placeholder for a car environment ---
class SimpleCarEnv(gym.Env):
    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.reset()

    def reset(self, seed=None, options=None):
        self.pos = np.array([0, 0])
        self.goal = np.array([self.grid_size-1, self.grid_size-1])
        return self.pos.copy(), {}

    def step(self, action):
        if action == 0:   # up
            self.pos[1] = min(self.grid_size-1, self.pos[1]+1)
        elif action == 1: # down
            self.pos[1] = max(0, self.pos[1]-1)
        elif action == 2: # left
            self.pos[0] = max(0, self.pos[0]-1)
        elif action == 3: # right
            self.pos[0] = min(self.grid_size-1, self.pos[0]+1)
        done = np.array_equal(self.pos, self.goal)
        reward = 1.0 if done else -0.01
        return self.pos.copy(), reward, done, False, {}

# --- Feature Extraction ---
def feature_fn(state):
    # Simple one-hot encoding of position for demonstration
    feat = np.zeros((env.grid_size, env.grid_size))
    feat[state[0], state[1]] = 1
    return feat.flatten()

# --- Generate Expert Demonstrations (shortest path) ---
def generate_expert_trajectories(env, n_trajectories=20):
    expert_states = []
    expert_actions = []
    for _ in range(n_trajectories):
        state, _ = env.reset()
        traj_states = []
        traj_actions = []
        while not np.array_equal(state, env.goal):
            if state[0] < env.goal[0]:
                action = 3  # right
            elif state[1] < env.goal[1]:
                action = 0  # up
            else:
                action = env.action_space.sample()
            traj_states.append(state.copy())
            traj_actions.append(action)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        expert_states.extend(traj_states)
        expert_actions.extend(traj_actions)
    return np.array(expert_states), np.array(expert_actions)

# --- Maximum Entropy IRL (simplified version) ---
class LinearRewardIRL(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.reward_weights = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, features):
        return torch.matmul(features, self.reward_weights)

def compute_feature_expectations(states, feature_fn):
    feats = np.array([feature_fn(s) for s in states])
    return feats.mean(axis=0)

def value_iteration(env, reward_fn, gamma=0.9, n_iters=100):
    V = np.zeros((env.grid_size, env.grid_size))
    for _ in range(n_iters):
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                state = np.array([x, y])
                values = []
                for a in range(env.action_space.n):
                    next_state, _, _, _, _ = env.step(a)
                    r = reward_fn(torch.tensor(feature_fn(next_state), dtype=torch.float32)).item()
                    values.append(r + gamma * V[next_state[0], next_state[1]])
                V[x, y] = max(values)
    return V

def irl(env, expert_states, feature_fn, n_iters=50, lr=0.1):
    feature_dim = feature_fn(env.reset()[0]).shape[0]
    model = LinearRewardIRL(feature_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    expert_feat_exp = compute_feature_expectations(expert_states, feature_fn)
    for it in tqdm(range(n_iters), desc="IRL Training"):
        # Policy evaluation (here, just random sampling for demonstration)
        sampled_states = []
        state, _ = env.reset()
        for _ in range(len(expert_states)):
            action = env.action_space.sample()
            state, _, done, _, _ = env.step(action)
            sampled_states.append(state.copy())
            if done:
                state, _ = env.reset()
        learner_feat_exp = compute_feature_expectations(sampled_states, feature_fn)

        # IRL loss: maximize expert - learner feature expectations
        loss = -torch.dot(model.reward_weights, torch.tensor(expert_feat_exp - learner_feat_exp, dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 10 == 0:
            print(f"Iteration {it}: Loss={loss.item():.4f}")

    return model

if __name__ == "__main__":
    env = SimpleCarEnv(grid_size=10)

    # Generate expert demonstrations
    expert_states, expert_actions = generate_expert_trajectories(env, n_trajectories=30)
    print(f"Collected {len(expert_states)} expert states.")

    # Run IRL to recover reward weights
    irl_model = irl(env, expert_states, feature_fn, n_iters=50, lr=0.1)
    print("Recovered reward weights:", irl_model.reward_weights.data.numpy())

    # Example: Evaluate the learned reward on a random state
    test_state = np.array([5, 5])
    test_feat = torch.tensor(feature_fn(test_state), dtype=torch.float32)
    print("Reward for state [5,5]:", irl_model(test_feat).item())