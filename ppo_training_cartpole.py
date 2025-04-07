import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.distributions import Normal
from torch.distributions.categorical import Categorical

# Hyperparameters
ENV_NAME = 'CartPole-v1'
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_loss_coef = 0.5
batch_size = 64
num_epochs = 10
total_timesteps = 20000
update_timesteps = 2048

# Helper function to initialize layers
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# PPO Agent
class PPOAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.prod(env.observation_space.shape)
        act_dim = env.action_space.n

        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)

# Rollout buffer for PPO
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

# PPO Training Loop
def train_ppo():
    env = gym.make(ENV_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    agent = PPOAgent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    buffer = RolloutBuffer()
    obs = env.reset()[0]
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    global_step = 0

    while global_step < total_timesteps:
        # Collect rollout data
        for _ in range(update_timesteps):
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs)
            next_obs, reward, done, _, _ = env.step(action.cpu().numpy())
            buffer.obs.append(obs.cpu().numpy())
            buffer.actions.append(action.cpu().numpy())
            buffer.log_probs.append(log_prob.cpu().numpy())
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            buffer.values.append(value.cpu().numpy())

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            global_step += 1

            if done:
                obs = torch.tensor(env.reset()[0], dtype=torch.float32, device=device)

        # Compute advantages and returns
        with torch.no_grad():
            next_value = agent.critic(obs).cpu().numpy()
        returns = []
        advantages = []
        gae = 0
        for step in reversed(range(len(buffer.rewards))):
            if buffer.dones[step]:
                delta = buffer.rewards[step] - buffer.values[step]
                gae = delta
            else:
                delta = buffer.rewards[step] + gamma * buffer.values[step + 1] - buffer.values[step]
                gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + buffer.values[step])

        # Convert buffer data to tensors
        obs = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(buffer.actions), dtype=torch.long, device=device)
        log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=device)
        returns = torch.tensor(np.array(returns), dtype=torch.float32, device=device)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32, device=device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(num_epochs):
            for i in range(0, len(obs), batch_size):
                batch_obs = obs[i:i + batch_size]
                batch_actions = actions[i:i + batch_size]
                batch_log_probs = log_probs[i:i + batch_size]
                batch_returns = returns[i:i + batch_size]
                batch_advantages = advantages[i:i + batch_size]

                _, new_log_probs, entropy, new_values = agent.get_action_and_value(batch_obs, batch_actions)
                new_values = new_values.view(-1)

                # Policy loss
                ratio = (new_log_probs - batch_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(batch_returns, new_values)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        buffer.clear()

    env.close()

if __name__ == "__main__":
    train_ppo()
