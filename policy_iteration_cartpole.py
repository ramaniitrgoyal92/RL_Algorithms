import gymnasium as gym
import numpy as np

ENV_NAME = 'CartPole-v1'

def policy_evaluation(env, policy, value_table, gamma=0.99, theta=1e-6):
    """
    Evaluate the value function for a given policy.
    """
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            v = value_table[state]
            value_table[state] = sum(
                [prob * (reward + gamma * value_table[next_state])
                 for prob, next_state, reward, _ in env.P[state][policy[state]]]
            )
            delta = max(delta, abs(v - value_table[state]))
        if delta < theta:
            break
    return value_table

def policy_improvement(env, value_table, gamma=0.99):
    """
    Improve the policy based on the value function.
    """
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        q_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            q_values[action] = sum(
                [prob * (reward + gamma * value_table[next_state])
                 for prob, next_state, reward, _ in env.P[state][action]]
            )
        policy[state] = np.argmax(q_values)
    return policy

def policy_iteration(env, gamma=0.99):
    """
    Perform policy iteration to find the optimal policy.
    """
    policy = np.zeros(env.observation_space.n, dtype=int)
    value_table = np.zeros(env.observation_space.n)

    while True:
        value_table = policy_evaluation(env, policy, value_table, gamma)
        new_policy = policy_improvement(env, value_table, gamma)
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy

    return policy, value_table

if __name__ == "__main__":
    # Create the environment
    env = gym.make(ENV_NAME)

    # Policy Iteration
    optimal_policy, optimal_value = policy_iteration(env)

    print("Optimal Policy:")
    print(optimal_policy)

    print("Optimal Value Function:")
    print(optimal_value)

    env.close()