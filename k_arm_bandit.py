import gymnasium as gym
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt

# Gymnasium implementation of k-armed bandit problem
class BanditEnv(gym.Env):
    def __init__(self, k_arms: int):
        super().__init__()
        self.k_arms = k_arms
        self.action_space = gym.spaces.Discrete(k_arms)
        self.observation_space = gym.spaces.Discrete(1)  # A single state
        self.q_star = self.np_random.normal(0, 1, k_arms)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.q_star = self.np_random.normal(0, 1, self.k_arms)
        return 0, {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        reward = self.np_random.normal(self.q_star[action], 1)
        return 0, reward, False, False, {}

class BanditAgent:
   
    def __init__(self, k_arms: int, epsilon: float):
        self.k_arms = k_arms
        self.epsilon = epsilon
        self.rng = np.random.default_rng()

        self.__setup()

    def act(self):
        """ε-greedy decision"""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.k_arms)  # Explore
        else:
            return np.argmax(self.q_estimates)  # Exploit

    def update(self, action: int, reward: float):
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]

    def reset(self):
        self.__setup()

    def __setup(self):
        self.q_estimates = np.zeros(self.k_arms)
        self.action_counts = np.zeros(self.k_arms)
        self.rng = np.random.default_rng()

class GreedyBanditAgent(BanditAgent):
    def __init__(self, k_arms):
        super().__init__(k_arms, 0) # Exploit Always

class ConstantLRBanditAgent(BanditAgent):
    def __init__(self, k_arms: int, epsilon: float, learning_rate: float):
        super().__init__(k_arms, epsilon)
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError(f"{learning_rate!r} is outside value thresholds, will not result in convergence")

        self.learning_rate = learning_rate

    def update(self, action: int, reward: float):
        self.action_counts[action] += 1
        self.q_estimates[action] += self.learning_rate * (reward - self.q_estimates[action]) 

if  __name__ == "__main__":
    n_steps = 1000
    n_episodes = 2000

    k_arms = 10
    learning_rate = 0.5
    explore_epsilon = 0.1
    exploit_epsilon = 0.01
    q_init = 5

    env = BanditEnv(k_arms)
    agents = [
        BanditAgent(k_arms, explore_epsilon),
        BanditAgent(k_arms, exploit_epsilon),
        GreedyBanditAgent(k_arms),
        ConstantLRBanditAgent(k_arms, explore_epsilon, learning_rate),
    ]

    all_rewards = np.zeros((n_episodes, len(agents),  n_steps))
    all_optimal_actions = np.zeros((n_episodes, len(agents), n_steps))

    # Run 2000 different k-armed bandits for n_steps
    for episode in tqdm(range(n_episodes)):
        env.reset()
        optimal_action = np.argmax(env.q_star)
        for a, agent in enumerate(agents):
            agent.reset()
            rewards = np.zeros(n_steps)
            optimal_action_counts = np.zeros(n_steps)
            for step in range(n_steps):
                action = agent.act()
                _, reward, _, _, _ = env.step(action)

                agent.update(action, reward)

                rewards[step] = reward
                if action == optimal_action:
                    optimal_action_counts[step] = 1

            all_rewards[episode][a] = rewards
            all_optimal_actions[episode][a] = optimal_action_counts


    avg_rewards = np.mean(all_rewards, axis=0)
    optimal_action_percentage = np.mean(all_optimal_actions, axis=0) * 100


    # Plot the results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards[0], label=f"ε={explore_epsilon}", color="blue")
    plt.plot(avg_rewards[1], label=f"ε={exploit_epsilon}", color="orange")
    plt.plot(avg_rewards[2], label=f"greedy", color="green")
    plt.plot(avg_rewards[3], label=f"ε={explore_epsilon}, α=0.5", color="red")
    # Add labels directly on the lines
    plt.title("Average Reward")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.legend()



    plt.subplot(1, 2, 2)
    plt.plot(optimal_action_percentage[0], label=f"ε={explore_epsilon}")
    plt.plot(optimal_action_percentage[1], label=f"ε={exploit_epsilon}")
    plt.plot(optimal_action_percentage[2], label=f"greedy")
    plt.plot(optimal_action_percentage[3], label=f"ε={explore_epsilon}, α={learning_rate}")
    plt.title("Optimal Action Percentage")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()

    plt.show()



