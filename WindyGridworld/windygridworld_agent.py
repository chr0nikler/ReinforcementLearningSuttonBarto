from windygridworld_env import WindyGridworldEnv
from tqdm import tqdm
import numpy as np
import random
import copy
import pdb



class WindyGridworldAgent:


    def __init__(self, env, target_policy):


        self.env = env

        self.Q = np.zeros((7,10,4)) # Initialize Q to zero
        self.discount = 1
        self.alpha = 0.5
        self.epsilon = 0.1
        self.pi = target_policy

        self.old_q = copy.deepcopy(self.Q)

    def learn(self):

        truncated = False
        terminated = False

        S, _ = self.env.reset()
        A = self.pi(self.env, self.Q, S, self.epsilon)

        while not terminated and not truncated:
            S_p, R, terminated, truncated, _ = self.env.step(A)
            A_p = self.pi(self.env, self.Q, S_p, self.epsilon)
            # SARSA
            # self.Q[*S,A] += self.alpha * (R + self.discount * self.Q[*S_p, A_p] - self.Q[*S, A])
            # Q-learning
            self.Q[*S,A] += self.alpha * (R + self.discount * np.max(self.Q[*S_p]) - self.Q[*S, A])
            S = S_p
            A = A_p

        print("done")
        q_changes = np.sum(self.old_q != self.Q)  
        total_states = self.Q.size
        change_percentage = (q_changes / total_states) * 100
        self.old_q = copy.deepcopy(self.Q)
        print(f"Q changed in {change_percentage:.2f}% of states")

        if change_percentage < 0.01:
            return True

    def exploit(self):

        truncated = False
        terminated = False

        S, _ = self.env.reset()

        while not terminated or truncated:
            A = self.pi(self.env, self.Q, S, 0)
            S, _, terminated, truncated, _ = self.env.step(A, 1)


def pi(env, q, s, eps):
    RNG = np.random.default_rng()

    if RNG.random() < eps:
        return env.action_space.sample()
    else:
        return np.argmax(q[*s])

ENV = WindyGridworldEnv('human')
EPISODES = 200

agent = WindyGridworldAgent(ENV, pi)

for ep in tqdm(range(0, EPISODES)):
    if agent.learn():
        break

agent.exploit()

ENV.close()

        






