"""

Exercise 5.12 Racetrack

Implementing a Monte Carlo agent using
off policy weighted importance sampling

We'll choose behavior policy b to be ε-soft

"""


from racetrack_env import RacetrackEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import pdb
import copy

class DriverAgent:


    def __init__(self, env, behavior_policy, discount_rate, epsilon):

        RNG = np.random.default_rng()

        self.env = env 
        self.b = behavior_policy
        
        track = env.track.shape
        self.Q = -100 * RNG.random((track[1],track[0],5,5,9))
        self.C = np.zeros((track[1],track[0],5,5,9))
        self.pi = np.zeros((track[1],track[0],5,5), dtype=int)

        self.discount_rate = discount_rate
        self.epsilon = epsilon

        self.states = np.array([])
        self.actions = np.array([], dtype=int)
        self.rewards = np.array([0])

        self.episodes = 0
        self.old_pi = copy.deepcopy(self.pi)
        self.old_q = copy.deepcopy(self.Q)

    def __explore(self):

        done = False
        truncated = False

        observation, info = self.env.reset(None,False)

        self.states = np.array([observation])
        self.actions = np.array([], dtype=int)
        self.rewards = np.array([0])

        while not done and not truncated:
            #print(observation)
            a = b(self.env, self.Q, observation, self.epsilon)
            self.actions = np.append(self.actions, a)

            observation, reward, done, truncated, _ = ENV.step(a,True,False)
            self.states = np.append(self.states, [observation], axis=0)
            self.rewards = np.append(self.rewards, reward)

        self.episodes += 1

    def race(self):
        done = False
        truncated = False
        observation, info = ENV.reset()
        states = np.array([observation])

        while not done and not truncated:
            a = self.pi[*observation]
            observation, _, done, truncated, _  = ENV.step(a,False,True,1)
            states = np.append(states, [observation], axis=0)

        return states


    def learn(self):
        G = 0
        W = 1

        self.__explore()
        
        for t in range(len(self.states)-2, -1, -1):
            St = self.states[t]
            At = self.actions[t]
            G = self.discount_rate * G + self.rewards[t+1]
            self.C[*St, At] += W
            self.Q[*St, At] += (W/self.C[*St, At]) * (G - self.Q[*St, At])

            self.pi[*St] = np.argmax(self.Q[*St])
            if At != self.pi[*St]:
                break
            W *= 1/(1-self.epsilon + (self.epsilon/9))
            W = max(W, 1e-6)

        if self.episodes % 1000 == 0:
            self.__log_policy_change()

        if self.episodes % 20000 == 0:
            self.epsilon /= 2

    def __log_policy_change(self):
        policy_changes = np.sum(self.old_pi != self.pi)  # Count number of states where policy changed
        total_states = self.pi.size
        change_percentage = (policy_changes / total_states) * 100
        print(f"Episode {self.episodes}: Policy changed in {change_percentage:.2f}% of states")
        #self.old_pi = copy.deepcopy(self.pi)


def b(env, q, s, epsilon):
    """
    policy randomly samples action space, ignoring state
    """
    RNG = np.random.default_rng()
    if RNG.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q[*s])



TRACK_NUMBER = 2
EPSILON = 0.5
DISCOUNT_RATE = 0.9
EPISODES = 100000
RENDER_MODE="human"


ENV = RacetrackEnv(TRACK_NUMBER,RENDER_MODE)

agent = DriverAgent(ENV, b, DISCOUNT_RATE, EPSILON)

for ep in tqdm(range(0, EPISODES)):
    agent.learn()


for i in range(0,10):
    agent.race()


ENV.close()
