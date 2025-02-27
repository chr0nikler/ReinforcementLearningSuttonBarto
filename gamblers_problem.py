"""

Example 4.3: Gambler's Problem


"""

import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

STATES = np.arange(101)

V = np.empty(101)
V[0] = 0
V[100] = 1

policy = np.zeros(101, dtype=int)


THETA = 0
DISCOUNT = 0

P_HEADS = 0.55

sweeps = np.empty((0,len(V)))

while True:
    delta = 0
    
    for s in tqdm(STATES):
        if s == 0 or s == 100:
            continue
        else:
            v = V[s]
            action_values = np.zeros(min(s, 100-s) + 1)
            for a in range(len(action_values)):
                action_values[a] = P_HEADS * V[s+a] + \
                    (1- P_HEADS) * V[s-a]
            max_a = np.argmax(action_values)
            V[s] = action_values[max_a]
            policy[s] = max_a
            delta = max(delta, abs(v-V[s]))
    sweeps = np.append(sweeps, np.array([V]), axis=0)
    print(policy)
    if delta <= THETA:
        print(delta)
        break

for i, sweep in enumerate(sweeps):
    plt.plot(np.arange(len(sweep)), sweep, linestyle='-')

plt.xlabel("Capital (State)")
plt.ylabel("Value Estimates")
plt.title("State-Value Estimates")

# Add a legend

# Display the plot
plt.show()

print(policy)

plt.plot(np.arange(len(policy)), policy, marker='o')

plt.xlabel("Capital (state)")
plt.ylabel("Stake (action)")
plt.title("Policy")
# Display the plot
plt.show()




