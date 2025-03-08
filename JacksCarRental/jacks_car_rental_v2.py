"""

Example 4.7

Jack's Car Rental
with nice employee
and parking lot limits

"""

import gymnasium as gym
import numpy as np
import pdb

from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import poisson

# Problem constants
MAX_CARS = 20  # Maximum cars per location
MAX_MOVE = 5   # Maximum cars moved per night
RENTAL_REWARD = 10
MOVE_COST = 2
DISCOUNT = 0.9  # Discount factor
EXTRA_LOT_COST = 4

# Poisson parameters
RENTAL_MEAN = [3, 4]  # Rental request means for locations 1 and 2
RETURN_MEAN = [3, 2]  # Return means for locations 1 and 2

# Not negative initialization because that is immediately a terminal state
V = np.zeros((MAX_CARS + 1) * (MAX_CARS + 1))  # State-value function

# Maximum movement is 5 cars, but we are starting with the no movement policy
policy = np.zeros((MAX_CARS + 1) * (MAX_CARS + 1), dtype=int)

# Poisson PMF helper
def poisson_pmf(n, lam):
    return poisson.pmf(n, lam)

# Precompute Poisson probabilities for efficiency
poisson_cache = {}
for lam in [3, 4, 2]:  # All lambda values
    poisson_cache[lam] = [poisson_pmf(n, lam) for n in range(11)]  # Up to 10 events

# Function to compute expected value for a state and action
def expected_return(state, action, V):
    
    s1 = state // (MAX_CARS + 1)
    s2 = state % (MAX_CARS + 1)

        
    s1 -= action  # Cars moved from location 1 to 2
    s2 += action

    # Enforce limits
    if s1 < 0 or s1 > MAX_CARS or s2 < 0 or s2 > MAX_CARS:
        return -np.inf

    # Reduce cost of actions by 1 because employee moves 1 car for free
    if action > 0:
        action -= 1

    # Immediate cost of moving cars
    reward = -MOVE_COST * abs(action)

    # Incurred cost of needing an extra lot at night
    if s1 > 10: 
        reward -= EXTRA_LOT_COST
    if s2 > 10:
        reward -= EXTRA_LOT_COST
    

    # Expected value calculation
    expected_value = 0.0
    for rent1 in range(min(11,s1)):  # Rentals at location 1
        for rent2 in range(min(11,s2)):  # Rentals at location 2
            for ret1 in range(11):  # Returns to location 1
                for ret2 in range(11):  # Returns to location 2
                    # Probabilities for the events
                    prob = (
                        poisson_cache[RENTAL_MEAN[0]][rent1]
                        * poisson_cache[RENTAL_MEAN[1]][rent2]
                        * poisson_cache[RETURN_MEAN[0]][ret1]
                        * poisson_cache[RETURN_MEAN[1]][ret2]
                    )

                    # Cars rented
                    rented1 = min(s1, rent1)
                    rented2 = min(s2, rent2)

                    # Revenue
                    total_reward = reward + RENTAL_REWARD * (rented1 + rented2)

                    # Cars remaining after rentals and returns
                    cars1 = min(MAX_CARS, s1 - rented1 + ret1)
                    cars2 = min(MAX_CARS, s2 - rented2 + ret2)

                    # Update expected value
                    expected_value += prob * (total_reward + DISCOUNT * V[cars1 * (MAX_CARS+1) + cars2])

    return expected_value

def maximum_return(state, V):

    s1 = state // (MAX_CARS + 1)
    s2 = state % (MAX_CARS + 1)

    a21 = -min(5,s2)
    a12 = min(5,s1)

    action_values = []
    for a in range(-5, 5+1):
        action_values = np.append(action_values, expected_return(state, a, V))

    return np.argmax(action_values) - 5




# Policy evaluation
def policy_evaluation(V, policy, theta=1e-4):
    while True:
        delta = 0
        for state in tqdm(range(0,len(V))):
            action = policy[state]
            v = V[state]
            V[state] = expected_return(state, action, V)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V


def policy_iteration(V, policy):
    stable = True
    for state in range(0, len(V)):
        a = policy[state]
        policy[state] = maximum_return(state, V)
        if a != policy[state]:
            stable = False

    return policy, stable

# Function to plot policy as a contour chart with lines and labels
def plot_all_policies(policies, max_cars=MAX_CARS):
    n_policies = len(policies)
    n_cols = 4  # Number of columns in the grid
    n_rows = int(np.ceil(n_policies / n_cols))  # Number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()  # Flatten the axes for easy indexing

    for i, policy in enumerate(policies):
        # Reshape policy for 2D visualization
        policy_reshaped = policy.reshape(max_cars + 1, max_cars + 1)

        ax = axes[i]
        c = ax.contourf(
            range(max_cars + 1),
            range(max_cars + 1),
            policy_reshaped,
            cmap="viridis",
            levels=np.arange(-MAX_MOVE, MAX_MOVE + 1)
        )
        fig.colorbar(c, ax=ax, label="Cars moved (action)")
        ax.set_title(f"Iteration {i}")
        ax.set_xlabel("Cars at location 2")
        ax.set_ylabel("Cars at location 1")

    # Hide unused subplots
    for j in range(len(policies), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


is_stable = False

policies = []

while True:
    policies.append(policy.copy())

    V = policy_evaluation(V, policy)
    print("State-value function after policy evaluation:")
    print(V)

    policy, is_stable = policy_iteration(V, policy)
    print("policy function after policy evaluation:")
    print(policy)

    if is_stable:
        policies.append(policy.copy())
        break
    

plot_all_policies(policies)
