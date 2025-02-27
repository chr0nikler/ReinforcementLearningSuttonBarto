import numpy as np
from scipy.stats import poisson
from tqdm import tqdm

# Problem constants
MAX_CARS = 20  # Maximum cars per location
MAX_MOVE = 5   # Maximum cars moved per night
RENTAL_REWARD = 10
MOVE_COST = 2
DISCOUNT = 0.9  # Discount factor

# Poisson parameters
RENTAL_MEAN = [3, 4]  # Rental request means for locations 1 and 2
RETURN_MEAN = [3, 2]  # Return means for locations 1 and 2

# Poisson PMF helper
def poisson_pmf(n, lam):
    return poisson.pmf(n, lam)

# Initialize state-value function and policy
V = np.zeros((MAX_CARS + 1, MAX_CARS + 1))  # State-value function
policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)  # Action for each state

# Precompute Poisson probabilities for efficiency
poisson_cache = {}
for lam in [3, 4, 2]:  # All lambda values
    poisson_cache[lam] = [poisson_pmf(n, lam) for n in range(11)]  # Up to 10 events

# Function to compute expected value for a state and action
def expected_return(state, action, V):
    s1, s2 = state
    s1 -= action  # Cars moved from location 1 to 2
    s2 += action

    # Enforce limits
    if s1 < 0 or s1 > MAX_CARS or s2 < 0 or s2 > MAX_CARS:
        return -np.inf

    # Immediate cost of moving cars
    reward = -MOVE_COST * abs(action)

    # Expected value calculation
    expected_value = 0.0
    for rent1 in range(11):  # Rentals at location 1
        for rent2 in range(11):  # Rentals at location 2
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
                    expected_value += prob * (total_reward + DISCOUNT * V[cars1, cars2])

    return expected_value

# Policy evaluation
def policy_evaluation(V, policy, theta=1e-4):
    while True:
        delta = 0
        for s1 in tqdm(range(MAX_CARS + 1)):
            for s2 in range(MAX_CARS + 1):
                state = (s1, s2)
                action = policy[s1, s2]
                v = V[s1, s2]
                V[s1, s2] = expected_return(state, action, V)
                delta = max(delta, abs(v - V[s1, s2]))
        print(delta)
        if delta < theta:
            break
    return V

# Run policy evaluation
V = policy_evaluation(V, policy)

# Display results
print("State-value function after policy evaluation:")
print(V)
