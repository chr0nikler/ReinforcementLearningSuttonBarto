import numpy as np
import matplotlib.pyplot as plt

class GamblersProblem:
    def __init__(self, goal=100, ph=0.4):
        self.goal = goal  # The goal amount of money
        self.ph = ph      # Probability of the coin coming up heads
        self.states = range(1, self.goal)  # States (1 to goal-1)
        self.V = np.zeros(self.goal + 1)  # State-value function
        self.policy = np.zeros(self.goal + 1, dtype=int)  # Policy (bet size)

    def value_iteration(self, theta=1e-9):
        """Perform value iteration to compute the optimal policy."""
        while True:
            delta = 0
            for s in self.states:
                # Compute the action-value function for all bets
                action_values = []
                for bet in range(1, min(s, self.goal - s) + 1):
                    win_state = s + bet
                    lose_state = s - bet
                    reward = 1 if win_state == self.goal else 0
                    action_value = (
                        self.ph * (reward + self.V[win_state]) +
                        (1 - self.ph) * self.V[lose_state]
                    )
                    action_values.append(action_value)

                # Update the state-value function
                best_value = max(action_values)
                delta = max(delta, abs(self.V[s] - best_value))
                self.V[s] = best_value

            if delta < theta:
                break

        # Derive the policy from the optimal value function
        for s in self.states:
            action_values = []
            for bet in range(1, min(s, self.goal - s) + 1):
                win_state = s + bet
                lose_state = s - bet
                reward = 1 if win_state == self.goal else 0
                action_value = (
                    self.ph * (reward + self.V[win_state]) +
                    (1 - self.ph) * self.V[lose_state]
                )
                action_values.append(action_value)

            self.policy[s] = np.argmax(action_values) + 1

    def display_results(self):
        """Display the computed policy and value function."""
        print("Optimal Policy (Bet Sizes):")
        print(self.policy)
        print("\nState-Value Function:")
        print(self.V)

    def plot_policy(self):
        """Plot the optimal policy."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.policy, drawstyle="steps-post")
        plt.xlabel("Capital")
        plt.ylabel("Final Bet Size")
        plt.title("Optimal Policy for Gambler's Problem")
        plt.grid(True)
        plt.show()

# Main Simulation
if __name__ == "__main__":
    gamblers = GamblersProblem(goal=100, ph=0.4)
    gamblers.value_iteration()
    gamblers.display_results()
    gamblers.plot_policy()
