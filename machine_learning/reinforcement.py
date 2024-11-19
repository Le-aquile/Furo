import numpy as np
import random

class RLAgent:
    def __init__(self, state_bins=(10, 10, 10, 10), actions=["LEFT", "RIGHT", "NONE"], learning_rate=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99):
        """
        Reinforcement Learning Agent using Q-learning to maximize points.
        """
        self.state_bins = state_bins  # Discretization bins for state dimensions
        self.actions = actions  # Action space
        self.learning_rate = learning_rate  # Learning rate (alpha)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration

        # Initialize Q-table with dimensions: state_bins + action_space
        self.q_table = np.zeros((*state_bins, len(actions)))

    def discretize_state(self, state):
        """
        Convert continuous state to discrete bins.
        """
        enemy_x, enemy_y, player_x, player_y = state["enemy_x"], state["enemy_y"], state["player_x"], state["player_y"]

        # Normalize state values to the range [0, 1]
        norm_enemy_x = enemy_x / 800
        norm_enemy_y = enemy_y / 600
        norm_player_x = player_x / 800
        norm_player_y = player_y / 600

        # Discretize using bins
        discrete_enemy_x = int(norm_enemy_x * (self.state_bins[0] - 1))
        discrete_enemy_y = int(norm_enemy_y * (self.state_bins[1] - 1))
        discrete_player_x = int(norm_player_x * (self.state_bins[2] - 1))
        discrete_player_y = int(norm_player_y * (self.state_bins[3] - 1))

        return (discrete_enemy_x, discrete_enemy_y, discrete_player_x, discrete_player_y)

    def act(self, state):
        """
        Choose an action based on ε-greedy policy.
        """
        discrete_state = self.discretize_state(state)

        if random.random() < self.epsilon:
            # Explore: Choose a random action
            return random.choice(self.actions)
        else:
            # Exploit: Choose the action with the highest Q-value
            action_index = np.argmax(self.q_table[discrete_state])
            return self.actions[action_index]

    def update(self, state, action, reward, next_state):
        """
        Update Q-values using the Bellman equation.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_index = self.actions.index(action)

        # Compute TD target and TD error
        td_target = reward + self.gamma * np.max(self.q_table[discrete_next_state])
        td_error = td_target - self.q_table[discrete_state][action_index]

        # Update Q-value
        self.q_table[discrete_state][action_index] += self.learning_rate * td_error

        # Decay exploration rate
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        """
        Save the Q-table to a file.
        """
        np.save(filename, self.q_table)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """
        Load the Q-table from a file.
        """
        self.q_table = np.load(filename)
        print(f"Model loaded from {filename}")
