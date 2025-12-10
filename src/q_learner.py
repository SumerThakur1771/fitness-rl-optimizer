"""
Tabular Q-Learning agent for workout workflow decisions.

This agent learns which action to take in each state:
- 'add_strength': Add a strength training exercise
- 'add_cardio': Add a cardio exercise
- 'add_flexibility': Add a flexibility exercise
- 'finalize': Stop and create the workout plan
"""

import random
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent for high-level workout planning decisions.
    
    Uses epsilon-greedy exploration and standard Q-Learning update rule:
    Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
    """
    
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize Q-Learning agent.
        
        Args:
            actions: List of possible actions (e.g., ['add_strength', 'add_cardio', ...])
            alpha: Learning rate (0.1 = learn slowly but steadily)
            gamma: Discount factor (0.9 = value future rewards highly)
            epsilon: Exploration rate (0.1 = explore 10% of the time)
        """
        self.actions = actions
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.q = defaultdict(float) # Q-table: q[(state, action)] = value
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        - With probability epsilon: choose random action (explore)
        - With probability 1-epsilon: choose best action (exploit)
        
        Args:
            state: Current state (e.g., (2, "intermediate", "medium_time"))
            
        Returns:
            action: Selected action (e.g., "add_strength")
        """
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Exploitation: best action based on Q-values
        q_values = [self.q[(state, a)] for a in self.actions]
        max_q = max(q_values)
        
        # Handle ties: if multiple actions have same max Q-value, pick randomly
        best_actions = [a for a in self.actions if self.q[(state, a)] == max_q]
        return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-value using the Q-Learning formula.
        
        Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after taking action
        """
        # Find the maximum Q-value for the next state
        max_next_q = max(self.q[(next_state, a)] for a in self.actions)
        
        # Current Q-value
        old_q = self.q[(state, action)]
        
        # Q-Learning update
        self.q[(state, action)] = old_q + self.alpha * (
            reward + self.gamma * max_next_q - old_q
        )
    
    def get_q_table_size(self):
        """Return the number of state-action pairs learned."""
        return len(self.q)