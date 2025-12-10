"""
UCB1 (Upper Confidence Bound) Bandit algorithm for exercise category selection.

This helps the agent balance:
- Exploitation: Use exercise categories that worked well before
- Exploration: Try new exercise categories to discover better options

Formula: UCB(category) = average_reward + c × sqrt(ln(total_pulls) / pulls_for_this_category)
"""

import math
import random
from collections import defaultdict


class UCBBandit:
    """
    UCB1 bandit for balancing exploration and exploitation.
    
    Each 'arm' represents an exercise category (strength, cardio, flexibility, etc.)
    The bandit learns which categories give best rewards for different scenarios.
    """
    
    def __init__(self, actions, c=2.0):
        """
        Initialize UCB bandit.
        
        Args:
            actions: List of exercise categories (e.g., ['strength', 'cardio', 'flexibility'])
            c: Exploration constant (higher = more exploration)
               Typical value: 2.0 (balances exploration/exploitation well)
        """
        self.actions = actions
        self.c = c
        self.q_values = defaultdict(float)      # Average reward for each category
        self.action_counts = defaultdict(int)   # How many times each category was tried
        self.total_steps = 0                    # Total number of selections made
    
    def select_action(self):
        """
        Select an exercise category using UCB1 criterion.
        
        Strategy:
        1. Try each category at least once (initialization)
        2. Then pick category with highest UCB score:
           UCB = average_reward + exploration_bonus
        
        Returns:
            action: Selected exercise category (e.g., "strength")
        """
        self.total_steps += 1
        
        # Phase 1: Try each action at least once
        for action in self.actions:
            if self.action_counts[action] == 0:
                return action
        
        # Phase 2: Use UCB formula to balance exploration/exploitation
        ucb_scores = {}
        for action in self.actions:
            n_action = self.action_counts[action]
            
            # Exploitation term: how good is this category on average?
            exploit = self.q_values[action]
            
            # Exploration term: bonus for under-explored categories
            explore = self.c * math.sqrt(math.log(self.total_steps) / n_action)
            
            ucb_scores[action] = exploit + explore
        
        # Pick the action with highest UCB score
        max_score = max(ucb_scores.values())
        best_actions = [a for a, score in ucb_scores.items() if score == max_score]
        
        return random.choice(best_actions)
    
    def update(self, action, reward):
        """
        Update the estimated value of an exercise category.
        
        Uses incremental mean update:
        new_average = old_average + (new_reward - old_average) / count
        
        Args:
            action: Exercise category that was used (e.g., "strength")
            reward: Reward received from using this category
        """
        self.action_counts[action] += 1
        n = self.action_counts[action]
        
        # Incremental update of average reward
        old_q = self.q_values[action]
        self.q_values[action] = old_q + (reward - old_q) / n
    
    def get_statistics(self):
        """
        Get current statistics for all exercise categories.
        
        Returns:
            dict: {category: {'avg_reward': float, 'count': int}}
        """
        stats = {}
        for action in self.actions:
            stats[action] = {
                'avg_reward': self.q_values[action],
                'count': self.action_counts[action]
            }
        return stats