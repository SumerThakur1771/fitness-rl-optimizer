"""
Main RL training script for the fitness workout optimizer.

This script:
1. Loads workout scenarios
2. Initializes agents and RL algorithms
3. Trains for 50 episodes
4. Logs metrics to CSV
5. Shows learning progress

Run with: python src/main.py
"""

import json
import csv
import os

from fitness_agents import WorkoutAgent, IntensityAgent, RLController
from ucb_selector import UCBBandit
from q_learner import QLearningAgent
from reward_calculator import compute_category_reward


def load_scenarios():
    """
    Load workout scenarios from JSON file.
    
    Returns:
        list: List of scenario dictionaries
    """
    with open("data/workout_scenarios.json") as f:
        return json.load(f)


def main():
    """
    Main training function.
    
    Trains the RL system for 50 episodes, logging performance metrics.
    """
    print("="*60)
    print("FITNESS WORKOUT OPTIMIZER - RL TRAINING")
    print("="*60)
    print()
    
    # Load scenarios
    print("Loading workout scenarios...")
    scenarios = load_scenarios()
    print(f"✓ Loaded {len(scenarios)} scenarios")
    print()
    
    # Initialize agents
    print("Initializing agents...")
    workout_agent = WorkoutAgent()
    intensity_agent = IntensityAgent()
    print("✓ WorkoutAgent ready")
    print("✓ IntensityAgent ready")
    print()
    
    # Initialize UCB Bandit for category selection
    print("Initializing UCB Bandit...")
    exercise_categories = [
        "strength",
        "cardio",
        "flexibility",
        "compound",
        "isolation",
        "hiit",
        "plyometric",
        "yoga",
        "low_impact"
    ]
    bandit = UCBBandit(
        actions=exercise_categories,
        c=2.0  # Exploration constant
    )
    print("✓ UCB Bandit ready with {} categories".format(len(exercise_categories)))
    print()
    
    # Initialize Q-Learning Agent for workflow decisions
    print("Initializing Q-Learning Agent...")
    q_agent = QLearningAgent(
        actions=["add_strength", "add_cardio", "add_flexibility", "finalize"],
        alpha=0.1,    # Learning rate
        gamma=0.9,    # Discount factor
        epsilon=0.1   # Exploration rate
    )
    print("✓ Q-Learning Agent ready")
    print("  - Learning rate (α): 0.1")
    print("  - Discount factor (γ): 0.9")
    print("  - Exploration rate (ε): 0.1")
    print()
    
    # Initialize RL Controller
    print("Initializing RL Controller...")
    controller = RLController(workout_agent, intensity_agent, bandit, q_agent)
    print("✓ RL Controller ready")
    print()
    
    # Training parameters
    num_episodes = 50
    print(f"Starting training for {num_episodes} episodes...")
    print("="*60)
    print()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Open CSV file for logging
    metrics_path = "logs/rl_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "avg_reward", "avg_exercises"])
        
        # Training loop
        for episode in range(1, num_episodes + 1):
            episode_total_reward = 0.0
            episode_total_exercises = 0
            
            # Run through all scenarios
            for scenario in scenarios:
                reward, exercises = controller.run_episode(
                    scenario,
                    compute_category_reward
                )
                episode_total_reward += reward
                episode_total_exercises += exercises
            
            # Calculate averages
            avg_reward = episode_total_reward / len(scenarios)
            avg_exercises = episode_total_exercises / len(scenarios)
            
            # Log to CSV
            writer.writerow([episode, avg_reward, avg_exercises])
            
            # Print progress
            print(f"Episode {episode:2d}/{num_episodes}: "
                  f"avg_reward = {avg_reward:6.3f}, "
                  f"avg_exercises = {avg_exercises:.2f}")
    
    print()
    print("="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print()
    print(f"✓ Metrics saved to: {metrics_path}")
    print(f"✓ Q-table size: {q_agent.get_q_table_size()} state-action pairs learned")
    print()
    print("Next steps:")
    print("1. Run: python src/plot_learning_curve.py  (to generate plots)")
    print("2. Run: python src/main_fixed.py           (to see baseline comparison)")
    print()
    
    # Show bandit statistics
    print("UCB Bandit Statistics (Category Selection):")
    print("-" * 60)
    stats = bandit.get_statistics()
    for category, info in sorted(stats.items(), key=lambda x: x[1]['avg_reward'], reverse=True):
        if info['count'] > 0:
            print(f"  {category:15s}: avg_reward = {info['avg_reward']:5.2f}, "
                  f"count = {info['count']:3d}")
    print()


if __name__ == "__main__":
    main()