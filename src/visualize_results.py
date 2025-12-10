"""
Utility script to generate learning curve visualizations.

This script:
1. Reads metrics from logs/rl_metrics.csv
2. Creates two plots:
   - Reward learning curve (shows improvement)
   - Exercise efficiency curve (shows optimization)
3. Saves plots as PNG images

Run AFTER training: python src/plot_learning_curve.py
"""

import csv
import matplotlib.pyplot as plt
import os


def load_metrics(path="logs/rl_metrics.csv"):
    """
    Load training metrics from CSV file.
    
    Args:
        path: Path to metrics CSV file
        
    Returns:
        tuple: (episodes, avg_rewards, avg_exercises)
    """
    episodes = []
    avg_rewards = []
    avg_exercises = []
    
    if not os.path.exists(path):
        print(f"Error: {path} not found!")
        print("Please run 'python src/main.py' first to generate training data.")
        return None, None, None
    
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            avg_rewards.append(float(row["avg_reward"]))
            avg_exercises.append(float(row["avg_exercises"]))
    
    return episodes, avg_rewards, avg_exercises


def plot_reward_curve(episodes, avg_rewards):
    """
    Create and save reward learning curve.
    
    Shows how average reward improves over episodes.
    Upward trend indicates successful learning!
    
    Args:
        episodes: List of episode numbers
        avg_rewards: List of average rewards per episode
    """
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, avg_rewards, marker="o", linewidth=2, markersize=4, color='#2E86AB')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title("RL Learning Curve: Average Reward per Episode", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = "logs/learning_curve_reward.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_efficiency_curve(episodes, avg_exercises):
    """
    Create and save exercise efficiency curve.
    
    Shows how many exercises the agent selects per workout.
    Stabilization indicates the agent learned optimal stopping point.
    
    Args:
        episodes: List of episode numbers
        avg_exercises: List of average exercises per episode
    """
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, avg_exercises, marker="s", linewidth=2, markersize=4, color='#A23B72')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Number of Exercises", fontsize=12)
    plt.title("Policy Efficiency: Exercises per Workout", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = "logs/learning_curve_exercises.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_statistics(episodes, avg_rewards, avg_exercises):
    """
    Print summary statistics of training.
    
    Args:
        episodes: List of episode numbers
        avg_rewards: List of average rewards
        avg_exercises: List of average exercises
    """
    print()
    print("="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print()
    
    # Reward statistics
    print("Reward Statistics:")
    print(f"  Initial (Episode 1):  {avg_rewards[0]:.3f}")
    print(f"  Final (Episode {episodes[-1]}):    {avg_rewards[-1]:.3f}")
    print(f"  Improvement:          {avg_rewards[-1] - avg_rewards[0]:+.3f}")
    print(f"  Min Reward:           {min(avg_rewards):.3f}")
    print(f"  Max Reward:           {max(avg_rewards):.3f}")
    print()
    
    # Exercise efficiency statistics
    print("Exercise Selection Statistics:")
    print(f"  Initial (Episode 1):  {avg_exercises[0]:.2f} exercises/workout")
    print(f"  Final (Episode {episodes[-1]}):    {avg_exercises[-1]:.2f} exercises/workout")
    print(f"  Average (all):        {sum(avg_exercises)/len(avg_exercises):.2f} exercises/workout")
    print()
    
    # Learning analysis
    early_avg = sum(avg_rewards[:10]) / 10
    late_avg = sum(avg_rewards[-10:]) / 10
    print("Learning Analysis:")
    print(f"  Early episodes (1-10):   {early_avg:.3f} avg reward")
    print(f"  Late episodes (41-50):   {late_avg:.3f} avg reward")
    print(f"  Improvement:             {late_avg - early_avg:+.3f} ({((late_avg/early_avg - 1)*100):+.1f}%)")
    print()


def main():
    """
    Main function to generate all visualizations.
    """
    print("="*60)
    print("LEARNING CURVE GENERATOR")
    print("="*60)
    print()
    
    # Load metrics
    print("Loading metrics from logs/rl_metrics.csv...")
    episodes, avg_rewards, avg_exercises = load_metrics()
    
    if episodes is None:
        return
    
    print(f"✓ Loaded {len(episodes)} episodes of data")
    print()
    
    # Generate plots
    print("Generating plots...")
    plot_reward_curve(episodes, avg_rewards)
    plot_efficiency_curve(episodes, avg_exercises)
    print()
    
    # Print statistics
    print_statistics(episodes, avg_rewards, avg_exercises)
    
    print("="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print()
    print("Generated files:")
    print("  1. logs/learning_curve_reward.png")
    print("  2. logs/learning_curve_exercises.png")
    print()
    print("These plots show:")
    print("  ✓ How the agent's performance improved over time")
    print("  ✓ How the agent learned optimal workout length")
    print()
    print("Use these plots in your:")
    print("  - Technical report")
    print("  - Video demonstration")
    print("  - Presentation slides")
    print()


if __name__ == "__main__":
    main()