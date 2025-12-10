"""
Baseline (non-RL) script for comparison.

This runs a simple fixed strategy that does NOT learn:
- Always picks 3 strength exercises
- No adaptation to user goals
- No learning from experience

This demonstrates what the system does WITHOUT RL.

Run with: python src/main_fixed.py
"""

import json
from fitness_agents import WorkoutAgent, IntensityAgent, BaselineController


def load_scenarios():
    """Load workout scenarios from JSON file."""
    with open("data/workout_scenarios.json") as f:
        return json.load(f)


def main():
    """
    Run baseline fixed pipeline for comparison.
    """
    print("="*60)
    print("BASELINE (NON-RL) WORKOUT PLANNER")
    print("="*60)
    print()
    print("This is the 'before RL' system that uses fixed rules.")
    print("Strategy: Always recommend 3 strength exercises.")
    print()
    
    # Load scenarios
    scenarios = load_scenarios()
    print(f"Testing on {len(scenarios)} user scenarios...")
    print()
    
    # Initialize agents
    workout_agent = WorkoutAgent()
    intensity_agent = IntensityAgent()
    controller = BaselineController(workout_agent, intensity_agent)
    
    # Run baseline for each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario['user_goal']}")
        print(f"  User Level: {scenario['fitness_level']}")
        print(f"  Time Available: {scenario['time_available']} min")
        print(f"  Best Category: {scenario['best_category']}")
        
        # Run fixed pipeline
        result = controller.run_fixed_pipeline(scenario)
        
        print(f"  → Baseline Plan: {len(result['exercises'])} exercises")
        print(f"  → Total Time: {result['total_time']} min")
        print(f"  → Method: {result['method']}")
        
        # Show what was planned
        print("  → Exercises:")
        for j, ex in enumerate(result['exercises'], 1):
            print(f"      {j}. {ex['category'].title()} "
                  f"(difficulty: {ex['difficulty']})")
        
        print()
    
    print("="*60)
    print("BASELINE COMPLETE")
    print("="*60)
    print()
    print("Observations:")
    print("- All users get the same 3 strength exercises")
    print("- No adaptation to user goals or time constraints")
    print("- No learning or improvement over time")
    print()
    print("Compare this with the RL system:")
    print("  → RL system learns optimal strategies")
    print("  → Adapts to different user goals")
    print("  → Improves performance over episodes")
    print()


if __name__ == "__main__":
    main()