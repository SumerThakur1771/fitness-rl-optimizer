"""
Environment and reward functions for the fitness workout optimizer.

This module defines:
- How to calculate rewards for exercise selection
- How to evaluate workout quality
- Helper functions for state representation
"""


def compute_category_reward(scenario, chosen_category):
    """
    Calculate reward for choosing a specific exercise category.
    
    Rewards are based on how well the category matches the user's goal:
    - Best category: +1.0 (perfect match!)
    - Good category: +0.5 (decent choice)
    - Bad category: -0.2 (poor choice, penalized)
    - Unknown: 0.0 (neutral)
    
    Args:
        scenario: User scenario dict with goal, fitness level, etc.
        chosen_category: Exercise category chosen (e.g., "strength")
        
    Returns:
        float: Reward value between -0.2 and 1.0
        
    Example:
        >>> scenario = {"best_category": "strength", "good_categories": ["compound"]}
        >>> compute_category_reward(scenario, "strength")
        1.0
        >>> compute_category_reward(scenario, "compound")
        0.5
    """
    if chosen_category == scenario["best_category"]:
        return 1.0
    elif chosen_category in scenario.get("good_categories", []):
        return 0.5
    elif chosen_category in scenario.get("bad_categories", []):
        return -0.2
    else:
        return 0.0


def compute_workout_quality(exercises_selected, scenario):
    """
    Evaluate overall workout quality based on exercises selected.
    
    Considers:
    - Variety: Having different exercise types is good
    - Time efficiency: Not going over time limit
    - Goal alignment: Matching user's fitness goal
    
    Args:
        exercises_selected: List of exercise category names
        scenario: User scenario dict
        
    Returns:
        float: Quality score (higher is better)
    """
    if not exercises_selected:
        return 0.0
    
    # Variety bonus: unique exercise types are better
    exercise_categories = [ex['category'] for ex in exercises_selected]
    unique_types = len(set(exercise_categories))
    variety_score = unique_types * 0.3
    
    # Time efficiency: assume each exercise takes ~15 minutes
    estimated_time = len(exercises_selected) * 15
    time_available = scenario["time_available"]
    
    if estimated_time <= time_available:
        time_score = 0.5
    else:
        # Penalty for going over time
        time_score = -0.3
    
    # Goal alignment: check if exercises match user goal
    goal_alignment = 0.0
    for exercise in exercises_selected:
        category = exercise['category']
    if category == scenario["best_category"]:
        goal_alignment += 0.4
    elif category in scenario.get("good_categories", []):
        goal_alignment += 0.2
    
    total_quality = variety_score + time_score + goal_alignment
    return total_quality


def get_intensity_level(avg_reward):
    """
    Convert average reward to intensity level descriptor.
    
    This creates discrete intensity buckets for state representation:
    - high: avg_reward > 0.7 (great workout so far!)
    - medium: avg_reward > 0.3 (decent workout)
    - low: avg_reward <= 0.3 (needs improvement)
    
    Args:
        avg_reward: Average reward from exercises so far
        
    Returns:
        str: "high", "medium", or "low"
    """
    if avg_reward > 0.7:
        return "high"
    elif avg_reward > 0.3:
        return "medium"
    else:
        return "low"


def get_time_bucket(exercises_count, time_available):
    """
    Discretize time remaining into buckets.
    
    Estimates time remaining based on exercises already selected.
    Assumption: Each exercise takes about 15 minutes.
    
    Args:
        exercises_count: Number of exercises selected so far
        time_available: Total time available for workout (minutes)
        
    Returns:
        str: "plenty", "medium", or "limited"
    """
    estimated_time_used = exercises_count * 15
    time_remaining = time_available - estimated_time_used
    
    if time_remaining > 30:
        return "plenty"
    elif time_remaining > 15:
        return "medium"
    else:
        return "limited"


def get_fitness_level_bucket(fitness_level):
    """
    Normalize fitness level string for state representation.
    
    Args:
        fitness_level: User's fitness level (e.g., "intermediate")
        
    Returns:
        str: Normalized fitness level
    """
    # Already in good format, just return it
    return fitness_level.lower()