"""
Agent definitions and RL-based controller for the fitness workout optimizer.

This module contains:
- WorkoutAgent: Selects exercises based on categories
- IntensityAgent: Adjusts workout difficulty
- RLController: Orchestrates agents using RL (Q-Learning + UCB Bandit)
"""

from reward_calculator import (
    compute_category_reward,
    compute_workout_quality,
    get_intensity_level,
    get_time_bucket,
    get_fitness_level_bucket
)


class WorkoutAgent:
    """
    Agent that selects exercises from different categories.
    
    In a real system, this would query an exercise database or API.
    Here we simulate it by returning exercise metadata.
    """
    
    def __init__(self):
        # Exercise categories available
        self.categories = [
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
    
    def select_exercise(self, category, fitness_level):
        """
        Select an exercise from a specific category.
        
        Args:
            category: Exercise category (e.g., "strength")
            fitness_level: User's fitness level (beginner/intermediate/advanced)
            
        Returns:
            dict: Exercise information
        """
        return {
            "category": category,
            "difficulty": fitness_level,
            "name": f"{category.title()} Exercise",
            "estimated_time": 15  # minutes
        }
    
    def get_available_categories(self):
        """Return list of all exercise categories."""
        return self.categories


class IntensityAgent:
    """
    Agent that adjusts workout intensity based on user's fitness level.
    
    This agent modifies exercise difficulty to match the user's capabilities.
    """
    
    def __init__(self):
        self.intensity_levels = ["beginner", "intermediate", "advanced"]
    
    def adjust_intensity(self, exercise, target_level):
        """
        Adjust exercise intensity to match user's fitness level.
        
        Args:
            exercise: Exercise dict
            target_level: Target fitness level
            
        Returns:
            dict: Modified exercise with adjusted intensity
        """
        exercise["adjusted_difficulty"] = target_level
        return exercise
    
    def get_intensity_recommendation(self, fitness_level, exercises_count):
        """
        Recommend intensity based on workout progress.
        
        As workout progresses (more exercises), may recommend lower intensity
        to avoid overtraining.
        
        Args:
            fitness_level: User's base fitness level
            exercises_count: Number of exercises already in workout
            
        Returns:
            str: Recommended intensity level
        """
        # Start at user's level, reduce if workout is getting long
        if exercises_count >= 4:
            # Long workout, reduce intensity slightly
            levels = {"advanced": "intermediate", "intermediate": "beginner", "beginner": "beginner"}
            return levels.get(fitness_level, fitness_level)
        return fitness_level


class BaselineController:
    """
    Baseline (non-RL) controller with fixed rules.
    
    This is the "before RL" system for comparison.
    Always uses the same simple strategy:
    - Pick strength exercises for everyone
    - Stop after 3 exercises
    - No learning or adaptation
    """
    
    def __init__(self, workout_agent, intensity_agent):
        self.workout_agent = workout_agent
        self.intensity_agent = intensity_agent
    
    def run_fixed_pipeline(self, scenario):
        """
        Execute a fixed workout planning strategy.
        
        Strategy:
        1. Always pick 3 strength exercises
        2. Adjust intensity to user's level
        3. Return workout
        
        Args:
            scenario: User scenario dict
            
        Returns:
            dict: Workout plan with exercises
        """
        exercises = []
        
        # Fixed strategy: always 3 strength exercises
        for _ in range(3):
            exercise = self.workout_agent.select_exercise(
                "strength", 
                scenario["fitness_level"]
            )
            exercise = self.intensity_agent.adjust_intensity(
                exercise, 
                scenario["fitness_level"]
            )
            exercises.append(exercise)
        
        return {
            "exercises": exercises,
            "total_time": len(exercises) * 15,
            "method": "baseline_fixed"
        }


class RLController:
    """
    RL-based controller using Q-Learning + UCB Bandit.
    
    This is the intelligent system that learns from experience:
    - Q-Learning decides: Which ACTION to take (add exercise? finalize?)
    - UCB Bandit decides: Which CATEGORY to pick (strength? cardio?)
    
    The controller learns over time which strategies work best for different users.
    """
    
    def __init__(self, workout_agent, intensity_agent, bandit, q_agent):
        """
        Initialize RL controller with agents and learning algorithms.
        
        Args:
            workout_agent: WorkoutAgent instance
            intensity_agent: IntensityAgent instance  
            bandit: UCBBandit instance (for category selection)
            q_agent: QLearningAgent instance (for action selection)
        """
        self.workout_agent = workout_agent
        self.intensity_agent = intensity_agent
        self.bandit = bandit
        self.q_agent = q_agent
        
        # Available high-level actions
        self.actions = [
            "add_strength",
            "add_cardio", 
            "add_flexibility",
            "finalize"
        ]
    
    def _get_state(self, exercises, exercise_rewards, scenario):
        """
        Convert current workout state into discrete representation for Q-Learning.
        
        State consists of:
        - Number of exercises (0, 1, 2, 3+)
        - Intensity level (low/medium/high based on avg reward)
        - Time availability (plenty/medium/limited)
        - Fitness level (beginner/intermediate/advanced)
        
        Args:
            exercises: List of exercises selected so far
            exercise_rewards: List of rewards from those exercises
            scenario: User scenario dict
            
        Returns:
            tuple: State representation (e.g., (2, "medium", "plenty", "intermediate"))
        """
        # Discretize exercise count
        exercise_count = len(exercises)
        if exercise_count >= 3:
            count_bucket = 3
        else:
            count_bucket = exercise_count
        
        # Calculate average reward quality
        if exercise_rewards:
            avg_reward = sum(exercise_rewards) / len(exercise_rewards)
        else:
            avg_reward = 0.0
        
        intensity = get_intensity_level(avg_reward)
        time_status = get_time_bucket(exercise_count, scenario["time_available"])
        fitness = get_fitness_level_bucket(scenario["fitness_level"])
        
        return (count_bucket, intensity, time_status, fitness)
    
    def _map_action_to_category(self, action):
        """
        Map high-level action to exercise category.
        
        Args:
            action: Action like "add_strength", "add_cardio", etc.
            
        Returns:
            str: Exercise category or None if action is "finalize"
        """
        action_to_category = {
            "add_strength": "strength",
            "add_cardio": "cardio",
            "add_flexibility": "flexibility"
        }
        return action_to_category.get(action, None)
    
    def run_episode(self, scenario, reward_fn):
        """
        Run one RL training episode for a single user scenario.
        
        This is where the learning happens! The agent:
        1. Observes current state
        2. Chooses action (Q-Learning)
        3. If adding exercise, picks category (UCB Bandit)
        4. Gets reward
        5. Updates Q-values and bandit estimates
        6. Repeats until workout is finalized
        
        Args:
            scenario: User scenario dict
            reward_fn: Function to compute rewards
            
        Returns:
            tuple: (total_reward, total_exercises_added)
        """
        exercises = []
        exercise_rewards = []
        done = False
        
        state = self._get_state(exercises, exercise_rewards, scenario)
        total_reward = 0.0
        total_exercises = 0
        max_exercises = 6  # Safety limit
        
        while not done and total_exercises < max_exercises:
            # Step 1: Q-Learning selects high-level action
            action = self.q_agent.select_action(state)
            
            if action == "finalize":
                # Agent decides to stop and finalize workout
                if exercises:
                    # Calculate final workout quality
                    quality = compute_workout_quality(exercises, scenario)
                    step_reward = quality
                else:
                    # Finalizing with no exercises is bad!
                    step_reward = -1.0
                done = True
            
            else:
                # Agent decides to add an exercise
                # Step 2: UCB Bandit picks which category
                category = self._map_action_to_category(action)
                
                if category:
                    # Get exercise from WorkoutAgent
                    exercise = self.workout_agent.select_exercise(
                        category,
                        scenario["fitness_level"]
                    )
                    
                    # Adjust intensity with IntensityAgent
                    exercise = self.intensity_agent.adjust_intensity(
                        exercise,
                        self.intensity_agent.get_intensity_recommendation(
                            scenario["fitness_level"],
                            len(exercises)
                        )
                    )
                    
                    # Calculate reward for this category choice
                    category_reward = reward_fn(scenario, category)
                    
                    # Update UCB Bandit
                    self.bandit.update(category, category_reward)
                    
                    # Add to workout
                    exercises.append(exercise)
                    exercise_rewards.append(category_reward)
                    total_exercises += 1
                    
                    # Step reward: small penalty for adding exercise (cost)
                    step_reward = category_reward - 0.05
                else:
                    # Invalid action, small penalty
                    step_reward = -0.1
            
            # Accumulate total reward
            total_reward += step_reward
            
            # Get next state
            next_state = self._get_state(exercises, exercise_rewards, scenario)
            
            # Step 3: Update Q-Learning agent
            self.q_agent.update(state, action, step_reward, next_state)
            
            # Move to next state
            state = next_state
        
        return total_reward, total_exercises