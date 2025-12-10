# Reinforcement Learning for Fitness Workout Optimization

## Take-Home Final – Northeastern University

**Author: Sumer**
**GitHub Repository:** https://github.com/SumerThakur1771/fitness-rl-optimizer
---

## Project Overview

This project implements a **Reinforcement Learning-powered agentic fitness system** that learns to create personalized workout plans based on user goals, fitness level, and available time.

The goal is to enable autonomous agents to learn through experience and optimize:

- Which exercise categories to select for different fitness goals
- When to add more exercises versus finalizing the workout
- How to balance workout quality with time constraints

The final system uses:

- **Q-Learning** (workout decision-making)
- **UCB1 Bandit** (exercise category selection)
- **Agent orchestration** (WorkoutAgent + IntensityAgent + RL Controller)

---

## Core Idea

Traditional workout planning systems follow fixed rules and cannot adapt to individual needs. This project replaces static logic with an **RL Controller** that improves over time.

| Component       | Learns                       |
| --------------- | ---------------------------- |
| Q-Learning      | Best workflow action         |
| UCB1 Bandit     | Best exercise category       |
| Reward Function | Balance quality & efficiency |

---

## System Architecture

```
User Profile Input
        ↓
RL Controller (Q-Learning + UCB Bandit)
        ↓                     ↓
 WorkoutAgent → Categories (strength / cardio / flexibility / etc.)
        ↓
 IntensityAgent → Difficulty Adjustment
        ↓
 Reward Computation (quality + variety - time penalty)
        ↓
Q-Table & UCB Update
        ↺ (feedback loop)
```

---

## Project Structure

```
fitness_rl_optimizer/
│
├── data/
│   └── workout_scenarios.json    # User scenarios
│
├── src/
│   ├── q_learner.py              # Q-Learning implementation
│   ├── ucb_selector.py           # UCB1 bandit implementation
│   ├── reward_calculator.py      # Reward model and helpers
│   ├── fitness_agents.py         # Agents + RL Controller
│   ├── train_rl.py               # RL training pipeline
│   ├── baseline_test.py          # Baseline system (no learning)
│   └── visualize_results.py      # Matplotlib learning plots
│
├── logs/
│   ├── rl_metrics.csv
│   ├── learning_curve_reward.png
│   └── learning_curve_exercises.png
│
├── requirements.txt
├── TechnicalDocumentation.pdf
├── PPT.pdf
├── VideoDemonstration.mp4
├── requirements.txt
└── README.md

```

---

## Installation

### Clone or Download

```bash
cd fitness_rl_optimizer
```

### Create Virtual Environment (Optional)

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Running the System

### Run Baseline (Non-RL Pipeline)

```bash
python src/baseline_test.py
```

This shows the fixed strategy that does not learn.

### Run Reinforcement Learning (Q-Learning + UCB)

```bash
python src/train_rl.py
```

**Outputs:**

- Episode rewards
- Average exercises per workout
- `logs/rl_metrics.csv`

### Generate Learning Curves

```bash
python src/visualize_results.py
```

**Creates images inside `logs/`:**

- `learning_curve_reward.png`
- `learning_curve_exercises.png`

---

## Experimental Setup

| Parameter       | Value           |
| --------------- | --------------- |
| Episodes        | 50              |
| Learning Rate   | 0.1             |
| Discount Factor | 0.9             |
| Max Exercises   | 6               |
| Exploration     | UCB1 Bandit     |
| Scenarios       | 8 user profiles |

---

## Results Summary

### 1. Reward Learning Curve

- Rewards increase steadily over episodes
- Shows policy improvement and Q-Learning convergence

### 2. Exercise Efficiency Curve

- Average exercises stabilize at approximately 2.0-2.1
- Indicates improved workflow and optimal stopping

### 3. Baseline vs RL Comparison

| Metric             | Baseline  | RL System        |
| ------------------ | --------- | ---------------- |
| Adaptation         | None      | Learns over time |
| Category Selection | Fixed     | Adaptive (UCB)   |
| Workflow Decisions | Hardcoded | Learned policy   |
| Performance        | Static    | **High**         |

---

## RL Math (Short Version)

### Q-Learning

```
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]
```

### UCB1

```
UCB = average_reward + sqrt(2 ln N / n)
```

---

## Technologies Used

- Python
- Q-Learning (value-based RL)
- UCB Multi-Armed Bandit
- Matplotlib
- NumPy
- JSON-based dataset

---

## Future Enhancements

- Add PPO or REINFORCE (policy gradient methods)
- Introduce multi-agent RL extensions
- Integrate real exercise database APIs
- Use RLHF (Reinforcement Learning from Human Feedback)
- Apply transfer learning to new fitness domains

---

## Author

**Sumer**  
MS in Information Systems  
Northeastern University

Background: 5+ years of fitness training and nutrition tracking experience, bringing domain expertise to this RL application.

---

## License

This project is part of the **Take-Home Final: Reinforcement Learning for Agentic AI Systems** assignment.
It is intended for academic use only.
