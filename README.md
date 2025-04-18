# VEX U High Stakes AI Strategy Simulator

## Overview
This project simulates a simplified version of the VEX U "High Stakes" robotics competition game. The simulation involves a randomly generated 24x24 tile game field with AI agents representing one team (Red or Blue), whose task is to determine the best sequence of moves to maximize their score under game-specific constraints. The system leverages advanced AI techniques including **Monte Carlo Tree Search (MCTS)**, **heuristic search**, **reinforcement learning**, **machine learning**, and **probabilistic reasoning**.

## Objective
Build a simulation tool where a user selects a team (Red or Blue) and, based on a randomly generated game field, the AI agent:
- Navigates the environment
- Collects their team's rings (Red or Blue)
- Stacks rings into mobile goals
- Moves mobile goals into scoring corners
- Calculates and displays the most optimal scoring strategy using AI-driven search and evaluation methods

## Game Rules & Simplifications
1. **Teams:** Red team and Blue team, each with 2 identical robots.
2. **Robot Abilities:**
   - Pick up rings of their own color
   - Stack rings into mobile goals
   - Move mobile goals across the field
   - Score on wall stakes
3. **Field Layout:**
   - Size: 24x24 grid
   - 2 wall stakes per alliance
   - 4 corners (2 positive, 2 negative)
   - 5 mobile goals
   - 48 rings (24 red, 24 blue)
   - Robots occupy 2x2 space; goals and rings occupy 1x1
   - Each tile can contain at most:
     - One object (robot, ring stack, or mobile goal)
     - Ring stacks: max 2 rings per stack (one red + one blue only)
4. **Object Classification:**
   - **Immovable:** Wall stakes, opponent robots
   - **Movable:** Mobile goals, rings
5. **Scoring:**
   - Score rings by placing them in mobile goals
   - Double points by placing goals in positive (own or opponent's) corners
   - Aim to fill 5 goals and score at least 2 in positive corners
6. **Fallback Condition:**
   - If the goal is not fully attainable (e.g., robot is trapped), the AI should return the best scoring path possible based on accessible resources

## AI Techniques
- **Monte Carlo Tree Search (MCTS):** Simulates thousands of move sequences to evaluate and select the optimal scoring strategy
- **Heuristic Search (e.g., A*, Greedy):** Prioritizes movement toward high-value goals and areas with scoring potential
- **Reinforcement Learning:** (Optional) Allows agents to learn scoring patterns and environmental layouts over repeated simulations
- **Machine Learning:** Predictive modeling for strategy evaluation and scoring efficiency
- **Probabilistic Reasoning:** Handles uncertainty and partial visibility (e.g., object positioning or opponent interference)

## User Interaction
- Select Red or Blue alliance
- Program generates random field layout with valid object placement
- AI calculates optimal route to maximize score
- Output:
  - Movement path and scoring sequence
  - Final score and whether full scoring configuration was achieved
  - Visual or terminal output of the simulated strategy

## Technologies
- **Language:** Python
- **Core Libraries:**
  - `numpy` — grid structure and calculations
  - `random` — object generation and environment randomness
  - `heapq` — priority queue for A* search
  - `matplotlib` or `pygame` (optional) — for field visualization
  - `scikit-learn` (optional) — for ML analysis of strategies
  - `json`, `argparse` — for configuration and input/output control