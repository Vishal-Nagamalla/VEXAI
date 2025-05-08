import heapq
import random
import time
import copy

# Import constants from field.py
from field import (
    GRID_SIZE, ROBOT_SIZE, MAX_SIMULATION_STEPS, MAX_RINGS_ALLIANCE_STAKE, MAX_RINGS_MOBILE_STAKE, MAX_RINGS_NEUTRAL_STAKE, POSITIVE_CORNERS, NEGATIVE_CORNERS, POSITIVE_CORNERS_BLUE, NEGATIVE_CORNERS_BLUE, ALLIANCE_WALL_STAKES, NEUTRAL_WALL_STAKES, DIRS, SPAWN_POSITION, RING_STANDARD_POINTS, RING_NEGATIVE_POINTS)

class State:
    """
    x,y           – robot anchor
    rings         – loose rings being carried (no upper bound)
    held_goals    – tuple of ints, one entry per mobile goal the robot is holding
                    (each int is that goal's current ring count, 0‥3)
    steps         – path length so far
    """
    __slots__ = ('x', 'y', 'rings', 'held_goals', 'steps')

    def __init__(self, x, y, rings=0, held_goals=(), steps=0):
        self.x, self.y   = x, y
        self.rings       = rings
        self.held_goals  = tuple(held_goals)      # hashable
        self.steps       = steps

    def copy(self):
        return State(self.x, self.y, self.rings, self.held_goals, self.steps)

    def __eq__(self, o):
        return (self.x, self.y, self.rings, self.held_goals) == \
               (o.x,  o.y,  o.rings,  o.held_goals)

    def __hash__(self):
        return hash((self.x, self.y, self.rings, self.held_goals))

class Node:
    def __init__(self, state, g=0, h=0, parent=None, action=None, score=0):
        self.state = state
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.action = action
        self.score = score  # Track score for this node
    
    def __lt__(self, other):
        # Prioritize score first, then f value
        if self.score != other.score:
            return self.score > other.score  # Higher score is better
        return self.f < other.f

class PlanResources:
    """Track resources that change during planning"""
    def __init__(self, ring_positions, goal_positions, stake_rings):
        self.ring_positions = ring_positions.copy()
        self.goal_positions = goal_positions.copy()
        self.stake_rings = copy.deepcopy(stake_rings)
    
    def copy(self):
        return PlanResources(
            self.ring_positions.copy(),
            self.goal_positions.copy(),
            copy.deepcopy(self.stake_rings)
        )

def robot_cells(x, y):
    """Return all cells occupied by the robot at position (x,y)"""
    cells = []
    for dx in range(ROBOT_SIZE):
        for dy in range(ROBOT_SIZE):
            cells.append((x+dx, y+dy))
    return cells

def distance_to_nearest(x, y, positions):
    """Calculate Manhattan distance to nearest position"""
    if not positions:
        return float('inf')
    return min(abs(x-px) + abs(y-py) for px, py in positions)



def heuristic(state, resources, team_color):
    """Estimate cost to reach a good state"""
    # Define both plus zones as available
    top_left_zone = [(0, 22), (1, 22), (0, 23), (1, 23)]  # POSITIVE_CORNERS_BLUE
    top_right_zone = [(22, 22), (23, 22), (22, 23), (23, 23)]  # POSITIVE_CORNERS
    all_positive_zones = top_left_zone + top_right_zone
    
    # Prioritize ring collection if we have < 2 rings
    if state.rings < 2 and resources.ring_positions:
        return distance_to_nearest(state.x, state.y, resources.ring_positions) * 2
    
    # If we have rings, prioritize placing them on alliance stake if it has room
    if state.rings > 0:
        alliance_stake = ALLIANCE_WALL_STAKES[team_color]
        # Check if alliance stake has room
        if alliance_stake in resources.stake_rings:
            total_rings = resources.stake_rings[alliance_stake]['red'] + resources.stake_rings[alliance_stake]['blue']
            if total_rings < MAX_RINGS_ALLIANCE_STAKE:
                return distance_to_nearest(state.x, state.y, [alliance_stake]) * 1.5
    
    # If we have a loaded goal, prioritize getting to the closest positive zone
    if state.held_goals:
        return distance_to_nearest(state.x, state.y, all_positive_zones)
    
    # Default: find nearest goal to load if any exist
    if resources.goal_positions:
        return distance_to_nearest(state.x, state.y, resources.goal_positions) * 1.2
    
    # Default fallback
    return 0

def check_valid_position(x, y, field):
    """Check if a position is valid and unoccupied by obstacles"""
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
        return False
    
    # Check for obstacles (stakes, rings, and mobile goals)
    cell = field[y][x]
    if ("red_stake" in cell or "blue_stake" in cell or "neutral_stake" in cell or
        'r' in cell or 'b' in cell or 'G' in cell):
        return False
        
    return True



def get_successors(state, field, resources, team_color):
    """Generate (next_state, action, score_change, new_resources) tuples for all possible actions"""
    succ = []
    
    # If we've reached max steps, no more successors
    if state.steps >= MAX_SIMULATION_STEPS:
        return []
    
    ring_symbol = 'r' if team_color == 'red' else 'b'
    
    # Define both plus zones as available
    top_left_zone = [(0, 22), (1, 22), (0, 23), (1, 23)]  # POSITIVE_CORNERS_BLUE
    top_right_zone = [(22, 22), (23, 22), (22, 23), (23, 23)]  # POSITIVE_CORNERS
    all_positive_zones = top_left_zone + top_right_zone
    
    # Define negative zones
    negative_zones = [(0, 0), (1, 0), (0, 1), (1, 1), (22, 0), (23, 0), (22, 1), (23, 1)]
    
    # Adjust scoring weights based on current step
    # Early game: prioritize ring collection, goal loading, and placement
    # Late game: prioritize moving goals to positive zones
    early_game = state.steps < (MAX_SIMULATION_STEPS // 3)  # First 1/3 of game
    mid_game = state.steps < (MAX_SIMULATION_STEPS * 2 // 3)  # Middle 1/3 of game
    late_game = state.steps >= (MAX_SIMULATION_STEPS * 2 // 3)  # Last 1/3 of game
    
    # Score changes adjusted by game phase
    score_changes = {
        "Move": 0,
        "PickUpRing": 5 if early_game else (3 if mid_game else 2),
        "PlaceRingOnGoal": 4 if early_game else (3 if mid_game else 2),
        "PlaceRingOnStake": 6 if early_game else (4 if mid_game else 3),  # Higher priority for stake placement
        "LoadGoal": 6 if early_game else (8 if mid_game else 10),  # Increased priority
        "DeliverGoal": 8 if early_game else (10 if mid_game else 12),  # Increased priority
    }
    
    # 1) Move in 4 directions - respect robot size
    for name, (dx, dy) in DIRS.items():
        nx, ny = state.x + dx, state.y + dy
        
        # Check if new position is valid
        if 0 <= nx <= GRID_SIZE - ROBOT_SIZE and 0 <= ny <= GRID_SIZE - ROBOT_SIZE:
            valid_move = True
            
            # Check all cells that would be occupied by the robot
            for rx, ry in robot_cells(nx, ny):
                if not check_valid_position(rx, ry, field):
                    valid_move = False
                    break
            
            if valid_move:
                new_state = State(nx, ny, state.rings, state.held_goals, state.steps + 1)
                # Add bonus for moving towards positive zone when loaded
                move_score = score_changes["Move"]
                
                # In late game, heavily prioritize moving goals to positive zone
                if state.held_goals:
                    # Calculate distance to closest positive zone
                    current_dist = min(abs(state.x-x) + abs(state.y-y) for x, y in all_positive_zones)
                    new_dist = min(abs(nx-x) + abs(ny-y) for x, y in all_positive_zones)
                    
                    if new_dist < current_dist:
                        # Bonus increases in late game
                        move_score += 1 if early_game else (2 if mid_game else 4)
                    
                    # Penalty for moving towards negative zone
                    current_neg_dist = min(abs(state.x-x) + abs(state.y-y) for x, y in negative_zones)
                    new_neg_dist = min(abs(nx-x) + abs(ny-y) for x, y in negative_zones)
                    if new_neg_dist < current_neg_dist:
                        move_score -= 2 if early_game else (3 if mid_game else 5)
                
                # In early game, prioritize moving towards rings
                elif state.rings < 2 and resources.ring_positions:
                    current_ring_dist = min(abs(state.x-x) + abs(state.y-y) for x, y in resources.ring_positions)
                    new_ring_dist = min(abs(nx-x) + abs(ny-y) for x, y in resources.ring_positions)
                    if new_ring_dist < current_ring_dist:
                        move_score += 2 if early_game else (1 if mid_game else 0)
                
                # Resources don't change for move actions
                succ.append((new_state, f"Move {name}", move_score, resources))
    
    # 2) PickUpRing from adjacent cells (not under the robot)
    # Get perimeter cells around robot
    perim = set()
    for rx, ry in robot_cells(state.x, state.y):
        for dx, dy in DIRS.values():
            nx, ny = rx + dx, ry + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                perim.add((nx, ny))
    
    # Remove cells under the robot
    perim = perim - set(robot_cells(state.x, state.y))
    
    # Check perimeter for rings to pick up
    for px, py in perim:
        if (px, py) in resources.ring_positions:
            # Create a new copy of resources with this ring removed
            new_resources = resources.copy()
            new_resources.ring_positions.remove((px, py))
            
            new_state = State(state.x, state.y,
                            state.rings + 1,
                            state.held_goals,
                            state.steps + 1)
            succ.append((new_state, "PickUpRing", score_changes["PickUpRing"], new_resources))
            break
    
    # 3) PlaceRingOnGoal if adjacent to a mobile goal or holding one and we have rings
    if state.rings > 0:
        # If robot is carrying a goal, prioritize placing rings on it directly
        if state.held_goals:
            # Check if the first held goal has room for more rings
            ring_count = state.held_goals[0]
            if ring_count < MAX_RINGS_MOBILE_STAKE:  # Enforce maximum ring limit
                goals_after = list(state.held_goals)
                goals_after[0] += 1
                new_state = State(state.x, state.y,
                                state.rings - 1,
                                tuple(goals_after),
                                state.steps + 1)
                # Resources don't change when placing on held goal
                succ.append((new_state, "PlaceRingOnGoal", score_changes["PlaceRingOnGoal"], resources))
        else:
            # Get perimeter cells if not already calculated
            if not perim:
                perim = set()
                for rx, ry in robot_cells(state.x, state.y):
                    for dx, dy in DIRS.values():
                        nx, ny = rx + dx, ry + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            perim.add((nx, ny))
                # Remove cells under the robot
                perim = perim - set(robot_cells(state.x, state.y))
            
            # Check for goals in perimeter with available capacity
            for px, py in perim:
                if (px, py) in resources.goal_positions:
                    # Check if this is a mobile goal (not a stake)
                    if (px, py) not in NEUTRAL_WALL_STAKES and (px, py) != ALLIANCE_WALL_STAKES['red'] and (px, py) != ALLIANCE_WALL_STAKES['blue']:
                        # Create a new state with one less ring
                        new_state = State(state.x, state.y, state.rings - 1, state.held_goals, state.steps + 1)
                        
                        # Bonus for placing rings on goals in plus zones
                        place_score = score_changes["PlaceRingOnGoal"]
                        if (px, py) in all_positive_zones:
                            place_score *= 1.5
                            
                        # Resources don't change when placing on external goal
                        succ.append((new_state, "PlaceRingOnGoal", place_score, resources))
                        break
    
    # --- 4) PlaceRingOnStake (only if the stake has room) -----------------------
    if state.rings > 0:
        # Get perimeter cells if not already calculated
        if not perim:
            perim = set()
            for rx, ry in robot_cells(state.x, state.y):
                for dx, dy in DIRS.values():
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        perim.add((nx, ny))
            # Remove cells under the robot
            perim = perim - set(robot_cells(state.x, state.y))

        # Alliance stake first - only if it has room
        alliance_stake = ALLIANCE_WALL_STAKES[team_color]
        if alliance_stake in perim and alliance_stake in resources.stake_rings:
            total = resources.stake_rings[alliance_stake]['red'] + resources.stake_rings[alliance_stake]['blue']
            if total < MAX_RINGS_ALLIANCE_STAKE:
                new_state = State(state.x, state.y,
                                state.rings - 1, state.held_goals,
                                state.steps + 1)
                
                # Update stake rings in a new resources copy
                new_resources = resources.copy()
                new_resources.stake_rings[alliance_stake]['red' if ring_symbol == 'r' else 'blue'] += 1
                
                # Higher priority for alliance stake placement
                place_score = score_changes["PlaceRingOnStake"] * 1.5
                
                succ.append((new_state, "PlaceRingOnStake", place_score, new_resources))

        # Neutral stakes - only if they have room
        for stake in NEUTRAL_WALL_STAKES:
            if stake in perim and stake in resources.stake_rings:
                total = resources.stake_rings[stake]['red'] + resources.stake_rings[stake]['blue']
                if total < MAX_RINGS_NEUTRAL_STAKE:
                    new_state = State(state.x, state.y,
                                    state.rings - 1, state.held_goals,
                                    state.steps + 1)
                    
                    # Update stake rings in a new resources copy
                    new_resources = resources.copy()
                    new_resources.stake_rings[stake]['red' if ring_symbol=='r' else 'blue'] += 1
                    
                    succ.append((new_state, "PlaceRingOnStake", score_changes["PlaceRingOnStake"], new_resources))
                    break
    
    # 5) LoadGoal if adjacent to a mobile goal and we don't have one loaded
    if not state.held_goals:  # Only if we're not already carrying a goal
        # Get perimeter cells if not already calculated
        if not perim:
            perim = set()
            for rx, ry in robot_cells(state.x, state.y):
                for dx, dy in DIRS.values():
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        perim.add((nx, ny))
            # Remove cells under the robot
            perim = perim - set(robot_cells(state.x, state.y))
            
        # Look for mobile goals in the perimeter
        for px, py in perim:
            if (px, py) in resources.goal_positions:
                # Verify this is a mobile goal (not a stake)
                if (px, py) not in NEUTRAL_WALL_STAKES and (px, py) != ALLIANCE_WALL_STAKES['red'] and (px, py) != ALLIANCE_WALL_STAKES['blue']:
                    # Update resources - remove goal from map
                    new_resources = resources.copy()
                    new_resources.goal_positions.remove((px, py))

                    # Calculate how many rings this goal already has
                    # For simplicity in planning, we'll just count the total number of rings
                    # (not distinguishing between red and blue)
                    goal_rings = 0
                    
                    # In the search, we don't have detailed ring information,
                    # so estimate this from the field
                    for color in ['red', 'blue']:
                        symbol = 'r' if color == 'red' else 'b'
                        if symbol in field[py][px]:
                            goal_rings += 1
                    
                    # Make sure we don't exceed the maximum
                    goal_rings = min(goal_rings, MAX_RINGS_MOBILE_STAKE)
                    
                    # Add penalty if we would lose rings by loading goal
                    load_score = score_changes["LoadGoal"]
                    if state.rings > 0:
                        load_score -= state.rings * 2  # Penalty for losing rings
                    
                    # But add bonus if goal has rings already
                    if goal_rings > 0:
                        load_score += goal_rings * 2
                    
                    # Only load if it's worthwhile
                    if load_score > 0 or goal_rings > 0:
                        new_state = State(state.x, state.y,
                                        0,  # rings in hand go to zero when loading a goal
                                        (goal_rings,),  # append the new goal with its rings
                                        state.steps + 1)
                        
                        succ.append((new_state, "LoadGoal", load_score, new_resources))
                        break
    
    # 6) DeliverGoal if in or near a positive zone and we have a goal
    robot_area = set(robot_cells(state.x, state.y))
    # Check if any part of the robot is in a positive zone
    in_positive_zone = any((rx, ry) in all_positive_zones for (rx, ry) in robot_area)
    
    if state.held_goals and in_positive_zone:
        delivered_goal_rings = state.held_goals[0]
        # Update resources - add goal back to map at robot's position
        new_resources = resources.copy()
        
        # Simulate placing the goal adjacent to the robot in a positive zone if possible
        placed_in_positive = False
        for cell in robot_area:
            for dx, dy in DIRS.values():
                nx, ny = cell[0] + dx, cell[1] + dy
                if (nx, ny) in all_positive_zones and (nx, ny) not in new_resources.goal_positions:
                    new_resources.goal_positions.append((nx, ny))
                    placed_in_positive = True
                    break
            if placed_in_positive:
                break
                
        # If no positive zone placement found, just put it somewhere adjacent
        if not placed_in_positive:
            new_resources.goal_positions.append((state.x, state.y))
        
        new_state = State(state.x, state.y,
                        state.rings,
                        tuple(state.held_goals[1:]),   # remaining ones still onboard
                        state.steps + 1)
        
        # Prioritize delivering loaded goals
        deliver_score = score_changes["DeliverGoal"]
        if delivered_goal_rings > 0:
            deliver_score *= 1.5  # Bonus for goals with rings
        if placed_in_positive:
            deliver_score *= 1.5  # Additional bonus for placement in positive zone
        
        succ.append((new_state, "DeliverGoal", deliver_score, new_resources))
    
    return succ

def find_ring_positions(field, team_color):
    """Find all ring positions of the specified color"""
    ring_symbol = 'r' if team_color == 'red' else 'b'
    positions = []
    
    for y in range(len(field)):
        for x in range(len(field[0])):
            if ring_symbol in field[y][x]:
                positions.append((x, y))
    
    return positions

def find_goal_positions(field):
    """Find all mobile goal positions"""
    positions = []
    
    for y in range(len(field)):
        for x in range(len(field[0])):
            if 'G' in field[y][x]:
                positions.append((x, y))
    
    return positions

def generate_greedy_plan(field, start_state, team_color, stake_rings):
    """Generate a greedy plan that tries to maximize score over 240 steps"""
    print("Generating greedy plan...")
    
    # Find all rings and goals
    ring_positions = find_ring_positions(field, team_color)
    goal_positions = find_goal_positions(field)
    
    print(f"Found {len(ring_positions)} rings and {len(goal_positions)} goals")
    
    # Initialize resources
    resources = PlanResources(ring_positions, goal_positions, stake_rings)
    
    # Start with initial state
    current_state = start_state.copy()
    plan = []
    
    # Continue until we reach the step limit
    while current_state.steps < MAX_SIMULATION_STEPS:
        # Get all possible successor states
        successors = get_successors(current_state, field, resources, team_color)
        
        if not successors:
            # No valid moves, just add random movement
            # Try each direction to find a valid one
            valid_dirs = []
            for name, (dx, dy) in DIRS.items():
                nx, ny = current_state.x + dx, current_state.y + dy
                if 0 <= nx <= GRID_SIZE - ROBOT_SIZE and 0 <= ny <= GRID_SIZE - ROBOT_SIZE:
                    valid = True
                    for rx, ry in robot_cells(nx, ny):
                        if not check_valid_position(rx, ry, field):
                            valid = False
                            break
                    if valid:
                        valid_dirs.append(name)
            
            # Choose a random valid direction
            if valid_dirs:
                action = f"Move {random.choice(valid_dirs)}"
                current_state.x += DIRS[action.split()[1]][0]
                current_state.y += DIRS[action.split()[1]][1]
            else:
                # No valid moves at all - this should not happen much
                action = f"Move {random.choice(list(DIRS.keys()))}"
            
            plan.append(action)
            current_state.steps += 1
            continue
        
        # Sort successors by score change (highest first)
        successors.sort(key=lambda x: x[2], reverse=True)
        
        # Take the best action
        best_state, best_action, _, new_resources = successors[0]
        
        # Update current state and resources
        current_state = best_state
        resources = new_resources
        
        # Add the action to the plan
        plan.append(best_action)
    
    return plan

def a_star_search(start_state, field, team_color, stake_rings):
    """Find a high-scoring plan using A* search, then extend to full length"""
    print(f"Starting search for {team_color} team...")
    
    # Verify field array structure
    if not field or not field[0]:
        print("Error: Invalid field structure")
        return generate_greedy_plan(field, start_state, team_color, stake_rings)
    
    # Find all rings and goals
    ring_positions = find_ring_positions(field, team_color)
    goal_positions = find_goal_positions(field)
    
    print(f"Found {len(ring_positions)} {team_color} rings and {len(goal_positions)} goals")
    
    # Start time for tracking performance
    start_time = time.time()
    
    # Create initial resources
    initial_resources = PlanResources(ring_positions, goal_positions, stake_rings)
    
    # Generate plan using A* search
    plan, final_resources = generate_improved_plan(start_state, field, team_color, 
                                   initial_resources, start_time)
    
    print(f"Generated plan with {len(plan)} steps")
    
    # If the plan is already 240 steps, return it
    if len(plan) >= MAX_SIMULATION_STEPS:
        return plan[:MAX_SIMULATION_STEPS]
    
    # Otherwise, extend the plan to 240 steps
    while len(plan) < MAX_SIMULATION_STEPS:
        # Try to generate more actions using the current state and resources
        current_state = simulate_plan_execution(start_state, plan)
        
        # Get all possible next actions
        next_actions = get_successors(current_state, field, final_resources, team_color)
        
        if next_actions:
            # Choose the best action
            next_actions.sort(key=lambda x: x[2], reverse=True)
            _, action, _, new_resources = next_actions[0]
            plan.append(action)
            final_resources = new_resources
        else:
            # No valid actions, add random movement
            valid_dirs = []
            for name, (dx, dy) in DIRS.items():
                nx, ny = current_state.x + dx, current_state.y + dy
                if 0 <= nx <= GRID_SIZE - ROBOT_SIZE and 0 <= ny <= GRID_SIZE - ROBOT_SIZE:
                    valid = True
                    for rx, ry in robot_cells(nx, ny):
                        if not check_valid_position(rx, ry, field):
                            valid = False
                            break
                    if valid:
                        valid_dirs.append(name)
            
            if valid_dirs:
                plan.append(f"Move {random.choice(valid_dirs)}")
            else:
                # If no valid moves, just repeat the last action or add a no-op
                if plan:
                    plan.append(plan[-1])
                else:
                    plan.append("Move Right")  # Fallback
    
    # Return exactly 240 steps
    return plan[:MAX_SIMULATION_STEPS]

def simulate_plan_execution(start_state, plan):
    """Simulate executing a plan to get the final state"""
    current_state = start_state.copy()
    
    for action in plan:
        current_state.steps += 1
        
        if action == "PickUpRing":
            current_state.rings += 1
        elif action.startswith("PlaceRingOn"):
            if current_state.rings > 0:
                current_state.rings -= 1
                if action == "PlaceRingOnGoal" and current_state.held_goals:
                    # Add ring to held goal
                    goal_rings = list(current_state.held_goals)
                    goal_rings[0] += 1
                    current_state.held_goals = tuple(goal_rings)
        elif action == "LoadGoal":
            current_state.rings = 0  # Rings are lost when loading a goal
            current_state.held_goals = (0,)  # Add a new goal with 0 rings
        elif action == "DeliverGoal":
            if current_state.held_goals:
                current_state.held_goals = current_state.held_goals[1:]  # Remove first goal
        elif action.startswith("Move"):
            # Extract direction from "Move X"
            direction = action.split(" ")[1]
            dx, dy = DIRS[direction]
            current_state.x += dx
            current_state.y += dy
    
    return current_state

def generate_improved_plan(start_state, field, team_color, initial_resources, start_time):
    """Generate an improved plan using A* search with resource tracking"""
    open_list = []
    closed = set()
    
    # Create start node
    start_h = heuristic(start_state, initial_resources, team_color)
    start_node = Node(start_state, g=0, h=start_h, score=0)
    
    # Track resources for each node
    node_resources = {start_node: initial_resources}
    
    heapq.heappush(open_list, start_node)
    
    # Track best solution found so far
    best_node = start_node
    best_score = 0
    best_resources = initial_resources
    
    # Search parameters - increase time limit to ensure better plans
    max_nodes = 30000  # Node limit
    nodes_expanded = 0
    time_limit = 5.0  # Time limit in seconds (increased)
    
    # Main search loop
    while open_list and nodes_expanded < max_nodes and time.time() - start_time < time_limit:
        current = heapq.heappop(open_list)
        
        # Get resources for current node
        current_resources = node_resources[current]
        
        # Update best solution if this one has a higher score
        if current.score > best_score:
            best_node = current
            best_score = current.score
            best_resources = current_resources
        
        # Only expand nodes below step limit
        if current.state.steps >= MAX_SIMULATION_STEPS:
            continue
        
        # Skip already visited states
        state_key = (current.state.x, current.state.y, current.state.rings, current.state.held_goals)
        if state_key in closed:
            continue
            
        closed.add(state_key)
        nodes_expanded += 1
        
        # Progress logging
        if nodes_expanded % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Expanded {nodes_expanded} nodes, queue size: {len(open_list)}, time: {elapsed:.2f}s")
        
        # Generate successors
        for succ_state, action, score_change, succ_resources in get_successors(
                current.state, field, current_resources, team_color):
            
            # Skip already visited states
            state_key = (succ_state.x, succ_state.y, succ_state.rings, succ_state.held_goals)
            if state_key in closed:
                continue
            
            # Calculate new costs and score
            g2 = current.g + 1
            h2 = heuristic(succ_state, succ_resources, team_color)
            score2 = current.score + score_change
            
            # Create new node
            node2 = Node(succ_state, g=g2, h=h2, parent=current, action=action, score=score2)
            
            # Store resources for this node
            node_resources[node2] = succ_resources
            
            heapq.heappush(open_list, node2)
    
    # Reconstruct the best path found
    plan = []
    if best_node:
        plan = reconstruct_path(best_node)
    
    return plan, best_resources

def reconstruct_path(node):
    """Reconstruct the path from the goal node back to the start"""
    actions = []
    while node.parent:
        actions.append(node.action)
        node = node.parent
    return list(reversed(actions))
