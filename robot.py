import tkinter as tk
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import numpy as np
import time
import copy
from field import (GRID_SIZE, ROBOT_SIZE, POSITIVE_CORNERS, NEGATIVE_CORNERS, 
                  POSITIVE_CORNERS_BLUE, NEGATIVE_CORNERS_BLUE, ALLIANCE_WALL_STAKES,
                  NEUTRAL_WALL_STAKES, DIRS, MAX_SIMULATION_STEPS, SPAWN_POSITION,
                  MAX_RINGS_ALLIANCE_STAKE, MAX_RINGS_NEUTRAL_STAKE, MAX_RINGS_MOBILE_STAKE)
from scoring_module import calculate_score, get_best_goal_placement

# ------------ GAME OBJECTS ------------
class FieldObject:
    def __init__(self, name, symbol, color, movable):
        self.name = name
        self.symbol = symbol
        self.color = color
        self.movable = movable

RING_RED    = FieldObject("ring_red",    "r", "red",        True)
RING_BLUE   = FieldObject("ring_blue",   "b", "#4da6ff",    True)
MOBILE_GOAL = FieldObject("goal",        "G", "lightgreen", True)
ROBOT_RED   = FieldObject("robot_red",   "R", "#ff9999",    False)
ROBOT_BLUE  = FieldObject("robot_blue",  "B", "#9999ff",    False)

# ------------ FIELD -------------------
class FieldGrid:
    def __init__(self):
        self.grid = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        # All occupied positions
        self.occupied = set(POSITIVE_CORNERS + NEGATIVE_CORNERS + 
                            POSITIVE_CORNERS_BLUE + NEGATIVE_CORNERS_BLUE +
                            NEUTRAL_WALL_STAKES + 
                            [ALLIANCE_WALL_STAKES['red'], ALLIANCE_WALL_STAKES['blue']])
        
        # Add neutral wall stakes
        for x, y in NEUTRAL_WALL_STAKES:
            self.grid[y][x].append("neutral_stake")
        
        # Add alliance wall stakes
        red_stake = ALLIANCE_WALL_STAKES['red']
        blue_stake = ALLIANCE_WALL_STAKES['blue']
        self.grid[red_stake[1]][red_stake[0]].append("red_stake")
        self.grid[blue_stake[1]][blue_stake[0]].append("blue_stake")
        
        # Initialize ring counters for all stake types
        self.stake_rings = {}
        
        # Add ring counters for alliance stakes
        self.stake_rings[ALLIANCE_WALL_STAKES['red']] = {'red': 0, 'blue': 0}
        self.stake_rings[ALLIANCE_WALL_STAKES['blue']] = {'red': 0, 'blue': 0}
        
        # Add ring counters for neutral stakes
        for stake_pos in NEUTRAL_WALL_STAKES:
            self.stake_rings[stake_pos] = {'red': 0, 'blue': 0}
        
        # Separate tracking for mobile goals and their rings
        self.goal_rings = {}  # (x,y) -> {'red': count, 'blue': count}

    def is_valid_position(self, x, y, size=1):
        """Check if a position is valid for object placement"""
        for dx in range(size):
            for dy in range(size):
                if not (0 <= x+dx < GRID_SIZE and 0 <= y+dy < GRID_SIZE):
                    return False
                # Check if position is in the occupied list (stakes, scoring zones)
                if (x+dx, y+dy) in self.occupied:
                    return False
                # Also check if this position has any objects already
                if self.grid[y+dy][x+dx]:
                    return False
        return True

    def place_object(self, obj, count, size=1, allow_stack=False):
        """Place objects on the field"""
        placed = 0
        attempts = 0
        max_attempts = count * 100  # Avoid infinite loops
        
        while placed < count and attempts < max_attempts:
            attempts += 1
            x = random.randint(0, GRID_SIZE - size)
            y = random.randint(0, GRID_SIZE - size)

            # For rings, never allow stacking
            if obj.symbol in ['r', 'b']:
                if self.is_valid_position(x, y, size):
                    self.grid[y][x].append(obj.symbol)
                    # Mark position as occupied to prevent other objects being placed here
                    self.occupied.add((x, y))
                    placed += 1
            # For other objects like goals, use the normal logic
            else:
                if self.is_valid_position(x, y, size):
                    for dx in range(size):
                        for dy in range(size):
                            self.grid[y+dy][x+dx].append(obj.symbol)
                            self.occupied.add((x+dx, y+dy))
                    
                    # Initialize goal ring counter if this is a mobile goal
                    if obj.symbol == 'G':
                        self.goal_rings[(x, y)] = {'red': 0, 'blue': 0}
                    
                    placed += 1
        
        if placed < count:
            print(f"Warning: Could only place {placed}/{count} of object {obj.name} after {max_attempts} attempts")

    def get_grid(self):
        return self.grid
    
class RobotSimulator:

    def __init__(self, field, team_color, robot_pos, plan):
        # field is deep‑copied so simulator can reset cleanly
        self.original_field = copy.deepcopy(field)
        self.field = copy.deepcopy(field)

        self.team_color = team_color
        self.original_robot_pos = robot_pos  # Store original position
        self.robot_positions = [robot_pos]
        self.plans = [plan]
        self.current_step = 0
        self.max_steps = min(len(plan), MAX_SIMULATION_STEPS)

        self.ring_symbol = 'r' if team_color == 'red' else 'b'
        self.robot_symbol = 'R' if team_color == 'red' else 'B'
        self.robot_states = [{'rings': 0, 'goal_loaded': False, 'held_goal_rings': {'red': 0, 'blue': 0}}]

        # Track last action for better UI feedback
        self.last_action_result = ""

        # Initialize field state from the original field
        self._initialize_field_state()
        
        # Add robot to grid at start position
        self._add_robot_to_grid(robot_pos[0], robot_pos[1])
        
        # Initialize history with initial state
        self.history = [self._snapshot()]
    
    def _initialize_field_state(self):
        """Initialize field state from the original field"""
        # gather original mobile goal locations and ring counts
        self.goal_positions = []
        self.goal_rings = {}
        
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if 'G' in self.field.grid[y][x]:
                    self.goal_positions.append((x, y))
                    if (x, y) in self.original_field.goal_rings:
                        self.goal_rings[(x, y)] = copy.deepcopy(self.original_field.goal_rings[(x, y)])
                    else:
                        self.goal_rings[(x, y)] = {'red': 0, 'blue': 0}
        
        # Copy stake ring counters
        self.stake_rings = copy.deepcopy(self.original_field.stake_rings)
        
        # Track all ring positions by color
        self.ring_positions = {
            'red': [],
            'blue': []
        }
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if 'r' in self.field.grid[y][x]:
                    self.ring_positions['red'].append((x, y))
                if 'b' in self.field.grid[y][x]:
                    self.ring_positions['blue'].append((x, y))
        
        # Track score
        self.current_score = {'red': 0, 'blue': 0}
        self.update_score()
    
    def _find_all_rings(self):
        """Find all rings of the robot's color on the field"""
        positions = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.ring_symbol in self.field.grid[y][x]:
                    positions.append((x, y))
        return positions

    # ---------- history helpers ----------
    def _snapshot(self):
        """Take a complete snapshot of the current state"""
        return {
            'field': copy.deepcopy(self.field),
            'robot_positions': copy.deepcopy(self.robot_positions),
            'robot_states': copy.deepcopy(self.robot_states),
            'goal_positions': copy.deepcopy(self.goal_positions),
            'goal_rings': copy.deepcopy(self.goal_rings),
            'stake_rings': copy.deepcopy(self.stake_rings),
            'ring_positions': copy.deepcopy(self.ring_positions),
            'score': copy.deepcopy(self.current_score),
            'last_action_result': self.last_action_result,
            'current_step': self.current_step
        }

    # ---------- public controls ----------
    def step_forward(self):
        """Execute the next step in the plan"""
        if self.current_step >= self.max_steps:
            print("End of plan reached")
            return False

        action = self.plans[0][self.current_step]
        self.last_action_result = ""  # Reset last action result
        
        # Execute the action and capture the result
        self._execute_action(action)
        self.current_step += 1
        self.update_score()
        self.history.append(self._snapshot())
        return True

    def step_backward(self):
        """Return to the previous step using the restart approach"""
        if self.current_step <= 0:
            print("Already at beginning of simulation")
            return False
            
        # Restart from beginning and execute steps until current_step - 1
        self.reset(quiet=True)
        target_step = self.current_step - 1
        
        for i in range(target_step):
            if i < len(self.plans[0]):
                action = self.plans[0][i]
                self._execute_action(action)
                self.current_step += 1
                self.update_score()
                self.history.append(self._snapshot())
        
        print("Stepped backward")
        return True

    def reset(self, quiet=False):
        """Reset the simulator to its initial state"""
        # Clear robot from current position
        self._clear_robot_from_grid()
        
        # Reset field to original state
        self.field = copy.deepcopy(self.original_field)
        self.current_step = 0
        
        # Reset to original position
        self.robot_positions = [self.original_robot_pos]
        self.robot_states = [{'rings': 0, 'goal_loaded': False, 'held_goal_rings': {'red': 0, 'blue': 0}}]
        
        # Reset the field state
        self._initialize_field_state()
        
        # Reset last action result
        self.last_action_result = ""
        
        # Make sure robot is on the grid
        self._add_robot_to_grid(self.original_robot_pos[0], self.original_robot_pos[1])
        
        # Reset history
        self.history = [self._snapshot()]
        
        if not quiet:
            print("Resetting simulation")
        return True

    def update_score(self):
        """Update the current score based on the field state"""
        self.current_score = calculate_score(self.field.grid, self.goal_positions, 
                                            {**self.goal_rings, **self.stake_rings})

    # ---------- action execution ----------
    def _clear_robot_from_grid(self):
        """Remove the robot from the entire grid"""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.robot_symbol in self.field.grid[y][x]:
                    self.field.grid[y][x].remove(self.robot_symbol)

    def _remove_robot_from_grid(self, x, y):
        """Remove the robot from its current position"""
        for dx in range(ROBOT_SIZE):
            for dy in range(ROBOT_SIZE):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if self.robot_symbol in self.field.grid[ny][nx]:
                        self.field.grid[ny][nx].remove(self.robot_symbol)

    def _add_robot_to_grid(self, x, y):
        """Add the robot to the grid at the specified position"""
        for dx in range(ROBOT_SIZE):
            for dy in range(ROBOT_SIZE):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    self.field.grid[ny][nx].append(self.robot_symbol)

    def _robot_cells(self):
        """Return all cells covered by the robot"""
        x, y = self.robot_positions[0]
        cells = []
        for dx in range(ROBOT_SIZE):
            for dy in range(ROBOT_SIZE):
                cells.append((x+dx, y+dy))
        return cells

    def _execute_action(self, action):
        x, y = self.robot_positions[0]
        state = self.robot_states[0]

        if action.startswith("Move"):
            _, direction = action.split()
            dx, dy = DIRS[direction]
            
            # Check if new position would be valid before moving
            new_x, new_y = x + dx, y + dy
            if not (0 <= new_x <= GRID_SIZE - ROBOT_SIZE and 0 <= new_y <= GRID_SIZE - ROBOT_SIZE):
                # Invalid move, don't update position
                error_msg = f"Invalid move: ({new_x}, {new_y}) would be out of bounds"
                print(error_msg)
                self.last_action_result = error_msg
                return
                    
            # Check if any part of the robot would be over an obstacle (ring, goal, stake)
            invalid_move = False
            obstacle_pos = None
            for rdx in range(ROBOT_SIZE):
                for rdy in range(ROBOT_SIZE):
                    nx, ny = new_x + rdx, new_y + rdy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        cell = self.field.grid[ny][nx]
                        # Don't allow moving over rings, goals, or stakes
                        if 'r' in cell or 'b' in cell or 'G' in cell or "red_stake" in cell or "blue_stake" in cell or "neutral_stake" in cell:
                            invalid_move = True
                            obstacle_pos = (nx, ny)
                            break
                    if invalid_move:
                        break
                        
            if invalid_move:
                error_msg = f"Invalid move: {obstacle_pos} contains an obstacle"
                print(error_msg)
                self.last_action_result = error_msg
                return
                    
            # Valid move, update position
            self._remove_robot_from_grid(x, y)
            x, y = new_x, new_y
            self.robot_positions[0] = (x, y)
            self._add_robot_to_grid(x, y)
            self.last_action_result = f"Moved {direction} to position ({x}, {y})"

        elif action == "PickUpRing":
            # Check perimeter around the robot for rings
            ring_found = False
            ring_pos = None
            
            # Get all cells adjacent to the robot
            perimeter = set()
            for dx in range(ROBOT_SIZE):
                for dy in range(ROBOT_SIZE):
                    rx, ry = x + dx, y + dy
                    if 0 <= rx < GRID_SIZE and 0 <= ry < GRID_SIZE:
                        # Check all 4 directions from this robot cell
                        for dir_x, dir_y in DIRS.values():
                            nx, ny = rx + dir_x, ry + dir_y
                            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                                # Don't include cells under the robot
                                if not (x <= nx < x + ROBOT_SIZE and y <= ny < y + ROBOT_SIZE):
                                    perimeter.add((nx, ny))
                
            # Check for rings in the perimeter
            for nx, ny in perimeter:
                if self.ring_symbol in self.field.grid[ny][nx]:
                    # Remove the ring from the field
                    self.field.grid[ny][nx].remove(self.ring_symbol)
                    
                    # Remove the ring from our tracking list
                    ring_type = 'red' if self.ring_symbol == 'r' else 'blue'
                    ring_pos = (nx, ny)
                    if ring_pos in self.ring_positions[ring_type]:
                        self.ring_positions[ring_type].remove(ring_pos)
                    
                    # Add ring to robot inventory
                    state['rings'] += 1
                    ring_found = True
                    success_msg = f"Picked up {self.team_color} ring at {ring_pos}."
                    print(success_msg)
                    self.last_action_result = success_msg
                    break
            
            if not ring_found:
                # No ring found - look for nearby rings to suggest
                nearby_msg = f"No {self.team_color} ring found adjacent to robot to pick up"
                print(nearby_msg)
                
                # Look for nearest ring to suggest a move
                nearest_ring = None
                nearest_dist = float('inf')
                for rx, ry in self.ring_positions[self.team_color]:
                    dist = abs(x - rx) + abs(y - ry)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_ring = (rx, ry)
                
                if nearest_ring:
                    suggestion = f"Consider moving toward the ring at {nearest_ring} instead"
                    print(suggestion)
                    self.last_action_result = f"{nearby_msg}. {suggestion}"
                else:
                    self.last_action_result = f"{nearby_msg}. No rings of your color remain on the field."

        elif action.startswith("PlaceRingOnGoal"):
            # Place a ring on a mobile goal
            if state['rings'] > 0:
                # If the robot is carrying a goal, place ring on that goal
                if state['goal_loaded']:
                    # Check if the held goal has reached maximum capacity
                    ring_type = 'red' if self.ring_symbol == 'r' else 'blue'
                    total_rings = state['held_goal_rings']['red'] + state['held_goal_rings']['blue']
                    
                    if total_rings < MAX_RINGS_MOBILE_STAKE:
                        # Add the ring to the goal's inventory
                        state['held_goal_rings'][ring_type] += 1
                        state['rings'] -= 1
                        success_msg = f"Placed {ring_type} ring on held goal. Goal now has {state['held_goal_rings']}"
                        print(success_msg)
                        self.last_action_result = success_msg
                    else:
                        error_msg = f"Held goal is already at maximum capacity ({MAX_RINGS_MOBILE_STAKE} rings)"
                        print(error_msg)
                        self.last_action_result = error_msg
                    return
                
                # Otherwise, check perimeter for goals
                perim = set()
                for rx, ry in self._robot_cells():
                    for dx, dy in DIRS.values():
                        nx, ny = rx + dx, ry + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            perim.add((nx, ny))
                
                # Remove cells under the robot
                perim = perim - set(self._robot_cells())
                
                # First check if there are any goals with available capacity
                goal_found = False
                for px, py in perim:
                    if (px, py) in self.goal_positions:
                        goal_pos = (px, py)
                        ring_type = 'red' if self.ring_symbol == 'r' else 'blue'
                        
                        # Check if goal has space for more rings
                        total_rings = self.goal_rings[goal_pos]['red'] + self.goal_rings[goal_pos]['blue']
                        if total_rings < MAX_RINGS_MOBILE_STAKE:
                            # Place ring on goal
                            self.goal_rings[goal_pos][ring_type] += 1
                            state['rings'] -= 1
                            goal_found = True
                            success_msg = f"Placed {ring_type} ring on goal at {goal_pos}. Goal now has {self.goal_rings[goal_pos]}"
                            print(success_msg)
                            self.last_action_result = success_msg
                            break
                        else:
                            error_msg = f"Goal at {goal_pos} is already at maximum capacity ({MAX_RINGS_MOBILE_STAKE} rings)"
                            print(error_msg)
                            self.last_action_result = error_msg
                
                if not goal_found and state['rings'] > 0:
                    error_msg = "No goal with available capacity found to place ring on"
                    print(error_msg)
                    self.last_action_result = error_msg
            else:
                error_msg = "Cannot place ring: no rings in inventory"
                print(error_msg)
                self.last_action_result = error_msg
                        
        elif action.startswith("PlaceRingOnStake"):
            # Place a ring on an alliance or neutral stake
            if state['rings'] > 0:
                # Check perimeter around robot for a stake
                perim = set()
                for rx, ry in self._robot_cells():
                    for dx, dy in DIRS.values():
                        nx, ny = rx + dx, ry + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            perim.add((nx, ny))
                
                # Remove cells under the robot
                perim = perim - set(self._robot_cells())
                
                stake_found = False
                
                # Try alliance stake first (higher priority)
                alliance_stake = ALLIANCE_WALL_STAKES[self.team_color]
                if alliance_stake in perim:
                    max_rings = MAX_RINGS_ALLIANCE_STAKE
                    if alliance_stake in self.stake_rings:
                        total_rings = self.stake_rings[alliance_stake]['red'] + self.stake_rings[alliance_stake]['blue']
                        if total_rings < max_rings:
                            ring_type = 'red' if self.ring_symbol == 'r' else 'blue'
                            self.stake_rings[alliance_stake][ring_type] += 1
                            state['rings'] -= 1
                            stake_found = True
                            success_msg = f"Placed {ring_type} ring on alliance stake at {alliance_stake}"
                            print(success_msg)
                            self.last_action_result = success_msg
                        else:
                            error_msg = f"Alliance stake at {alliance_stake} is already at maximum capacity ({MAX_RINGS_ALLIANCE_STAKE} rings)"
                            print(error_msg)
                            self.last_action_result = error_msg
                
                # Try neutral stakes if alliance stake wasn't available
                if not stake_found:
                    for stake_pos in NEUTRAL_WALL_STAKES:
                        if stake_pos in perim:
                            max_rings = MAX_RINGS_NEUTRAL_STAKE
                            if stake_pos in self.stake_rings:
                                total_rings = self.stake_rings[stake_pos]['red'] + self.stake_rings[stake_pos]['blue']
                                if total_rings < max_rings:
                                    ring_type = 'red' if self.ring_symbol == 'r' else 'blue'
                                    self.stake_rings[stake_pos][ring_type] += 1
                                    state['rings'] -= 1
                                    stake_found = True
                                    success_msg = f"Placed {ring_type} ring on neutral stake at {stake_pos}"
                                    print(success_msg)
                                    self.last_action_result = success_msg
                                    break
                                else:
                                    error_msg = f"Neutral stake at {stake_pos} is already at maximum capacity ({MAX_RINGS_NEUTRAL_STAKE} rings)"
                                    print(error_msg)
                                    self.last_action_result = error_msg
                
                if not stake_found and state['rings'] > 0:
                    error_msg = "No stake with available capacity found to place ring on"
                    print(error_msg)
                    self.last_action_result = error_msg
            else:
                error_msg = "Cannot place ring: no rings in inventory"
                print(error_msg)
                self.last_action_result = error_msg

        elif action == "LoadGoal":
            if state['goal_loaded']:
                error_msg = "Cannot load goal: already carrying a goal"
                print(error_msg)
                self.last_action_result = error_msg
                return
                
            # Examine perimeter cells 1 tile away
            perim = []
            for i in range(ROBOT_SIZE):
                perim.append((x + i, y - 1))
                perim.append((x + ROBOT_SIZE, y + i))
                perim.append((x + i, y + ROBOT_SIZE))
                perim.append((x - 1, y + i))

            goal_found = False
            for nx, ny in perim:
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 'G' in self.field.grid[ny][nx]:
                    # Cannot load goals from stationary stakes
                    stake_pos = (nx, ny)
                    if stake_pos in NEUTRAL_WALL_STAKES or stake_pos == ALLIANCE_WALL_STAKES['red'] or stake_pos == ALLIANCE_WALL_STAKES['blue']:
                        error_msg = f"Cannot load stationary stake at {stake_pos}"
                        print(error_msg)
                        self.last_action_result = error_msg
                        continue
                        
                    # We can load mobile goals even if they're in scoring zones
                    goal_pos = (nx, ny)
                    self.field.grid[ny][nx].remove('G')
                    if goal_pos in self.goal_positions:
                        self.goal_positions.remove(goal_pos)
                        # Store the rings on this goal in the robot state
                        if goal_pos in self.goal_rings:
                            # Track the rings on the loaded goal
                            state['held_goal_rings'] = copy.deepcopy(self.goal_rings[goal_pos])
                            self.goal_rings.pop(goal_pos)
                    
                    # Loading a goal causes the robot to drop any rings it was carrying
                    previous_rings = state['rings']
                    state['goal_loaded'] = True
                    state['rings'] = 0  # Reset rings when loading goal
                    
                    # Add status message about dropped rings
                    goal_found = True
                    ring_msg = f" (Dropped {previous_rings} rings)" if previous_rings > 0 else ""
                    success_msg = f"Loaded goal from {goal_pos} with rings: {state['held_goal_rings']}{ring_msg}"
                    print(success_msg)
                    self.last_action_result = success_msg
                    break
            
            if not goal_found:
                error_msg = "No mobile goal found to load"
                print(error_msg)
                self.last_action_result = error_msg

        elif action == "DeliverGoal":
            if not state['goal_loaded']:
                error_msg = "Cannot deliver goal: no goal loaded"
                print(error_msg)
                self.last_action_result = error_msg
                return
                
            # Get all cells adjacent to the robot (not inside it)
            perimeter = []
            robot_cells = set(self._robot_cells())
            
            # Check all cells around the robot's perimeter
            for rx, ry in robot_cells:
                for dx, dy in DIRS.values():
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        # Don't include cells that are part of the robot
                        if (nx, ny) not in robot_cells:
                            # Don't include cells that already have objects
                            if not ('r' in self.field.grid[ny][nx] or 
                                    'b' in self.field.grid[ny][nx] or 
                                    'G' in self.field.grid[ny][nx] or 
                                    "red_stake" in self.field.grid[ny][nx] or 
                                    "blue_stake" in self.field.grid[ny][nx] or 
                                    "neutral_stake" in self.field.grid[ny][nx]):
                                perimeter.append((nx, ny))
            
            # Get the positive zones
            top_left_zone = [(0, 22), (1, 22), (0, 23), (1, 23)]
            top_right_zone = [(22, 22), (23, 22), (22, 23), (23, 23)]
            
            # Check if any part of the robot is in a positive zone
            in_positive_zone = any(cell in top_left_zone or cell in top_right_zone for cell in robot_cells)
            
            if in_positive_zone and perimeter:
                # We're in a positive zone and have valid positions to place the goal
                
                # First try to find a position that's also in a positive zone
                positive_positions = [pos for pos in perimeter if pos in top_left_zone or pos in top_right_zone]
                
                if positive_positions:
                    # Place the goal in a positive zone position
                    cx, cy = positive_positions[0]
                    self.field.grid[cy][cx].append('G')
                    self.goal_positions.append((cx, cy))
                    
                    # Transfer rings from robot's held goal to the placed goal
                    self.goal_rings[(cx, cy)] = copy.deepcopy(state['held_goal_rings'])
                    state['held_goal_rings'] = {'red': 0, 'blue': 0}
                    state['goal_loaded'] = False
                    success_msg = f"Delivered goal to positive zone at ({cx}, {cy}) with rings: {self.goal_rings[(cx, cy)]}"
                    print(success_msg)
                    self.last_action_result = success_msg
                else:
                    # Place the goal in any valid position near the robot
                    cx, cy = perimeter[0]
                    self.field.grid[cy][cx].append('G')
                    self.goal_positions.append((cx, cy))
                    
                    # Transfer rings from robot's held goal to the placed goal
                    self.goal_rings[(cx, cy)] = copy.deepcopy(state['held_goal_rings'])
                    state['held_goal_rings'] = {'red': 0, 'blue': 0}
                    state['goal_loaded'] = False
                    success_msg = f"Delivered goal adjacent to robot at ({cx}, {cy}) with rings: {self.goal_rings[(cx, cy)]}"
                    print(success_msg)
                    self.last_action_result = success_msg
            elif perimeter:
                # We're not in a positive zone but have valid positions to place the goal
                
                # Make sure we don't place in negative zones
                negative_zones = [(0, 0), (1, 0), (0, 1), (1, 1), (22, 0), (23, 0), (22, 1), (23, 1)]
                safe_positions = [pos for pos in perimeter if pos not in negative_zones]
                
                if safe_positions:
                    cx, cy = safe_positions[0]
                    self.field.grid[cy][cx].append('G')
                    self.goal_positions.append((cx, cy))
                    
                    # Transfer rings from robot's held goal to the placed goal
                    self.goal_rings[(cx, cy)] = copy.deepcopy(state['held_goal_rings'])
                    state['held_goal_rings'] = {'red': 0, 'blue': 0}
                    state['goal_loaded'] = False
                    success_msg = f"Delivered goal to ({cx}, {cy}) with rings: {self.goal_rings[(cx, cy)]}"
                    print(success_msg)
                    self.last_action_result = success_msg
                else:
                    error_msg = "Cannot deliver goal: no valid placement positions outside negative zones"
                    print(error_msg)
                    self.last_action_result = error_msg
            else:
                error_msg = "Cannot deliver goal: no valid placement positions around robot"
                print(error_msg)
                self.last_action_result = error_msg

# ------------ VISUALISATION -----------
def draw_field(sim):
    fig, ax = plt.subplots(figsize=(12, 10))
    step_text = plt.figtext(0.02, 0.95, f"Step: 0 / {sim.max_steps}")
    score_text = plt.figtext(0.5, 0.02, "", ha='center', fontsize=12)
    debug_text = plt.figtext(0.98, 0.02, "", ha='right', fontsize=8, color='gray')
    action_text = None  # Keep track of action text
    status_text = None  # Keep track of action result status

    def repaint():
        nonlocal action_text, status_text
        ax.clear()
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_xticks(np.arange(0, GRID_SIZE, 1))
        ax.set_yticks(np.arange(0, GRID_SIZE, 1))
        ax.set_facecolor("#d3d3d3")
        ax.grid(True, which='both', color='black', linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

        grid = sim.field.grid

        # Corner zones - all yellow with + and - markers
        for zone, label, name in [
            (POSITIVE_CORNERS, '+', 'Red Alliance Positive'),
            (NEGATIVE_CORNERS, '-', 'Red Alliance Negative'),
            (POSITIVE_CORNERS_BLUE, '+', 'Blue Alliance Positive'),
            (NEGATIVE_CORNERS_BLUE, '-', 'Blue Alliance Negative')]:
            for x, y in zone:
                ax.add_patch(patches.Rectangle((x, y), 1, 1, color='yellow'))
            cx, cy = np.mean(zone, axis=0)
            ax.text(cx+0.25, cy+0.25, label, color='black', fontsize=14, fontweight='bold')
            
        # Neutral wall stakes (yellow)
        for x, y in NEUTRAL_WALL_STAKES:
            ax.add_patch(patches.Rectangle((x, y), 1, 1, color='yellow'))
            ax.text(x+0.25, y+0.25, "S", color='black', fontsize=14, fontweight='bold')
            
            # Display ring counters for neutral stakes
            if (x, y) in sim.stake_rings:
                red_count = sim.stake_rings[(x, y)]['red']
                blue_count = sim.stake_rings[(x, y)]['blue']
                if red_count > 0 or blue_count > 0:
                    ax.text(x+0.5, y+0.75, f"{red_count}R/{blue_count}B", 
                           fontsize=8, ha='center', va='center', color='black')
        
        # Alliance stakes
        red_stake = ALLIANCE_WALL_STAKES['red']
        blue_stake = ALLIANCE_WALL_STAKES['blue']
        
        ax.add_patch(patches.Rectangle((red_stake[0], red_stake[1]), 1, 1, color='darkred'))
        ax.text(red_stake[0]+0.25, red_stake[1]+0.25, "S", color='white', fontsize=14, fontweight='bold')
        
        # Display ring counters for red alliance stake
        if red_stake in sim.stake_rings:
            red_count = sim.stake_rings[red_stake]['red']
            blue_count = sim.stake_rings[red_stake]['blue']
            if red_count > 0 or blue_count > 0:
                ax.text(red_stake[0]+0.5, red_stake[1]+0.75, f"{red_count}R/{blue_count}B", 
                       fontsize=8, ha='center', va='center', color='white')
        
        ax.add_patch(patches.Rectangle((blue_stake[0], blue_stake[1]), 1, 1, color='darkblue'))
        ax.text(blue_stake[0]+0.25, blue_stake[1]+0.25, "S", color='white', fontsize=14, fontweight='bold')
        
        # Display ring counters for blue alliance stake
        if blue_stake in sim.stake_rings:
            red_count = sim.stake_rings[blue_stake]['red']
            blue_count = sim.stake_rings[blue_stake]['blue']
            if red_count > 0 or blue_count > 0:
                ax.text(blue_stake[0]+0.5, blue_stake[1]+0.75, f"{red_count}R/{blue_count}B", 
                       fontsize=8, ha='center', va='center', color='white')

        # highlight robot zone
        rx, ry = sim.robot_positions[0]
        edge_col = 'red' if sim.team_color == 'red' else 'blue'
        ax.add_patch(patches.Rectangle((rx, ry), ROBOT_SIZE, ROBOT_SIZE,
                                        fill=False, edgecolor=edge_col, linestyle='--', linewidth=1.5))

        # pieces
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cell = grid[y][x]
                for sym in cell:
                    if sym == 'G':
                        ax.add_patch(patches.Circle((x+0.5, y+0.5), 0.35, facecolor=MOBILE_GOAL.color, edgecolor='#aaaaaa'))
                        # Display ring counters for goals
                        if (x, y) in sim.goal_rings:
                            red_count = sim.goal_rings[(x, y)]['red']
                            blue_count = sim.goal_rings[(x, y)]['blue']
                            if red_count > 0 or blue_count > 0:
                                ax.text(x+0.5, y+0.5, f"{red_count}R/{blue_count}B", 
                                       fontsize=8, ha='center', va='center', color='black')
                    elif sym == 'R':
                        ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=ROBOT_RED.color, edgecolor='black'))
                    elif sym == 'B':
                        ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=ROBOT_BLUE.color, edgecolor='black'))
                    elif sym == 'r':
                        ax.add_patch(patches.Circle((x+0.5, y+0.5), 0.3, color=RING_RED.color))
                    elif sym == 'b':
                        ax.add_patch(patches.Circle((x+0.5, y+0.5), 0.3, color=RING_BLUE.color))

        # overlay robot state text
        state = sim.robot_states[0]
        cx, cy = rx + ROBOT_SIZE/2 - 0.5, ry + ROBOT_SIZE/2 - 0.5
        if state['rings'] > 0:
            ax.text(cx, cy, f"{state['rings']}", fontsize=14, fontweight='bold', color='white')
        if state['goal_loaded']:
            ax.add_patch(patches.Circle((cx, cy), 0.25, facecolor=MOBILE_GOAL.color, edgecolor='black'))
            
            # If the goal has rings, show their count
            if state['held_goal_rings']['red'] > 0 or state['held_goal_rings']['blue'] > 0:
                red_count = state['held_goal_rings']['red']
                blue_count = state['held_goal_rings']['blue'] 
                ax.text(cx, cy+0.4, f"{red_count}R/{blue_count}B", 
                       fontsize=8, ha='center', va='center', color='black',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        plt.gca().invert_yaxis()
        step_text.set_text(f"Step: {sim.current_step} / {sim.max_steps}")
        score_text.set_text(f"Score - Red: {sim.current_score['red']} | Blue: {sim.current_score['blue']}")
        debug_text.set_text(f"Updated: {time.strftime('%H:%M:%S')}")

        # Remove previous text elements if they exist
        if action_text is not None:
            action_text.remove()
            action_text = None
        if status_text is not None:
            status_text.remove()
            status_text = None

        # Show active action
        if sim.current_step > 0 and sim.current_step-1 < sim.max_steps:
            action = sim.plans[0][sim.current_step-1]
            action_text = plt.figtext(0.5, 0.95, f"Action: {action}", ha='center', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            
            # Display action result status if available
            if hasattr(sim, 'last_action_result') and sim.last_action_result:
                status_color = 'red' if any(x in sim.last_action_result.lower() for x in ['invalid', 'error', 'cannot', 'no']) else 'green'
                status_text = plt.figtext(0.5, 0.91, sim.last_action_result, ha='center', fontsize=10,
                              color=status_color, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        plt.title(f"VEX U High Stakes — {sim.team_color.capitalize()} Alliance")
        fig.canvas.draw_idle()

    # ---- buttons ----
    btn_y = 0.05
    prev_ax = plt.axes([0.15, btn_y, 0.15, 0.05])
    reset_ax = plt.axes([0.425, btn_y, 0.15, 0.05])
    next_ax = plt.axes([0.7, btn_y, 0.15, 0.05])
    prev_btn = Button(prev_ax, '← Previous')
    reset_btn = Button(reset_ax, 'Reset')
    next_btn = Button(next_ax, 'Next Step →')

    def on_prev_click(event):
        success = sim.step_backward()
        if success:
            repaint()
        else:
            print("Already at beginning of simulation")

    def on_next_click(event):
        success = sim.step_forward()
        if success:
            repaint()
        else:
            print("End of plan reached")

    def on_reset_click(event):
        sim.reset()
        repaint()

    prev_btn.on_clicked(on_prev_click)
    next_btn.on_clicked(on_next_click)
    reset_btn.on_clicked(on_reset_click)

    repaint()
    plt.tight_layout(rect=[0, 0.1, 1, 0.93])
    plt.show()

# ------------ UI ----------------------
def on_team_select(team):
    print(f"You selected the {team.upper()} alliance.")
    window.destroy()

    try:
        from ai_search_engine import a_star_search, State
    except ImportError:
        print("ERROR: Could not import ai_search_engine module!")
        print("Make sure you have created the file 'ai_search_engine.py'")
        return

    # Use designated spawn positions instead of end zones
    spawn = SPAWN_POSITION[team]
    sx, sy = spawn
    start_state = State(x=sx, y=sy, rings=0, held_goals=(), steps=0)
    
    print("Starting search for plan...")
    t0 = time.time()
    plan = a_star_search(start_state, field.get_grid(), team, field.stake_rings)
    t1 = time.time()
    
    if not plan:
        print("No plan found!")
        # Provide a simple default plan for testing
        plan = ["Move Right", "Move Down", "PickUpRing", "Move Up", "Move Left"]
        
    print(f"Planned {len(plan)} actions in {t1 - t0:.2f}s")
    print(f"Maximum simulation steps: {MAX_SIMULATION_STEPS}")
    for step in plan[:10]:  # Show first 10 steps
        print("-", step)
    if len(plan) > 10:
        print(f"... and {len(plan) - 10} more steps")

    sim = RobotSimulator(field, team, spawn, plan)
    draw_field(sim)

def launch_ui():
    global window
    window = tk.Tk()
    window.title("VEX U High Stakes")
    w, h = 360, 200
    ws, hs = window.winfo_screenwidth(), window.winfo_screenheight()
    window.geometry(f"{w}x{h}+{(ws-w)//2}+{(hs-h)//2}")
    window.configure(bg="#f0f0f0")

    tk.Label(window, text="Select Your Alliance", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333").pack(pady=(20, 10))

    tk.Button(window, text="Red Alliance", font=("Arial", 14, "bold"), bg="#ff4d4d", fg="white",
              activebackground="#cc0000", bd=0, width=13,
              command=lambda: on_team_select("red")).pack(pady=5)

    tk.Button(window, text="Blue Alliance", font=("Arial", 14, "bold"), bg="#4da6ff", fg="white",
              activebackground="#007acc", bd=0, width=13,
              command=lambda: on_team_select("blue")).pack(pady=5)

    window.mainloop()

# ------------ MAIN --------------------
if __name__ == "__main__":
    field = FieldGrid()
    field.place_object(RING_RED,  24, allow_stack=True)
    field.place_object(RING_BLUE, 24, allow_stack=True)
    field.place_object(MOBILE_GOAL, 5)
    # No random robots — they will be spawned only after team selection.
    print("Launching UI...")
    launch_ui()
