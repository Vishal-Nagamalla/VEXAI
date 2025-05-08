import tkinter as tk
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import numpy as np
import time
import copy
from field import (GRID_SIZE, ROBOT_SIZE, POSITIVE_CORNERS, NEGATIVE_CORNERS, 
                  POSITIVE_CORNERS_BLUE, NEGATIVE_CORNERS_BLUE, RED_STAKE, 
                  BLUE_STAKE, FIXED_STAKES, DIRS, MAX_SIMULATION_STEPS,
                  MAX_RINGS_PER_GOAL)

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
                            FIXED_STAKES + [RED_STAKE, BLUE_STAKE])
        
        # Add fixed stakes
        for x, y in FIXED_STAKES:
            self.grid[y][x].append("stake")
        
        # Add colored stakes
        self.grid[RED_STAKE[1]][RED_STAKE[0]].append("red_stake")
        self.grid[BLUE_STAKE[1]][BLUE_STAKE[0]].append("blue_stake")
        
        # Initialize goal ring counters
        self.goal_rings = {}  # (x,y) -> {'red': count, 'blue': count}

    def is_valid_position(self, x, y, size=1):
        for dx in range(size):
            for dy in range(size):
                if not (0 <= x+dx < GRID_SIZE and 0 <= y+dy < GRID_SIZE):
                    return False
                if (x+dx, y+dy) in self.occupied:
                    return False
        return True

    def place_object(self, obj, count, size=1, allow_stack=False):
        placed = 0
        while placed < count:
            x = random.randint(0, GRID_SIZE - size)
            y = random.randint(0, GRID_SIZE - size)

            if allow_stack:
                if (x, y) in self.occupied:
                    continue

                stack = self.grid[y][x]
                if len(stack) == 0:
                    stack.append(obj.symbol)
                    placed += 1
                elif len(stack) == 1 and stack[0] in ['r', 'b']:
                    stack.append(obj.symbol)
                    placed += 1
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

    def get_grid(self):
        return self.grid

# ------------ SIMULATOR --------------
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
        self.robot_states = [{'rings': 0, 'goal_loaded': False}]

        # gather original mobile goal locations and ring counts
        self.goal_positions = []
        self.goal_rings = {}
        
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if 'G' in field.grid[y][x]:
                    self.goal_positions.append((x, y))
                    if (x, y) in field.goal_rings:
                        self.goal_rings[(x, y)] = copy.deepcopy(field.goal_rings[(x, y)])
                    else:
                        self.goal_rings[(x, y)] = {'red': 0, 'blue': 0}

        # Add robot to grid at start position
        self._add_robot_to_grid(robot_pos[0], robot_pos[1])
        
        self.history = [self._snapshot()]

    # ---------- history helpers ----------
    def _snapshot(self):
        return {
            'field': copy.deepcopy(self.field),
            'robot_positions': copy.deepcopy(self.robot_positions),
            'robot_states': copy.deepcopy(self.robot_states),
            'goal_positions': copy.deepcopy(self.goal_positions),
            'goal_rings': copy.deepcopy(self.goal_rings)
        }

    # ---------- public controls ----------
    def step_forward(self):
        if self.current_step >= self.max_steps:
            return False

        action = self.plans[0][self.current_step]
        self._execute_action(action)
        self.current_step += 1
        self.history.append(self._snapshot())
        return True

    def step_backward(self):
        if self.current_step <= 0:
            return False
        self.current_step -= 1
        state = self.history[self.current_step]
        self.field = state['field']
        self.robot_positions = state['robot_positions']
        self.robot_states = state['robot_states']
        self.goal_positions = state['goal_positions']
        self.goal_rings = state['goal_rings']
        return True

    def reset(self):
        self.field = copy.deepcopy(self.original_field)
        self.current_step = 0
        # Reset to original position
        self.robot_positions = [self.original_robot_pos]
        self.robot_states = [{'rings': 0, 'goal_loaded': False}]
        
        # Reset goal positions and ring counts from original field
        self.goal_positions = []
        self.goal_rings = {}
        
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if 'G' in self.original_field.grid[y][x]:
                    self.goal_positions.append((x, y))
                    if (x, y) in self.original_field.goal_rings:
                        self.goal_rings[(x, y)] = copy.deepcopy(self.original_field.goal_rings[(x, y)])
                    else:
                        self.goal_rings[(x, y)] = {'red': 0, 'blue': 0}
        
        # Make sure robot is on the grid
        self._clear_robot_from_grid()
        self._add_robot_to_grid(self.original_robot_pos[0], self.original_robot_pos[1])
        
        self.history = [self._snapshot()]

    # ---------- action execution ----------
    def _clear_robot_from_grid(self):
        # Remove robot from entire grid (to handle reset properly)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.robot_symbol in self.field.grid[y][x]:
                    self.field.grid[y][x].remove(self.robot_symbol)

    def _remove_robot_from_grid(self, x, y):
        for dx in range(ROBOT_SIZE):
            for dy in range(ROBOT_SIZE):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if self.robot_symbol in self.field.grid[ny][nx]:
                        self.field.grid[ny][nx].remove(self.robot_symbol)

    def _add_robot_to_grid(self, x, y):
        for dx in range(ROBOT_SIZE):
            for dy in range(ROBOT_SIZE):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    self.field.grid[ny][nx].append(self.robot_symbol)

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
                print(f"Invalid move: ({new_x}, {new_y}) would be out of bounds")
                return
                
            self._remove_robot_from_grid(x, y)
            x, y = new_x, new_y
            self.robot_positions[0] = (x, y)
            self._add_robot_to_grid(x, y)

        elif action == "PickUpRing":
            # Check all cells under the robot for rings
            ring_found = False
            for dx in range(ROBOT_SIZE):
                for dy in range(ROBOT_SIZE):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        if self.ring_symbol in self.field.grid[ny][nx]:
                            self.field.grid[ny][nx].remove(self.ring_symbol)
                            state['rings'] += 1
                            ring_found = True
                            break  # Only pick up one ring per action
                if ring_found:
                    break

        elif action.startswith("PlaceRingOnGoal"):
            # New action to place a ring on a mobile goal
            if state['rings'] > 0:
                # Check perimeter around robot for a goal
                perim = set()
                for rx, ry in self._robot_cells():
                    for dx, dy in DIRS.values():
                        nx, ny = rx + dx, ry + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            perim.add((nx, ny))
                
                # Remove cells under the robot
                perim = perim - set(self._robot_cells())
                
                for px, py in perim:
                    if (px, py) in self.goal_positions:
                        # We found a goal to place the ring on
                        goal_pos = (px, py)
                        ring_type = 'red' if self.ring_symbol == 'r' else 'blue'
                        
                        # Check if goal has space for more rings
                        total_rings = self.goal_rings[goal_pos]['red'] + self.goal_rings[goal_pos]['blue']
                        if total_rings < MAX_RINGS_PER_GOAL:
                            # Place ring on goal
                            self.goal_rings[goal_pos][ring_type] += 1
                            state['rings'] -= 1
                            print(f"Placed {ring_type} ring on goal at {goal_pos}. Goal now has {self.goal_rings[goal_pos]}")
                            break

        elif action == "LoadGoal":
            # examine perimeter cells 1 tile away
            perim = []
            for i in range(ROBOT_SIZE):
                perim.append((x + i, y - 1))
                perim.append((x + ROBOT_SIZE, y + i))
                perim.append((x + i, y + ROBOT_SIZE))
                perim.append((x - 1, y + i))

            for nx, ny in perim:
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 'G' in self.field.grid[ny][nx]:
                    # We can load goals even if they're in scoring zones
                    goal_pos = (nx, ny)
                    self.field.grid[ny][nx].remove('G')
                    if goal_pos in self.goal_positions:
                        self.goal_positions.remove(goal_pos)
                        # Store the rings on this goal in the robot state
                        if goal_pos in self.goal_rings:
                            # Could track these rings in robot state if desired
                            self.goal_rings.pop(goal_pos)
                    state['goal_loaded'] = True
                    state['rings'] = 0  # Reset rings when loading goal
                    break

        elif action == "DeliverGoal":
            # Check if any part of the robot is in a scoring zone
            in_scoring_zone = False
            
            # Get the appropriate scoring zones based on team color
            if self.team_color == 'red':
                positive_corners = POSITIVE_CORNERS
                negative_corners = NEGATIVE_CORNERS
            else:
                positive_corners = POSITIVE_CORNERS_BLUE
                negative_corners = NEGATIVE_CORNERS_BLUE
            
            # Check if any part of the robot is in a scoring zone
            robot_cells = self._robot_cells()
            for rx, ry in robot_cells:
                if (rx, ry) in positive_corners or (rx, ry) in negative_corners:
                    in_scoring_zone = True
                    break
            
            if in_scoring_zone and state['goal_loaded']:
                # Place the goal entirely within the scoring zone
                # Find the center of the robot
                cx, cy = x + ROBOT_SIZE//2 - 1, y + ROBOT_SIZE//2 - 1
                
                # If we're at the edge, adjust to ensure goal is in scoring zone
                for zone in [positive_corners, negative_corners]:
                    if any((rx, ry) in zone for rx, ry in robot_cells):
                        # Find center point of the zone
                        zone_x = sum(x for x, y in zone) // len(zone)
                        zone_y = sum(y for x, y in zone) // len(zone)
                        
                        # Adjust goal placement to be inside zone
                        if (cx, cy) not in zone:
                            # Find closest point in zone
                            for zx, zy in zone:
                                if abs(zx - cx) <= 1 and abs(zy - cy) <= 1:
                                    cx, cy = zx, zy
                                    break
                
                # Place the goal
                self.field.grid[cy][cx].append('G')
                self.goal_positions.append((cx, cy))
                self.goal_rings[(cx, cy)] = {'red': 0, 'blue': 0}  # Initialize ring counter
                state['goal_loaded'] = False
                print(f"Delivered goal to ({cx}, {cy})")
            else:
                print("Cannot deliver goal: not in scoring zone or no goal loaded")

    def _robot_cells(self):
        """Return all cells covered by the robot"""
        x, y = self.robot_positions[0]
        cells = []
        for dx in range(ROBOT_SIZE):
            for dy in range(ROBOT_SIZE):
                cells.append((x+dx, y+dy))
        return cells

# ------------ VISUALISATION -----------
def draw_field(sim):
    fig, ax = plt.subplots(figsize=(10, 10))
    step_text = plt.figtext(0.02, 0.95, f"Step: 0 / {sim.max_steps}")
    debug_text = plt.figtext(0.98, 0.02, "", ha='right', fontsize=8, color='gray')
    action_text = None  # Keep track of action text

    def repaint():
        nonlocal action_text
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
            
        # Fixed stakes (yellow)
        for x, y in FIXED_STAKES:
            ax.add_patch(patches.Rectangle((x, y), 1, 1, color='yellow'))
            ax.text(x+0.25, y+0.25, "S", color='black', fontsize=14, fontweight='bold')
        
        # Colored stakes
        ax.add_patch(patches.Rectangle((RED_STAKE[0], RED_STAKE[1]), 1, 1, color='darkred'))
        ax.text(RED_STAKE[0]+0.25, RED_STAKE[1]+0.25, "S", color='white', fontsize=14, fontweight='bold')
        
        ax.add_patch(patches.Rectangle((BLUE_STAKE[0], BLUE_STAKE[1]), 1, 1, color='darkblue'))
        ax.text(BLUE_STAKE[0]+0.25, BLUE_STAKE[1]+0.25, "S", color='white', fontsize=14, fontweight='bold')

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

        plt.gca().invert_yaxis()
        step_text.set_text(f"Step: {sim.current_step} / {sim.max_steps}")
        debug_text.set_text(f"Updated: {time.strftime('%H:%M:%S')}")

        # Remove previous action text if it exists
        if action_text is not None:
            action_text.remove()
            action_text = None

        # show active action
        if sim.current_step > 0 and sim.current_step-1 < sim.max_steps:
            action = sim.plans[0][sim.current_step-1]
            action_text = plt.figtext(0.5, 0.95, f"Action: {action}", ha='center', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        plt.title(f"VEX U High Stakes — {sim.team_color.capitalize()} Alliance")
        fig.canvas.draw_idle()

    # ---- buttons ----
    btn_y = 0.02
    prev_ax = plt.axes([0.15, btn_y, 0.15, 0.05])
    reset_ax = plt.axes([0.425, btn_y, 0.15, 0.05])
    next_ax = plt.axes([0.7, btn_y, 0.15, 0.05])
    prev_btn = Button(prev_ax, '← Previous')
    reset_btn = Button(reset_ax, 'Reset')
    next_btn = Button(next_ax, 'Next Step →')

    def on_prev_click(event):
        success = sim.step_backward()
        if success:
            print("Stepped backward")
            repaint()
        else:
            print("Cannot step backward further")

    def on_next_click(event):
        success = sim.step_forward()
        if success:
            print("Stepped forward")
            repaint()
        else:
            print("End of plan reached")

    def on_reset_click(event):
        print("Resetting simulation")
        sim.reset()
        repaint()

    prev_btn.on_clicked(on_prev_click)
    next_btn.on_clicked(on_next_click)
    reset_btn.on_clicked(on_reset_click)

    repaint()
    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
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

    spawn = (0, 21) if team == "blue" else (21, 0)
    sx, sy = spawn
    start_state = State(x=sx, y=sy, rings=0, delivered=0, goal_loaded=False)
    
    print("Starting search for plan...")
    t0 = time.time()
    plan = a_star_search(start_state, field.get_grid(), team)
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
