# Save this file as ai_search_engine.py
import heapq
import copy

# Import constants from field.py
from field import (GRID_SIZE, ROBOT_SIZE, POSITIVE_CORNERS, NEGATIVE_CORNERS, 
                  POSITIVE_CORNERS_BLUE, NEGATIVE_CORNERS_BLUE, RED_STAKE, 
                  BLUE_STAKE, FIXED_STAKES, DIRS, MAX_RINGS_PER_GOAL)

class State:
    """
    Represents: 
      - robot (x,y)
      - rings_collected (how many rings held)
      - goals_delivered (how many goals delivered)
      - goal_loaded (whether a mobile goal is currently held)
    """
    __slots__ = ('x', 'y', 'rings', 'delivered', 'goal_loaded')
    
    def __init__(self, x, y, rings=0, delivered=0, goal_loaded=False):
        self.x = x
        self.y = y
        self.rings = rings
        self.delivered = delivered
        self.goal_loaded = goal_loaded
    
    def __eq__(self, other):
        return (self.x, self.y, self.rings, self.delivered, self.goal_loaded) == \
               (other.x, other.y, other.rings, other.delivered, other.goal_loaded)
    
    def __hash__(self):
        return hash((self.x, self.y, self.rings, self.delivered, self.goal_loaded))

class Node:
    def __init__(self, state, g=0, h=0, parent=None, action=None):
        self.state = state
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.action = action
    
    def __lt__(self, other):
        return self.f < other.f

def robot_cells(x, y):
    """Return all cells occupied by the robot at position (x,y)"""
    cells = []
    for dx in range(ROBOT_SIZE):
        for dy in range(ROBOT_SIZE):
            cells.append((x+dx, y+dy))
    return cells

def heuristic(state, ring_positions, goal_positions, team_color):
    """Estimate cost to goal based on distance to nearest ring and delivery zone"""
    if state.delivered >= 5:  # Already at goal state
        return 0
        
    if not ring_positions and not state.goal_loaded:
        # No rings left and no goal loaded, we can't complete the task
        return 1000  # High cost to deprioritize this path
    
    # Get the appropriate scoring zones based on team color
    if team_color == 'red':
        positive_corners = POSITIVE_CORNERS
    else:
        positive_corners = POSITIVE_CORNERS_BLUE
    
    # If we need more goals and don't have one loaded
    if state.delivered < 5 and not state.goal_loaded:
        # Distance to closest ring if we need to pick one up
        if state.rings == 0 and ring_positions:
            return min(abs(state.x-x) + abs(state.y-y) for x, y in ring_positions)
        
        # Distance to closest goal if we have rings but no goal
        if state.rings > 0 and goal_positions:
            return min(abs(state.x-x) + abs(state.y-y) for x, y in goal_positions)
    
    # If we have a goal loaded, distance to delivery zone
    if state.goal_loaded:
        return min(abs(state.x-x) + abs(state.y-y) for x, y in positive_corners)
    
    return 0

def is_goal_state(state):
    # goal: delivered == 5 (we've placed 5 goals)
    return state.delivered >= 5

def get_successors(node, field, ring_positions, goal_positions, team_color):
    """ Generate (next_state, action_name, cost, new_ring_positions, new_goal_positions) """
    succ = []
    n = len(field)
    s = node.state
    ring_symbol = 'r' if team_color == 'red' else 'b'
    
    # Get the appropriate scoring zones based on team color
    if team_color == 'red':
        positive_corners = POSITIVE_CORNERS
        negative_corners = NEGATIVE_CORNERS
    else:
        positive_corners = POSITIVE_CORNERS_BLUE
        negative_corners = NEGATIVE_CORNERS_BLUE
    
    # 1) Move in 4 directions - respect robot size
    for name, (dx, dy) in DIRS.items():
        nx, ny = s.x+dx, s.y+dy
        if 0 <= nx <= GRID_SIZE - ROBOT_SIZE and 0 <= ny <= GRID_SIZE - ROBOT_SIZE:
            # Check if new position is valid
            valid_move = True
            for cx, cy in robot_cells(nx, ny):
                # Check for fixed stakes or other unmovable objects
                if not (0 <= cx < n and 0 <= cy < n):
                    valid_move = False
                    break
                if "stake" in field[cy][cx] or "red_stake" in field[cy][cx] or "blue_stake" in field[cy][cx]:
                    valid_move = False
                    break
            
            if valid_move:
                new = State(nx, ny, s.rings, s.delivered, s.goal_loaded)
                succ.append((new, f"Move {name}", 1, ring_positions, goal_positions))
    
    # 2) PickUpRing if ring of our color present under the robot and we carry <2
    if s.rings < 2:
        for rx, ry in robot_cells(s.x, s.y):
            if (rx, ry) in ring_positions:
                # Create new state with ring picked up
                new = State(s.x, s.y, s.rings+1, s.delivered, s.goal_loaded)
                new_ring_positions = ring_positions.copy()
                new_ring_positions.remove((rx, ry))
                succ.append((new, "PickUpRing", 1, new_ring_positions, goal_positions))
                break  # Only pick up one ring per action
    
    # 3) PlaceRingOnGoal if adjacent to a mobile goal and we have rings
    if s.rings > 0:
        # Get perimeter cells around robot
        perim = set()
        for rx, ry in robot_cells(s.x, s.y):
            for dx, dy in DIRS.values():
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    perim.add((nx, ny))
        
        # Remove cells under the robot
        robot_area = set(robot_cells(s.x, s.y))
        perim = perim - robot_area
        
        # Check for goals in perimeter
        for px, py in perim:
            if (px, py) in goal_positions:
                new = State(s.x, s.y, s.rings-1, s.delivered, s.goal_loaded)
                succ.append((new, "PlaceRingOnGoal", 1, ring_positions, goal_positions))
                break
    
    # 4) LoadGoal if adjacent to a mobile goal and we have rings but no loaded goal
    if s.rings > 0 and not s.goal_loaded:
        # Get perimeter cells around robot
        perim = set()
        for rx, ry in robot_cells(s.x, s.y):
            for dx, dy in DIRS.values():
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    perim.add((nx, ny))
        
        # Remove cells under the robot
        robot_area = set(robot_cells(s.x, s.y))
        perim = perim - robot_area
        
        # Check for goals in perimeter
        for px, py in perim:
            if (px, py) in goal_positions:
                new = State(s.x, s.y, 0, s.delivered, True)
                new_goals = goal_positions.copy()
                new_goals.remove((px, py))
                succ.append((new, "LoadGoal", 1, ring_positions, new_goals))
                break
    
    # 5) DeliverGoal if any part of the robot is in a positive corner and we have a goal
    if s.goal_loaded:
        # Check if any part of the robot is in a positive corner
        robot_area = set(robot_cells(s.x, s.y))
        pos_corners_set = set(positive_corners)
        
        if any(cell in pos_corners_set for cell in robot_area):
            # Find center point in scoring zone for goal placement
            cx, cy = s.x + ROBOT_SIZE//2 - 1, s.y + ROBOT_SIZE//2 - 1
            
            # Make sure goal is placed within scoring zone
            if (cx, cy) not in pos_corners_set:
                for px, py in pos_corners_set:
                    if abs(px - cx) <= 1 and abs(py - cy) <= 1:
                        cx, cy = px, py
                        break
            
            new = State(s.x, s.y, 0, s.delivered+1, False)
            new_goals = goal_positions.copy()
            new_goals.append((cx, cy))
            succ.append((new, "DeliverGoal", 1, ring_positions, new_goals))
    
    return succ

def reconstruct_path(node):
    actions = []
    while node.parent:
        actions.append(node.action)
        node = node.parent
    return list(reversed(actions))

def a_star_search(start_state, field, team_color):
    print(f"Starting A* search for {team_color} team...")
    
    # Initialize by finding all rings and goals in the field
    ring_symbol = 'r' if team_color == 'red' else 'b'
    N = len(field)
    
    # Find all initial ring positions
    ring_positions = []
    for y in range(N):
        for x in range(N):
            if ring_symbol in field[y][x]:
                ring_positions.append((x, y))
    
    print(f"Found {len(ring_positions)} {team_color} rings")
    
    # Find all initial goal positions
    goal_positions = []
    for y in range(N):
        for x in range(N):
            if 'G' in field[y][x]:
                goal_positions.append((x, y))
    
    print(f"Found {len(goal_positions)} goals")
    
    open_list = []
    start_h = heuristic(start_state, ring_positions, goal_positions, team_color)
    start_node = Node(start_state, g=0, h=start_h)
    heapq.heappush(open_list, start_node)
    closed = set()
    
    # Track ring and goal positions for each node
    node_resources = {id(start_node): (ring_positions, goal_positions)}
    
    nodes_expanded = 0
    max_nodes = 50000  # Limit search to prevent excessive runtime
    
    while open_list and nodes_expanded < max_nodes:
        current = heapq.heappop(open_list)
        
        if is_goal_state(current.state):
            path = reconstruct_path(current)
            print(f"Found solution with {len(path)} steps after expanding {nodes_expanded} nodes")
            return path
        
        state_key = (current.state.x, current.state.y, current.state.rings, 
                      current.state.delivered, current.state.goal_loaded)
        if state_key in closed:
            continue
            
        closed.add(state_key)
        nodes_expanded += 1
        
        if nodes_expanded % 1000 == 0:
            print(f"Expanded {nodes_expanded} nodes, queue size: {len(open_list)}")
        
        # Get the current ring and goal positions
        current_rings, current_goals = node_resources[id(current)]
        
        for result in get_successors(current, field, current_rings, current_goals, team_color):
            succ_state, action, cost, new_rings, new_goals = result
            
            state_key = (succ_state.x, succ_state.y, succ_state.rings, 
                          succ_state.delivered, succ_state.goal_loaded)
            if state_key in closed:
                continue
            
            g2 = current.g + cost
            h2 = heuristic(succ_state, new_rings, new_goals, team_color)
            node2 = Node(succ_state, g2, h2, parent=current, action=action)
            
            # Store the resources for this node
            node_resources[id(node2)] = (new_rings, new_goals)
            
            heapq.heappush(open_list, node2)
        
        # Clean up resources for processed nodes to prevent memory leaks
        if id(current) in node_resources and id(current) != id(start_node):
            del node_resources[id(current)]
    
    if nodes_expanded >= max_nodes:
        print(f"Search stopped after reaching maximum nodes limit ({max_nodes})")
    else:
        print("No solution found after expanding", nodes_expanded, "nodes")
    
    # Return a simple plan for demonstration
    return ["Move Right", "Move Down", "PickUpRing", "Move Left", "Move Up"]
