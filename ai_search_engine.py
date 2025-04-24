import heapq
from field import RED_POSITIVE, BLUE_POSITIVE

# Directions for movement
DIRS = {
    'Up':    (0, -1),
    'Down':  (0,  1),
    'Left':  (-1, 0),
    'Right': (1,  0),
}

class State:
    """
    Represents: 
      - robot (x,y)
      - rings_collected (how many rings held)
      - goals_delivered (list of goals already placed in corners)
      - grid snapshot if you need to mutate (optional)
    """
    __slots__ = ('x','y','rings','delivered','goal_loaded')
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
        self.state   = state
        self.g       = g
        self.h       = h
        self.f       = g + h
        self.parent  = parent
        self.action  = action

    def __lt__(self, other):
        return self.f < other.f

def heuristic(state, field, team_color):
    """
    Estimate: remaining rings to collect + distance to nearest ring +
              distance from ring to mobile goal + to corner.
    For simplicity: use 0 (or Manhattan to closest ring).
    """
    # find all ring positions of our color
    ring_positions = []
    sym = 'r' if team_color=='red' else 'b'
    for y in range(len(field)):
        for x in range(len(field)):
            if field[y][x].count(sym)>0:
                ring_positions.append((x,y))
    if not ring_positions:
        return 0
    # distance to closest ring
    d1 = min(abs(state.x-x)+abs(state.y-y) for x,y in ring_positions)
    return d1

def is_goal_state(state):
    # goal: delivered == 5 (we’ve placed 5 goals)
    return state.delivered >= 5

def get_successors(node, field, team_color):
    """ Generate (next_state, action_name, cost) """
    succ = []
    s = node.state
    N = len(field)

    # 1) Move in 4 directions
    for name,(dx,dy) in DIRS.items():
        nx, ny = s.x+dx, s.y+dy
        if 0 <= nx < N and 0 <= ny < N:
            new = State(nx, ny, s.rings, s.delivered, s.goal_loaded)
            succ.append((new, f"Move {name}", 1))

    # 2) PickUpRing if ring of our color present and we carry <2
    sym = 'r' if team_color=='red' else 'b'
    if s.rings < 2 and field[s.y][s.x].count(sym) > 0:
        new = State(s.x, s.y, s.rings+1, s.delivered, s.goal_loaded)
        succ.append((new, "PickUpRing", 1))

    # 3) DropRing into mobile goal if adjacent
    #    (simplify: assume drop always succeeds)
    #    We won’t track goal-by-goal inventory here
    if s.rings > 0:
        for dx,dy in DIRS.values():
            ax, ay = s.x+dx, s.y+dy
            if 0 <= ax < N and 0 <= ay < N and field[ay][ax].count('G')>0:
                new = State(s.x, s.y, 0, s.delivered, s.goal_loaded)
                new.goal_loaded = True
                succ.append((new, "LoadGoal", 1))

    # 4) DeliverGoal if in a positive corner
    corners = RED_POSITIVE+BLUE_POSITIVE
    if (s.x,s.y) in corners and s.goal_loaded:  # if we have loaded
        new = State(s.x, s.y, 0, s.delivered+1, s.goal_loaded)
        new.goal_loaded = False
        succ.append((new, "DeliverGoal", 1))

    return succ

def reconstruct_path(node):
    actions = []
    while node.parent:
        actions.append(node.action)
        node = node.parent
    return list(reversed(actions))

def a_star_search(start_state, field, team_color):
    open_list = []
    start = Node(start_state, g=0, h=heuristic(start_state, field, team_color))
    heapq.heappush(open_list, start)
    closed = set()

    while open_list:
        current = heapq.heappop(open_list)
        if is_goal_state(current.state):
            return reconstruct_path(current)
        closed.add(current.state)

        for succ_state, action, cost in get_successors(current, field, team_color):
            if succ_state in closed:
                continue
            g2 = current.g + cost
            h2 = heuristic(succ_state, field, team_color)
            node2 = Node(succ_state, g2, h2, parent=current, action=action)
            heapq.heappush(open_list, node2)

    return []  # no plan found
