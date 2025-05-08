# Game constants and field definitions
GRID_SIZE = 24
ROBOT_SIZE = 2  # robot footprint is 2×2
MAX_SIMULATION_STEPS = 240  # maximum moves the robot can make (2 per second for 2 minutes)

# Corner zones - color neutral with + and -
POSITIVE_CORNERS = [(22,22),(23,22),(22,23),(23,23)]  # Positive scoring zone
NEGATIVE_CORNERS = [(22,0),(23,0),(22,1),(23,1)]      # Negative scoring zone
POSITIVE_CORNERS_BLUE = [(0,22),(1,22),(0,23),(1,23)]  # Positive scoring zone (blue side)
NEGATIVE_CORNERS_BLUE = [(0,0),(1,0),(0,1),(1,1)]      # Negative scoring zone (blue side)

# Wall stakes
ALLIANCE_WALL_STAKES = {
    'red': (23, 12),    # Red alliance wall stake
    'blue': (0, 12)     # Blue alliance wall stake
}

# Neutral wall stakes
NEUTRAL_WALL_STAKES = [(12,0), (12,23)]

# Movement directions (in grid coordinates, y increases downward)
DIRS = {
    "Up":    (0, -1),
    "Down":  (0, 1),
    "Left":  (-1, 0),
    "Right": (1, 0),
}

# Maximum rings per stake type
MAX_RINGS_ALLIANCE_STAKE = 2   # Alliance wall stakes can hold up to 2 rings
MAX_RINGS_NEUTRAL_STAKE = 6    # Neutral wall stakes can hold up to 6 rings
MAX_RINGS_MOBILE_STAKE = 6     # Mobile stakes can hold up to 6 rings

# Spawn positions
SPAWN_POSITION = {
    'red': (21, 12),    # Red robot starts in the middle of red side
    'blue': (2, 12)     # Blue robot starts in the middle of blue side
}

# Scoring rules
RING_STANDARD_POINTS = 1      # Points for each ring
RING_NEGATIVE_POINTS = -1     # Points for rings in negative zone
