# Save this file as field.py

# Game constants and field definitions
GRID_SIZE = 24
ROBOT_SIZE = 2  # robot footprint is 2×2
MAX_SIMULATION_STEPS = 500  # maximum steps for the simulation

# Corner zones - color neutral but with + and -
POSITIVE_CORNERS = [(22,22),(23,22),(22,23),(23,23)]  # Red Alliance Positive
NEGATIVE_CORNERS = [(22,0),(23,0),(22,1),(23,1)]      # Red Alliance Negative
POSITIVE_CORNERS_BLUE = [(0,22),(1,22),(0,23),(1,23)]  # Blue Alliance Positive  
NEGATIVE_CORNERS_BLUE = [(0,0),(1,0),(0,1),(1,1)]      # Blue Alliance Negative

# Colored stakes - one per alliance
RED_STAKE = (23, 12)
BLUE_STAKE = (0, 12)

# Fixed common stakes
FIXED_STAKES = [(12,0), (12,23)]

# Movement directions (Note: these are in grid coordinates, not screen coordinates)
DIRS = {
    # Corrected directions: Up should be (0, -1) and Down should be (0, 1)
    # in grid coordinates (y increases downward)
    "Up":    (0, -1),
    "Down":  (0, 1),
    "Left":  (-1, 0),
    "Right": (1, 0),
}

# Maximum rings per mobile goal
MAX_RINGS_PER_GOAL = 6
