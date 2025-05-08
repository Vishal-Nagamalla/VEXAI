from field import (POSITIVE_CORNERS, NEGATIVE_CORNERS, POSITIVE_CORNERS_BLUE, 
                  NEGATIVE_CORNERS_BLUE, ALLIANCE_WALL_STAKES, NEUTRAL_WALL_STAKES,
                  RING_STANDARD_POINTS, RING_NEGATIVE_POINTS)

def calculate_score(field_grid, goal_positions, goal_rings):
    """
    Calculate the score for both red and blue teams based on the current field state.
    
    Args:
        field_grid: The 2D grid representation of the field
        goal_positions: List of (x,y) coordinates of all mobile goals
        goal_rings: Dictionary mapping (x,y) positions to ring counts {'red': n, 'blue': m}
        
    Returns:
        A dictionary with scores for each team: {'red': score_red, 'blue': score_blue}
    """
    scores = {'red': 0, 'blue': 0}
    
    # Score rings on alliance wall stakes
    red_stake = ALLIANCE_WALL_STAKES['red']
    blue_stake = ALLIANCE_WALL_STAKES['blue']
    
    # Check if there are ring counters for the alliance stakes
    if red_stake in goal_rings:
        # Red rings on red alliance stake score points
        scores['red'] += goal_rings[red_stake]['red'] * RING_STANDARD_POINTS
        # Blue rings on red alliance stake score 0
    
    if blue_stake in goal_rings:
        # Blue rings on blue alliance stake score points
        scores['blue'] += goal_rings[blue_stake]['blue'] * RING_STANDARD_POINTS
        # Red rings on blue alliance stake score 0
    
    # Score rings on neutral wall stakes
    for stake_pos in NEUTRAL_WALL_STAKES:
        if stake_pos in goal_rings:
            scores['red'] += goal_rings[stake_pos]['red'] * RING_STANDARD_POINTS
            scores['blue'] += goal_rings[stake_pos]['blue'] * RING_STANDARD_POINTS
    
    # Score rings on mobile goals
    for goal_pos in goal_positions:
        if goal_pos in goal_rings:
            # Check if goal is in a scoring zone
            if goal_pos in POSITIVE_CORNERS or goal_pos in POSITIVE_CORNERS_BLUE:
                # Positive zone - score standard points
                scores['red'] += goal_rings[goal_pos]['red'] * RING_STANDARD_POINTS
                scores['blue'] += goal_rings[goal_pos]['blue'] * RING_STANDARD_POINTS
            elif goal_pos in NEGATIVE_CORNERS or goal_pos in NEGATIVE_CORNERS_BLUE:
                # Negative zone - score negative points
                scores['red'] += goal_rings[goal_pos]['red'] * RING_NEGATIVE_POINTS
                scores['blue'] += goal_rings[goal_pos]['blue'] * RING_NEGATIVE_POINTS
            # Goals not in scoring zones score 0 points (no action needed)
    
    return scores

def is_in_positive_zone(pos):
    """Check if a position is in any positive scoring zone"""
    return pos in POSITIVE_CORNERS or pos in POSITIVE_CORNERS_BLUE

def is_in_negative_zone(pos):
    """Check if a position is in any negative scoring zone"""
    return pos in NEGATIVE_CORNERS or pos in NEGATIVE_CORNERS_BLUE

def is_in_scoring_zone(pos):
    """Check if a position is in any scoring zone"""
    return is_in_positive_zone(pos) or is_in_negative_zone(pos)

def get_best_goal_placement(robot_pos, team, occupied_positions):
    """
    Determine the best position to place a goal in a scoring zone.
    
    Uses priority order for corners in each plus zone:
    - Top Left Plus Zone: top left → bottom left → top right → bottom right
    - Top Right Plus Zone: top right → bottom right → top left → bottom left
    
    Args:
        robot_pos: (x,y) position of robot
        team: 'red' or 'blue' alliance (only affects which zone to prioritize)
        occupied_positions: list of positions that already have goals
        
    Returns:
        (x,y) coordinates for best goal placement, or None if no valid placement
    """
    # Define the plus zones
    top_left_zone = [(0, 22), (1, 22), (0, 23), (1, 23)]  # POSITIVE_CORNERS_BLUE
    top_right_zone = [(22, 22), (23, 22), (22, 23), (23, 23)]  # POSITIVE_CORNERS
    
    # Define corner priorities for each zone
    top_left_priority = [(0, 22), (0, 23), (1, 22), (1, 23)]  # top-left, bottom-left, top-right, bottom-right
    top_right_priority = [(23, 22), (23, 23), (22, 22), (22, 23)]  # top-right, bottom-right, top-left, bottom-left
    
    # Check if robot is in or near a plus zone
    robot_x, robot_y = robot_pos
    robot_cells = []
    for dx in range(2):  # 2x2 robot
        for dy in range(2):
            robot_cells.append((robot_x + dx, robot_y + dy))
    
    # Check which zone the robot is in or closer to
    in_top_left = any(cell in top_left_zone for cell in robot_cells)
    in_top_right = any(cell in top_right_zone for cell in robot_cells)
    
    # If robot is in a specific zone, use that zone's priority
    if in_top_left:
        zone_priority = top_left_priority
    elif in_top_right:
        zone_priority = top_right_priority
    else:
        # If not in any zone, use the zone that's closer
        dist_to_top_left = min(abs(robot_x - x) + abs(robot_y - y) for x, y in top_left_zone)
        dist_to_top_right = min(abs(robot_x - x) + abs(robot_y - y) for x, y in top_right_zone)
        
        if dist_to_top_left <= dist_to_top_right:
            zone_priority = top_left_priority
        else:
            zone_priority = top_right_priority
    
    # Try corners in priority order
    for pos in zone_priority:
        if pos not in occupied_positions:
            return pos
    
    # If all corners in the priority list are occupied, try any other position in either zone
    all_plus_positions = top_left_zone + top_right_zone
    for pos in all_plus_positions:
        if pos not in occupied_positions:
            return pos
    
    # If all positions in both zones are occupied, check if we can place adjacent to the robot
    # but not in a negative zone
    perim = set()
    for rx, ry in robot_cells:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Adjacent cells
            nx, ny = rx + dx, ry + dy
            if 0 <= nx < 24 and 0 <= ny < 24:  # Grid boundaries
                perim.add((nx, ny))
    
    # Remove cells under the robot
    perim = perim - set(robot_cells)
    
    # Remove cells that are already occupied
    perim = perim - set(occupied_positions)
    
    # Remove cells in negative zones
    negative_zones = [(0, 0), (1, 0), (0, 1), (1, 1), (22, 0), (23, 0), (22, 1), (23, 1)]
    perim = perim - set(negative_zones)
    
    # If there's an available perimeter position, return the first one
    if perim:
        return next(iter(perim))
    
    # No valid placement found
    return None
