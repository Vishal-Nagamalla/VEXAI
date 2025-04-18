import tkinter as tk
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

GRID_SIZE = 24
NUM_RINGS_PER_COLOR = 24

class FieldObject:
    def __init__(self, name, symbol, color, movable):
        self.name = name
        self.symbol = symbol
        self.color = color
        self.movable = movable

RING_RED    = FieldObject("ring_red",    "r", "red",        True)
RING_BLUE   = FieldObject("ring_blue",   "b", "#4da6ff",   True)
MOBILE_GOAL = FieldObject("goal",        "G", "lightgreen",True)
ROBOT_RED   = FieldObject("robot_red",   "R", "#ff4d4d",   False)
ROBOT_BLUE  = FieldObject("robot_blue",  "B", "#4da6ff",   False)

RED_POSITIVE   = [(22,22),(23,22),(22,23),(23,23)]
BLUE_POSITIVE  = [(0,22),(1,22),(0,23),(1,23)]
RED_NEGATIVE   = [(22,0),(23,0),(22,1),(23,1)]
BLUE_NEGATIVE  = [(0,0),(1,0),(0,1),(1,1)]

# Now only one high wall stake square at x=12 on top and bottom
FIXED_STAKES = [(12,0), (12,23)]

class FieldGrid:
    def __init__(self):
        self.grid = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        # pre‑occupy the fixed stakes and corners
        self.occupied = set(RED_POSITIVE + BLUE_POSITIVE + RED_NEGATIVE + BLUE_NEGATIVE + FIXED_STAKES)
        for x,y in FIXED_STAKES:
            self.grid[y][x].append("stake")

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
                stack = self.grid[y][x]
                if len(stack) < 2 or (obj.symbol not in stack and len(stack)==1):
                    stack.append(obj.symbol)
                    placed += 1
            else:
                if self.is_valid_position(x, y, size):
                    for dx in range(size):
                        for dy in range(size):
                            self.grid[y+dy][x+dx].append(obj.symbol)
                            self.occupied.add((x+dx, y+dy))
                    placed += 1

    def get_grid(self):
        return self.grid

def draw_field(grid, team_color="red"):
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(np.arange(0, GRID_SIZE, 1))
    ax.set_yticks(np.arange(0, GRID_SIZE, 1))
    ax.set_facecolor("#d3d3d3")  # lighter grey
    ax.grid(True, which='both', color='black', linewidth=0.5)

    # draw corners
    for zone,label in [
        (RED_POSITIVE, '+'), (BLUE_POSITIVE, '+'),
        (RED_NEGATIVE, '-'), (BLUE_NEGATIVE, '-')
    ]:
        for x,y in zone:
            ax.add_patch(patches.Rectangle((x,y), 1,1, color='black'))
        cx,cy = np.mean(zone, axis=0)
        ax.text(cx+0.25, cy+0.25, label,
                color='white', fontsize=14, fontweight='bold')

    # draw objects
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            cell = grid[y][x]
            if "stake" in cell:
                # yellow 1x1 stake at (12,0) and (12,23)
                ax.add_patch(patches.Rectangle((x,y),1,1, color='yellow'))

            # ring stacks
            if cell.count("r")==2 or cell.count("b")==2:
                ax.text(x+0.3, y+0.3, "S",
                        fontsize=12, fontweight='bold', color='white')

            for sym in cell:
                if sym=="G":
                    ax.add_patch(patches.Rectangle((x,y),1,1, color="lightgreen"))
                elif sym=="R":
                    ax.add_patch(patches.Rectangle((x,y),1,1, color="#ff4d4d"))
                elif sym=="B":
                    ax.add_patch(patches.Rectangle((x,y),1,1, color="#4da6ff"))
                elif sym=="r":
                    ax.add_patch(patches.Circle((x+0.5,y+0.5),0.3, color="red"))
                elif sym=="b":
                    ax.add_patch(patches.Circle((x+0.5,y+0.5),0.3, color="#4da6ff"))

    plt.gca().invert_yaxis()
    plt.title("VEX U High Stakes Field View")
    plt.show()

def on_team_select(team):
    print(f"You selected the {team.upper()} alliance.")
    window.destroy()
    draw_field(field.get_grid(), team_color=team)

def launch_ui():
    global window
    window = tk.Tk()
    window.title("VEX U High Stakes")
    # make window reasonably sized and centered
    w, h = 360, 200
    ws, hs = window.winfo_screenwidth(), window.winfo_screenheight()
    x, y = (ws - w)//2, (hs - h)//2
    window.geometry(f"{w}x{h}+{x}+{y}")
    window.configure(bg="#f0f0f0")

    # Prompt label
    label = tk.Label(
        window,
        text="Select Your Alliance",
        font=("Arial", 16, "bold"),
        bg="#f0f0f0",
        fg="#333"
    )
    label.pack(pady=(20, 10))

    # Red button
    red_btn = tk.Button(
        window,
        text="Red Alliance",
        font=("Arial", 14, "bold"),
        bg="#ff4d4d", fg="white",
        activebackground="#cc0000", activeforeground="white",
        highlightbackground="#ff4d4d", highlightcolor="#ff4d4d",
        bd=0, relief="raised",
        padx=20, pady=8,
        command=lambda: on_team_select("red")
    )
    red_btn.pack(pady=5)

    # Blue button
    blue_btn = tk.Button(
        window,
        text="Blue Alliance",
        font=("Arial", 14, "bold"),
        bg="#4da6ff", fg="white",
        activebackground="#007acc", activeforeground="white",
        highlightbackground="#4da6ff", highlightcolor="#4da6ff",
        bd=0, relief="raised",
        padx=20, pady=8,
        command=lambda: on_team_select("blue")
    )
    blue_btn.pack(pady=5)

    window.mainloop()

if __name__ == "__main__":
    field = FieldGrid()
    field.place_object(RING_RED,    NUM_RINGS_PER_COLOR, allow_stack=True)
    field.place_object(RING_BLUE,   NUM_RINGS_PER_COLOR, allow_stack=True)
    field.place_object(MOBILE_GOAL, 5)
    field.place_object(ROBOT_RED,   2, size=2)
    field.place_object(ROBOT_BLUE,  2, size=2)
    launch_ui()