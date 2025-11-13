import numpy as np
import tkinter as tk
import time

# Constants for the maze dimensions
UNIT = 40   # Pixel size for each grid unit
MAZE_H = 4  # Maze height (number of rows)
MAZE_W = 4  # Maze width (number of columns)

class Maze(tk.Tk, object):
    def __init__(self):
        super().__init__()

        # Define possible actions
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)

        # Set up the window title and size
        self.title('Maze Game')
        self.geometry(f'{MAZE_W * UNIT}x{MAZE_H * UNIT}')

        # Initialize the maze
        self._build_maze()

    def _build_maze(self):
        """Build the maze grid and place the agent and special items."""
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)
        self._create_grid_lines()
        self._place_items()
        self.canvas.pack()

    def _create_grid_lines(self):
        """Create the grid lines for the maze."""
        # Vertical grid lines
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT)
        # Horizontal grid lines
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r)

    def _place_items(self):
        """Place the special items (hells, paradise, and agent) in the maze."""
        origin = np.array([20, 20])  # Starting position of the agent

        # Place hells (black squares)
        self.hell1 = self._create_rectangle(origin + np.array([UNIT * 2, UNIT]), 'black')
        self.hell2 = self._create_rectangle(origin + np.array([UNIT, UNIT * 2]), 'black')

        # Place paradise (yellow circle)
        self.oval = self._create_oval(origin + UNIT * 2, 'yellow')

        # Place the agent (red rectangle)
        self.rect = self._create_rectangle(origin, 'red')

    def _create_rectangle(self, center, color):
        """Create a rectangle with the given center and color."""
        return self.canvas.create_rectangle(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill=color
        )

    def _create_oval(self, center, color):
        """Create an oval with the given center and color."""
        return self.canvas.create_oval(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill=color
        )

    def reset(self):
        """Reset the agent's position to the origin and return the initial state."""
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self._create_rectangle(origin, 'red')
        return self.canvas.coords(self.rect)

    def step(self, action):
        """Move the agent according to the action and return the next state, reward, and whether the game is over."""
        current_coords = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        # Update the agent's position based on the action
        if action == 0:   # Up
            if current_coords[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # Down
            if current_coords[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # Right
            if current_coords[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # Left
            if current_coords[0] > UNIT:
                base_action[0] -= UNIT

        # Move the agent on the canvas
        self.canvas.move(self.rect, base_action[0], base_action[1])
        next_coords = self.canvas.coords(self.rect)

        # Determine the reward and whether the game is over
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return next_coords, reward, done

    def render(self):
        """Update the GUI to reflect the current state of the maze."""
        self.update()

def main():
    """Main function to run the maze simulation."""
    maze = Maze()
    maze.mainloop()

if __name__ == "__main__":
    main()
