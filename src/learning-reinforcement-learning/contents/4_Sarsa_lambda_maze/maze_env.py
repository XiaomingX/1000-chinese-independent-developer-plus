import numpy as np
import time
import sys

# Check Python version for compatibility with Tkinter
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40  # Pixel size for each grid cell
MAZE_H = 4  # Maze height (number of rows)
MAZE_W = 4  # Maze width (number of columns)

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        
        # Define possible actions (up, down, left, right)
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        
        # Set up the window
        self.title('Maze Environment')
        self.geometry(f'{MAZE_W * UNIT}x{MAZE_H * UNIT}')
        
        # Build the maze grid and components
        self._build_maze()

    def _build_maze(self):
        """Initialize the maze grid and its elements."""
        # Create the canvas to draw the maze
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # Draw grid lines for the maze
        self._draw_grid()

        # Define the origin for placing elements
        origin = np.array([20, 20])

        # Create 'hell' locations (reward = -1)
        self.hell1 = self._create_hell(origin + np.array([UNIT * 2, UNIT]))
        self.hell2 = self._create_hell(origin + np.array([UNIT, UNIT * 2]))

        # Create 'paradise' location (reward = +1)
        self.paradise = self._create_paradise(origin + UNIT * 2)

        # Create the red rectangle representing the explorer
        self.explorer = self._create_explorer(origin)

        # Pack the canvas for display
        self.canvas.pack()

    def _draw_grid(self):
        """Draw the grid lines on the canvas."""
        # Vertical grid lines
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        # Horizontal grid lines
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

    def _create_hell(self, center):
        """Create a 'hell' location (black rectangle)."""
        return self.canvas.create_rectangle(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill='black'
        )

    def _create_paradise(self, center):
        """Create the 'paradise' location (yellow oval)."""
        return self.canvas.create_oval(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill='yellow'
        )

    def _create_explorer(self, origin):
        """Create the red rectangle (explorer)."""
        return self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )

    def reset(self):
        """Reset the maze to the initial state."""
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.explorer)
        origin = np.array([20, 20])
        self.explorer = self._create_explorer(origin)
        return self.canvas.coords(self.explorer)

    def step(self, action):
        """Move the explorer based on the action and return the result."""
        # Get current position of the explorer
        current_position = self.canvas.coords(self.explorer)

        # Define action mapping
        action_delta = np.array([0, 0])
        if action == 0:  # Move up
            if current_position[1] > UNIT:
                action_delta[1] -= UNIT
        elif action == 1:  # Move down
            if current_position[1] < (MAZE_H - 1) * UNIT:
                action_delta[1] += UNIT
        elif action == 2:  # Move right
            if current_position[0] < (MAZE_W - 1) * UNIT:
                action_delta[0] += UNIT
        elif action == 3:  # Move left
            if current_position[0] > UNIT:
                action_delta[0] -= UNIT

        # Move the explorer
        self.canvas.move(self.explorer, action_delta[0], action_delta[1])

        # Get new position after the move
        new_position = self.canvas.coords(self.explorer)

        # Check reward and termination conditions
        if new_position == self.canvas.coords(self.paradise):
            reward = 1
            done = True
            new_position = 'terminal'
        elif new_position in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            new_position = 'terminal'
        else:
            reward = 0
            done = False

        return new_position, reward, done

    def render(self):
        """Render the maze (update the GUI)."""
        time.sleep(0.05)
        self.update()

def main():
    """Main function to run the maze environment."""
    # Create the maze environment
    maze = Maze()

    # Reset the maze to initial state
    state = maze.reset()
    print("Initial State:", state)

    # Example of moving the explorer in the maze
    actions = [0, 2, 1, 3]  # List of actions (up, right, down, left)
    for action in actions:
        print(f"Action: {maze.action_space[action]}")
        next_state, reward, done = maze.step(action)
        print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")
        maze.render()

        if done:
            print("Episode finished!")
            break

if __name__ == '__main__':
    main()
