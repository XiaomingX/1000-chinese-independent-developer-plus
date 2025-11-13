import numpy as np
import time
import sys
import tkinter as tk

# Constants for the maze dimensions and grid unit size
UNIT = 40   # Size of each grid unit in pixels
MAZE_HEIGHT = 4  # Number of rows in the maze
MAZE_WIDTH = 4   # Number of columns in the maze

class Maze(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Define the possible actions: up, down, left, right
        self.action_space = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.action_space)
        self.num_features = 2
        
        # Set up the window
        self.title('Maze')
        self.geometry(f'{MAZE_WIDTH * UNIT}x{MAZE_HEIGHT * UNIT}')
        self._build_maze()

    def _build_maze(self):
        """Build the maze grid and the key elements (explorer, hells, and paradise)."""
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_HEIGHT * UNIT, width=MAZE_WIDTH * UNIT)
        
        # Draw grid lines
        self._draw_grid()
        
        # Define the positions for the maze elements
        self.origin = np.array([20, 20])
        
        # Add hell (negative reward areas)
        self.hell1_center = self.origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            self.hell1_center[0] - 15, self.hell1_center[1] - 15,
            self.hell1_center[0] + 15, self.hell1_center[1] + 15,
            fill='black'
        )
        
        # Add paradise (positive reward area)
        self.paradise_center = self.origin + UNIT * 2
        self.paradise = self.canvas.create_oval(
            self.paradise_center[0] - 15, self.paradise_center[1] - 15,
            self.paradise_center[0] + 15, self.paradise_center[1] + 15,
            fill='yellow'
        )
        
        # Add explorer (red rectangle)
        self.explorer = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red'
        )
        
        self.canvas.pack()

    def _draw_grid(self):
        """Draw the grid lines for the maze."""
        for c in range(0, MAZE_WIDTH * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_HEIGHT * UNIT)
        for r in range(0, MAZE_HEIGHT * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_WIDTH * UNIT, r)

    def reset(self):
        """Reset the maze, putting the explorer back at the origin."""
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.explorer)  # Remove the old explorer
        self.explorer = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red'
        )
        return self._get_state()

    def step(self, action):
        """Take a step based on the action and return the new state, reward, and whether the episode is done."""
        current_coords = self.canvas.coords(self.explorer)
        action_offset = np.array([0, 0])

        # Apply the action to move the explorer
        if action == 0:  # up
            if current_coords[1] > UNIT:
                action_offset[1] -= UNIT
        elif action == 1:  # down
            if current_coords[1] < (MAZE_HEIGHT - 1) * UNIT:
                action_offset[1] += UNIT
        elif action == 2:  # right
            if current_coords[0] < (MAZE_WIDTH - 1) * UNIT:
                action_offset[0] += UNIT
        elif action == 3:  # left
            if current_coords[0] > UNIT:
                action_offset[0] -= UNIT
        
        # Move the explorer based on the action
        self.canvas.move(self.explorer, action_offset[0], action_offset[1])
        new_coords = self.canvas.coords(self.explorer)
        
        # Compute the reward and determine if the episode is done
        reward, done = self._get_reward_and_done(new_coords)
        next_state = self._get_state(new_coords)
        
        return next_state, reward, done

    def _get_state(self, coords=None):
        """Get the state (normalized position relative to the paradise)."""
        if coords is None:
            coords = self.canvas.coords(self.explorer)
        return (np.array(coords[:2]) - np.array(self.canvas.coords(self.paradise)[:2])) / (MAZE_HEIGHT * UNIT)

    def _get_reward_and_done(self, coords):
        """Evaluate the reward and check if the episode is done."""
        if coords == self.canvas.coords(self.paradise):
            return 1, True  # Paradise reached
        elif coords in [self.canvas.coords(self.hell1)]:
            return -1, True  # Hell reached
        return 0, False  # Continue the game

    def render(self):
        """Update the canvas to show the current state."""
        self.update()


def main():
    """Main function to run the maze and simulate the agent's movements."""
    # Create a Maze instance
    maze = Maze()
    
    # Reset the maze to the starting state
    state = maze.reset()
    done = False
    
    # Simulate a simple agent movement sequence
    actions = [0, 2, 1, 3]  # up, right, down, left
    for action in actions:
        if done:
            break
        # Take a step based on the action
        state, reward, done = maze.step(action)
        print(f"Action: {maze.action_space[action]}, Reward: {reward}, Done: {done}")
        maze.render()
        time.sleep(0.5)  # Pause to visualize the movement


if __name__ == "__main__":
    main()
