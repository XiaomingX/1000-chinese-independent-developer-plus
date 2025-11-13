import numpy as np
import time
import sys
import tkinter as tk

# Constants for the maze setup
UNIT = 40  # Size of each grid cell (in pixels)
MAZE_H = 4  # Height of the maze (number of rows)
MAZE_W = 4  # Width of the maze (number of columns)

class Maze(tk.Tk):
    def __init__(self):
        """Initialize the maze environment and set up the maze GUI."""
        super().__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # Possible actions: up, down, left, right
        self.n_actions = len(self.action_space)  # Number of possible actions
        self.title('Maze Game')
        self.geometry(f'{MAZE_W * UNIT}x{MAZE_H * UNIT}')  # Set the window size
        self._build_maze()  # Build the maze using tkinter Canvas

    def _build_maze(self):
        """Set up the visual maze grid and objects (explorer, hells, and paradise)."""
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # Create grid lines
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT)
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r)

        # Define maze objects (hells, paradise, and the explorer)
        origin = np.array([20, 20])  # Starting position of the explorer

        # Hells (bad places with -1 reward)
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self._create_rectangle(hell1_center, 'black')
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self._create_rectangle(hell2_center, 'black')

        # Paradise (good place with +1 reward)
        paradise_center = origin + UNIT * 2
        self.paradise = self._create_oval(paradise_center, 'yellow')

        # Explorer (red rectangle)
        self.explorer = self._create_rectangle(origin, 'red')

        # Pack everything into the canvas
        self.canvas.pack()

    def _create_rectangle(self, center, color):
        """Create a rectangle on the canvas with a given center and color."""
        return self.canvas.create_rectangle(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill=color
        )

    def _create_oval(self, center, color):
        """Create an oval on the canvas with a given center and color."""
        return self.canvas.create_oval(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill=color
        )

    def reset(self):
        """Reset the environment, bringing the explorer back to the origin."""
        self.update()  # Update the GUI
        time.sleep(0.5)  # Wait for the animation
        self.canvas.delete(self.explorer)  # Remove the old explorer
        origin = np.array([20, 20])  # Reset the position of the explorer
        self.explorer = self._create_rectangle(origin, 'red')
        return self.canvas.coords(self.explorer)

    def step(self, action):
        """Move the explorer based on the given action and return the new state and reward."""
        current_position = self.canvas.coords(self.explorer)
        move = np.array([0, 0])

        # Determine the movement direction based on the action
        if action == 0:  # Move up
            if current_position[1] > UNIT:
                move[1] -= UNIT
        elif action == 1:  # Move down
            if current_position[1] < (MAZE_H - 1) * UNIT:
                move[1] += UNIT
        elif action == 2:  # Move right
            if current_position[0] < (MAZE_W - 1) * UNIT:
                move[0] += UNIT
        elif action == 3:  # Move left
            if current_position[0] > UNIT:
                move[0] -= UNIT

        # Move the explorer on the canvas
        self.canvas.move(self.explorer, move[0], move[1])

        # Get the new position of the explorer
        new_position = self.canvas.coords(self.explorer)

        # Check for rewards based on the new position
        if new_position == self.canvas.coords(self.paradise):
            return new_position, 1, True  # Reward = +1 for reaching paradise
        elif new_position in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            return new_position, -1, True  # Reward = -1 for falling into hell
        else:
            return new_position, 0, False  # Reward = 0 for regular move

    def render(self):
        """Render the current state of the environment."""
        time.sleep(0.1)  # Slow down for visibility
        self.update()


def run_game():
    """Run the maze game with a simple agent that moves right."""
    for _ in range(10):  # Run for 10 episodes
        state = env.reset()
        while True:
            env.render()  # Update the canvas
            action = 1  # Example action: move down
            state, reward, done = env.step(action)  # Take a step
            if done:
                break  # Stop if the episode is done


def main():
    """Main function to set up and run the maze environment."""
    global env
    env = Maze()  # Initialize the maze environment
    env.after(100, run_game)  # Start the game after 100 ms
    env.mainloop()  # Start the tkinter event loop


if __name__ == '__main__':
    main()
