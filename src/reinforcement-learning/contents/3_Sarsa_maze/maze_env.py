import numpy as np
import time
import sys
import tkinter as tk

# Constants for the maze setup
UNIT = 40  # Size of each grid cell in pixels
MAZE_H = 4  # Maze height (number of rows)
MAZE_W = 4  # Maze width (number of columns)

class Maze(tk.Tk):
    """
    A class representing a simple maze environment for reinforcement learning.
    The maze consists of a red agent, black hell zones (penalty), and a yellow
    paradise (reward).
    """
    def __init__(self):
        super().__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # Possible actions (up, down, left, right)
        self.n_actions = len(self.action_space)
        self.title('Maze')  # Set window title
        self.geometry(f'{MAZE_W * UNIT}x{MAZE_H * UNIT}')  # Set window size
        self._build_maze()  # Call function to draw the maze

    def _build_maze(self):
        """
        Create the maze environment, including the agent, hell zones, and paradise.
        """
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # Draw grid lines
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT)
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r)

        # Starting position (top-left corner)
        origin = np.array([20, 20])

        # Hell zones (penalty areas)
        self.hell1 = self._create_hell_zone(origin + np.array([UNIT * 2, UNIT]))
        self.hell2 = self._create_hell_zone(origin + np.array([UNIT, UNIT * 2]))

        # Paradise zone (reward area)
        self.oval = self._create_paradise(origin + UNIT * 2)

        # Red agent rectangle (explorer)
        self.rect = self._create_agent(origin)

        # Pack the canvas
        self.canvas.pack()

    def _create_hell_zone(self, position):
        """
        Helper function to create a hell zone (penalty area).
        """
        return self.canvas.create_rectangle(
            position[0] - 15, position[1] - 15,
            position[0] + 15, position[1] + 15,
            fill='black'
        )

    def _create_paradise(self, position):
        """
        Helper function to create the paradise (reward area).
        """
        return self.canvas.create_oval(
            position[0] - 15, position[1] - 15,
            position[0] + 15, position[1] + 15,
            fill='yellow'
        )

    def _create_agent(self, position):
        """
        Helper function to create the agent (explorer).
        """
        return self.canvas.create_rectangle(
            position[0] - 15, position[1] - 15,
            position[0] + 15, position[1] + 15,
            fill='red'
        )

    def reset(self):
        """
        Reset the maze to the initial state by placing the agent back at the start.
        """
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)  # Remove the current agent
        origin = np.array([20, 20])  # Reset position
        self.rect = self._create_agent(origin)
        return self.canvas.coords(self.rect)  # Return agent's current position

    def step(self, action):
        """
        Execute one step in the environment based on the action.
        """
        current_position = self.canvas.coords(self.rect)  # Get current position
        movement = np.array([0, 0])  # Movement delta (x, y)

        # Define action effects (up, down, right, left)
        if action == 0:  # up
            if current_position[1] > UNIT:
                movement[1] -= UNIT
        elif action == 1:  # down
            if current_position[1] < (MAZE_H - 1) * UNIT:
                movement[1] += UNIT
        elif action == 2:  # right
            if current_position[0] < (MAZE_W - 1) * UNIT:
                movement[0] += UNIT
        elif action == 3:  # left
            if current_position[0] > UNIT:
                movement[0] -= UNIT

        # Move the agent
        self.canvas.move(self.rect, movement[0], movement[1])

        # Get new position after movement
        new_position = self.canvas.coords(self.rect)

        # Reward function
        if new_position == self.canvas.coords(self.oval):  # Paradise reached
            reward = 1
            done = True
            new_position = 'terminal'
        elif new_position in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:  # Hell reached
            reward = -1
            done = True
            new_position = 'terminal'
        else:
            reward = 0
            done = False

        return new_position, reward, done

    def render(self):
        """
        Render the maze (for visualization).
        """
        time.sleep(0.1)
        self.update()


def main():
    """
    Main function to run the maze environment.
    """
    # Create the maze environment
    maze = Maze()

    # Reset the environment and get initial state
    state = maze.reset()
    print(f"Initial state: {state}")

    # Simulate taking random actions in the environment
    for step in range(10):
        action = np.random.choice(maze.n_actions)  # Random action
        new_state, reward, done = maze.step(action)
        print(f"Step {step + 1}: Action {action}, New state {new_state}, Reward {reward}")

        # Render the environment
        maze.render()

        # If the episode is done (paradise or hell reached), reset the environment
        if done:
            print("Episode finished. Resetting...")
            state = maze.reset()

if __name__ == '__main__':
    main()
