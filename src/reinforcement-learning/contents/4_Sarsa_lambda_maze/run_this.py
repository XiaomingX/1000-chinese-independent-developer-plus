"""
Sarsa is an online updating method for Reinforcement Learning.

Unlike Q-learning, which updates offline, Sarsa updates during the current trajectory.
Sarsa is more cautious when punishment is near because it considers all behaviors,
while Q-learning is bolder, focusing only on the maximum behavior.
"""

from maze_env import Maze
from RL_brain import SarsaLambdaTable


def run_episode(env, RL):
    """
    Runs a single episode of the game where the agent interacts with the environment
    and updates its learning using the Sarsa algorithm.
    """
    # Initialize the environment and get the initial observation
    observation = env.reset()

    # Choose an action based on the initial observation
    action = RL.choose_action(str(observation))

    # Initialize eligibility trace (set to zero at the start)
    RL.eligibility_trace *= 0

    # Loop until the episode ends
    while True:
        env.render()  # Render the environment (visualize the agent's actions)

        # Take an action, get the next observation, and the reward
        observation_, reward, done = env.step(action)

        # Choose the next action based on the new observation
        action_ = RL.choose_action(str(observation_))

        # Update the learning model using the Sarsa algorithm
        RL.learn(str(observation), action, reward, str(observation_), action_)

        # Update the current observation and action
        observation, action = observation_, action_

        # End the episode if the 'done' flag is True
        if done:
            break

    print("Episode finished.")

    
def main():
    """
    Main function to set up the environment, initialize the RL agent, 
    and start the learning process.
    """
    # Create the maze environment
    env = Maze()

    # Initialize the RL agent (SarsaLambdaTable)
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    # Run the agent through multiple episodes
    for episode in range(100):
        print(f"Starting episode {episode + 1}")
        run_episode(env, RL)

    # End of game
    print('Game over')
    env.destroy()  # Clean up the environment


if __name__ == "__main__":
    main()  # Start the main function
