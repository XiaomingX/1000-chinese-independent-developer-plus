"""
Sarsa is an online updating method for reinforcement learning.

Unlike Q-learning, which updates the Q-values after completing an episode, Sarsa updates them during the current trajectory (i.e., while the agent is taking actions).

In comparison, Sarsa is more cautious when rewards or punishments are close, as it evaluates the entire trajectory, while Q-learning focuses on maximizing the future reward from the current state.
"""

from maze_env import Maze
from RL_brain import SarsaTable


def run_episode():
    """
    Run a single episode of the game using the Sarsa algorithm.
    """
    # Initialize the environment and get the starting observation
    observation = env.reset()

    # Choose an initial action based on the current observation
    action = RL.choose_action(str(observation))

    while True:
        # Render the current state of the environment
        env.render()

        # Take the action, observe the next state, and get the reward
        next_observation, reward, done = env.step(action)

        # Choose the next action based on the new observation
        next_action = RL.choose_action(str(next_observation))

        # Update the Sarsa table with the current state, action, reward, next state, and next action
        RL.learn(str(observation), action, reward, str(next_observation), next_action)

        # Update the observation and action for the next iteration
        observation = next_observation
        action = next_action

        # Break the loop if the episode is finished (done is True)
        if done:
            break

    print(f"Episode {episode+1} finished.")


def main():
    """
    Main function to initialize the environment, RL agent, and run multiple episodes.
    """
    # Initialize the Maze environment
    global env
    env = Maze()

    # Initialize the Sarsa reinforcement learning agent with the available actions
    RL = SarsaTable(actions=list(range(env.n_actions)))

    # Run 100 episodes of the game
    for episode in range(100):
        run_episode()

    # End of all episodes
    print("All episodes finished. Game over.")
    env.destroy()


if __name__ == "__main__":
    main()
