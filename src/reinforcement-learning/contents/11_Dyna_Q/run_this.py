"""
Simplest model-based Reinforcement Learning: Dyna-Q.

- Red rectangle: Explorer.
- Black rectangles: Hells (reward = -1).
- Yellow bin circle: Paradise (reward = +1).
- Other states: Ground (reward = 0).

This script manages the main control flow of the Dyna-Q algorithm, which combines
model-free Q-learning with a model-based approach for updating the agent's knowledge.
The RL logic resides in RL_brain.py.

Visit the tutorial here: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable, EnvModel


def run_episode(env, RL, env_model, max_steps=40, learn_times=10):
    """
    Runs a single episode of the Dyna-Q algorithm.

    Args:
        env: The environment where the agent operates.
        RL: The Q-learning agent.
        env_model: The environment model for Dyna-Q.
        max_steps: Maximum number of steps per episode.
        learn_times: Number of times to learn from the model per step.

    Returns:
        None
    """
    state = env.reset()  # Reset the environment and get the initial state
    
    for step in range(max_steps):
        env.render()  # Render the environment to visualize the agent
        action = RL.choose_action(str(state))  # Choose an action based on the current state
        next_state, reward, done = env.step(action)  # Take the action and observe the result
        
        # Learn from the actual experience (state, action, reward, next_state)
        RL.learn(str(state), action, reward, str(next_state))

        # Store the transition in the environment model (like a memory replay buffer)
        env_model.store_transition(str(state), action, reward, next_state)
        
        # Perform additional learning based on the model's experiences
        for _ in range(learn_times):
            model_state, model_action = env_model.sample_s_a()  # Sample a (state, action) pair from the model
            model_reward, model_next_state = env_model.get_r_s_(model_state, model_action)  # Get the model's predicted reward and next state
            RL.learn(model_state, model_action, model_reward, str(model_next_state))  # Learn from the model's experience

        state = next_state  # Move to the next state
        if done:
            break  # Exit the loop if the episode ends


def main():
    """
    Main function to initialize the environment, Q-learning agent, and model,
    then run the Dyna-Q episodes.
    
    Returns:
        None
    """
    # Initialize the environment, Q-learning agent, and model
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env_model = EnvModel(actions=list(range(env.n_actions)))

    # Run the update function which controls the learning process
    env.after(0, lambda: run_episode(env, RL, env_model))
    env.mainloop()  # Start the environment's main loop to visualize the agent's learning

if __name__ == "__main__":
    main()
