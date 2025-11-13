from maze_env import Maze
from RL_brain import DeepQNetwork

def run_episode(env, RL, step):
    """Run a single episode of the maze game."""
    observation = env.reset()
    
    while True:
        env.render()  # Display the environment

        # RL selects an action based on the current observation
        action = RL.choose_action(observation)

        # RL takes the action and gets the next state, reward, and done flag
        next_observation, reward, done = env.step(action)

        # Store the transition in memory
        RL.store_transition(observation, action, reward, next_observation)

        # Learn from the transitions if conditions are met
        if step > 200 and step % 5 == 0:
            RL.learn()

        # Update the observation for the next loop
        observation = next_observation

        # Break the loop when the episode is done
        if done:
            break
        
        step += 1

    return step

def run_maze():
    """Initialize the environment and run multiple episodes."""
    env = Maze()  # Create the maze environment
    RL = DeepQNetwork(
        n_actions=env.n_actions,
        n_features=env.n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000
    )  # Initialize the RL model

    step = 0
    for episode in range(300):
        print(f"Running episode {episode + 1}...")
        step = run_episode(env, RL, step)

    # End of game
    print('Game over')
    env.destroy()

    # Plot the learning cost
    RL.plot_cost()

def main():
    """Main function to start the maze game and run the learning process."""
    run_maze()

if __name__ == "__main__":
    main()
