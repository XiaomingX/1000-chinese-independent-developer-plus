import gym
from gym import wrappers
from DuelingDQNPrioritizedReplay import DuelingDQNPrioritizedReplay

# Define constants
ACTION_SPACE = 4  # The number of possible actions in LunarLander-v2
OBSERVATION_SPACE = 8  # The number of features in the state space
MEMORY_CAPACITY = 50000  # Memory size for storing experiences
TARGET_REP_ITER = 2000  # Frequency of target network updates
MAX_EPISODES = 900  # Maximum number of training episodes
EPSILON_GREEDY = 0.95  # Initial epsilon for exploration-exploitation tradeoff
EPSILON_INCREMENT = 0.00001  # Increment in epsilon after each episode
GAMMA = 0.99  # Discount factor for future rewards
LEARNING_RATE = 0.0001  # Learning rate for the model
BATCH_SIZE = 32  # Number of samples in a batch for learning
HIDDEN_LAYERS = [400, 400]  # Hidden layer sizes for the neural network
RENDER = True  # Whether to render the environment during training

def initialize_environment():
    """Initialize the LunarLander environment."""
    env = gym.make('LunarLander-v2')
    env.seed(1)  # Set random seed for reproducibility
    return env

def initialize_rl_agent(env):
    """Initialize the Dueling DQN agent."""
    agent = DuelingDQNPrioritizedReplay(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=LEARNING_RATE,
        e_greedy=EPSILON_GREEDY,
        reward_decay=GAMMA,
        hidden=HIDDEN_LAYERS,
        batch_size=BATCH_SIZE,
        replace_target_iter=TARGET_REP_ITER,
        memory_size=MEMORY_CAPACITY,
        e_greedy_increment=EPSILON_INCREMENT,
    )
    return agent

def train_agent(env, agent):
    """Train the agent on the LunarLander environment."""
    total_steps = 0
    running_reward = 0
    reward_scale = 100  # Scale rewards to make them more manageable

    for episode in range(MAX_EPISODES):
        state = env.reset()  # Reset the environment at the start of each episode
        episode_reward = 0
        
        while True:
            if total_steps > MEMORY_CAPACITY and RENDER:
                env.render()  # Render the environment for visualization

            action = agent.choose_action(state)  # Select an action based on the state
            next_state, reward, done, _ = env.step(action)  # Take the action and observe the next state and reward

            # Penalize the agent less for crashing
            if reward == -100:
                reward = -30

            # Normalize the reward
            reward /= reward_scale

            episode_reward += reward
            agent.store_transition(state, action, reward, next_state)  # Store the experience in memory

            if total_steps > MEMORY_CAPACITY:
                agent.learn()  # Update the agent's neural network

            if done:  # End of episode
                land_status = '| Landed' if reward == 100 / reward_scale else '| Crashed'
                running_reward = 0.99 * running_reward + 0.01 * episode_reward  # Update running average of rewards
                print(f'Episode: {episode}, {land_status}, Episode Reward: {round(episode_reward, 2)}, '
                      f'Running Reward: {round(running_reward, 2)}, Epsilon: {round(agent.epsilon, 3)}')
                break

            state = next_state  # Update the state for the next step
            total_steps += 1

def main():
    """Main function to run the training loop."""
    # Initialize environment and RL agent
    env = initialize_environment()
    agent = initialize_rl_agent(env)

    # Start training the agent
    train_agent(env, agent)

if __name__ == "__main__":
    main()
