import gym
import matplotlib.pyplot as plt
from RL_brain import PolicyGradient

# Constants
DISPLAY_REWARD_THRESHOLD = -2000  # Show render if the total episode reward exceeds this threshold
RENDER = False  # Disable rendering for performance

def initialize_environment():
    """
    Initialize the MountainCar environment with fixed seed for reproducibility.
    """
    env = gym.make('MountainCar-v0')
    env.seed(1)
    env = env.unwrapped
    return env

def setup_policy_gradient(env):
    """
    Set up the Policy Gradient agent with necessary parameters.
    """
    return PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.995,
    )

def train_agent(env, RL):
    """
    Train the agent using Policy Gradient method over multiple episodes.
    """
    running_reward = None  # Initialize running reward

    for i_episode in range(1000):
        observation = env.reset()  # Reset environment for the new episode

        while True:
            # Render environment if the condition is met
            if RENDER:
                env.render()

            # Choose an action based on the current observation
            action = RL.choose_action(observation)

            # Take the action in the environment
            observation_, reward, done, _ = env.step(action)

            # Store the transition in the agent's memory
            RL.store_transition(observation, action, reward)

            if done:
                # Calculate the sum of rewards for the episode
                ep_rs_sum = sum(RL.ep_rs)

                # Update the running reward
                if running_reward is None:
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

                # Enable rendering if the running reward exceeds the threshold
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True

                print(f"Episode: {i_episode}, Reward: {int(running_reward)}")

                # Learn from the stored transitions
                vt = RL.learn()

                # Plot the learning progress after the 30th episode
                if i_episode == 30:
                    plot_learning_progress(vt)

                break

            # Update the observation for the next step
            observation = observation_

def plot_learning_progress(vt):
    """
    Plot the state-action value over episodes for visualization.
    """
    plt.plot(vt)
    plt.xlabel('Episode Steps')
    plt.ylabel('Normalized State-Action Value')
    plt.show()

def main():
    """
    Main function to tie together environment setup, policy gradient setup, and training.
    """
    # Initialize the environment
    env = initialize_environment()

    # Setup the Policy Gradient agent
    RL = setup_policy_gradient(env)

    # Start training the agent
    train_agent(env, RL)

if __name__ == "__main__":
    main()
