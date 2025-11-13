import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from RL_brain import DQNPrioritizedReplay

# Constants for the environment and memory size
MEMORY_SIZE = 10000
ENV_NAME = 'MountainCar-v0'
SEED = 21
EPISODES = 20
ACTION_SPACE = 3
FEATURE_SPACE = 2
E_GREEDY_INCREMENT = 0.00005

def create_rl_model(sess, n_actions, n_features, memory_size, e_greedy_increment, prioritized=False):
    """Creates and returns a DQN model with or without prioritized replay."""
    return DQNPrioritizedReplay(
        n_actions=n_actions,
        n_features=n_features,
        memory_size=memory_size,
        e_greedy_increment=e_greedy_increment,
        sess=sess,
        prioritized=prioritized,
        output_graph=True if prioritized else False,
    )

def train_rl_model(RL, env, episodes=EPISODES, memory_size=MEMORY_SIZE):
    """Trains the RL model on the environment."""
    total_steps = 0
    steps = []
    episode_numbers = []

    for i_episode in range(episodes):
        observation = env.reset()
        while True:
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            if done: 
                reward = 10  # Reward for finishing the episode

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > memory_size:
                RL.learn()

            if done:
                print(f'Episode {i_episode} finished')
                steps.append(total_steps)
                episode_numbers.append(i_episode)
                break

            observation = observation_
            total_steps += 1

    return np.vstack((episode_numbers, steps))

def plot_training_results(his_natural, his_prio):
    """Plots the training results for comparison between natural DQN and DQN with prioritized replay."""
    plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='Natural DQN')
    plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with Prioritized Replay')
    plt.legend(loc='best')
    plt.ylabel('Total Training Time')
    plt.xlabel('Episode')
    plt.grid()
    plt.show()

def main():
    # Initialize environment and random seed
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(SEED)

    # Initialize TensorFlow session
    sess = tf.Session()

    # Create two DQN models: one with natural replay and one with prioritized replay
    with tf.variable_scope('natural_DQN'):
        RL_natural = create_rl_model(
            sess, 
            n_actions=ACTION_SPACE, 
            n_features=FEATURE_SPACE, 
            memory_size=MEMORY_SIZE, 
            e_greedy_increment=E_GREEDY_INCREMENT,
            prioritized=False
        )

    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL_prio = create_rl_model(
            sess, 
            n_actions=ACTION_SPACE, 
            n_features=FEATURE_SPACE, 
            memory_size=MEMORY_SIZE, 
            e_greedy_increment=E_GREEDY_INCREMENT,
            prioritized=True
        )

    # Initialize global variables in TensorFlow
    sess.run(tf.global_variables_initializer())

    # Train both models
    print("Training Natural DQN...")
    his_natural = train_rl_model(RL_natural, env)

    print("Training DQN with Prioritized Replay...")
    his_prio = train_rl_model(RL_prio, env)

    # Compare results
    plot_training_results(his_natural, his_prio)

if __name__ == "__main__":
    main()
