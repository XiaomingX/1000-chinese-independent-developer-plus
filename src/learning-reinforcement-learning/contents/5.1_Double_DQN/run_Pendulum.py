import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from RL_brain import DoubleDQN

# Define constants
MEMORY_SIZE = 3000
ACTION_SPACE = 11
ENV_NAME = 'Pendulum-v0'
SEED = 1

# Initialize the gym environment
def create_env():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(SEED)
    return env

# Initialize RL agents: Natural DQN and Double DQN
def create_agents(sess):
    with tf.variable_scope('Natural_DQN'):
        natural_DQN = DoubleDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, double_q=False, sess=sess
        )

    with tf.variable_scope('Double_DQN'):
        double_DQN = DoubleDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True
        )
    
    return natural_DQN, double_DQN

# Train the RL agent
def train_agent(agent, env):
    total_steps = 0
    observation = env.reset()
    q_values = []
    
    while True:
        action = agent.choose_action(observation)
        # Convert action to continuous space (range: -2 to 2)
        continuous_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
        
        # Step in the environment
        observation_, reward, done, _ = env.step(np.array([continuous_action]))
        
        # Normalize reward to range (-1, 0)
        reward /= 10

        # Store transition in memory
        agent.store_transition(observation, action, reward, observation_)
        
        # Learn if enough transitions are collected
        if total_steps > MEMORY_SIZE:
            agent.learn()

        if total_steps - MEMORY_SIZE > 20000:  # Stop after 20,000 steps
            break

        observation = observation_
        total_steps += 1
        q_values.append(agent.q)
    
    return q_values

# Main function to run the training and plot results
def main():
    # Create environment
    env = create_env()
    
    # Initialize TensorFlow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Create RL agents
        natural_DQN, double_DQN = create_agents(sess)
        
        # Train both agents
        q_natural = train_agent(natural_DQN, env)
        q_double = train_agent(double_DQN, env)
        
        # Plot the Q-values for both agents
        plt.plot(np.array(q_natural), c='r', label='Natural DQN')
        plt.plot(np.array(q_double), c='b', label='Double DQN')
        plt.legend(loc='best')
        plt.ylabel('Q eval')
        plt.xlabel('Training steps')
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()
