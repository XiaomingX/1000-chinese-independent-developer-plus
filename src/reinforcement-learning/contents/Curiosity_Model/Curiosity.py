import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt


class CuriosityDrivenAgent:
    def __init__(self, n_actions, n_states, learning_rate=0.01, gamma=0.98, epsilon=0.95,
                 target_update_interval=300, memory_size=10000, batch_size=128, output_graph=False):
        """
        Initialize the agent with necessary parameters.
        """
        self.n_actions = n_actions  # Number of actions
        self.n_states = n_states  # Number of state variables
        self.learning_rate = learning_rate  # Learning rate for optimization
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate for epsilon-greedy strategy
        self.target_update_interval = target_update_interval  # How often to update target network
        self.memory_size = memory_size  # Size of experience replay memory
        self.batch_size = batch_size  # Batch size for learning

        self.learn_step_counter = 0  # Counter for learning steps
        self.memory_counter = 0  # Counter for memory

        # Initialize memory buffer
        self.memory = np.zeros((self.memory_size, n_states * 2 + 2))

        # Build the networks
        self.states, self.actions, self.rewards, self.next_states, self.dyn_train, self.dqn_train, self.q_values, self.intrinsic_rewards = self._build_networks()

        # Initialize target replacement operation
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # Create a session for TensorFlow
        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

    def _build_networks(self):
        """
        Build the dynamics network and DQN (Deep Q-Network).
        """
        states = tf.placeholder(tf.float32, [None, self.n_states], name="states")
        actions = tf.placeholder(tf.int32, [None, ], name="actions")
        rewards = tf.placeholder(tf.float32, [None, ], name="rewards")
        next_states = tf.placeholder(tf.float32, [None, self.n_states], name="next_states")

        # Build the dynamics model
        dyn_next_state, curiosity_reward, dyn_train_op = self._build_dynamics_network(states, actions, next_states)

        # Combine curiosity-driven reward and external reward
        total_reward = tf.add(curiosity_reward, rewards, name="total_reward")

        # Build the Q-network and loss function
        q_values, dqn_loss, dqn_train_op = self._build_q_network(states, actions, total_reward, next_states)

        return states, actions, rewards, next_states, dyn_train_op, dqn_train_op, q_values, curiosity_reward

    def _build_dynamics_network(self, states, actions, next_states):
        """
        Build the dynamics model (predicts next state based on current state and action).
        """
        with tf.variable_scope("dyn_net"):
            action_float = tf.expand_dims(tf.cast(actions, dtype=tf.float32), axis=1)
            state_action = tf.concat((states, action_float), axis=1)
            dyn_layer = tf.layers.dense(state_action, 32, activation=tf.nn.relu)
            predicted_next_state = tf.layers.dense(dyn_layer, self.n_states)

        with tf.name_scope("intrinsic_reward"):
            intrinsic_reward = tf.reduce_sum(tf.square(next_states - predicted_next_state), axis=1)

        # Optimize the dynamics model using RMSProp
        dyn_train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(tf.reduce_mean(intrinsic_reward))

        return predicted_next_state, intrinsic_reward, dyn_train_op

    def _build_q_network(self, states, actions, rewards, next_states):
        """
        Build the Q-network (evaluates state-action pairs).
        """
        with tf.variable_scope('eval_net'):
            hidden_layer = tf.layers.dense(states, 128, activation=tf.nn.relu)
            q_values = tf.layers.dense(hidden_layer, self.n_actions)

        with tf.variable_scope('target_net'):
            target_hidden = tf.layers.dense(next_states, 128, activation=tf.nn.relu)
            target_q_values = tf.layers.dense(target_hidden, self.n_actions)

        with tf.variable_scope('q_target'):
            q_target = rewards + self.gamma * tf.reduce_max(target_q_values, axis=1)

        with tf.variable_scope('q_wrt_actions'):
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            q_values_for_action = tf.gather_nd(q_values, action_indices)

        # Loss is mean squared error between Q-values and target Q-values
        loss = tf.losses.mean_squared_error(labels=q_target, predictions=q_values_for_action)

        # Optimize the Q-network using RMSProp
        dqn_train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "eval_net"))

        return q_values, loss, dqn_train_op

    def store_transition(self, state, action, reward, next_state):
        """
        Store a transition in memory.
        """
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy.
        """
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            q_values = self.sess.run(self.q_values, feed_dict={self.states: state})
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        """
        Learn from experiences in memory and update the model.
        """
        if self.learn_step_counter % self.target_update_interval == 0:
            self.sess.run(self.target_replace_op)

        # Sample a batch from memory
        sample_indices = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
        batch_memory = self.memory[sample_indices, :]

        states_batch = batch_memory[:, :self.n_states]
        actions_batch = batch_memory[:, self.n_states]
        rewards_batch = batch_memory[:, self.n_states + 1]
        next_states_batch = batch_memory[:, -self.n_states:]

        # Train on the batch using Q-network
        self.sess.run(self.dqn_train, feed_dict={self.states: states_batch, self.actions: actions_batch,
                                                 self.rewards: rewards_batch, self.next_states: next_states_batch})

        # Train on the dynamics model after every 1000 steps
        if self.learn_step_counter % 1000 == 0:
            self.sess.run(self.dyn_train, feed_dict={self.states: states_batch, self.actions: actions_batch,
                                                     self.next_states: next_states_batch})

        self.learn_step_counter += 1


def main():
    # Create the environment
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    # Initialize the agent
    agent = CuriosityDrivenAgent(n_actions=3, n_states=2, learning_rate=0.01, output_graph=False)

    episode_steps = []
    for episode in range(200):
        state = env.reset()
        steps = 0
        while True:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state)
            agent.learn()

            if done:
                print(f'Episode: {episode} | Steps: {steps}')
                episode_steps.append(steps)
                break

            state = next_state
            steps += 1

    # Plot the results
    plt.plot(episode_steps)
    plt.ylabel("Steps per Episode")
    plt.xlabel("Episode")
    plt.show()


if __name__ == "__main__":
    main()
