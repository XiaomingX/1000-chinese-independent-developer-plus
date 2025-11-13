import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False
    ):
        """
        Initialize the Deep Q Network (DQN) with parameters.
        """
        # Initialize hyperparameters
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # Initialize learning step counter
        self.learn_step_counter = 0

        # Initialize memory (store state, action, reward, next_state)
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # Build neural networks (evaluation and target networks)
        self._build_net()

        # Prepare operations for target network update
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # Initialize TensorFlow session
        self.sess = tf.Session()

        # Optionally write TensorFlow graph to visualize in TensorBoard
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # List to store cost history for plotting
        self.cost_his = []

    def _build_net(self):
        """
        Build the evaluation and target neural networks.
        """
        # Placeholders for input states, next states, actions, and rewards
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input state
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # next state
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # action

        # Weight and bias initializers
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # Build evaluation network
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # Build target network
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        # Calculate target Q value for training
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)  # Prevent gradient flow through the target

        # Calculate Q value for chosen action (from evaluation network)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        # Loss function: Mean squared error between Q target and Q value for the chosen action
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        # Training operation: RMSProp optimizer
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        """
        Store the transition (state, action, reward, next_state) in memory.
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # Replace old memory with new memory in a cyclic manner
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        """
        Choose an action based on epsilon-greedy policy.
        """
        observation = observation[np.newaxis, :]  # Add batch dimension
        if np.random.uniform() < self.epsilon:
            # Select the action with the highest Q value
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            # Choose a random action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        """
        Perform learning (training) step.
        """
        # Update target network every `replace_target_iter` steps
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\nTarget network parameters replaced.\n')

        # Sample a batch of experiences from memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # Train the network on the batch
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # Gradually increase epsilon to reduce exploration over time
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        """
        Plot the cost (loss) history over training steps.
        """
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()

def main():
    """
    Main function to instantiate the DQN and run training.
    """
    # Initialize the Deep Q Network with the appropriate parameters
    dqn = DeepQNetwork(n_actions=3, n_features=4, output_graph=True)

    # Example: Store some random transitions and train the network
    for episode in range(1000):  # Example: 1000 episodes
        state = np.random.rand(4)  # Random initial state (4 features)
        action = dqn.choose_action(state)  # Choose an action
        reward = np.random.rand()  # Random reward
        next_state = np.random.rand(4)  # Random next state

        dqn.store_transition(state, action, reward, next_state)  # Store transition
        dqn.learn()  # Perform learning step

    # Plot the cost (loss) history after training
    dqn.plot_cost()

if __name__ == '__main__':
    main()
