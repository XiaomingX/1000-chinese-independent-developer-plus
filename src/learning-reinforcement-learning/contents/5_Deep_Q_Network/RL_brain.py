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
        output_graph=False,
    ):
        # Initialize class parameters
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate  # Learning rate
        self.gamma = reward_decay  # Reward decay (discount factor)
        self.epsilon_max = e_greedy  # Max epsilon for exploration
        self.replace_target_iter = replace_target_iter  # Number of steps before replacing target network
        self.memory_size = memory_size  # Size of the memory
        self.batch_size = batch_size  # Batch size for training
        self.epsilon_increment = e_greedy_increment  # Increment for epsilon
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # Initial epsilon
        self.learn_step_counter = 0  # Counter for the number of learning steps
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # Memory to store experiences

        # Build the neural network
        self._build_net()

        # Initialize target and evaluation network parameters
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # Start TensorFlow session
        self.sess = tf.Session()

        # If output_graph is true, save the computation graph for TensorBoard
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        # Initialize all variables in the TensorFlow session
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # List to keep track of the cost during training

    def _build_net(self):
        """
        Builds the evaluation and target networks.
        The evaluation network is used for selecting actions, 
        and the target network is used for calculating the target Q values.
        """
        # Placeholders for state inputs and Q target outputs
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # Input state
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # Target Q-values

        # Build the evaluation network (for selecting actions)
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        # Loss function for evaluating the network
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        # Training operation (RMSProp optimizer)
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # Build the target network (for computing the target Q values)
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # Next state

        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        """
        Stores a new transition (state, action, reward, next state) in memory.
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # Replace old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        """
        Chooses an action based on the current observation.
        Uses epsilon-greedy strategy for exploration vs exploitation.
        """
        observation = observation[np.newaxis, :]  # Add batch dimension

        if np.random.uniform() < self.epsilon:
            # Choose the action with the highest Q-value
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            # Choose a random action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        """
        Updates the evaluation network by performing a training step.
        It calculates the target Q-values and minimizes the loss between
        the target Q-values and the predicted Q-values.
        """
        if self.learn_step_counter % self.replace_target_iter == 0:
            # Replace the target network with the evaluation network
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # Sample a batch of experiences from memory
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # Get the next Q-values and current Q-values for the batch
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # Next states
                self.s: batch_memory[:, :self.n_features],  # Current states
            })

        # Calculate the target Q-values
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # Update the target Q-values based on the reward and the next Q-value
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # Train the evaluation network by minimizing the loss
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # Update epsilon for exploration
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        """
        Plots the cost over training steps to visualize the training progress.
        """
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()

# Main function to demonstrate the agent's functionality
def main():
    # Create the Deep Q Network (DQN) agent
    dqn = DeepQNetwork(n_actions=4, n_features=8)

    # Example of training the agent (in practice, you'd interact with an environment like OpenAI Gym)
    for episode in range(100):
        state = np.random.random(8)  # Random state as placeholder
        action = dqn.choose_action(state)  # Choose an action based on the state
        reward = np.random.random()  # Random reward as placeholder
        next_state = np.random.random(8)  # Random next state as placeholder
        dqn.store_transition(state, action, reward, next_state)  # Store the transition
        dqn.learn()  # Learn from stored experiences

        if episode % 10 == 0:
            print(f'Episode {episode} complete.')

    # Plot the cost to visualize the training progress
    dqn.plot_cost()

if __name__ == "__main__":
    main()
