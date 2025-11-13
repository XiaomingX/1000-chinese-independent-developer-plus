import numpy as np
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(1)
tf.set_random_seed(1)

class DuelingDQN:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False,
                 dueling=True,
                 sess=None):
        # Initialize parameters
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
        self.dueling = dueling  # Whether to use dueling architecture

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # (state, action, reward, next_state)

        self._build_network()

        # Create target and evaluation networks and their replacement operations
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # Initialize the session
        self.sess = tf.Session() if sess is None else sess
        self.sess.run(tf.global_variables_initializer())

        # If output_graph is True, write the graph to TensorBoard
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        # To store cost history
        self.cost_history = []

    def _build_network(self):
        # Build layers for the DQN model (both evaluation and target networks)
        def build_layers(input_data, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(input_data, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                # Q-value = Value + Advantage
                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s, a)
            else:
                # Non-dueling DQN
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out

        # Define placeholders for input and output
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # state input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # target Q-values

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # Evaluation network (Q-value prediction)
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            # Loss function (mean squared error between target and predicted Q-values)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            # Optimizer
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # Define target network (for computing the next Q-values)
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # next state input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        # Store the transition in memory
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # Choose an action based on epsilon-greedy policy
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # Exploration
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:  # Exploitation
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # Update the target network every few steps
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\nTarget network parameters updated.')

        # Sample a batch of transitions from memory
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # Compute Q-values for the next state
        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:, -self.n_features:]})
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        # Update Q-values using the Bellman equation
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # Perform one step of gradient descent to minimize loss
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})

        # Store the cost for later analysis
        self.cost_history.append(self.cost)

        # Update epsilon (exploration rate)
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        self.learn_step_counter += 1

# Main function to run the Dueling DQN
def main():
    # Define environment parameters (replace with your actual environment)
    n_actions = 4  # Example: 4 possible actions
    n_features = 10  # Example: 10 features in the state

    # Initialize Dueling DQN agent
    agent = DuelingDQN(n_actions=n_actions, n_features=n_features)

    # Simulate some learning (replace with actual environment interaction)
    for episode in range(100):
        state = np.random.randn(n_features)  # Random initial state
        total_reward = 0

        for step in range(100):  # Max steps per episode
            action = agent.choose_action(state)
            next_state = np.random.randn(n_features)  # Random next state
            reward = np.random.random()  # Random reward

            agent.store_transition(state, action, reward, next_state)

            # Learn every step
            agent.learn()

            state = next_state
            total_reward += reward

        print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
