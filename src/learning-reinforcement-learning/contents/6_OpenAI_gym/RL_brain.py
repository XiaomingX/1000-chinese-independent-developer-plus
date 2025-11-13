import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
        # Initialization of hyperparameters
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

        # Step counter and memory initialization
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # Build the neural network models for evaluation and target
        self._build_network()

        # Initialize TensorFlow session
        self.sess = tf.Session()

        # Optionally log for TensorBoard
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        # Initialize all TensorFlow variables
        self.sess.run(tf.global_variables_initializer())

        # Track the cost history for analysis
        self.cost_history = []

    def _build_network(self):
        """Builds the evaluation and target networks."""
        # Placeholder for the state input and Q target (for calculating loss)
        self.state_input = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        # Build the evaluation network
        with tf.variable_scope('eval_net'):
            self.q_eval = self._build_layers(self.state_input)

        # Loss and optimization for the evaluation network
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # Build the target network
        self.next_state_input = tf.placeholder(tf.float32, [None, self.n_features], name='next_state')
        with tf.variable_scope('target_net'):
            self.q_next = self._build_layers(self.next_state_input)

        # Copy parameters from the evaluation network to the target network
        e_params = tf.get_collection('eval_net_params')
        t_params = tf.get_collection('target_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def _build_layers(self, state_input):
        """Defines the layers for the network."""
        n_l1 = 10  # Number of neurons in the first hidden layer
        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        # First hidden layer
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer)
            l1 = tf.nn.relu(tf.matmul(state_input, w1) + b1)

        # Output layer
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer)
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer)
            return tf.matmul(l1, w2) + b2

    def store_transition(self, state, action, reward, next_state):
        """Stores the agent's experiences in memory."""
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        """Chooses an action based on the current policy (epsilon-greedy)."""
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # Use the evaluation network to predict Q values for all actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.state_input: observation})
            action = np.argmax(actions_value)
        else:
            # Choose a random action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        """Performs a learning step (i.e., training the evaluation network)."""
        # Update the target network every `replace_target_iter` steps
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\nTarget parameters replaced')

        # Sample a batch of memories for training
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # Get Q values for next states and current states
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.state_input: batch_memory[:, :self.n_features],
                                                  self.next_state_input: batch_memory[:, -self.n_features:]})

        # Update Q target with the Bellman equation
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # Train the evaluation network
        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.state_input: batch_memory[:, :self.n_features],
                                                                      self.q_target: q_target})

        self.cost_history.append(cost)

        # Increment epsilon (for epsilon-greedy exploration)
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        self.learn_step_counter += 1

    def plot_cost(self):
        """Plots the training cost over time."""
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()

def main():
    # Initialize the environment and agent
    n_actions = 4  # Example: 4 possible actions
    n_features = 8  # Example: 8 features per state
    agent = DeepQNetwork(n_actions, n_features)

    # Simulate training
    for episode in range(1000):
        state = np.random.rand(n_features)  # Simulating a random initial state
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state = np.random.rand(n_features)  # Simulating the next state
            reward = np.random.rand()  # Simulating a random reward
            done = np.random.rand() < 0.1  # Simulating a random end condition
            agent.store_transition(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            # Learn every few steps
            if episode % 10 == 0:
                agent.learn()

        print(f"Episode {episode} total reward: {total_reward}")

    # After training, plot the cost history
    agent.plot_cost()

if __name__ == "__main__":
    main()
