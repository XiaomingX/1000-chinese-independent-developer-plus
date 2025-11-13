import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class SumTree:
    """
    A SumTree data structure that stores data with priorities.
    This is used for prioritized experience replay in DQN.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # Capacity of the tree
        self.tree = np.zeros(2 * capacity - 1)  # Tree to store priorities
        self.data = np.zeros(capacity, dtype=object)  # Data storage for transitions

    def add(self, priority, data):
        """
        Add a new transition to the tree with a specified priority.
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # Store the transition data
        self.update(tree_idx, priority)  # Update the priority in the tree

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # Reset the pointer if we exceed capacity
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        """
        Update the priority of a specific leaf node and propagate the change up the tree.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:  # Propagate the change upwards
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value):
        """
        Retrieve the leaf with the highest priority based on the value.
        """
        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            if left_idx >= len(self.tree):  # Reached the leaf level
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_idx]:
                    parent_idx = left_idx
                else:
                    value -= self.tree[left_idx]
                    parent_idx = right_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]  # The root node stores the total sum of priorities


class Memory:
    """
    The Memory class implements prioritized experience replay using SumTree.
    """
    epsilon = 0.01  # Small value to avoid zero priority
    alpha = 0.6  # The degree to which TD error affects priority
    beta = 0.4  # The importance-sampling correction term
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.0  # Clipped absolute error value

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        """
        Store a new transition with the highest priority (initially).
        """
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        self.tree.add(max_priority, transition)

    def sample(self, batch_size):
        """
        Sample a batch of transitions based on their priority.
        """
        batch_idx = np.empty((batch_size,), dtype=np.int32)
        batch_memory = np.empty((batch_size, self.tree.data[0].size))
        IS_weights = np.empty((batch_size, 1))

        priority_segment = self.tree.total_priority / batch_size
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority

        for i in range(batch_size):
            low = priority_segment * i
            high = priority_segment * (i + 1)
            value = np.random.uniform(low, high)
            idx, priority, data = self.tree.get_leaf(value)

            prob = priority / self.tree.total_priority
            IS_weights[i, 0] = np.power(prob / min_prob, -self.beta)
            batch_idx[i], batch_memory[i, :] = idx, data

        return batch_idx, batch_memory, IS_weights

    def batch_update(self, tree_idx, abs_errors):
        """
        Update the priorities of the sampled transitions based on their absolute errors.
        """
        abs_errors += self.epsilon  # Ensure the error is non-zero
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        priorities = np.power(clipped_errors, self.alpha)
        for idx, priority in zip(tree_idx, priorities):
            self.tree.update(idx, priority)


class DQNWithPrioritizedReplay:
    def __init__(self, n_actions, n_features, learning_rate=0.005, reward_decay=0.9, e_greedy=0.9, 
                 replace_target_iter=500, memory_size=10000, batch_size=32, e_greedy_increment=None, 
                 output_graph=False, prioritized=True, sess=None):
        """
        Initialize the DQN with prioritized experience replay.
        """
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
        self.prioritized = prioritized

        self.learn_step_counter = 0
        self._build_network()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.memory = Memory(capacity=memory_size) if prioritized else np.zeros((memory_size, n_features * 2 + 2))
        
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_history = []

    def _build_network(self):
        """
        Build the evaluation and target networks.
        """
        def build_layers(inputs, n_l1, w_initializer, b_initializer, trainable):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, trainable=trainable)
                out = tf.matmul(l1, w2) + b2
            return out

        # Evaluate Network
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        if self.prioritized:
            self.IS_weights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        with tf.variable_scope('eval_net'):
            self.q_eval = build_layers(self.s, 20, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1), True)

        # Loss Function
        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)
                self.loss = tf.reduce_mean(self.IS_weights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        # Optimizer
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # Target Network
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            self.q_next = build_layers(self.s_, 20, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1), False)

    def store_transition(self, s, a, r, s_):
        """
        Store a new transition in the memory.
        """
        if self.prioritized:
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)
        else:
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        """
        Choose an action based on the current observation.
        """
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        """
        Learn from the memory and update the networks.
        """
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, IS_weights = self.memory.sample(self.batch_size)
        else:
            sample_idx = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_idx, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:], self.s: batch_memory[:, :self.n_features]}
        )

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self.train_op, self.abs_errors, self.loss],
                                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                                self.q_target: q_target,
                                                                self.IS_weights: IS_weights})
            self.memory.batch_update(tree_idx, abs_errors)
        else:
            _, self.cost = self.sess.run([self.train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target})

        self.cost_history.append(self.cost)

        # Update epsilon
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        self.learn_step_counter += 1


def main():
    # Example of how to use the DQN with prioritized replay
    n_actions = 4
    n_features = 10
    agent = DQNWithPrioritizedReplay(n_actions=n_actions, n_features=n_features)

    # Dummy training loop (replace with actual environment interactions)
    for episode in range(1000):
        state = np.random.rand(1, n_features)  # Example initial state
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state = np.random.rand(1, n_features)  # Simulated next state
            reward = np.random.rand()  # Simulated reward
            agent.store_transition(state, action, reward, next_state)
            state = next_state
            agent.learn()
            total_reward += reward
            done = total_reward > 10  # Example termination condition

if __name__ == "__main__":
    main()
