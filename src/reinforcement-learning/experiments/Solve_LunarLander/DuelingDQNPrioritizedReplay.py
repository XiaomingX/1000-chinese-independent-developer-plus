import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class SumTree:
    """
    A modified SumTree class that stores data with priorities in a binary tree.
    The tree supports updating priorities and sampling data with importance sampling.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree of priorities
        self.data = np.zeros(capacity, dtype=object)  # Data storage
        self.data_pointer = 0

    def add_new_priority(self, priority, data):
        """Add new data with its priority to the tree."""
        leaf_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(leaf_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx, priority):
        """Update the priority of a tree node and propagate the change."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """Propagate priority change to the parent nodes."""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        """Retrieve a leaf based on the priority range."""
        leaf_idx = self._retrieve(lower_bound)
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """Recursively retrieve the leaf index based on the priority."""
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):  # No more children, return the leaf
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound - self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        """The root of the tree, representing the sum of all priorities."""
        return self.tree[0]


class Memory:
    """
    Memory class using SumTree to store transitions with priorities for Prioritized Experience Replay (PER).
    """
    epsilon = 0.001
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 1e-4
    abs_err_upper = 1

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, error, transition):
        """Store a transition with the calculated priority based on error."""
        priority = self._get_priority(error)
        self.tree.add_new_priority(priority, transition)

    def sample(self, n):
        """Sample a batch of transitions with importance sampling weights."""
        batch_idx, batch_memory, ISWeights = [], [], []
        segment = self.tree.root_priority / n
        self.beta = min(1, self.beta + self.beta_increment_per_sampling)

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)

        for i in range(n):
            lower_bound = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights.append(self.tree.capacity * prob)
            batch_idx.append(idx)
            batch_memory.append(data)

        ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # Normalize IS weights
        return batch_idx, np.vstack(batch_memory), ISWeights

    def update(self, idx, error):
        """Update the priority of a transition in the tree."""
        priority = self._get_priority(error)
        self.tree.update(idx, priority)

    def _get_priority(self, error):
        """Compute priority based on error."""
        error += self.epsilon
        return np.clip(np.power(error, self.alpha), 0, self.abs_err_upper)


class DuelingDQNPrioritizedReplay:
    """
    DQN with Dueling Network Architecture and Prioritized Experience Replay.
    """
    def __init__(self, n_actions, n_features, learning_rate=0.005, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=500, memory_size=10000, batch_size=32, hidden=[100, 50], output_graph=False, sess=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.hidden = hidden
        self.epsilon_increment = None
        self.epsilon = 0.5 if e_greedy is not None else self.epsilon_max

        self.learn_step_counter = 0
        self._build_net()
        self.memory = Memory(capacity=memory_size)

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        """Build the neural network model for Q-learning with dueling architecture."""
        def build_layers(s, c_names, w_initializer, b_initializer):
            for i, h in enumerate(self.hidden):
                in_units = self.n_features if i == 0 else self.hidden[i-1]
                with tf.variable_scope(f'l{i}'):
                    w = tf.get_variable('w', [in_units, h], initializer=w_initializer, collections=c_names)
                    b = tf.get_variable('b', [1, h], initializer=b_initializer, collections=c_names)
                    s = tf.nn.relu(tf.matmul(s, w) + b)
            with tf.variable_scope('Value'):
                w = tf.get_variable('w', [self.hidden[-1], 1], initializer=w_initializer, collections=c_names)
                b = tf.get_variable('b', [1, 1], initializer=b_initializer, collections=c_names)
                V = tf.matmul(s, w) + b
            with tf.variable_scope('Advantage'):
                w = tf.get_variable('w', [self.hidden[-1], self.n_actions], initializer=w_initializer, collections=c_names)
                b = tf.get_variable('b', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                A = tf.matmul(s, w) + b
            with tf.variable_scope('Q'):
                out = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))
            return out

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0., 0.01)
            b_initializer = tf.constant_initializer(0.01)
            self.q_eval = build_layers(self.s, c_names, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.abs_errors = tf.abs(tf.reduce_sum(self.q_target - self.q_eval, axis=1))
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        max_p = np.max(self.memory.tree.tree[-self.memory.tree.capacity:])
        self.memory.store(max_p, transition)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            return np.argmax(actions_value)
        else:
            return np.random.randint(0, self.n_actions)

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        batch_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        batch_s = np.array(batch_memory[:, :self.n_features])
        batch_a = np.array(batch_memory[:, self.n_features:self.n_features + 1], dtype=int)
        batch_r = np.array(batch_memory[:, self.n_features + 1:self.n_features + 2])
        batch_s_ = np.array(batch_memory[:, -self.n_features:])

        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_s_})
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch_s})

        batch_q_target = q_eval.copy()
        batch_idx_ = np.array(batch_idx, dtype=int)

        batch_q_target[range(self.batch_size), batch_a[:, 0]] = batch_r[:, 0] + self.gamma * np.max(q_next, axis=1)

        _, abs_errors = self.sess.run([self._train_op, self.abs_errors],
                                      feed_dict={self.s: batch_s, self.q_target: batch_q_target, self.ISWeights: ISWeights})

        for i in range(self.batch_size):
            self.memory.update(batch_idx[i], abs_errors[i])

        self.learn_step_counter += 1

def main():
    env = Maze()  # Assuming the environment is defined elsewhere
    agent = DuelingDQNPrioritizedReplay(n_actions=env.action_space.n, n_features=env.observation_space.shape[0])

    for episode in range(1000):  # Example number of episodes
        s = env.reset()
        total_reward = 0
        done = False
        while not done:
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            agent.store_transition(s, a, r, s_)
            agent.learn()
            total_reward += r
            s = s_

        print(f"Episode {episode} - Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
