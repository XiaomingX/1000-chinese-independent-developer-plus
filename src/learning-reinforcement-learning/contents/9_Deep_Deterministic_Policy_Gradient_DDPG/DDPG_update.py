import tensorflow as tf
import numpy as np
import gym
import time


# Hyperparameters
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # Learning rate for the actor
LR_C = 0.002    # Learning rate for the critic
GAMMA = 0.9     # Reward discount factor
TAU = 0.01      # Soft replacement for target networks
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
RENDER = False
ENV_NAME = 'Pendulum-v0'


class DDPG:
    def __init__(self, a_dim, s_dim, a_bound):
        # Initialize memory, session, and network parameters
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self._build_network()

    def _build_network(self):
        """Build the actor and critic networks."""
        with tf.variable_scope('Actor'):
            self.a = self._build_actor(self.S, scope='eval', trainable=True)
            a_ = self._build_actor(self.S_, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            q = self._build_critic(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_critic(self.S_, a_, scope='target', trainable=False)

        # Network parameters
        self._initialize_parameters()

        # Target network soft replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        # Compute TD error and loss
        q_target = self.R + GAMMA * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # Maximize the Q-value
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

    def _initialize_parameters(self):
        """Initialize network parameters."""
        # Actor and critic parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

    def _build_actor(self, s, scope, trainable):
        """Build the actor network."""
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_critic(self, s, a, scope, trainable):
        """Build the critic network."""
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def choose_action(self, s):
        """Choose an action based on the current state."""
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        """Perform one learning step by training the networks."""
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # Train actor and critic networks
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        """Store a transition in memory."""
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1


def main():
    """Run the DDPG algorithm on the Pendulum environment."""
    # Initialize the environment
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    # Get the dimensions of state and action space
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    # Initialize the DDPG agent
    ddpg = DDPG(a_dim, s_dim, a_bound)

    var = 3  # Control exploration (exploration noise)
    start_time = time.time()

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Choose action with added exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)  # Add randomness for exploration

            # Take a step in the environment
            s_, r, done, info = env.step(a)

            # Store the transition in memory
            ddpg.store_transition(s, a, r / 10, s_)

            # Learn if enough transitions are stored
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995  # Decay exploration noise
                ddpg.learn()

            s = s_
            ep_reward += r

            # Print the episode result
            if j == MAX_EP_STEPS - 1:
                print(f'Episode: {i}, Reward: {int(ep_reward)}, Explore: {var:.2f}')
                break

    print(f'Running time: {time.time() - start_time} seconds')


if __name__ == "__main__":
    main()
