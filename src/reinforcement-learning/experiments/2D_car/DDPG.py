import tensorflow as tf
import numpy as np
import os
import shutil
from car_env import CarEnv

# Set random seeds for reproducibility
np.random.seed(1)
tf.set_random_seed(1)

# Hyperparameters
MAX_EPISODES = 500
MAX_EP_STEPS = 600
LR_A = 1e-4  # Learning rate for actor
LR_C = 1e-4  # Learning rate for critic
GAMMA = 0.9  # Discount factor for rewards
REPLACE_ITER_A = 800  # Frequency of replacing target networks for actor
REPLACE_ITER_C = 700  # Frequency of replacing target networks for critic
MEMORY_CAPACITY = 2000  # Memory size
BATCH_SIZE = 16  # Batch size for training
VAR_MIN = 0.1  # Minimum exploration noise
RENDER = True  # Whether to render the environment
LOAD = False  # Whether to load a pre-trained model
DISCRETE_ACTION = False  # Whether the action space is discrete

# Initialize environment and get state and action dimensions
env = CarEnv(discrete_action=DISCRETE_ACTION)
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

# TensorFlow placeholders for states, rewards, and next states
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor:
    """Actor network to decide actions based on states."""
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            self.a = self._build_net(S, scope='eval_net', trainable=True)
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        """Builds the neural network."""
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 100, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='l2', trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')
        return scaled_a

    def learn(self, s):
        """Update the actor's weights."""
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        """Choose action based on current state."""
        s = s[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={S: s})[0]

    def add_grad_to_graph(self, a_grads):
        """Add gradient operations to the graph."""
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)
        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic:
    """Critic network to evaluate the actions taken by the actor."""
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]

    def _build_net(self, s, a, scope, trainable):
        """Builds the critic network."""
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)
            net = tf.nn.relu6(tf.matmul(s, init_w) + tf.matmul(a, init_w) + init_b)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='l2', trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
        return q

    def learn(self, s, a, r, s_):
        """Update the critic's weights."""
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory:
    """Memory buffer to store experience tuples."""
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        """Store a new transition in memory."""
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        """Sample a batch of transitions from memory."""
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


def train():
    """Train the actor-critic model."""
    var = 2.0  # Control exploration
    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_step = 0

        for t in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)  # Add randomness to action for exploration
            s_, r, done = env.step(a)
            memory.store_transition(s, a, r, s_)

            if memory.pointer > MEMORY_CAPACITY:
                var = max([var * 0.9995, VAR_MIN])  # Decay exploration noise
                b_M = memory.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_step += 1

            if done or t == MAX_EP_STEPS - 1:
                print(f"Episode {ep} | Steps: {ep_step} | Explore: {var:.2f}")
                break

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    save_path = saver.save(sess, os.path.join(path, 'DDPG.ckpt'), write_meta_graph=False)
    print(f"\nModel saved at: {save_path}\n")


def eval():
    """Evaluate the trained model."""
    env.set_fps(30)
    while True:
        s = env.reset()
        while True:
            env.render()
            a = actor.choose_action(s)
            s_, r, done = env.step(a)
            s = s_
            if done:
                break


def main():
    """Main function to control the flow of training or evaluation."""
    with tf.Session() as sess:
        actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
        critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
        actor.add_grad_to_graph(critic.a_grads)
        memory = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)
        saver = tf.train.Saver()
        path = './discrete' if DISCRETE_ACTION else './continuous'

        if LOAD:
            saver.restore(sess, tf.train.latest_checkpoint(path))
            eval()
        else:
            sess.run(tf.global_variables_initializer())
            train()


if __name__ == '__main__':
    main()
