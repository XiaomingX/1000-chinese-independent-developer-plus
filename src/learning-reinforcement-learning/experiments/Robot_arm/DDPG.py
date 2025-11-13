import tensorflow as tf
import numpy as np
import os
import shutil
from arm_env import ArmEnv

# Set random seeds for reproducibility
np.random.seed(1)
tf.set_random_seed(1)

# Hyperparameters
MAX_EPISODES = 600  # Maximum number of episodes
MAX_EP_STEPS = 200  # Maximum steps per episode
LR_A = 1e-4  # Actor learning rate
LR_C = 1e-4  # Critic learning rate
GAMMA = 0.9  # Discount factor
REPLACE_ITER_A = 1100  # Actor network replacement frequency
REPLACE_ITER_C = 1000  # Critic network replacement frequency
MEMORY_CAPACITY = 5000  # Memory capacity for experience replay
BATCH_SIZE = 16  # Batch size for learning
VAR_MIN = 0.1  # Minimum exploration noise
RENDER = True  # Whether to render the environment
LOAD = False  # Whether to load a pre-trained model
MODE = ['easy', 'hard']  # Modes for the environment
n_model = 1  # Selected model (easy or hard mode)

# Initialize the environment
env = ArmEnv(mode=MODE[n_model])
STATE_DIM = env.state_dim  # State dimension
ACTION_DIM = env.action_dim  # Action dimension
ACTION_BOUND = env.action_bound  # Action boundaries

# TensorFlow placeholders for state, reward, and next state
S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
R = tf.placeholder(tf.float32, [None, 1], name='r')
S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')

class Actor:
    def __init__(self, sess, action_dim, action_bound, learning_rate, replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replace_iter = replace_iter
        self.replace_counter = 0

        with tf.variable_scope('Actor'):
            self.a = self._build_net(S, scope='eval_net', trainable=True)
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w, trainable=trainable)
            scaled_actions = tf.multiply(actions, self.action_bound, name='scaled_a')
        return scaled_actions

    def learn(self, s):
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.replace_counter % self.replace_iter == 0:
            self.sess.run(self.replace)
        self.replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={S: s})[0]

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

class Critic:
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replace_iter = replace_iter
        self.replace_counter = 0

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
        self.replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replace_counter % self.replace_iter == 0:
            self.sess.run(self.replace)
        self.replace_counter += 1

class Memory:
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory is not full'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

def train():
    var = 2.  # Control exploration
    memory = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for t in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)
            s_, r, done = env.step(a)
            memory.store_transition(s, a, r, s_)

            if memory.pointer > MEMORY_CAPACITY:
                var = max([var * 0.9999, VAR_MIN])
                batch = memory.sample(BATCH_SIZE)
                b_s = batch[:, :STATE_DIM]
                b_a = batch[:, STATE_DIM:STATE_DIM + ACTION_DIM]
                b_r = batch[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = batch[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS - 1 or done:
                result = '| done' if done else '| ----'
                print(f'Ep: {ep}, {result}, | R: {int(ep_reward)}, | Explore: {var:.2f}')
                break

    save_model()

def save_model():
    if os.path.isdir(path): 
        shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join(path, 'DDPG.ckpt')
    saver.save(sess, ckpt_path, write_meta_graph=False)
    print(f"\nModel saved to {ckpt_path}\n")

def eval():
    env.set_fps(30)
    s = env.reset()
    while True:
        if RENDER:
            env.render()
        a = actor.choose_action(s)
        s_, r, done = env.step(a)
        s = s_

def main():
    # Start a TensorFlow session
    sess = tf.Session()

    # Create Actor and Critic networks
    actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
    critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)

    # Initialize variables and restore model if needed
    saver = tf.train.Saver()
    path = './' + MODE[n_model]
    if LOAD:
        saver.restore(sess, tf.train.latest_checkpoint(path))
    else:
        sess.run(tf.global_variables_initializer())

    # Train or evaluate based on the LOAD flag
    if LOAD:
        eval()
    else:
        train()

if __name__ == '__main__':
    main()
