import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

# Constants
GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 1500
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # Actor learning rate
LR_C = 0.001     # Critic learning rate
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

# Environment
env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]

class ACNet:
    def __init__(self, scope, globalAC=None):
        """ACNet constructor initializes either the global or local network."""
        if scope == GLOBAL_NET_SCOPE:
            self._initialize_global_network(scope)
        else:
            self._initialize_local_network(scope, globalAC)

    def _initialize_global_network(self, scope):
        """Initialize global network and placeholders."""
        with tf.variable_scope(scope):
            self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
            self.a_params, self.c_params = self._build_net(scope)[-2:]

    def _initialize_local_network(self, scope, globalAC):
        """Initialize local network, calculate losses and gradients."""
        with tf.variable_scope(scope):
            self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
            self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

            mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)
            td = self.v_target - self.v

            # Critic loss
            self.c_loss = tf.reduce_mean(tf.square(td))

            # Actor loss (using entropy for exploration)
            normal_dist = tf.distributions.Normal(mu * A_BOUND[1], sigma + 1e-4)
            log_prob = normal_dist.log_prob(self.a_his)
            exp_v = log_prob * tf.stop_gradient(td)
            entropy = normal_dist.entropy()
            self.exp_v = ENTROPY_BETA * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)

            # Gradients
            self.a_grads = tf.gradients(self.a_loss, self.a_params)
            self.c_grads = tf.gradients(self.c_loss, self.c_params)

            # Action selection
            self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])

            # Sync operations
            self._initialize_sync_ops(globalAC)

    def _initialize_sync_ops(self, globalAC):
        """Synchronize parameters between local and global networks."""
        with tf.name_scope('sync'):
            self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
            self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
            self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
            self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        """Build the actor-critic network with RNN for sequential decision-making."""
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('critic'):
            cell_size = 64
            s = tf.expand_dims(self.s, axis=1)  # Add time dimension
            rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(rnn_cell, s, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size])
            l_c = tf.layers.dense(cell_out, 50, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # Critic output: state value

        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(cell_out, 80, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):
        """Update global network using local gradients."""
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        """Pull updated global network parameters to local network."""
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, cell_state):
        """Choose an action based on current state using the local network."""
        s = s[np.newaxis, :]
        a, cell_state = SESS.run([self.A, self.final_state], {self.s: s, self.init_state: cell_state})
        return a, cell_state


class Worker:
    def __init__(self, name, globalAC):
        """Initialize worker with its environment and associated actor-critic network."""
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        """Worker's core logic: interact with environment, collect data, and update global network."""
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            rnn_state = SESS.run(self.AC.init_state)  # Initialize RNN state
            keep_state = rnn_state.copy()
            for ep_t in range(MAX_EP_STEP):
                if self.name == 'W_0':
                    self.env.render()

                a, rnn_state_ = self.AC.choose_action(s, rnn_state)  # Get action
                s_, r, done, info = self.env.step(a)
                done = done or ep_t == MAX_EP_STEP - 1  # Terminal condition

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # Normalize reward

                # Update global network periodically
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    v_s_ = 0 if done else SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :], self.AC.init_state: rnn_state_})[0, 0]
                    buffer_v_target = self._calculate_v_target(buffer_r, v_s_)
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.init_state: keep_state,
                    }

                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                    keep_state = rnn_state_.copy()

                s = s_
                rnn_state = rnn_state_
                total_step += 1

                if done:
                    self._log_episode(ep_r)
                    break

    def _calculate_v_target(self, buffer_r, v_s_):
        """Calculate the value target for each timestep in the episode."""
        buffer_v_target = []
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            buffer_v_target.append(v_s_)
        return buffer_v_target[::-1]

    def _log_episode(self, ep_r):
        """Log the running reward for the current episode."""
        if len(GLOBAL_RUNNING_R) == 0:
            GLOBAL_RUNNING_R.append(ep_r)
        else:
            GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
        print(self.name, "Ep:", GLOBAL_EP, "| Ep_r:", GLOBAL_RUNNING_R[-1])
        GLOBAL_EP += 1


def main():
    """Initialize session, workers, and coordinator. Start training process."""
    global SESS, COORD, OPT_A
