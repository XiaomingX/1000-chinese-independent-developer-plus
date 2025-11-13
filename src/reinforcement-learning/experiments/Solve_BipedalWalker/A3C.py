"""
Asynchronous Advantage Actor Critic (A3C) Reinforcement Learning

This code implements the A3C algorithm to train a reinforcement learning agent on the BipedalWalker-v2 environment from OpenAI's gym.
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

# Game and training parameters
GAME = 'BipedalWalker-v2'
LOG_DIR = './log'
MAX_GLOBAL_EP = 8000  # Maximum global episodes
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10  # Update frequency of the global network
GAMMA = 0.99  # Discount factor for rewards
ENTROPY_BETA = 0.005  # Entropy regularization for exploration
LR_A = 0.00005  # Learning rate for the actor
LR_C = 0.0001  # Learning rate for the critic

# Environment parameters
env = gym.make(GAME)
N_S = env.observation_space.shape[0]  # Number of state dimensions
N_A = env.action_space.shape[0]  # Number of action dimensions
A_BOUND = [env.action_space.low, env.action_space.high]
del env

# Global variables for tracking rewards and episode counts
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0


class ACNet(object):
    """Actor-Critic Network class that defines both the actor and the critic."""

    def __init__(self, scope, globalAC=None):
        """
        Initializes the network, either as a global network or a local network.
        
        Args:
            scope: Scope for variable names.
            globalAC: Reference to the global network for parameter synchronization.
        """
        if scope == GLOBAL_NET_SCOPE:
            # Global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:
            # Local network (used for training and updating)
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                
                # Build the actor and critic networks
                mu, sigma, self.v = self._build_net()

                # Temporal Difference (TD) error and loss functions
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                self.c_loss = tf.reduce_mean(tf.square(td))  # Critic loss (squared TD error)

                # Actor loss: Entropy regularization + expected value
                normal_dist = tf.contrib.distributions.Normal(mu, sigma)
                log_prob = normal_dist.log_prob(self.a_his)
                exp_v = log_prob * td
                entropy = normal_dist.entropy()  # Encourages exploration
                self.exp_v = ENTROPY_BETA * entropy + exp_v
                self.a_loss = tf.reduce_mean(-self.exp_v)  # Minimize actor loss

                # Sample actions from the policy
                self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1)), *A_BOUND)

                # Calculate gradients for both actor and critic
                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

            # Sync with the global network
            with tf.name_scope('sync'):
                self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        """Builds the actor and critic networks."""
        w_init = tf.contrib.layers.xavier_initializer()

        # Actor network
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 500, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dense(l_a, 300, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')

        # Critic network
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 500, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_c = tf.layers.dense(l_c, 300, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')

        return mu, sigma, v

    def update_global(self, feed_dict):
        """Updates the global network parameters based on the local gradients."""
        _, _, t = SESS.run([self.update_a_op, self.update_c_op, self.test], feed_dict)
        return t

    def pull_global(self):
        """Pull the global network parameters to the local network."""
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        """Choose an action based on the current state using the policy."""
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})


class Worker(object):
    """Worker that interacts with the environment and updates the global network."""

    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        """Main worker loop for training."""
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0

            while True:
                if self.name == 'W_0' and total_step % 30 == 0:
                    self.env.render()

                # Choose action based on current state
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if r == -100: r = -2  # Adjust reward if necessary

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                # Update global network and reset local buffers
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    v_s_ = 0 if done else SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []

                    # Calculate the discounted rewards
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    # Update global network
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {self.AC.s: buffer_s, self.AC.a_his: buffer_a, self.AC.v_target: buffer_v_target}
                    test = self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    # Print progress and record episode reward
                    achieve = '| Achieve' if self.env.unwrapped.hull.position[0] >= 88 else '| -------'
                    GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r) if GLOBAL_RUNNING_R else [ep_r]
                    print(f"{self.name} Ep: {GLOBAL_EP} {achieve} | Pos: {self.env.unwrapped.hull.position[0]} | RR: {GLOBAL_RUNNING_R[-1]:.1f} | EpR: {ep_r:.1f}")
                    GLOBAL_EP += 1
                    break


def main():
    """Main function to start training with multiple workers."""
    global SESS, OPT_A, OPT_C, COORD

    # Start a TensorFlow session
    SESS = tf.Session()

    # Initialize optimizers and global network
    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # Global network
        workers = [Worker(f'W_{i}', GLOBAL_AC) for i in range(multiprocessing.cpu_count())]

    # Start TensorFlow coordinator and initialize variables
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    # Start worker threads
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work)
        t.start()
        worker_threads.append(t)

    # Wait for all threads to finish
    COORD.join(worker_threads)

    # Plot the global running rewards
    plt.plot(GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Global Running Reward')
    plt.show()


if __name__ == "__main__":
    main()
