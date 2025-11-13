import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

# Define constants for the environment and training parameters
GAME = 'BipedalWalker-v2'
OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 8000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.00002  # Learning rate for actor
LR_C = 0.0001   # Learning rate for critic

# Initialize global variables for tracking progress
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

# Create the environment
env = gym.make(GAME)
N_S = env.observation_space.shape[0]  # Number of states
N_A = env.action_space.shape[0]       # Number of actions
A_BOUND = [env.action_space.low, env.action_space.high]
del env  # Delete the environment after initialization

# Define the Actor-Critic Network class
class ACNet:
    def __init__(self, scope, globalAC=None):
        """
        Initialize the Actor-Critic Network. If this is the global network (scope == GLOBAL_NET_SCOPE),
        it builds the network for the actor and critic. Otherwise, it builds a local network to compute gradients.
        """
        if scope == GLOBAL_NET_SCOPE:
            self._build_global_net(scope)
        else:
            self._build_local_net(scope, globalAC)

    def _build_global_net(self, scope):
        """
        Build the global network: This includes the placeholder for states, and the actor and critic networks.
        """
        with tf.variable_scope(scope):
            self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
            self._build_net()
            # Collect trainable parameters for the actor and critic
            self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
            self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

    def _build_local_net(self, scope, globalAC):
        """
        Build the local network for computing gradients, including loss functions and synchronization operations.
        """
        with tf.variable_scope(scope):
            self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
            self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

            mu, sigma, self.v = self._build_net()

            # Compute temporal difference error
            td = tf.subtract(self.v_target, self.v, name='TD_error')

            # Critic loss (mean squared error)
            self.c_loss = tf.reduce_mean(tf.square(td))

            # Actor loss (entropy + advantage)
            normal_dist = tf.contrib.distributions.Normal(mu, sigma)
            log_prob = normal_dist.log_prob(self.a_his)
            exp_v = log_prob * td
            entropy = normal_dist.entropy()  # Encourage exploration
            self.exp_v = ENTROPY_BETA * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)

            # Define the action selection mechanism
            self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1)), A_BOUND[0], A_BOUND[1])

            # Compute gradients for the actor and critic
            self.a_grads = tf.gradients(self.a_loss, self.a_params)
            self.c_grads = tf.gradients(self.c_loss, self.c_params)

            # Define operations for synchronizing parameters between local and global networks
            with tf.name_scope('sync'):
                self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        """
        Build the neural network layers for the actor and critic models.
        """
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('critic'):
            cell_size = 126
            s = tf.expand_dims(self.s, axis=1, name='timely_input')
            rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(rnn_cell, s, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size])
            l_c = tf.layers.dense(cell_out, 512, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')

        with tf.variable_scope('actor'):
            cell_out = tf.stop_gradient(cell_out)
            l_a = tf.layers.dense(cell_out, 512, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')

        return mu, sigma, v

    def update_global(self, feed_dict):
        """
        Update the global network using the gradients from the local network.
        """
        _, _, t = SESS.run([self.update_a_op, self.update_c_op, self.test], feed_dict)
        return t

    def pull_global(self):
        """
        Pull the global parameters to the local network.
        """
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, cell_state):
        """
        Choose an action based on the current state.
        """
        s = s[np.newaxis, :]
        a, cell_state = SESS.run([self.A, self.final_state], {self.s: s, self.init_state: cell_state})
        return a, cell_state

# Worker class for training on each worker's environment
class Worker:
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            rnn_state = SESS.run(self.AC.init_state)
            keep_state = rnn_state.copy()

            while True:
                if self.name == 'W_0' and total_step % 30 == 0:
                    self.env.render()

                a, rnn_state_ = self.AC.choose_action(s, rnn_state)
                s_, r, done, _ = self.env.step(a)
                if r == -100: r = -2  # Adjust reward for failure

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    v_s_ = 0 if done else SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :], self.AC.init_state: rnn_state_})[0, 0]

                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.init_state: keep_state,
                    }

                    test = self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                    keep_state = rnn_state_.copy()

                s = s_
                rnn_state = rnn_state_
                total_step += 1

                if done:
                    achieve = '| Achieve' if self.env.unwrapped.hull.position[0] >= 88 else '| -------'
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)

                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        achieve,
                        "| Pos: %i" % self.env.unwrapped.hull.position[0],
                        "| RR: %.1f" % GLOBAL_RUNNING_R[-1],
                        '| EpR: %.1f' % ep_r,
                        '| var:', test,
                    )
                    GLOBAL_EP += 1
                    break

# Main function to start training and visualizing the results
def main():
    global SESS, OPT_A, OPT_C, COORD

    SESS = tf.Session()

    # Set up the optimizer and global network
    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA', decay=0.95)
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC', decay=0.95)
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
        workers = [Worker(f'W_{i}', GLOBAL_AC) for i in range(N_WORKERS)]

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    # Start worker threads
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work)
        t.start()
        worker_threads.append(t)

    COORD.join(worker_threads)

    # Plot the results
    plt.plot(GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Global Running Reward')
    plt.show()

if __name__ == "__main__":
    main()
