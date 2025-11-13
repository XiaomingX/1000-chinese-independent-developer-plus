import os
import shutil
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing
import threading

# Game and training parameters
GAME = 'CartPole-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9  # Discount factor for future rewards
ENTROPY_BETA = 0.001  # Entropy coefficient for exploration
LR_A = 0.001  # Actor learning rate
LR_C = 0.001  # Critic learning rate

# Global variables to track the running reward and episode count
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

# Gym environment setup
env = gym.make(GAME)
N_S = env.observation_space.shape[0]  # Number of state features
N_A = env.action_space.n  # Number of possible actions


class ACNet:
    """Class defining the Actor-Critic network with both actor and critic components."""

    def __init__(self, scope, globalAC=None):
        """Initialize the Actor-Critic network."""
        self.scope = scope

        # Global network: just store the parameters
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            # Local network: calculate losses and gradients
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                self.c_loss = tf.reduce_mean(tf.square(td))  # Critic loss

                log_prob = tf.reduce_sum(
                    tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                    axis=1, keepdims=True
                )
                exp_v = log_prob * tf.stop_gradient(td)
                entropy = -tf.reduce_sum(
                    self.a_prob * tf.log(self.a_prob + 1e-5),
                    axis=1, keepdims=True
                )  # Encourage exploration
                self.exp_v = ENTROPY_BETA * entropy + exp_v
                self.a_loss = tf.reduce_mean(-self.exp_v)  # Actor loss

                # Gradients
                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

            # Synchronization between local and global networks
            with tf.name_scope('sync'):
                self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        """Build the actor and critic networks."""
        w_init = tf.random_normal_initializer(0., .1)

        # Actor network
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')

        # Critic network
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # State value

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):
        """Update the global network with local gradients."""
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        """Pull global network parameters to local network."""
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        """Choose an action based on the current state."""
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action


class Worker:
    """Worker that interacts with the environment and trains the model."""

    def __init__(self, name, globalAC):
        """Initialize the worker."""
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        """Main worker function that interacts with the environment and updates the network."""
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0

            while True:
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done: r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0  # Terminal state has no value
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    # Update global network with the calculated gradients
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(self.name, "Ep:", GLOBAL_EP, "| Ep_r:", GLOBAL_RUNNING_R[-1])
                    GLOBAL_EP += 1
                    break


def main():
    """Main function to initialize and run the training process."""
    global SESS, COORD, OPT_A, OPT_C, GLOBAL_AC

    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # Global network
        workers = []

        # Create worker threads
        for i in range(N_WORKERS):
            worker_name = f'W_{i}'
            workers.append(Worker(worker_name, GLOBAL_AC))

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

    # Plot the reward graph
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Step')
    plt.ylabel('Total Moving Reward')
    plt.show()


if __name__ == "__main__":
    main()
