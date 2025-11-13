import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

# Hyperparameters
GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()  # Number of parallel workers (usually equals the number of CPU cores)
MAX_EP_STEP = 200  # Maximum steps per episode
MAX_GLOBAL_EP = 2000  # Maximum global episodes
GLOBAL_NET_SCOPE = 'Global_Net'  # Scope for the global network
UPDATE_GLOBAL_ITER = 10  # Update global network every N steps
GAMMA = 0.9  # Discount factor for future rewards
ENTROPY_BETA = 0.01  # Entropy regularization parameter
LR_A = 0.0001  # Actor learning rate
LR_C = 0.001  # Critic learning rate
GLOBAL_RUNNING_R = []  # Global running reward
GLOBAL_EP = 0  # Global episode counter

# Initialize environment
env = gym.make(GAME)
N_S = env.observation_space.shape[0]  # Number of states
N_A = env.action_space.shape[0]  # Number of actions
A_BOUND = [env.action_space.low, env.action_space.high]  # Action boundaries

class ACNet:
    def __init__(self, scope, globalAC=None):
        """
        Initialize the actor-critic network. 
        This includes both the global network (if scope is GLOBAL_NET_SCOPE)
        and local networks (for each worker).
        """
        if scope == GLOBAL_NET_SCOPE:
            # Global network (shared by all workers)
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            # Local network for each worker
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                self.c_loss = tf.reduce_mean(tf.square(td))  # Critic loss

                # Actor loss
                normal_dist = tf.distributions.Normal(mu, sigma + 1e-4)
                log_prob = normal_dist.log_prob(self.a_his)
                exp_v = log_prob * tf.stop_gradient(td)  # Advantage
                entropy = normal_dist.entropy()  # Encourages exploration
                self.exp_v = ENTROPY_BETA * entropy + exp_v
                self.a_loss = tf.reduce_mean(-self.exp_v)

                # Action selection
                self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])

                # Gradients for both actor and critic
                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

            # Synchronization with global network
            self._setup_sync(globalAC)

    def _build_net(self, scope):
        """
        Builds the actor-critic network.
        Actor outputs mean and variance of action distribution.
        Critic outputs state value.
        """
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # State value

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def _setup_sync(self, globalAC):
        """Sets up the synchronization operations between local and global networks."""
        with tf.name_scope('sync'):
            with tf.name_scope('pull'):
                # Pull global parameters to local
                self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
            with tf.name_scope('push'):
                # Push local gradients to global network
                self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def update_global(self, feed_dict):
        """Updates the global network with local gradients."""
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        """Pulls the latest global network parameters."""
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        """Selects an action based on the current state."""
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})


class Worker:
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        """Worker function to interact with the environment and update the global network."""
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # Normalize reward

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    v_s_ = 0 if done else SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # Reverse buffer_r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
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
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(self.name, "Ep:", GLOBAL_EP, "| Ep_r:", GLOBAL_RUNNING_R[-1])
                    GLOBAL_EP += 1
                    break

def main():
    """Main function to initialize and start training."""
    global SESS, OPT_A, OPT_C, GLOBAL_AC, COORD
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # Global network
        workers = []

        # Create workers
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i
            workers.append(Worker(i_name, GLOBAL_AC))

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
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Step')
    plt.ylabel('Total Moving Reward')
    plt.show()

if __name__ == "__main__":
    main()
