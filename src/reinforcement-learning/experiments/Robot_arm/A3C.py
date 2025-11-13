"""
This script trains a reinforcement learning model for a robot arm that attempts to reach a blue point in an environment.
The environment returns a reward based on how close the arm is to the blue point.

The training model will be saved locally and can be loaded later for evaluation. 
The script is customizable, and you can adjust hyperparameters or the environment as needed.

Dependencies:
pyglet >= 1.2.4
numpy >= 1.12.1
tensorflow >= 1.0.1
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from arm_env import ArmEnv

# Hyperparameters
MAX_GLOBAL_EP = 2000  # Max number of global episodes
MAX_EP_STEP = 300  # Max steps per episode
UPDATE_GLOBAL_ITER = 5  # Update global network every X iterations
N_WORKERS = multiprocessing.cpu_count()  # Number of workers
LR_A = 1e-4  # Actor learning rate
LR_C = 2e-4  # Critic learning rate
GAMMA = 0.9  # Discount factor for reward
ENTROPY_BETA = 0.01  # Entropy regularization coefficient
MODE = ['easy', 'hard']  # Environment modes
n_model = 1  # Select model (0 for easy, 1 for hard)
GLOBAL_NET_SCOPE = 'Global_Net'  # Scope for global network
GLOBAL_RUNNING_R = []  # Running rewards
GLOBAL_EP = 0  # Global episode count

# Initialize environment
env = ArmEnv(mode=MODE[n_model])
N_S = env.state_dim  # State dimensions
N_A = env.action_dim  # Action dimensions
A_BOUND = env.action_bound  # Action boundaries
del env  # Delete env as it's no longer needed here


class ACNet:
    """
    This class defines the Actor-Critic Network.
    It contains methods for building the network, choosing actions, 
    and updating the global network from local workers.
    """
    
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:
            # Global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:
            # Local network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                mu, sigma, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                self.c_loss = tf.reduce_mean(tf.square(td))

                normal_dist = tf.contrib.distributions.Normal(mu, sigma + 1e-5)
                log_prob = normal_dist.log_prob(self.a_his)
                exp_v = log_prob * td
                entropy = normal_dist.entropy()

                self.exp_v = ENTROPY_BETA * entropy + exp_v
                self.a_loss = tf.reduce_mean(-self.exp_v)

                self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)

                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

            # Synchronization operations
            with tf.name_scope('sync'):
                self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        """
        Build the actor-critic network.
        The actor produces actions, and the critic estimates the state value.
        """
        w_init = tf.contrib.layers.xavier_initializer()
        
        # Actor Network
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dense(l_a, 300, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')

        # Critic Network
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_c = tf.layers.dense(l_c, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

        return mu, sigma, v

    def update_global(self, feed_dict):
        """
        Update the global network using local network gradients.
        """
        _, _, t = SESS.run([self.update_a_op, self.update_c_op, self.test], feed_dict)
        return t

    def pull_global(self):
        """
        Pull the global network parameters into the local network.
        """
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        """
        Choose an action based on the current state using the actor network.
        """
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})[0]


class Worker:
    """
    Worker class that runs in parallel to train the model.
    It interacts with the environment, collects experiences, and updates the global model.
    """
    
    def __init__(self, name, globalAC):
        self.env = ArmEnv(mode=MODE[n_model])
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                if self.name == 'W_0':
                    self.env.render()

                a = self.AC.choose_action(s)
                s_, r, done = self.env.step(a)

                if ep_t == MAX_EP_STEP - 1:  # Mark episode end
                    done = True

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0  # Terminal state
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]

                    # Calculate targets
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)

                    buffer_v_target.reverse()
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    
                    # Update global network
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    test = self.AC.update_global(feed_dict)

                    # Clear buffers
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1

                if done:
                    # Record running episode rewards
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    
                    print(f"{self.name} Ep: {GLOBAL_EP} | Ep_r: {GLOBAL_RUNNING_R[-1]} | Var: {test}")
                    GLOBAL_EP += 1
                    break


def main():
    """
    Main function to initialize the session, create global and worker networks, 
    and run the training process in parallel.
    """
    # Initialize session
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # Global network
        workers = [Worker(f'W_{i}', GLOBAL_AC) for i in range(N_WORKERS)]

    # Initialize variables
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    # Start worker threads
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work)
        t.start()
        worker_threads.append(t)

    # Wait for all workers to finish
    COORD.join(worker_threads)


if __name__ == "__main__":
    main()
