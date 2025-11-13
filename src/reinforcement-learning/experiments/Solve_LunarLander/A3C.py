import os
import shutil
import numpy as np
import tensorflow as tf
import gym
import multiprocessing
import threading
import matplotlib.pyplot as plt

# Constants
GAME = 'LunarLander-v2'
OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 5000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99
ENTROPY_BETA = 0.001
LR_A = 0.0005    # Actor learning rate
LR_C = 0.001     # Critic learning rate

# Global variables
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n
del env  # Free up memory


class ACNet(object):
    """The actor-critic network used in the reinforcement learning model."""
    
    def __init__(self, scope, globalAC=None):
        """Initialize network layers and define operations for training."""
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_network(N_A)
                self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.actions = tf.placeholder(tf.int32, [None, ], 'A')
                self.target_value = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.action_probabilities, self.value = self._build_network(N_A)

                td_error = tf.subtract(self.target_value, self.value, name='TD_error')
                with tf.name_scope('critic_loss'):
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))

                with tf.name_scope('actor_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.action_probabilities) * tf.one_hot(self.actions, N_A, dtype=tf.float32), axis=1, keepdims=True)
                    expected_value = log_prob * td_error
                    entropy = -tf.reduce_sum(self.action_probabilities * tf.log(self.action_probabilities), axis=1, keepdims=True)
                    self.expected_value = ENTROPY_BETA * entropy + expected_value
                    self.actor_loss = tf.reduce_mean(-self.expected_value)

                with tf.name_scope('gradients'):
                    self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                    self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_actor_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.actor_params, globalAC.actor_params)]
                    self.pull_critic_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.critic_params, globalAC.critic_params)]
                with tf.name_scope('push'):
                    self.update_actor_op = OPT_A.apply_gradients(zip(self.actor_grads, globalAC.actor_params))
                    self.update_critic_op = OPT_C.apply_gradients(zip(self.critic_grads, globalAC.critic_params))

    def _build_network(self, n_a):
        """Build the actor-critic neural network."""
        w_init = tf.random_normal_initializer(0., .01)

        # Critic Network (Value function)
        with tf.variable_scope('critic'):
            cell_size = 64
            state_input = tf.expand_dims(self.state, axis=1)
            rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(rnn_cell, state_input, initial_state=self.init_state, time_major=True)
            cell_output = tf.reshape(outputs, [-1, cell_size])
            critic_layer = tf.layers.dense(cell_output, 200, tf.nn.relu6, kernel_initializer=w_init, name='critic_layer')
            value = tf.layers.dense(critic_layer, 1, kernel_initializer=w_init, name='value')
        
        # Actor Network (Policy)
        with tf.variable_scope('actor'):
            actor_layer = tf.layers.dense(cell_output, 300, tf.nn.relu6, kernel_initializer=w_init, name='actor_layer')
            action_probabilities = tf.layers.dense(actor_layer, n_a, tf.nn.softmax, kernel_initializer=w_init, name='action_probabilities')

        return action_probabilities, value

    def update_global(self, feed_dict):
        """Update global network using local network's gradients."""
        SESS.run([self.update_actor_op, self.update_critic_op], feed_dict)

    def pull_global(self):
        """Pull parameters from the global network to the local network."""
        SESS.run([self.pull_actor_params_op, self.pull_critic_params_op])

    def choose_action(self, state, rnn_state):
        """Choose an action based on the current state using the actor network."""
        prob_weights, rnn_state = SESS.run([self.action_probabilities, self.final_state],
                                           feed_dict={self.state: state[np.newaxis, :], self.init_state: rnn_state})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action, rnn_state


class Worker(object):
    """Worker class that interacts with the environment and trains the actor-critic network."""
    
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        r_scale = 100
        buffer_s, buffer_a, buffer_r = [], [], []

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            state = self.env.reset()
            episode_reward = 0
            rnn_state = SESS.run(self.AC.init_state)
            keep_state = rnn_state.copy()

            while True:
                action, rnn_state_ = self.AC.choose_action(state, rnn_state)
                next_state, reward, done, _ = self.env.step(action)
                reward = reward if reward != -100 else -10  # Handle environment-specific cases

                episode_reward += reward
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward / r_scale)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        value_next_state = 0  # Terminal state
                    else:
                        value_next_state = SESS.run(self.AC.value, {self.AC.state: next_state[np.newaxis, :], self.AC.init_state: rnn_state_})[0, 0]

                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        value_next_state = r + GAMMA * value_next_state
                        buffer_v_target.append(value_next_state)
                    buffer_v_target.reverse()

                    buffer_s = np.vstack(buffer_s)
                    buffer_a = np.array(buffer_a)
                    buffer_v_target = np.vstack(buffer_v_target)

                    feed_dict = {
                        self.AC.state: buffer_s,
                        self.AC.actions: buffer_a,
                        self.AC.target_value: buffer_v_target,
                        self.AC.init_state: keep_state,
                    }

                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                    keep_state = rnn_state_.copy()

                state = next_state
                total_step += 1
                rnn_state = rnn_state_
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(episode_reward)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * episode_reward)
                    print(f"{self.name} | Episode {GLOBAL_EP} | Reward: {GLOBAL_RUNNING_R[-1]}")
                    GLOBAL_EP += 1
                    break


def main():
    """Main function to initialize the environment and start training."""
    global SESS, COORD, OPT_A, OPT_C, GLOBAL_AC, workers

    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
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

    # Plotting results
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


if __name__ == "__main__":
    main()
