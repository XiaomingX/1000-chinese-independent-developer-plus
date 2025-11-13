import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import threading
import queue

# Constants
EP_MAX = 1000          # Maximum number of episodes
EP_LEN = 500           # Length of each episode
N_WORKER = 4           # Number of parallel workers
GAMMA = 0.9            # Reward discount factor
A_LR = 0.0001          # Actor learning rate
C_LR = 0.0001          # Critic learning rate
MIN_BATCH_SIZE = 64    # Minimum batch size for PPO update
UPDATE_STEP = 15       # Steps for PPO update
EPSILON = 0.2          # Clipping for surrogate objective
GAME = 'CartPole-v0'   # Game environment

# Initialize environment
env = gym.make(GAME)
S_DIM = env.observation_space.shape[0]  # Observation space dimension
A_DIM = env.action_space.n              # Action space dimension

class PPONet:
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')  # State placeholder

        # Critic network
        self.v = self.build_critic(self.tfs)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # Actor network
        self.pi, pi_params = self.build_actor(self.tfs, 'pi', True)
        oldpi, oldpi_params = self.build_actor(self.tfs, 'oldpi', False)

        # Copy pi to oldpi
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # Actor loss (PPO objective)
        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(self.pi, a_indices)
        oldpi_prob = tf.gather_nd(oldpi, a_indices)
        ratio = pi_prob / (oldpi_prob + 1e-5)  # Avoid division by zero
        surr = ratio * self.tfadv  # Surrogate loss

        self.aloss = -tf.reduce_mean(
            tf.minimum(surr, tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv)
        )
        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        self.sess.run(tf.global_variables_initializer())

    def build_critic(self, input_tensor):
        """Build the critic network."""
        w_init = tf.random_normal_initializer(0., .1)
        lc = tf.layers.dense(input_tensor, 200, tf.nn.relu, kernel_initializer=w_init, name='lc')
        return tf.layers.dense(lc, 1)

    def build_actor(self, input_tensor, name, trainable):
        """Build the actor network."""
        with tf.variable_scope(name):
            l_a = tf.layers.dense(input_tensor, 200, tf.nn.relu, trainable=trainable)
            a_prob = tf.layers.dense(l_a, A_DIM, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def update(self):
        """Update the global PPO model."""
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # Wait until data is collected
                self.sess.run(self.update_oldpi_op)  # Copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # Collect data from workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM:S_DIM + 1].ravel(), data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # Update actor and critic
                for _ in range(UPDATE_STEP):
                    self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv})
                    self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})

                UPDATE_EVENT.clear()  # Finish updating
                GLOBAL_UPDATE_COUNTER = 0  # Reset counter
                ROLLING_EVENT.set()  # Allow workers to roll out

    def choose_action(self, state):
        """Choose an action based on the current state."""
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: state[None, :]})
        return np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

    def get_value(self, state):
        """Get value of the current state from the critic network."""
        if state.ndim < 2: 
            state = state[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: state})[0, 0]


class Worker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        """Run the worker's rollout and training loop."""
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            state = self.env.reset()
            episode_reward = 0
            buffer_state, buffer_action, buffer_reward = [], [], []

            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():
                    ROLLING_EVENT.wait()  # Wait until PPO is updated
                    buffer_state, buffer_action, buffer_reward = [], [], []  # Clear history
                action = self.ppo.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                if done: 
                    reward = -10  # Negative reward for failure

                buffer_state.append(state)
                buffer_action.append(action)
                buffer_reward.append(reward - 1)  # Adjust reward

                state = next_state
                episode_reward += reward

                GLOBAL_UPDATE_COUNTER += 1  # Count towards minimum batch size
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    if done:
                        value_next_state = 0
                    else:
                        value_next_state = self.ppo.get_value(state)

                    discounted_rewards = []
                    for r in buffer_reward[::-1]:
                        value_next_state = r + GAMMA * value_next_state
                        discounted_rewards.append(value_next_state)
                    discounted_rewards.reverse()

                    states = np.vstack(buffer_state)
                    actions = np.vstack(buffer_action)
                    rewards = np.array(discounted_rewards)[:, None]

                    QUEUE.put(np.hstack((states, actions, rewards)))  # Put data in queue

                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # Stop data collection
                        UPDATE_EVENT.set()  # Trigger PPO update

                    if GLOBAL_EP >= EP_MAX:
                        COORD.request_stop()  # Stop training
                        break

                    if done: 
                        break

            # Record reward changes for later plotting
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(episode_reward)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + episode_reward * 0.1)

            GLOBAL_EP += 1
            print(f'{GLOBAL_EP / EP_MAX * 100:.1f}% | Worker {self.worker_id} | Episode Reward: {episode_reward:.2f}')


def main():
    """Main function to start training and testing."""
    global GLOBAL_PPO, COORD, ROLLING_EVENT, UPDATE_EVENT, GLOBAL_EP, GLOBAL_UPDATE_COUNTER, GLOBAL_RUNNING_R, QUEUE

    # Initialize PPO network and global variables
    GLOBAL_PPO = PPONet()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # No updates initially
    ROLLING_EVENT.set()   # Start collecting data

    workers = [Worker(i) for i in range(N_WORKER)]
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()

    # Start worker threads
    threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work)
        t.start()
        threads.append(t)

    # Start PPO update thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update))
    threads[-1].start()

    COORD.join(threads)

    # Plot reward changes
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.ion()
    plt.show()

    # Test the trained model
    env = gym.make(GAME)
    while True:
        state = env.reset()
        for t in range(1000):
            env.render()
            state, reward, done, _ = env.step(GLOBAL_PPO.choose_action(state))
            if done:
                break


if __name__ == '__main__':
    main()
