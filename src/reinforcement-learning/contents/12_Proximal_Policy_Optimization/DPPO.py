import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import threading
import queue

# Hyperparameters
EP_MAX = 1000
EP_LEN = 200
N_WORKER = 4  # number of parallel workers
GAMMA = 0.9  # reward discount factor
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
MIN_BATCH_SIZE = 64  # minimum batch size for PPO update
UPDATE_STEP = 10  # steps to update PPO
EPSILON = 0.2  # clipping for surrogate objective
GAME = 'Pendulum-v0'
S_DIM, A_DIM = 3, 1  # state and action dimensions

# Global variables
GLOBAL_EP = 0
GLOBAL_UPDATE_COUNTER = 0
GLOBAL_RUNNING_R = []


class PPO:
    def __init__(self):
        """Initialize PPO model with actor and critic networks."""
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # Critic: value network
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # Actor: policy network
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # Choose action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # Compute surrogate loss
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv
        self.aloss = -tf.reduce_mean(tf.minimum(
            surr, tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def _build_anet(self, name, trainable):
        """Build actor network (policy) for training."""
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def update(self):
        """Update the actor and critic networks periodically."""
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # Wait for data collection
                self.sess.run(self.update_oldpi_op)  # Update old policy
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # Collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # Update actor and critic
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()  # Update finished
                GLOBAL_UPDATE_COUNTER = 0  # Reset counter
                ROLLING_EVENT.set()  # Resume data collection

    def choose_action(self, s):
        """Select action based on current policy."""
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        """Get value of state."""
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker:
    def __init__(self, wid):
        """Initialize worker with environment and PPO model."""
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        """Worker function to interact with environment and collect data."""
        global GLOBAL_EP, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():
                    ROLLING_EVENT.wait()  # Wait for PPO update
                    buffer_s, buffer_a, buffer_r = [], [], []
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # Normalize reward
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))

                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()
                        UPDATE_EVENT.set()  # Trigger global PPO update

                    if GLOBAL_EP >= EP_MAX:
                        COORD.request_stop()
                        break

            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print(f'{GLOBAL_EP / EP_MAX * 100:.1f}% | Worker {self.wid} | Ep_r: {ep_r:.2f}')


def main():
    """Main function to initialize and run PPO with workers."""
    global GLOBAL_PPO, GLOBAL_UPDATE_COUNTER, GLOBAL_EP

    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # Not updating yet
    ROLLING_EVENT.set()  # Start rolling out

    workers = [Worker(wid=i) for i in range(N_WORKER)]
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []

    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()  # Queue for workers to put data
    threads = []

    for worker in workers:
        t = threading.Thread(target=worker.work, args=())
        t.start()
        threads.append(t)

    # Add PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update))
    threads[-1].start()

    # Wait for all threads to finish
    COORD.join(threads)

    # Plot reward changes
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.ion()
    plt.show()

    # Test trained policy
    env = gym.make(GAME)
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]


if __name__ == '__main__':
    main()
