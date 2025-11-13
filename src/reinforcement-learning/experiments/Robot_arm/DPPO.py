import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import threading
import queue
from arm_env import ArmEnv

# Constants
EP_MAX = 2000           # Maximum number of episodes
EP_LEN = 300            # Length of each episode
N_WORKERS = 4           # Number of parallel workers
GAMMA = 0.9            # Discount factor for rewards
A_LR = 0.0001          # Learning rate for actor
C_LR = 0.0005          # Learning rate for critic
MIN_BATCH_SIZE = 64    # Minimum batch size for PPO update
UPDATE_STEP = 5        # Number of steps to update PPO
EPSILON = 0.2          # Clipped surrogate objective for PPO
MODE = ['easy', 'hard'] # Mode options for the environment
n_model = 1            # Select model type

# Initialize environment and dimensions
env = ArmEnv(mode=MODE[n_model])
S_DIM = env.state_dim
A_DIM = env.action_dim
A_BOUND = env.action_bound[1]

class PPO(object):
    """Proximal Policy Optimization (PPO) Agent"""
    def __init__(self):
        self.sess = tf.Session()

        # Placeholder for state input
        self.state_input = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # Critic: Estimate state value function
        critic_hidden = tf.layers.dense(self.state_input, 100, tf.nn.relu)
        self.state_value = tf.layers.dense(critic_hidden, 1)
        self.discounted_reward = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.discounted_reward - self.state_value
        self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
        self.critic_train_op = tf.train.AdamOptimizer(C_LR).minimize(self.critic_loss)

        # Actor: Policy network (stochastic)
        self.policy, self.policy_params = self._build_policy_network('policy', trainable=True)
        self.old_policy, self.old_policy_params = self._build_policy_network('old_policy', trainable=False)

        # Sampling and updating policy
        self.sample_action = tf.squeeze(self.policy.sample(1), axis=0)
        self.update_old_policy_op = [old.assign(new) for old, new in zip(self.old_policy_params, self.policy_params)]

        # Placeholders for action and advantage (for policy gradient)
        self.action_input = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.advantage_input = tf.placeholder(tf.float32, [None, 1], 'advantage')
        
        # Calculate probability ratio for PPO objective
        ratio = self.policy.prob(self.action_input) / (self.old_policy.prob(self.action_input) + 1e-5)
        surrogate_loss = ratio * self.advantage_input
        self.actor_loss = -tf.reduce_mean(tf.minimum(surrogate_loss, tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.advantage_input))

        # Training operations for actor
        self.actor_train_op = tf.train.AdamOptimizer(A_LR).minimize(self.actor_loss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        """Update the global PPO model using collected data from workers"""
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # Wait until data batch is collected
                self.sess.run(self.update_old_policy_op)  # Update old policy
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # Collect all data
                data = np.vstack(data)
                states, actions, rewards = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                advantages = self.sess.run(self.advantage, {self.state_input: states, self.discounted_reward: rewards})

                # Train actor and critic
                for _ in range(UPDATE_STEP):
                    self.sess.run(self.actor_train_op, {self.state_input: states, self.action_input: actions, self.advantage_input: advantages})
                    self.sess.run(self.critic_train_op, {self.state_input: states, self.discounted_reward: rewards})

                UPDATE_EVENT.clear()  # Update finished
                GLOBAL_UPDATE_COUNTER = 0  # Reset counter
                ROLLING_EVENT.set()  # Allow new data collection

    def _build_policy_network(self, name, trainable):
        """Build a policy network (actor) with a mean and standard deviation for actions"""
        with tf.variable_scope(name):
            hidden = tf.layers.dense(self.state_input, 200, tf.nn.relu, trainable=trainable)
            mean = A_BOUND * tf.layers.dense(hidden, A_DIM, tf.nn.tanh, trainable=trainable)
            stddev = tf.layers.dense(hidden, A_DIM, tf.nn.softplus, trainable=trainable)
            dist = Normal(loc=mean, scale=stddev)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, params

    def choose_action(self, state):
        """Choose an action based on the current state"""
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_action, {self.state_input: state})[0]
        return np.clip(action, -2, 2)

    def get_value(self, state):
        """Get the value of a state (critic estimate)"""
        if state.ndim < 2: state = state[np.newaxis, :]
        return self.sess.run(self.state_value, {self.state_input: state})[0, 0]


class Worker(object):
    """Worker that interacts with the environment and collects data"""
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.env = ArmEnv(mode=MODE[n_model])
        self.ppo = GLOBAL_PPO

    def work(self):
        """Worker's main loop to interact with environment and collect data"""
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            state = self.env.reset()
            episode_reward = 0
            buffer_state, buffer_action, buffer_reward = [], [], []

            for timestep in range(EP_LEN):
                if not ROLLING_EVENT.is_set():
                    ROLLING_EVENT.wait()  # Wait for PPO update
                    buffer_state, buffer_action, buffer_reward = [], [], []  # Reset buffer
                
                action = self.ppo.choose_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Collect experience
                buffer_state.append(state)
                buffer_action.append(action)
                buffer_reward.append(reward)

                state = next_state
                episode_reward += reward
                GLOBAL_UPDATE_COUNTER += 1

                if timestep == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    value_next_state = self.ppo.get_value(next_state)
                    discounted_rewards = []

                    # Compute discounted rewards
                    for r in buffer_reward[::-1]:
                        value_next_state = r + GAMMA * value_next_state
                        discounted_rewards.append(value_next_state)
                    discounted_rewards.reverse()

                    # Prepare data for update
                    states, actions, rewards = np.vstack(buffer_state), np.vstack(buffer_action), np.array(discounted_rewards)[:, np.newaxis]
                    QUEUE.put(np.hstack((states, actions, rewards)))

                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # Stop rolling out
                        UPDATE_EVENT.set()     # Trigger PPO update

                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if GLOBAL_EP >= EP_MAX:
                        COORD.request_stop()
                        break

            # Record reward changes for plotting
            GLOBAL_RUNNING_R.append(episode_reward) if len(GLOBAL_RUNNING_R) == 0 else GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + episode_reward * 0.1)
            GLOBAL_EP += 1
            print(f'{GLOBAL_EP / EP_MAX * 100:.1f}% | Worker {self.worker_id} | Episode Reward: {episode_reward:.2f}')


def main():
    """Main function to initialize PPO, workers, and training process"""
    global GLOBAL_PPO, UPDATE_EVENT, ROLLING_EVENT, COORD, QUEUE
    GLOBAL_PPO = PPO()  # Initialize global PPO agent
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # No update initially
    ROLLING_EVENT.set()   # Start collecting data

    # Initialize workers
    workers = [Worker(worker_id=i) for i in range(N_WORKERS)]
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []  # Running rewards
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()

    # Start worker threads
    threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work)
        t.start()
        threads.append(t)

    # Add PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update))
    threads[-1].start()

    # Wait for threads to finish
    COORD.join(threads)

    # Plot reward progress
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.ion()
    plt.show()

    # Test the trained model
    env.set_fps(30)
    while True:
        state = env.reset()
        for _ in range(400):
            env.render()
            state = env.step(GLOBAL_PPO.choose_action(state))[0]

if __name__ == '__main__':
    main()
