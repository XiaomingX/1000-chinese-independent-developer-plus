import tensorflow as tf
import numpy as np
import gym

# Set random seeds for reproducibility
np.random.seed(2)
tf.set_random_seed(2)

# Hyperparameters
MAX_EPISODE = 1000
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = -100  # Renders environment if total episode reward exceeds this threshold
RENDER = False  # Whether to render environment
GAMMA = 0.9  # Discount factor for future rewards
LR_A = 0.001  # Learning rate for the Actor
LR_C = 0.01  # Learning rate for the Critic

class Actor:
    def __init__(self, sess, n_features, action_bound, lr=0.0001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="action")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # Temporal difference error

        # Define the neural network layers
        l1 = tf.layers.dense(self.s, 30, activation=tf.nn.relu, name="l1")
        mu = tf.layers.dense(l1, 1, activation=tf.nn.tanh, name="mu")  # Mean of the action distribution
        sigma = tf.layers.dense(l1, 1, activation=tf.nn.softplus, name="sigma")  # Standard deviation of action distribution

        self.mu, self.sigma = tf.squeeze(mu * 2), tf.squeeze(sigma + 0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        # Action is sampled from a normal distribution with mean and stddev
        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])

        # Loss function with TD-error and entropy (for exploration)
        log_prob = self.normal_dist.log_prob(self.a)
        self.exp_v = log_prob * self.td_error  # Advantage-guided loss
        self.exp_v += 0.01 * self.normal_dist.entropy()  # Add entropy for exploration

        # Optimizer for training
        self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td_error):
        s = s[np.newaxis, :]  # Add batch dimension
        feed_dict = {self.s: s, self.a: a, self.td_error: td_error}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})


class Critic:
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")  # Value of next state
        self.r = tf.placeholder(tf.float32, name='reward')  # Reward

        # Define the neural network layers
        l1 = tf.layers.dense(self.s, 30, activation=tf.nn.relu, name="l1")
        self.v = tf.layers.dense(l1, 1, activation=None, name="value")

        # Temporal difference error (TD-error)
        self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_ - self.v)
        self.loss = tf.square(self.td_error)  # MSE loss for TD-error

        # Optimizer for training
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})  # Estimate value of the next state
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})
        return td_error


def main():
    # Set up the environment and the neural network models
    env = gym.make('Pendulum-v0')
    env.seed(1)
    env = env.unwrapped

    N_S = env.observation_space.shape[0]  # Number of state features
    A_BOUND = env.action_space.high  # Action boundaries (for continuous action space)

    # Initialize the TensorFlow session
    sess = tf.Session()

    # Instantiate Actor and Critic models
    actor = Actor(sess, n_features=N_S, action_bound=[-A_BOUND, A_BOUND], lr=LR_A)
    critic = Critic(sess, n_features=N_S, lr=LR_C)

    sess.run(tf.global_variables_initializer())

    # Training loop
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        ep_rs = []  # List to store rewards in this episode

        while True:
            # Optionally render the environment (can be disabled for speed)
            if RENDER:
                env.render()

            # Choose action based on current state using the Actor model
            a = actor.choose_action(s)

            # Interact with the environment
            s_, r, done, info = env.step(a)
            r /= 10  # Scale the reward

            # Learn from the Critic (TD error)
            td_error = critic.learn(s, r, s_)

            # Learn from the Actor (policy gradient with TD error)
            actor.learn(s, a, td_error)

            # Update the state and track rewards
            s = s_
            t += 1
            ep_rs.append(r)

            # End episode if maximum steps are reached or done flag is triggered
            if t > MAX_EP_STEPS:
                ep_rs_sum = sum(ep_rs)
                running_reward = ep_rs_sum if 'running_reward' not in globals() else running_reward * 0.9 + ep_rs_sum * 0.1
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True  # Start rendering if reward exceeds threshold
                print(f"Episode: {i_episode}, Reward: {int(running_reward)}")
                break


if __name__ == '__main__':
    main()
