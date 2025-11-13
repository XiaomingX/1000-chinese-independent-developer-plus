import numpy as np
import tensorflow as tf
import gym

# Set random seeds for reproducibility
np.random.seed(2)
tf.set_random_seed(2)

# Hyperparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # Render if the reward exceeds this threshold
MAX_EP_STEPS = 1000   # Max steps in one episode
RENDER = False        # Disable rendering by default
GAMMA = 0.9           # Discount factor for TD error
LR_A = 0.001          # Learning rate for actor
LR_C = 0.01           # Learning rate for critic

# Initialize environment
env = gym.make('CartPole-v0')
env.seed(1)  # Set the seed for reproducibility
env = env.unwrapped

N_F = env.observation_space.shape[0]  # Number of features (state space size)
N_A = env.action_space.n  # Number of actions


class Actor:
    """
    The Actor class represents the policy network.
    It takes in a state and outputs a probability distribution over possible actions.
    It learns by adjusting the policy to maximize expected reward using the TD-error.
    """
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        # Placeholders for input state, chosen action, and TD error
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "action")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # Temporal Difference error

        with tf.variable_scope('Actor'):
            # Hidden layer
            l1 = tf.layers.dense(self.s, 20, activation=tf.nn.relu, 
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1), name='l1')

            # Output layer - action probabilities
            self.acts_prob = tf.layers.dense(l1, n_actions, activation=tf.nn.softmax, 
                                             kernel_initializer=tf.random_normal_initializer(0., .1),
                                             bias_initializer=tf.constant_initializer(0.1), name='acts_prob')

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])  # Log of chosen action's probability
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # Advantage-guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # Maximize exp_v by minimizing its negative

    def learn(self, s, a, td):
        """
        Perform one step of learning by updating the policy (actor).
        """
        s = s[np.newaxis, :]  # Add batch dimension
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        """
        Choose an action based on the current state and the learned policy.
        """
        s = s[np.newaxis, :]  # Add batch dimension
        probs = self.sess.run(self.acts_prob, {self.s: s})  # Get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # Sample action


class Critic:
    """
    The Critic class represents the value network.
    It takes in a state and estimates the expected future reward (value).
    It learns by minimizing the TD error.
    """
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        # Placeholders for state, next state's value, and reward
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")  # Value of the next state
        self.r = tf.placeholder(tf.float32, None, 'reward')  # Reward

        with tf.variable_scope('Critic'):
            # Hidden layer
            l1 = tf.layers.dense(self.s, 20, activation=tf.nn.relu, 
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1), name='l1')

            # Output layer - value estimation
            self.v = tf.layers.dense(l1, 1, activation=None, 
                                     kernel_initializer=tf.random_normal_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(0.1), name='V')

        with tf.variable_scope('squared_TD_error'):
            # TD error: difference between actual reward + discounted next value and predicted value
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD error squared loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        """
        Perform one step of learning by updating the value function (critic).
        """
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]  # Add batch dimension
        v_ = self.sess.run(self.v, {self.s: s_})  # Get value of next state
        td_error, _ = self.sess.run([self.td_error, self.train_op], 
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


def main():
    # Create TensorFlow session
    sess = tf.Session()

    # Initialize actor and critic
    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=N_F, lr=LR_C)

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    # Start training episodes
    for i_episode in range(MAX_EPISODE):
        s = env.reset()  # Reset environment to initial state
        track_r = []  # Track rewards for each episode
        t = 0  # Time step counter

        while True:
            if RENDER: 
                env.render()  # Render environment if enabled

            # Actor chooses an action based on the current state
            a = actor.choose_action(s)

            # Take action in the environment
            s_, r, done, info = env.step(a)

            if done: 
                r = -20  # Negative reward for termination (failure)

            track_r.append(r)

            # Critic learns from the experience and computes TD error
            td_error = critic.learn(s, r, s_)

            # Actor learns from the TD error (advantage-guided)
            actor.learn(s, a, td_error)

            s = s_  # Update state
            t += 1  # Increment time step

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)  # Sum of rewards for the episode

                # Update running reward for smoothing
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                # Render the environment if the reward is above the threshold
                if running_reward > DISPLAY_REWARD_THRESHOLD: 
                    RENDER = True

                print(f"Episode {i_episode}, Reward: {int(running_reward)}")
                break


if __name__ == "__main__":
    main()
