import tensorflow as tf
import numpy as np
import gym
import time


##################### Hyperparameters #####################

MAX_EPISODES = 200      # Maximum number of episodes
MAX_EP_STEPS = 200      # Maximum steps per episode
LR_A = 0.001            # Learning rate for the actor
LR_C = 0.002            # Learning rate for the critic
GAMMA = 0.9             # Discount factor for future rewards
TAU = 0.01              # Soft replacement for target network
MEMORY_CAPACITY = 10000 # Maximum memory capacity
BATCH_SIZE = 32         # Batch size for training

RENDER = False          # Whether to render the environment or not
ENV_NAME = 'Pendulum-v0'  # The environment to train on


############################### DDPG #####################################

class DDPG:
    def __init__(self, a_dim, s_dim, a_bound):
        # Initialize memory, session, and environment parameters
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')   # State placeholder
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_') # Next state placeholder
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')      # Reward placeholder

        # Build Actor and Critic networks
        self.a = self._build_actor(self.S)
        q = self._build_critic(self.S, self.a)
        
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # Soft target network update
        
        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))
        
        target_update = [ema.apply(a_params), ema.apply(c_params)]  # Soft update of target parameters
        a_ = self._build_actor(self.S_, reuse=True, custom_getter=ema_getter)
        q_ = self._build_critic(self.S_, a_, reuse=True, custom_getter=ema_getter)

        # Actor loss (maximize Q-value)
        a_loss = - tf.reduce_mean(q)  
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        # Critic loss (TD error)
        with tf.control_dependencies(target_update):
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        # Choose action based on state s
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # Sample a batch of experiences from memory and learn
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        # Store a transition in memory
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # Replace old memory if necessary
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_actor(self, s, reuse=None, custom_getter=None):
        # Build the Actor network
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1')
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a')
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_critic(self, s, a, reuse=None, custom_getter=None):
        # Build the Critic network
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1])
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1])
            b1 = tf.get_variable('b1', [1, n_l1])
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1)  # Q-value


############################### Training Loop #####################################

def train_ddpg():
    # Initialize environment and agent
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)

    var = 3  # Exploration noise
    t1 = time.time()
    
    # Training loop
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Select action with noise for exploration
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)  # Add noise to action

            # Take a step in the environment
            s_, r, done, info = env.step(a)

            # Store the transition in memory
            ddpg.store_transition(s, a, r / 10, s_)

            # Learn if enough transitions are stored
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= 0.9995  # Decay exploration noise
                ddpg.learn()

            s = s_
            ep_reward += r

            # Print episode info
            if j == MAX_EP_STEPS - 1:
                print(f'Episode: {i}, Reward: {int(ep_reward)}, Exploration: {var:.2f}')
                break

    print(f'Training finished in {time.time() - t1:.2f} seconds')


############################### Main Function #####################################

if __name__ == '__main__':
    train_ddpg()
