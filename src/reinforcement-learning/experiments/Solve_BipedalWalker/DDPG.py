import tensorflow as tf
import numpy as np
import gym
import os
import shutil

# Constants
MAX_EPISODES = 2000
LR_A = 0.0005  # learning rate for actor
LR_C = 0.0005  # learning rate for critic
GAMMA = 0.999  # reward discount factor
REPLACE_ITER_A = 1700  # update interval for actor network
REPLACE_ITER_C = 1500  # update interval for critic network
MEMORY_CAPACITY = 200000  # memory capacity for experience replay
BATCH_SIZE = 32  # batch size for training
DISPLAY_THRESHOLD = 100  # threshold for enabling environment rendering
DATA_PATH = './data'
LOAD_MODEL = False
SAVE_MODEL_ITER = 100000
RENDER = False
OUTPUT_GRAPH = False
ENV_NAME = 'BipedalWalker-v2'

# TensorFlow session initialization
sess = tf.Session()

# Set random seeds for reproducibility
np.random.seed(1)
tf.set_random_seed(1)

# Environment setup
env = gym.make(ENV_NAME)
env.seed(1)

STATE_DIM = env.observation_space.shape[0]  # state dimension (24)
ACTION_DIM = env.action_space.shape[0]  # action dimension (4)
ACTION_BOUND = env.action_space.high  # action bounds ([1, 1, 1, 1])

# Global step for learning rate decay
GLOBAL_STEP = tf.Variable(0, trainable=False)
INCREASE_GS = GLOBAL_STEP.assign(tf.add(GLOBAL_STEP, 1))
LR_A = tf.train.exponential_decay(LR_A, GLOBAL_STEP, 10000, 0.97, staircase=True)
LR_C = tf.train.exponential_decay(LR_C, GLOBAL_STEP, 10000, 0.97, staircase=True)

# Define placeholders for state, reward, and next state
S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='state')
R = tf.placeholder(tf.float32, [None, 1], name='reward')
S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='next_state')


# Actor class definition
class Actor:
    def __init__(self, sess, action_dim, action_bound, learning_rate, replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replace_iter = replace_iter
        self.replace_counter = 0

        with tf.variable_scope('Actor'):
            # Evaluation network (outputs action given state)
            self.a = self.build_network(S, scope='eval_net', trainable=True)
            # Target network (used for actor target actions)
            self.a_ = self.build_network(S_, scope='target_net', trainable=False)

        # Variables for updating networks
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def build_network(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # Initialize weights and biases
            init_w = tf.random_normal_initializer(0., 0.01)
            init_b = tf.constant_initializer(0.01)
            # Build the network with two dense layers
            net = tf.layers.dense(s, 500, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w, bias_initializer=init_b)
                scaled_actions = tf.multiply(actions, self.action_bound)  # Scale the output to within action bounds
        return scaled_actions

    def learn(self, s):
        """Batch update for actor network"""
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.replace_counter % self.replace_iter == 0:
            # Update target network with evaluation network weights
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.replace_counter += 1

    def choose_action(self, s):
        """Choose an action given a state"""
        s = s[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={S: s})[0]

    def add_grad_to_graph(self, a_grads):
        """Add gradient calculations for policy"""
        with tf.variable_scope('policy_grads'):
            self.policy_grads_and_vars = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            # Use RMSProp optimizer for actor update
            opt = tf.train.RMSPropOptimizer(-self.lr)  # Negative learning rate for gradient ascent
            self.train_op = opt.apply_gradients(zip(self.policy_grads_and_vars, self.e_params), global_step=GLOBAL_STEP)


# Critic class definition
class Critic:
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replace_iter, actor_a, actor_a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replace_iter = replace_iter
        self.replace_counter = 0

        with tf.variable_scope('Critic'):
            # Build evaluation network for Q-values
            self.q = self.build_network(S, actor_a, 'eval_net', trainable=True)
            # Build target network for Q-values
            self.q_ = self.build_network(S_, actor_a_, 'target_net', trainable=False)

            # Variables for target and evaluation networks
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            # Target Q-values for learning
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('abs_TD'):
            # Temporal difference error
            self.abs_td = tf.abs(self.target_q - self.q)

        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        with tf.variable_scope('TD_error'):
            # Loss function: mean squared error between target Q and predicted Q
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            # Optimizer for critic network
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=GLOBAL_STEP)

        with tf.variable_scope('a_grad'):
            # Compute gradients for actions
            self.a_grads = tf.gradients(self.q, actor_a)[0]

    def build_network(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            # Initialize weights and biases
            init_w = tf.random_normal_initializer(0., 0.01)
            init_b = tf.constant_initializer(0.01)

            # Combine state and action inputs
            w1_s = tf.get_variable('w1_s', [self.s_dim, 700], initializer=init_w, trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, 700], initializer=init_w, trainable=trainable)
            b1 = tf.get_variable('b1', [1, 700], initializer=init_b, trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            # Additional dense layer
            net = tf.layers.dense(net, 20, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b)

            # Output Q-value
            q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
        return q

    def learn(self, s, a, r, s_, ISW):
        """Batch update for critic network"""
        _, abs_td = self.sess.run([self.train_op, self.abs_td], feed_dict={S: s, a: a, R: r, S_: s_, self.ISWeights: ISW})
        if self.replace_counter % self.replace_iter == 0:
            # Update target network with evaluation network weights
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.replace_counter += 1
        return abs_td


# Main function to initialize and run the DDPG algorithm
def main():
    # Create actor and critic networks
    actor = Actor(sess, ACTION_DIM, ACTION_BOUND, LR_A, REPLACE_ITER_A)
    critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)

    # Set up memory for experience replay
    M = Memory(MEMORY_CAPACITY)

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    total_reward_list = []

    for episode in range(MAX_EPISODES):
        # Reset environment and initialize state
        s = env.reset()
        ep_reward = 0

        while True:
            if episode > DISPLAY_THRESHOLD:
                RENDER = True
            else:
                RENDER = False

            if RENDER:
                env.render()

            # Actor chooses action
            a = actor.choose_action(s)

            # Interact with the environment and get the next state and reward
            s_, r, done, _ = env.step(a)
            ep_reward += r

            # Save experience to memory
            M.store_transition(s, a, r, s_)

            # Learn from memory if it's time
            if M.size > BATCH_SIZE:
                s_batch, a_batch, r_batch, s_batch_ = M.sample(BATCH_SIZE)
                ISW = np.ones_like(r_batch)
                td_error = critic.learn(s_batch, a_batch, r_batch, s_batch_, ISW)
                actor.learn(s_batch)

            # If the episode is done
            if done:
                print(f'Episode: {episode}, Reward: {ep_reward}')
                total_reward_list.append(ep_reward)
                break

        # Save model periodically
        if episode % SAVE_MODEL_ITER == 0:
            actor.save_model(f'{DATA_PATH}/actor_{episode}')
            critic.save_model(f'{DATA_PATH}/critic_{episode}')

    env.close()


if __name__ == '__main__':
    main()
