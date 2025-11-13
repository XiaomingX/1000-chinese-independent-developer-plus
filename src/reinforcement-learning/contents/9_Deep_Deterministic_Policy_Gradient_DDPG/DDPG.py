import tensorflow as tf
import numpy as np
import gym
import time

# Set seeds for reproducibility
np.random.seed(1)
tf.set_random_seed(1)

#####################  Hyperparameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_ACTOR = 0.001    # Learning rate for actor
LR_CRITIC = 0.001   # Learning rate for critic
GAMMA = 0.9        # Reward discount factor
REPLACEMENT_STRATEGY = dict(name='soft', tau=0.01)  # Target replacement strategy
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'


#################### Actor Class ####################

class Actor:
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement_strategy):
        self.sess = sess
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.replacement_strategy = replacement_strategy
        self.t_replace_counter = 0

        # Build Actor networks
        self.eval_net = self._build_network(scope='eval_net', trainable=True)
        self.target_net = self._build_network(scope='target_net', trainable=False)

        self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        self._set_target_replacement()

    def _build_network(self, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(S, 30, activation=tf.nn.relu, kernel_initializer=init_w,
                                  bias_initializer=init_b, name='l1', trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.action_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')
        return scaled_a

    def _set_target_replacement(self):
        if self.replacement_strategy['name'] == 'hard':
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.target_params, self.eval_params)]
        else:
            self.soft_replace = [
                tf.assign(t, (1 - self.replacement_strategy['tau']) * t + self.replacement_strategy['tau'] * e)
                for t, e in zip(self.target_params, self.eval_params)
            ]

    def learn(self, state):
        self.sess.run(self.train_op, feed_dict={S: state})

        if self.replacement_strategy['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % 600 == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, state):
        state = state[np.newaxis, :]
        return self.sess.run(self.eval_net, feed_dict={S: state})[0]

    def add_grad_to_graph(self, critic_gradients):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.eval_net, xs=self.eval_params, grad_ys=critic_gradients)

        with tf.variable_scope('A_train'):
            optimizer = tf.train.AdamOptimizer(-self.learning_rate)  # Negative sign for gradient ascent
            self.train_op = optimizer.apply_gradients(zip(self.policy_grads, self.eval_params))


#################### Critic Class ####################

class Critic:
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement_strategy, actor_action, target_actor_action):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replacement_strategy = replacement_strategy

        # Build Critic networks
        self.q_eval = self._build_network(scope='eval_net', action=actor_action, trainable=True)
        self.q_target = self._build_network(scope='target_net', action=target_actor_action, trainable=False)

        self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        self._set_target_replacement()

        # Define loss and optimization steps
        self.target_q = R + self.gamma * self.q_target
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q_eval))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.action_gradients = tf.gradients(self.q_eval, self.q_eval)[0]

    def _build_network(self, scope, action, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.action_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(S, w1_s) + tf.matmul(action, w1_a) + b1)

            with tf.variable_scope('q'):
                q_value = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
        return q_value

    def learn(self, state, action, reward, next_state):
        self.sess.run(self.train_op, feed_dict={S: state, self.action: action, R: reward, S_: next_state})

        if self.replacement_strategy['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % 500 == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1


#################### Memory Class ####################

class Memory:
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, action, [reward], next_state))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, batch_size):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=batch_size)
        return self.data[indices, :]


#################### Main Function ####################

def main():
    # Set up the environment and variables
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    # Placeholders for TensorFlow
    with tf.name_scope('S'):
        S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
    with tf.name_scope('R'):
        R = tf.placeholder(tf.float32, [None, 1], name='r')
    with tf.name_scope('S_'):
        S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

    # Create session
    sess = tf.Session()

    # Initialize actor and critic
    actor = Actor(sess, action_dim, action_bound, LR_ACTOR, REPLACEMENT_STRATEGY)
    critic = Critic(sess, state_dim, action_dim, LR_CRITIC, GAMMA, REPLACEMENT_STRATEGY, actor.eval_net, actor.target_net)
    actor.add_grad_to_graph(critic.action_gradients)

    sess.run(tf.global_variables_initializer())

    # Create memory
    memory = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    # Training loop
    exploration_noise = 3  # Control exploration
    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0

        for step in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise to the action
            action = actor.choose_action(state)
            action = np.clip(np.random.normal(action, exploration_noise), -2, 2)  # Add noise for exploration
            next_state, reward, done, info = env.step(action)

            # Store the transition in memory
            memory.store_transition(state, action, reward / 10, next_state)

            # If memory is sufficiently filled, sample and learn
            if memory.pointer > MEMORY_CAPACITY:
                exploration_noise *= 0.9995  # Decay exploration noise
                batch = memory.sample(BATCH_SIZE)
                batch_state = batch[:, :state_dim]
                batch_action = batch[:, state_dim: state_dim + action_dim]
                batch_reward = batch[:, -state_dim - 1: -state_dim]
                batch_next_state = batch[:, -state_dim:]

                critic.learn(batch_state, batch_action, batch_reward, batch_next_state)
                actor.learn(batch_state)

            state = next_state
            episode_reward += reward

            if step == MAX_EP_STEPS - 1:
                print(f'Episode: {episode}, Reward: {int(episode_reward)}, Explore: {exploration_noise:.2f}')
                if episode_reward > -300:
                    RENDER = True
                break

    print(f'Running time: {time.time() - t1}')


if __name__ == '__main__':
    main()
