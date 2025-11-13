import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

# Hyperparameters
EP_MAX = 1000  # Maximum episodes
EP_LEN = 200   # Maximum length of each episode
GAMMA = 0.9    # Discount factor for future rewards
A_LR = 0.0001  # Learning rate for the actor
C_LR = 0.0002  # Learning rate for the critic
BATCH = 32     # Batch size for updating
A_UPDATE_STEPS = 10  # Number of steps to update the actor
C_UPDATE_STEPS = 10  # Number of steps to update the critic
S_DIM, A_DIM = 3, 1  # State and action dimensions
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty method
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective (better method)
][1]  # Choose the 'clip' method for optimization


class PPO(object):
    """
    Proximal Policy Optimization (PPO) Class using TensorFlow.
    """

    def __init__(self):
        # Initialize session and placeholders
        self.sess = tf.Session()
        self.state_placeholder = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # Create the critic network (Value function)
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.state_placeholder, 100, tf.nn.relu)
            self.value = tf.layers.dense(l1, 1)
            self.discounted_reward_placeholder = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.discounted_reward_placeholder - self.value
            self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
            self.critic_train_op = tf.train.AdamOptimizer(C_LR).minimize(self.critic_loss)

        # Create the actor network (Policy)
        pi, pi_params = self._build_network('pi', trainable=True)
        oldpi, oldpi_params = self._build_network('oldpi', trainable=False)
        
        # Sample action from policy
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # Choose action

        # Update old policy network
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        # Placeholder for action and advantage
        self.action_placeholder = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.advantage_placeholder = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # Loss calculation (Surrogate objective)
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.action_placeholder) / (oldpi.prob(self.action_placeholder) + 1e-5)
                surrogate_loss = ratio * self.advantage_placeholder

            # Apply optimization method (KL penalty or clipping)
            if METHOD['name'] == 'kl_pen':
                self.lambda_placeholder = tf.placeholder(tf.float32, None, 'lambda')
                kl_divergence = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl_divergence)
                self.actor_loss = -(tf.reduce_mean(surrogate_loss - self.lambda_placeholder * kl_divergence))
            else:  # Clipping method (recommended)
                self.actor_loss = -tf.reduce_mean(tf.minimum(
                    surrogate_loss,
                    tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.advantage_placeholder))

        # Actor training operation
        with tf.variable_scope('atrain'):
            self.actor_train_op = tf.train.AdamOptimizer(A_LR).minimize(self.actor_loss)

        # Initialize TensorFlow summary for visualization
        tf.summary.FileWriter("log/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def update(self, states, actions, rewards):
        """
        Update both actor and critic networks.
        """
        # Update old policy
        self.sess.run(self.update_oldpi_op)

        # Calculate advantage using the discounted rewards
        advantage = self.sess.run(self.advantage, {self.state_placeholder: states, self.discounted_reward_placeholder: rewards})

        # Update actor network
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run([self.actor_train_op, self.kl_mean],
                                      {self.state_placeholder: states, self.action_placeholder: actions,
                                       self.advantage_placeholder: advantage, self.lambda_placeholder: METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:
                    break

            # Adjust lambda for KL penalty
            if kl < METHOD['kl_target'] / 1.5:
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)

        else:  # Use clipping method
            for _ in range(A_UPDATE_STEPS):
                self.sess.run(self.actor_train_op, {self.state_placeholder: states, self.action_placeholder: actions,
                                                     self.advantage_placeholder: advantage})

        # Update critic network (Value function)
        for _ in range(C_UPDATE_STEPS):
            self.sess.run(self.critic_train_op, {self.state_placeholder: states, self.discounted_reward_placeholder: rewards})

    def _build_network(self, name, trainable):
        """
        Build a neural network for either actor or critic.
        """
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.state_placeholder, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)  # Mean of the action distribution
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)  # Standard deviation
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, state):
        """
        Choose an action based on the current policy.
        """
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.state_placeholder: state})[0]
        return np.clip(action, -2, 2)

    def get_value(self, state):
        """
        Get the value of a given state (critic).
        """
        if state.ndim < 2:
            state = state[np.newaxis, :]
        return self.sess.run(self.value, {self.state_placeholder: state})[0, 0]


def main():
    # Initialize environment and PPO agent
    env = gym.make('Pendulum-v0').unwrapped
    ppo = PPO()

    # Store episode rewards for plotting
    all_episode_rewards = []

    # Run episodes
    for episode in range(EP_MAX):
        state = env.reset()
        buffer_state, buffer_action, buffer_reward = [], [], []
        episode_reward = 0

        for timestep in range(EP_LEN):
            # Render the environment (optional)
            env.render()

            # Choose action using the current policy
            action = ppo.choose_action(state)

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)

            # Normalize the reward
            buffer_state.append(state)
            buffer_action.append(action)
            buffer_reward.append((reward + 8) / 8)
            state = next_state
            episode_reward += reward

            # Update the agent after every batch or at the end of the episode
            if (timestep + 1) % BATCH == 0 or timestep == EP_LEN - 1:
                value_next_state = ppo.get_value(state)
                discounted_rewards = []
                for r in buffer_reward[::-1]:
                    value_next_state = r + GAMMA * value_next_state
                    discounted_rewards.append(value_next_state)
                discounted_rewards.reverse()

                # Update PPO using the stored data
                ppo.update(np.vstack(buffer_state), np.vstack(buffer_action), np.array(discounted_rewards)[:, np.newaxis])
                buffer_state, buffer_action, buffer_reward = [], [], []

        # Keep track of the moving average of episode rewards
        if episode == 0:
            all_episode_rewards.append(episode_reward)
        else:
            all_episode_rewards.append(all_episode_rewards[-1] * 0.9 + episode_reward * 0.1)

        # Print progress
        print(f"Episode {episode}, Reward: {episode_reward}, Lambda: {METHOD['lam']}")

    # Plot the moving average of episode rewards
    plt.plot(np.arange(len(all_episode_rewards)), all_episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Moving average of episode reward')
    plt.show()


# Run the main function
if __name__ == '__main__':
    main()
