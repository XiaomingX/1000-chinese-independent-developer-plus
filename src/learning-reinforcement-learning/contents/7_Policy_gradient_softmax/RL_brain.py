import numpy as np
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    """
    Implements a simple Policy Gradient reinforcement learning agent.
    This agent learns by adjusting its policy to maximize the expected cumulative reward.
    """

    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, output_graph=False):
        """
        Initializes the PolicyGradient class.

        :param n_actions: Number of possible actions in the environment
        :param n_features: Number of features in the observation space
        :param learning_rate: Learning rate for training
        :param reward_decay: Discount factor for future rewards
        :param output_graph: If True, it will output a TensorFlow graph for TensorBoard
        """
        # Store the parameters
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        # Initialize lists to store experiences
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # Build the neural network model
        self._build_net()

        # Initialize TensorFlow session
        self.sess = tf.Session()

        # Optionally output graph for visualization in TensorBoard
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        # Initialize all TensorFlow variables
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        """
        Builds the neural network that approximates the policy.
        The network takes the observation as input and outputs action probabilities.
        """

        # Define placeholders for input data
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_taken")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="rewards")

        # Define a fully connected layer (fc1) with tanh activation
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        # Define the output layer (fc2) to produce action probabilities
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,  # No activation here, we will apply softmax later
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # Apply softmax to get probabilities for each action
        self.all_act_prob = tf.nn.softmax(all_act, name='action_probabilities')

        # Define the loss function (negative log-likelihood)
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # Reward-weighted loss

        # Define the optimizer and the training step
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        """
        Choose an action based on the current policy.

        :param observation: The current observation of the environment
        :return: The chosen action
        """
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        """
        Store the transition (state, action, reward) in the episode memory.

        :param s: State (observation)
        :param a: Action taken
        :param r: Reward received
        """
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        """
        Update the policy based on the collected episode data.
        The rewards are discounted and normalized before being used for training.
        """
        # Discount and normalize the episode rewards
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # Train the network on the collected episode data
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs_norm,
        })

        # Clear the episode data for the next iteration
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        """
        Discount and normalize the rewards.

        :return: The discounted and normalized rewards
        """
        # Discount the rewards for each time step in the episode
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # Normalize the discounted rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

def main():
    """
    Main function to run the policy gradient agent.
    """
    # Define the number of actions and features (based on the environment)
    n_actions = 4  # Example: 4 possible actions
    n_features = 4  # Example: 4 features in the observation

    # Create a PolicyGradient agent
    agent = PolicyGradient(n_actions=n_actions, n_features=n_features)

    # Simulate an environment interaction (you can replace this with a Gym environment or your custom environment)
    for episode in range(1000):  # Run for 1000 episodes
        # Reset environment and initialize variables (example)
        observation = np.random.randn(n_features)  # Random observation
        done = False
        total_reward = 0

        # Run one episode
        while not done:
            # Choose an action based on the current observation
            action = agent.choose_action(observation)

            # Take the action and get the next state and reward (you should replace this with environment interaction)
            next_observation = np.random.randn(n_features)  # Random next observation
            reward = np.random.randn()  # Random reward (you should replace with environment's reward)
            done = np.random.rand() > 0.95  # Random done condition (you should replace with environment's done flag)

            # Store the transition
            agent.store_transition(observation, action, reward)

            # Update the total reward
            total_reward += reward

            # Move to the next observation
            observation = next_observation

        # After each episode, the agent learns from the stored transitions
        agent.learn()

        # Print the total reward for this episode
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

if __name__ == '__main__':
    main()
