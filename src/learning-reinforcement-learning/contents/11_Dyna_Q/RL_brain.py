import numpy as np
import pandas as pd

class QLearningTable:
    """
    Q-learning table for the agent, which helps the agent choose actions 
    and learn from the environment based on Q-learning algorithm.
    """
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """
        Initialize Q-learning parameters and Q-table.
        
        :param actions: List of possible actions the agent can take.
        :param learning_rate: The learning rate for updating the Q-values.
        :param reward_decay: Discount factor for future rewards.
        :param e_greedy: Probability of selecting the best action (epsilon-greedy policy).
        """
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        # Initialize an empty Q-table with columns for each action
        self.q_table = pd.DataFrame(columns=self.actions).astype('float32')

    def choose_action(self, observation):
        """
        Choose an action based on the current observation using epsilon-greedy strategy.
        
        :param observation: Current state the agent is in.
        :return: Chosen action.
        """
        self._check_state_exist(observation)

        # Epsilon-greedy action selection
        if np.random.uniform() < self.epsilon:
            # Choose the best action based on Q-values
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # Handle ties
            action = state_action.argmax()  # Choose action with max Q-value
        else:
            # Choose a random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        """
        Update the Q-table based on the agent's experience.
        
        :param s: Current state.
        :param a: Action taken.
        :param r: Reward received.
        :param s_: Next state.
        """
        self._check_state_exist(s_)

        # Predict Q-value for the current state-action pair
        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            # If next state is not terminal, use Bellman equation to update Q-value
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            # If next state is terminal, no future reward to consider
            q_target = r

        # Update the Q-value for the current state-action pair
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def _check_state_exist(self, state):
        """
        Check if the state exists in the Q-table. If not, add it with initial Q-values.
        
        :param state: State to check.
        """
        if state not in self.q_table.index:
            # Add a new state to the Q-table with initial Q-values (0 for all actions)
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            )


class EnvModel:
    """
    Environment model stores past transitions and can generate next state and reward signal.
    Similar to the experience replay in DQN.
    """
    def __init__(self, actions):
        """
        Initialize the environment model.
        
        :param actions: List of possible actions the agent can take.
        """
        self.actions = actions
        # Database to store state-action transitions and their corresponding rewards and next states
        self.database = pd.DataFrame(columns=actions, dtype=np.object)

    def store_transition(self, s, a, r, s_):
        """
        Store a state-action-reward-next_state tuple in the environment model.
        
        :param s: Current state.
        :param a: Action taken.
        :param r: Reward received.
        :param s_: Next state.
        """
        if s not in self.database.index:
            self.database = self.database.append(
                pd.Series([None] * len(self.actions), index=self.database.columns, name=s)
            )
        # Store the reward and next state for the given state-action pair
        self.database.loc[s, a] = (r, s_)

    def sample_s_a(self):
        """
        Sample a random state-action pair from the database.
        
        :return: A random state and a random action.
        """
        s = np.random.choice(self.database.index)
        a = np.random.choice(self.database.loc[s].dropna().index)  # Filter out None values
        return s, a

    def get_r_s_(self, s, a):
        """
        Get the reward and next state for a given state-action pair.
        
        :param s: State.
        :param a: Action.
        :return: A tuple containing reward and next state.
        """
        return self.database.loc[s, a]

def main():
    """
    Main function to tie everything together: Initialize Q-learning and environment model,
    simulate agent-environment interaction, and learn from experience.
    """
    actions = ['left', 'right', 'up', 'down']  # Example actions
    agent = QLearningTable(actions)
    env_model = EnvModel(actions)

    # Example simulation loop (this would be inside a real environment interaction)
    for episode in range(100):  # Number of episodes
        state = 'start'  # Starting state
        total_reward = 0

        while state != 'terminal':
            action = agent.choose_action(state)  # Choose an action based on the current state
            reward = np.random.random()  # Simulate receiving a reward (replace with real reward)
            next_state = 'terminal' if np.random.rand() < 0.1 else 'next_state'  # Simulate next state transition

            # Store the transition in the environment model
            env_model.store_transition(state, action, reward, next_state)

            # Learn from the experience
            agent.learn(state, action, reward, next_state)

            # Move to the next state
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
