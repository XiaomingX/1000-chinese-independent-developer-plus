import numpy as np
import pandas as pd

class ReinforcementLearningAgent:
    """
    Base class for the Reinforcement Learning agent.
    Handles the Q-table, action selection, and state checking.
    """
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, epsilon=0.9):
        """
        Initialize the agent with given parameters.
        
        :param action_space: List of available actions.
        :param learning_rate: Learning rate for Q-table updates.
        :param reward_decay: Discount factor for future rewards.
        :param epsilon: Probability of choosing the best action (epsilon-greedy).
        """
        self.actions = action_space  # List of available actions
        self.lr = learning_rate  # Learning rate for updating Q-values
        self.gamma = reward_decay  # Discount factor for future rewards
        self.epsilon = epsilon  # Epsilon for epsilon-greedy action selection

        # Initialize Q-table as a DataFrame with actions as columns
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exists(self, state):
        """
        Check if the state exists in the Q-table. If not, add it with initial Q-values (0).
        
        :param state: The state to check or add to the Q-table.
        """
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)
            )

    def choose_action(self, state):
        """
        Choose an action based on epsilon-greedy strategy.
        
        :param state: The current state from which the action is chosen.
        :return: The chosen action.
        """
        self.check_state_exists(state)

        if np.random.rand() < self.epsilon:
            # Epsilon-greedy: choose the best action based on the Q-table
            state_action = self.q_table.loc[state, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # Choose a random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        """
        Placeholder method for learning, to be implemented in child classes.
        """
        pass


class QLearningAgent(ReinforcementLearningAgent):
    """
    Q-learning algorithm (off-policy).
    """
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, epsilon=0.9):
        super().__init__(actions, learning_rate, reward_decay, epsilon)

    def learn(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning algorithm.
        
        :param state: The current state.
        :param action: The action taken in the current state.
        :param reward: The reward received after taking the action.
        :param next_state: The next state after the action is taken.
        """
        self.check_state_exists(next_state)
        
        q_predict = self.q_table.loc[state, action]
        if next_state != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward  # Terminal state, no future rewards
        
        # Update Q-value for the current state-action pair
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)


class SarsaAgent(ReinforcementLearningAgent):
    """
    SARSA algorithm (on-policy).
    """
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, epsilon=0.9):
        super().__init__(actions, learning_rate, reward_decay, epsilon)

    def learn(self, state, action, reward, next_state, next_action):
        """
        Update the Q-table using the SARSA algorithm.
        
        :param state: The current state.
        :param action: The action taken in the current state.
        :param reward: The reward received after taking the action.
        :param next_state: The next state after the action is taken.
        :param next_action: The action taken in the next state.
        """
        self.check_state_exists(next_state)
        
        q_predict = self.q_table.loc[state, action]
        if next_state != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[next_state, next_action]
        else:
            q_target = reward  # Terminal state, no future rewards
        
        # Update Q-value for the current state-action pair
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)


def main():
    # Define the action space and other parameters
    actions = ['left', 'right', 'up', 'down']
    learning_rate = 0.1
    reward_decay = 0.9
    epsilon = 0.8

    # Initialize Q-learning agent and SARSA agent
    q_learning_agent = QLearningAgent(actions, learning_rate, reward_decay, epsilon)
    sarsa_agent = SarsaAgent(actions, learning_rate, reward_decay, epsilon)

    # Example of training the agents in a simple environment (states are arbitrary here)
    states = ['state1', 'state2', 'state3']
    for episode in range(100):  # Run 100 episodes of training
        state = np.random.choice(states)
        action = q_learning_agent.choose_action(state)
        reward = np.random.random()  # Random reward
        next_state = np.random.choice(states)

        # Q-learning update
        q_learning_agent.learn(state, action, reward, next_state)

        # SARSA update (using SARSA agent)
        next_action = sarsa_agent.choose_action(next_state)
        sarsa_agent.learn(state, action, reward, next_state, next_action)

    # Print Q-tables after training
    print("Q-learning Q-table:")
    print(q_learning_agent.q_table)
    print("\nSARSA Q-table:")
    print(sarsa_agent.q_table)


if __name__ == "__main__":
    main()
