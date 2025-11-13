import numpy as np
import pandas as pd


class QLearningAgent:
    """
    This class represents the Q-learning agent, which makes decisions based on the Q-table.
    The agent updates the Q-table during learning by interacting with the environment.
    """

    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """
        Initialize the Q-learning agent with the necessary parameters.

        :param action_space: List of available actions.
        :param learning_rate: Rate at which the agent learns.
        :param reward_decay: Discount factor for future rewards.
        :param e_greedy: Probability of choosing the best action (vs. random action).
        """
        self.actions = action_space
        self.lr = learning_rate  # Learning rate
        self.gamma = reward_decay  # Discount factor
        self.epsilon = e_greedy  # Epsilon-greedy strategy for action selection

        # Initialize Q-table with zeros, rows are states and columns are actions
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        """
        Ensure the state exists in the Q-table. If not, add it with initial Q-values.

        :param state: The state to check and possibly add to the Q-table.
        """
        if state not in self.q_table.index:
            # Add new state with initial Q-values (0 for all actions)
            self.q_table.loc[state] = [0] * len(self.actions)

    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy strategy.

        :param state: Current state to base the action on.
        :return: Chosen action.
        """
        self.check_state_exist(state)

        if np.random.rand() < self.epsilon:
            # Choose the best action (with the highest Q-value)
            state_action = self.q_table.loc[state, :]
            # If multiple actions have the same value, choose one randomly
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # Choose a random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        """
        This method is a placeholder for learning in the base Q-learning agent.
        In the derived class (SarsaLambda), this method will be implemented.
        """
        pass


class SarsaLambdaAgent(QLearningAgent):
    """
    This class implements the Sarsa(λ) algorithm, an extension of Q-learning using eligibility traces.
    """

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        """
        Initialize the Sarsa(λ) agent with the necessary parameters.

        :param actions: List of available actions.
        :param learning_rate: Rate at which the agent learns.
        :param reward_decay: Discount factor for future rewards.
        :param e_greedy: Probability of choosing the best action.
        :param trace_decay: Decay factor for eligibility traces.
        """
        super().__init__(actions, learning_rate, reward_decay, e_greedy)
        self.lambda_ = trace_decay  # Lambda for eligibility traces
        self.eligibility_trace = self.q_table.copy()  # Initialize eligibility trace as a copy of Q-table

    def check_state_exist(self, state):
        """
        Ensure the state exists in both Q-table and eligibility trace.

        :param state: The state to check and possibly add to both Q-table and eligibility trace.
        """
        if state not in self.q_table.index:
            # Add new state to Q-table and eligibility trace with initial Q-values (0)
            new_state = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(new_state)
            self.eligibility_trace = self.eligibility_trace.append(new_state)

    def learn(self, s, a, r, s_, a_):
        """
        Update Q-table using the Sarsa(λ) algorithm with eligibility traces.

        :param s: Current state.
        :param a: Action taken in the current state.
        :param r: Reward received after taking action a in state s.
        :param s_: Next state.
        :param a_: Action taken in the next state.
        """
        self.check_state_exist(s_)

        # Predict Q-value for current state-action pair
        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            # If the next state is not terminal, calculate the target
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            # If the next state is terminal, the target is just the reward
            q_target = r

        # Calculate the error (TD-error)
        error = q_target - q_predict

        # Update eligibility trace: reset all except for the current state-action pair
        self.eligibility_trace.loc[s, :] = 0
        self.eligibility_trace.loc[s, a] = 1

        # Update Q-table using the eligibility trace and learning rate
        self.q_table += self.lr * error * self.eligibility_trace

        # Decay the eligibility trace after each update
        self.eligibility_trace *= self.gamma * self.lambda_


def main():
    # Define the environment's action space and other parameters
    actions = ['up', 'down', 'left', 'right']
    learning_rate = 0.1
    reward_decay = 0.9
    epsilon = 0.8
    trace_decay = 0.9

    # Create a SarsaLambda agent
    agent = SarsaLambdaAgent(actions, learning_rate, reward_decay, epsilon, trace_decay)

    # Example of agent learning
    states = ['state1', 'state2', 'state3', 'terminal']
    for _ in range(100):  # Run 100 learning steps
        state = np.random.choice(states)  # Randomly select a state
        action = agent.choose_action(state)  # Choose an action based on the state

        # Simulate environment interaction (reward and next state)
        reward = np.random.random()  # Simulate reward
        next_state = np.random.choice(states)  # Simulate next state
        next_action = agent.choose_action(next_state)  # Simulate next action

        # Update Q-table based on the agent's experience
        agent.learn(state, action, reward, next_state, next_action)

        # Optionally, print the updated Q-table
        print(agent.q_table)

if __name__ == "__main__":
    main()
