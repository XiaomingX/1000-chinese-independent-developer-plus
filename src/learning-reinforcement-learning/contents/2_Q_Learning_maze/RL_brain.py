import numpy as np
import pandas as pd

class QLearningAgent:
    """
    Q-learning agent that learns and makes decisions based on Q-learning.
    """
    def __init__(self, actions, learning_rate=0.01, discount_factor=0.9, exploration_rate=0.9):
        self.actions = actions  # List of possible actions
        self.learning_rate = learning_rate  # Learning rate for Q-value update
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.exploration_rate = exploration_rate  # Probability of exploring (random action)
        
        # Initialize the Q-table with actions as columns
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        """
        Select an action based on the current state using epsilon-greedy strategy.
        """
        self._ensure_state_exists(state)
        
        if np.random.uniform() < self.exploration_rate:
            # Exploration: Choose a random action
            action = np.random.choice(self.actions)
        else:
            # Exploitation: Choose the best action based on current Q-values
            state_actions = self.q_table.loc[state, :]
            max_q_value = np.max(state_actions)
            best_actions = state_actions[state_actions == max_q_value].index
            action = np.random.choice(best_actions)  # In case of tie, randomly choose one

        return action

    def learn(self, state, action, reward, next_state):
        """
        Update the Q-value for the given state-action pair based on the received reward.
        """
        self._ensure_state_exists(next_state)
        
        current_q_value = self.q_table.loc[state, action]
        
        if next_state != 'terminal':
            # If next state is not terminal, calculate Q-target using future rewards
            future_q_value = self.q_table.loc[next_state, :].max()
            q_target = reward + self.discount_factor * future_q_value
        else:
            # If next state is terminal, no future reward
            q_target = reward
        
        # Update the Q-table using the Q-learning formula
        self.q_table.loc[state, action] += self.learning_rate * (q_target - current_q_value)

    def _ensure_state_exists(self, state):
        """
        Ensure the state exists in the Q-table. If not, add it with initial Q-values of 0.
        """
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            )

def main():
    # Define the set of actions available to the agent
    actions = ['left', 'right', 'up', 'down']
    
    # Create a Q-learning agent with the specified actions
    agent = QLearningAgent(actions)

    # Example states and rewards (this would typically come from an environment)
    states = ['state1', 'state2', 'state3']
    rewards = {'state1': 1, 'state2': 2, 'state3': -1}
    
    # Simulate the agent's learning process
    for epoch in range(1000):
        current_state = np.random.choice(states)
        action = agent.choose_action(current_state)
        reward = rewards.get(current_state, 0)
        
        # Assume next state is either terminal or another state
        next_state = np.random.choice(states + ['terminal'], p=[0.7, 0.3])
        
        # Let the agent learn from this experience
        agent.learn(current_state, action, reward, next_state)

    # Print the final Q-table after learning
    print("Final Q-table:")
    print(agent.q_table)

if __name__ == "__main__":
    main()
