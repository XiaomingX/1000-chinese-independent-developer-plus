## 逆向经验回放 (HER) 详解

逆向经验回放 (Hindsight Experience Replay, HER) 是一种强化学习技巧，专门用于提升在稀疏奖励环境中的学习效率，尤其是在目标导向的任务中。它的核心思想是，即使智能体未能达成最初设定的目标，通过将失败的经验重新解释为“如果达成了另一个目标”，智能体仍然可以从中学习。

**核心思想**

HER 的核心在于，即使智能体 *未能* 达到预定目标，这次尝试仍然有价值。我们可以假设智能体实际上 *成功* 达到了它最终所处的状态。通过这种“事后诸葛亮”的方式，智能体可以学习到 *如何* 到达这个状态，从而将失败的经验转化为成功的经验。

例如，假设一个机械臂尝试将一个杯子放到桌子的特定位置，但失败了，杯子落在了桌子的另一个地方。传统的强化学习方法会认为这是一个失败的经验，没有提供任何有用的信息。但是，HER 会重新审视这次尝试，假设机械臂的目标 *本来就是* 将杯子放到它实际落下的位置。这样，智能体就可以学习到将杯子放到 *那个* 位置所需的动作序列。

**实现步骤**

1.  **经验收集:** 智能体使用当前策略与环境互动，生成轨迹 τ，该轨迹由状态、动作和奖励序列组成：(s, a, r(s, a, g), s’, g)。其中：
    *   `s`: 状态 (state)，例如机械臂各个关节的角度，杯子的位置等等。
    *   `a`: 动作 (action)，例如机械臂各个关节的移动指令。
    *   `r(s, a, g)`: 奖励 (reward)，基于当前状态 `s`、动作 `a` 和目标 `g` 计算得出。通常，如果机械臂成功将杯子移动到目标位置，则奖励为 1，否则为 0（稀疏奖励）。
    *   `s'`: 下一个状态 (next state)，执行动作 `a` 之后，环境进入的新状态。
    *   `g`: 目标 (goal)，例如杯子在桌子上的目标坐标。

2.  **回放缓存存储:** 将收集到的经验 (s, a, r(s, a, g), s’, g) 存储在回放缓存中。回放缓存就像一个经验池，智能体可以从中随机抽取经验进行学习。

3.  **目标修改:** 选择额外的目标来修改原始轨迹。有几种方法可以选择这些目标：

    *   **Final (最终):** 将 episode 的最终状态用作已实现的目标。
    *   **Future (未来):** 在同一 episode 中，replay 当前 transition 之后观察到的 *k* 个随机状态。已被证明对 replay 最有价值，因为目标很可能在不久的将来实现。通常 *k* 取值为 4。
    *   **Episode (Episode):** Replay 同一 episode 中的 *k* 个随机状态。
    *   **Random (随机):** Replay 到目前为止在整个训练过程中遇到的 *k* 个随机状态。

    *举例说明 Future 策略：* 假设一个 episode 包含 10 个时间步。在第 3 步时，智能体失败了。使用 Future 策略，我们可能会从第 4-10 步中随机选择一个状态作为“ hindsight goal ”。 这样，智能体就可以学习到 *如何* 从第 3 步到达这个未来的状态。

4.  **奖励调整:** 根据新目标修改奖励。在二元奖励问题中，仅当状态等于目标时才更改奖励。也就是说，如果智能体 *实际上* 达到了 hindsight goal，则将奖励设为 1，否则为 0。

5.  **回放和训练:** 从回放缓冲区（包含原始经验和 hindsight 经验）中采样，并使用 off-policy 强化学习算法训练策略。例如，可以使用 Deep Q-Network (DQN) 或 Deep Deterministic Policy Gradient (DDPG) 算法。

**算法细节**

*   HER 适用于多目标场景，将每个系统状态视为一个独立目标。
*   它训练 universal policies，将当前状态和目标状态都作为输入。这意味着智能体可以学习到 *如何* 达到 *任何* 给定的目标，而不仅仅是预先设定的目标。
*   HER 可以与 off-policy RL 算法集成，例如 DDPG（深度确定性策略梯度）。

**HER 与 DDPG**

将 HER 与 DDPG 结合使用时，可以考虑以下配置：

*   Adam 优化器。
*   多层感知器 (MLP)。
*   ReLU 激活函数。
*   并行处理（例如，8 个内核），并在更新后进行参数平均。
*   具有 3 个隐藏层的 Actor-critic 网络，每个隐藏层具有 64 个隐藏节点，以及用于 Actor 输出层的 tanh 激活函数。
*   经验回放缓冲区大小为 $$10^6$$，折扣因子 $$\gamma = 0.98$$，学习率 $$\alpha = 0.001$$，探索因子 $$\epsilon = 0.2$$。

**优点**

*   **提高样本效率:** HER 允许从不成功的轨迹中学习，从而增加了有用数据的数量。
*   **稀疏奖励环境:** HER 在奖励稀疏或二元的环境中特别有效。
*   **通用性:** HER 可以与各种 off-policy 强化学习算法结合使用。

**局限性**

*   HER 假设奖励和目标可以直接控制，但情况并非总是如此。
*   目标之间的关系可能并非总是直接的，这会影响 HER 的适用性。
*   状态表示应理想地反映目标，但这在所有环境中都可能无法实现。

**代码示例 (PyTorch)**

以下是一个简化的 HER 实现示例，使用 PyTorch 框架：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义 Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x)) # 动作输出范围通常在 [-1, 1]
        return x

# 定义 Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义 HER 回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, goal):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, goal)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, goal = zip(*[self.buffer[i] for i in batch])
        return (
            torch.FloatTensor(np.array(state)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.array(next_state)),
             torch.FloatTensor(np.array(goal))
        )

    def __len__(self):
        return len(self.buffer)

# 假设环境交互函数 (简化)
def interact_with_environment(actor, state, goal, epsilon):
    """
    与环境交互， epsilon-greedy 探索策略
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0) # 增加batch维度
    if np.random.rand() < epsilon:
        # 探索: 随机选择动作
        action = np.random.uniform(-1, 1, size=actor.fc3.out_features)  # 假设动作空间范围 [-1, 1]
    else:
        # 利用: 使用Actor网络选择动作
        actor.eval() # 设置为评估模式
        with torch.no_grad():
            action = actor(state_tensor).squeeze(0).numpy()
        actor.train() # 设置为训练模式
    next_state, reward, done = fake_env_step(state, action, goal) # 模拟环境交互
    return action, next_state, reward, done

def fake_env_step(state, action, goal):
    """
    模拟环境步骤 (非常简化).  实际环境中需要与真实环境交互.
    """
    # 简单的模拟: 动作直接影响状态
    next_state = state + action * 0.1 # 假设动作对状态有线性影响
    #  如果 next_state 接近 goal， 则奖励为 1, 否则为 0
    if np.linalg.norm(next_state - goal) < 0.1:
        reward = 1.0
        done = True
    else:
        reward = 0.0
        done = False
    return next_state, reward, done

# HER 目标重采样
def her_sample(replay_buffer, k=4):
    """
    实现 HER 的 "future" 策略
    """
    her_samples = []
    for episode in range(len(replay_buffer.buffer)): #简化
        state, action, reward, next_state, goal = replay_buffer.buffer[episode]

        # 添加原始样本
        her_samples.append((state, action, reward, next_state, goal))

        # Future 策略: 从 episode 剩下的 steps 中选择 k 个作为 hindsight goals
        future_states = next_state #简化
        for _ in range(k):
            hindsight_goal = future_states #Simplified
            hindsight_reward = 1.0 if np.linalg.norm(next_state - hindsight_goal) < 0.1 else 0.0
            her_samples.append((state, action, hindsight_reward, next_state, hindsight_goal))
    return her_samples

# 训练循环
def train(actor, critic, actor_optimizer, critic_optimizer, replay_buffer, batch_size, gamma, her=True):
    if len(replay_buffer) < batch_size:
        return # 经验不足， 无法训练

    # 1. 采样
    if her:
        #应用her采样
        her_samples = her_sample(replay_buffer)
        #转换为tensor
        state, action, reward, next_state, goal = zip(*her_samples)
        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        goal = torch.FloatTensor(np.array(goal))
    else:
        state, action, reward, next_state, goal = replay_buffer.sample(batch_size)

    # 2. 更新 Critic
    critic_optimizer.zero_grad()
    Q_values = critic(state, action)
    next_actions = actor(next_state)
    next_Q_values = critic(next_state, next_actions)
    expected_Q_values = reward + gamma * next_Q_values
    critic_loss = nn.MSELoss()(Q_values, expected_Q_values)
    critic_loss.backward()
    critic_optimizer.step()

    # 3. 更新 Actor
    actor_optimizer.zero_grad()
    actions = actor(state)
    actor_loss = -critic(state, actions).mean() #  使用 critic 网络的输出作为 actor 的 loss
    actor_loss.backward()
    actor_optimizer.step()

if __name__ == '__main__':
    # 1. 初始化
    state_dim = 3  # 例如: x, y, z 坐标
    action_dim = 3 # 例如:  x, y, z 方向的力
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.002)
    replay_buffer = ReplayBuffer(capacity=10000)

    # 2. 训练参数
    num_episodes = 1000
    batch_size = 64
    gamma = 0.99
    epsilon = 0.2 # 探索率

    # 3. 训练循环
    for episode in range(num_episodes):
        # 3.1 初始化环境
        state = np.random.rand(state_dim) # 随机初始化状态
        goal = np.random.rand(state_dim)  # 随机初始化目标
        done = False
        total_reward = 0

        while not done:
            # 3.2  与环境交互
            action, next_state, reward, done = interact_with_environment(actor, state, goal, epsilon)

            # 3.3  存储经验
            replay_buffer.push(state, action, reward, next_state, goal)

            # 3.4 更新状态
            state = next_state
            total_reward += reward

            # 3.5 训练
            train(actor, critic, actor_optimizer, critic_optimizer, replay_buffer, batch_size, gamma, her=True)

        print(f"Episode {episode}, Total Reward: {total_reward}")
```

**代码解释:**

1.  **网络定义:**  `Actor` 网络负责根据当前状态输出动作， `Critic` 网络负责评估在给定状态下执行特定动作的价值。

2.  **回放缓存:** `ReplayBuffer` 用于存储经验元组 (状态, 动作, 奖励, 下一个状态, 目标)。

3.  **环境交互:**  `interact_with_environment` 函数模拟智能体与环境的交互。 实际应用中，这部分代码会与真实环境 (例如，物理引擎或机器人硬件) 进行交互。

4.  **HER 采样:**  `her_sample` 函数实现了 HER 的 "future" 策略，用于生成 hindsight 经验。

5.  **训练:** `train` 函数从回放缓存中采样，并使用采样的数据更新 Actor 和 Critic 网络。

**实际应用**

*   **机器人控制:**  可以使用 HER 训练机器人完成复杂的任务，例如物体抓取、装配等。在这些任务中，机器人通常只能在成功完成任务时获得奖励，因此 HER 可以显著提高学习效率。
*   **游戏 AI:**  HER 可以用于训练游戏 AI，使其能够更好地完成目标导向的任务，例如在游戏中拾取特定物品、到达指定地点等。
*   **推荐系统:**  HER 可以用于优化推荐策略。 例如，可以将用户的历史行为视为状态，将推荐的物品视为动作，将用户的点击、购买等行为视为奖励。 通过 HER，可以学习到更好的推荐策略，提高用户的满意度。

**总结**

HER 是一种简单而有效的强化学习技巧，可以显著提高在稀疏奖励环境中的学习效率。它通过“事后诸葛亮”的方式，将失败的经验转化为成功的经验，从而使智能体能够更快地学习到目标导向的任务。 通过结合实际应用案例和代码示例，希望能帮助你更好地理解 HER 的原理和应用。