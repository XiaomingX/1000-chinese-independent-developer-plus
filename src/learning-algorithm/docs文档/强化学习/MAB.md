## 多臂老虎机问题：简单理解与应用

多臂老虎机（Multi-Armed Bandit, MAB）问题就像一个赌徒面对多个老虎机，每个老虎机的中奖概率不同，但赌徒不知道具体概率。目标是在有限的尝试次数内，找到收益最大的老虎机，并尽可能多地玩它，从而最大化总收益。解决这个问题的关键在于如何在*探索*（尝试不同的老虎机）和*利用*（玩目前看起来收益最高的老虎机）之间找到平衡。

以下介绍几种常见的MAB算法，并结合实际的例子和Python代码，帮助大家理解。

### 1. Epsilon-Greedy 算法

**原理**：

*   以较小的概率 ε （比如10%）随机选择一个老虎机（探索）。
*   以较大的概率 1-ε （比如90%）选择目前为止平均收益最高的老虎机（利用）。

**实际应用：**

假设一个电商平台要推荐商品，有3个不同的广告位（可以理解为3个老虎机）。Epsilon-Greedy 算法会以 10% 的概率随机展示一个广告，以 90% 的概率展示点击率最高的广告。

**代码示例：**

```python
import random

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = [0] * n_arms  # 每个臂选择的次数
        self.values = [0.0] * n_arms # 每个臂的平均收益

    def select_arm(self):
        if random.random() < self.epsilon:
            # 探索：随机选择一个臂
            return random.randint(0, self.n_arms - 1)
        else:
            # 利用：选择当前平均收益最高的臂
            return self.values.index(max(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        old_value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / float(n)) * old_value + (1 / float(n)) * reward

# 模拟老虎机
def simulate_bandit(algorithm, rewards, steps=1000):
    cumulative_rewards = 0
    for _ in range(steps):
        chosen_arm = algorithm.select_arm()
        reward = 1 if random.random() < rewards[chosen_arm] else 0 # 模拟奖励
        algorithm.update(chosen_arm, reward)
        cumulative_rewards += reward
    return cumulative_rewards

# 测试
n_arms = 3
probabilities = [0.2, 0.5, 0.3] # 每个臂的中奖概率
epsilon_greedy = EpsilonGreedy(n_arms, epsilon=0.1)
total_reward = simulate_bandit(epsilon_greedy, probabilities)
print(f"Epsilon-Greedy 总奖励: {total_reward}")
```

**数值指标：**

*   **累计奖励**：算法运行一段时间后获得的总奖励。
*   **平均奖励**：总奖励除以尝试次数。
*   **最优臂选择比例**：选择到最佳老虎机的次数占总次数的比例。

一般来说，ε 的取值在0.01到0.1之间比较常见。ε 过大，则探索过多，收敛速度慢；ε 过小，则可能陷入局部最优。

### 2. UCB (Upper Confidence Bound) 算法

**原理**：

UCB算法在选择时，会考虑每个臂的*平均收益*和*不确定性*。它给每个臂的平均收益加上一个置信区间，置信区间的宽度与臂被选择的次数成反比。也就是说，被选择次数越少的臂，置信区间越大，越有可能被选中。

**公式**：
$$
UCB_i(t) = \frac{r_i(t-1)}{n_i(t-1)} + c \cdot \sqrt{\frac{2 \cdot \log(t)}{n_i(t-1)}}
$$
其中：

*   $$UCB_i(t)$$是臂 $$i$$ 在时间 $$t$$ 的 UCB 值。
*   $$r_i(t-1)$$ 是臂 $$i$$ 在时间 $$t-1$$ 之前的累积回报。
*   $$n_i(t-1)$$ 是臂 $$i$$ 在时间 $$t-1$$ 之前被选择的次数。
*   $$c$$ 是一个超参数，控制置信区间的宽度。

**实际应用：**

在线广告投放：UCB算法可以用于选择要展示的广告。对于展示次数较少的广告，UCB 会给它一个较高的置信度，鼓励更多展示，以便更好地评估其效果。

**代码示例：**

```python
import math

class UCB:
    def __init__(self, n_arms, c=2):
        self.n_arms = n_arms
        self.c = c  # 控制置信区间的宽度
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_plays = 0

    def select_arm(self):
        self.total_plays += 1
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i  # 优先选择未探索的臂

        ucb_values = [0.0] * self.n_arms
        for i in range(self.n_arms):
            exploration_term = math.sqrt((2 * math.log(self.total_plays)) / self.counts[i])
            ucb_values[i] = self.values[i] + self.c * exploration_term

        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        old_value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / float(n)) * old_value + (1 / float(n)) * reward

# 测试
n_arms = 3
probabilities = [0.2, 0.5, 0.3]
ucb = UCB(n_arms, c=2)
total_reward = simulate_bandit(ucb, probabilities)
print(f"UCB 总奖励: {total_reward}")
```

**数值指标：**

*   **超参数 c 的影响**：c 值越大，越倾向于探索；c 值越小，越倾向于利用。通常需要通过实验来选择合适的 c 值。

### 3. Thompson Sampling 算法

**原理**：

Thompson Sampling 是一种基于贝叶斯方法的算法。它假设每个臂的奖励都服从一个概率分布（比如 Beta 分布），然后根据已有的数据不断更新这个分布。每次选择臂时，从每个臂的分布中抽取一个样本，选择样本值最高的臂。

**步骤**：

1.  **初始化**：对每个臂，假设其奖励服从一个 Beta 分布（Beta(α=1, β=1)）。
2.  **采样**：对每个臂，从其 Beta 分布中抽取一个样本。
3.  **选择**：选择样本值最高的臂。
4.  **更新**：根据选择的臂获得的奖励，更新该臂的 Beta 分布。如果获得奖励，则 α 加 1；否则，β 加 1。

**实际应用：**

新闻推荐：Thompson Sampling 可以用于选择要展示的新闻。每个新闻都有一个点击率的先验分布，算法根据用户的点击反馈不断更新这个分布，并选择点击率可能性最高的新闻展示。

**代码示例：**

```python
import numpy as np
import random

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = [1] * n_arms  # Beta 分布的 alpha 参数
        self.beta = [1] * n_arms   # Beta 分布的 beta 参数

    def select_arm(self):
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        return samples.index(max(samples))

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1

# 测试
n_arms = 3
probabilities = [0.2, 0.5, 0.3]
thompson_sampling = ThompsonSampling(n_arms)
total_reward = simulate_bandit(thompson_sampling, probabilities)
print(f"Thompson Sampling 总奖励: {total_reward}")
```

**数值指标：**

*   **Beta 分布的参数**：α 和 β 的值越大，说明对该臂的奖励估计越准确。

### 4. LinUCB 算法

**原理**：

LinUCB 算法是一种基于线性模型的 UCB 算法，它假设臂的奖励与臂的特征之间存在线性关系。

**公式**：
$$
UCB_i(t) = \theta_i^T \cdot x_t + \alpha \cdot \sqrt{x_t^T \cdot A_i^{-1} \cdot x_t}
$$
其中：

*   $$x_t$$ 是臂 $$i$$ 在时间 $$t$$ 的特征向量。
*   $$\theta_i$$ 是臂 $$i$$ 的线性模型参数。
*   $$A_i$$ 是臂 $$i$$ 的特征向量的协方差矩阵。
*   $$\alpha$$ 是一个超参数，控制置信区间的宽度。

**实际应用：**

个性化推荐：LinUCB 可以用于个性化推荐。每个用户和每个物品都有一个特征向量，LinUCB 算法根据用户的历史行为和物品的特征，预测用户对该物品的兴趣，并选择 UCB 值最高的物品推荐给用户。

**代码示例：**

```python
import numpy as np

class LinUCB:
    def __init__(self, n_arms, dimension, alpha=1.0):
        self.n_arms = n_arms
        self.dimension = dimension
        self.alpha = alpha
        self.A = [np.identity(dimension) for _ in range(n_arms)] # 臂的协方差矩阵
        self.b = [np.zeros(dimension) for _ in range(n_arms)]    # 臂的线性模型参数
        self.theta = [np.zeros(dimension) for _ in range(n_arms)]# 线性模型参数估计

    def select_arm(self, features):
        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            theta = np.linalg.solve(self.A[i], self.b[i]) # 求解线性方程组，得到模型参数
            ucb_values[i] = np.dot(theta, features[i]) + self.alpha * np.sqrt(np.dot(features[i], np.dot(np.linalg.inv(self.A[i]), features[i])))
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward, feature):
        self.A[chosen_arm] += np.outer(feature, feature)
        self.b[chosen_arm] += reward * feature

# 示例
n_arms = 3
dimension = 2  # 特征向量的维度
lin_ucb = LinUCB(n_arms, dimension)

# 模拟特征向量和奖励
features = [np.random.rand(n_arms, dimension) for _ in range(1000)]
rewards = np.random.randint(0, 2, size=(1000, n_arms))  # 0 或 1

cumulative_reward = 0
for t in range(1000):
    chosen_arm = lin_ucb.select_arm(features[t])
    reward = rewards[t][chosen_arm]
    lin_ucb.update(chosen_arm, reward, features[t][chosen_arm])
    cumulative_reward += reward

print(f"LinUCB 累计奖励: {cumulative_reward}")
```

**数值指标：**

*   **特征向量的维度**：特征向量的维度越高，模型表达能力越强，但也更容易过拟合。
*   **超参数 α 的影响**：α 值越大，越倾向于探索；α 值越小，越倾向于利用。

总的来说，多臂老虎机问题是一个在探索和利用之间寻找平衡的问题。选择合适的算法取决于具体的应用场景和数据特征。Epsilon-Greedy 算法简单易懂，但可能收敛速度较慢；UCB 算法考虑了不确定性，能更快地找到最优臂；Thompson Sampling 算法基于贝叶斯方法，能更好地处理不确定性；LinUCB 算法则适用于有特征信息的场景。