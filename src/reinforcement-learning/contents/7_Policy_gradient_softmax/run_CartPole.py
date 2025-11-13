import gym
import matplotlib.pyplot as plt
from RL_brain import PolicyGradient

# 常量配置
DISPLAY_REWARD_THRESHOLD = 400  # 如果总奖励超过此阈值，则渲染环境
RENDER = False  # 渲染会浪费时间，所以默认关闭

def initialize_environment():
    """
    初始化并返回Gym环境，设置随机种子以保证可复现性
    """
    env = gym.make('CartPole-v0')
    env.seed(1)  # 设置种子使得每次运行环境结果相同
    env = env.unwrapped  # 解封装环境，获得更直观的控制
    return env

def initialize_rl_agent(env):
    """
    初始化Policy Gradient RL代理
    """
    return PolicyGradient(
        n_actions=env.action_space.n,  # 动作空间的数量
        n_features=env.observation_space.shape[0],  # 状态空间的维度
        learning_rate=0.02,  # 学习率
        reward_decay=0.99,  # 奖励衰减因子
    )

def train_rl_agent(env, rl_agent):
    """
    训练RL代理，进行多个回合的强化学习
    """
    running_reward = None  # 初始化累计奖励
    for i_episode in range(3000):
        observation = env.reset()  # 重置环境，获得初始状态

        episode_rewards = []  # 当前回合的奖励记录

        while True:
            if RENDER:
                env.render()  # 如果需要渲染，则进行渲染

            action = rl_agent.choose_action(observation)  # 选择动作

            observation_, reward, done, info = env.step(action)  # 执行动作，得到反馈

            rl_agent.store_transition(observation, action, reward)  # 存储当前转移

            episode_rewards.append(reward)  # 记录当前回合的奖励

            if done:  # 回合结束时
                ep_rs_sum = sum(episode_rewards)  # 计算当前回合的总奖励

                # 更新累计奖励
                if running_reward is None:
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

                # 如果奖励超过阈值，则开始渲染环境
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True

                print(f"Episode {i_episode}, Total Reward: {int(running_reward)}")

                # 更新值函数
                vt = rl_agent.learn()

                # 绘制学习曲线
                if i_episode == 0:
                    plt.plot(vt)
                    plt.xlabel('Episode Steps')
                    plt.ylabel('Normalized State-Action Value')
                    plt.show()

                break

            observation = observation_  # 更新状态

def main():
    # 初始化环境和RL代理
    env = initialize_environment()
    rl_agent = initialize_rl_agent(env)

    # 打印环境信息
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Space High: {env.observation_space.high}")
    print(f"Observation Space Low: {env.observation_space.low}")

    # 开始训练
    train_rl_agent(env, rl_agent)

if __name__ == '__main__':
    main()
