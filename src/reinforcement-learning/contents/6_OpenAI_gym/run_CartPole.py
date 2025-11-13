import gym
from RL_brain import DeepQNetwork

def create_environment():
    """
    创建并初始化Gym环境。
    """
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    return env

def initialize_rl_agent(env):
    """
    初始化DeepQNetwork强化学习智能体。
    
    :param env: Gym环境
    :return: DeepQNetwork智能体
    """
    rl_agent = DeepQNetwork(
        n_actions=env.action_space.n,  # 动作空间的大小
        n_features=env.observation_space.shape[0],  # 状态空间的维度
        learning_rate=0.01,  # 学习率
        e_greedy=0.9,  # epsilon-greedy策略的初始值
        replace_target_iter=100,  # 每100步更新一次目标网络
        memory_size=2000,  # 经验回放的大小
        e_greedy_increment=0.001  # epsilon的增量
    )
    return rl_agent

def calculate_reward(observation_):
    """
    根据环境的状态计算奖励。奖励取决于小车的位置和杆子的角度。
    
    :param observation_: 当前环境的状态
    :return: 奖励值
    """
    x, x_dot, theta, theta_dot = observation_
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  # 小车位置的奖励
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5  # 杆子角度的奖励
    return r1 + r2

def train_rl_agent(env, rl_agent):
    """
    使用DeepQNetwork训练智能体。
    
    :param env: Gym环境
    :param rl_agent: 强化学习智能体
    """
    total_steps = 0

    for i_episode in range(100):  # 训练100个回合
        observation = env.reset()  # 重置环境
        ep_r = 0  # 当前回合的总奖励

        while True:
            env.render()  # 渲染当前环境

            action = rl_agent.choose_action(observation)  # 选择动作

            # 执行动作并获取反馈
            observation_, reward, done, info = env.step(action)

            # 根据环境状态计算奖励
            reward = calculate_reward(observation_)

            # 存储当前转移
            rl_agent.store_transition(observation, action, reward, observation_)

            ep_r += reward  # 累加奖励

            if total_steps > 1000:  # 超过1000步就开始学习
                rl_agent.learn()

            if done:  # 结束回合
                print(f'Episode: {i_episode}, Total Reward: {round(ep_r, 2)}, Epsilon: {round(rl_agent.epsilon, 2)}')
                break

            # 更新观察值
            observation = observation_
            total_steps += 1

def main():
    """
    主函数，初始化环境、智能体并开始训练。
    """
    env = create_environment()  # 创建并初始化环境
    rl_agent = initialize_rl_agent(env)  # 初始化强化学习智能体
    train_rl_agent(env, rl_agent)  # 开始训练智能体
    rl_agent.plot_cost()  # 绘制成本图

if __name__ == "__main__":
    main()  # 调用主函数
