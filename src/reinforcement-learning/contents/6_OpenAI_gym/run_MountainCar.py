import gym
from RL_brain import DeepQNetwork


# 初始化环境并打印环境信息
def initialize_environment():
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    print("Action Space: ", env.action_space)
    print("Observation Space: ", env.observation_space)
    print("Observation Space High: ", env.observation_space.high)
    print("Observation Space Low: ", env.observation_space.low)
    return env


# 计算奖励函数
def calculate_reward(position):
    # 奖励设计：目标是让小车靠近目标位置 (-0.5)
    return abs(position - (-0.5))  # 奖励越高越好


# 训练一轮
def train_episode(env, RL, total_steps, i_episode):
    observation = env.reset()  # 重置环境，获取初始观测
    ep_r = 0  # 当前回合的累计奖励

    while True:
        env.render()  # 可视化环境

        # 选择行动
        action = RL.choose_action(observation)

        # 执行动作，并获得反馈
        observation_, reward, done, info = env.step(action)

        # 计算奖励（使用自定义的奖励函数）
        reward = calculate_reward(observation_[0])

        # 存储当前转移
        RL.store_transition(observation, action, reward, observation_)

        # 学习：当步数超过一定阈值后开始训练
        if total_steps > 1000:
            RL.learn()

        # 累计奖励
        ep_r += reward

        # 如果结束了，就打印结果并跳出
        if done:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print(f'Episode: {i_episode} {get} | Ep_r: {round(ep_r, 4)} | Epsilon: {round(RL.epsilon, 2)}')
            break

        # 更新当前观测，继续下一步
        observation = observation_
        total_steps += 1

    return total_steps, ep_r


# 主函数
def main():
    # 初始化环境和强化学习模型
    env = initialize_environment()
    
    RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                      replace_target_iter=300, memory_size=3000,
                      e_greedy_increment=0.0002)
    
    total_steps = 0  # 总步数，用于控制学习进度

    # 训练多个回合
    for i_episode in range(10):
        total_steps, ep_r = train_episode(env, RL, total_steps, i_episode)

    # 训练完成后，绘制成本曲线
    RL.plot_cost()


# 程序入口
if __name__ == '__main__':
    main()
