import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from RL_brain import DuelingDQN


# 创建环境
def create_environment():
    """
    创建并初始化 gym 环境
    """
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    return env


# 初始化 DQN 算法
def initialize_dqn(sess, action_space, memory_size):
    """
    初始化两个 DQN 算法：自然 DQN 和 对抗 DQN
    """
    with tf.variable_scope('natural'):
        natural_dqn = DuelingDQN(
            n_actions=action_space, 
            n_features=3, 
            memory_size=memory_size,
            e_greedy_increment=0.001, 
            sess=sess, 
            dueling=False
        )
        
    with tf.variable_scope('dueling'):
        dueling_dqn = DuelingDQN(
            n_actions=action_space, 
            n_features=3, 
            memory_size=memory_size,
            e_greedy_increment=0.001, 
            sess=sess, 
            dueling=True, 
            output_graph=True
        )
        
    return natural_dqn, dueling_dqn


# 训练过程
def train_dqn(RL, env, memory_size):
    """
    训练 DQN 网络
    """
    accumulated_rewards = [0]
    total_steps = 0
    observation = env.reset()

    while True:
        # 选择动作
        action = RL.choose_action(observation)

        # 转换动作空间
        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # 将动作映射到 [-2, 2] 区间
        
        # 执行动作，获取下一步的状态和奖励
        observation_, reward, done, info = env.step(np.array([f_action]))

        # 奖励归一化
        reward /= 10  # 归一化到 (-1, 0) 范围
        accumulated_rewards.append(reward + accumulated_rewards[-1])  # 累计奖励

        # 存储状态、动作和奖励
        RL.store_transition(observation, action, reward, observation_)

        # 如果经历的步数超过 MEMORY_SIZE，开始学习
        if total_steps > memory_size:
            RL.learn()

        # 如果训练超过15000步，终止训练
        if total_steps - memory_size > 15000:
            break

        observation = observation_
        total_steps += 1

    return RL.cost_his, accumulated_rewards


# 绘制训练结果
def plot_results(c_natural, c_dueling, r_natural, r_dueling):
    """
    绘制训练过程中的损失和累计奖励
    """
    # 绘制损失图
    plt.figure(1)
    plt.plot(np.array(c_natural), c='r', label='Natural DQN')
    plt.plot(np.array(c_dueling), c='b', label='Dueling DQN')
    plt.legend(loc='best')
    plt.ylabel('Cost')
    plt.xlabel('Training Steps')
    plt.grid()

    # 绘制累计奖励图
    plt.figure(2)
    plt.plot(np.array(r_natural), c='r', label='Natural DQN')
    plt.plot(np.array(r_dueling), c='b', label='Dueling DQN')
    plt.legend(loc='best')
    plt.ylabel('Accumulated Reward')
    plt.xlabel('Training Steps')
    plt.grid()

    plt.show()


# 主函数
def main():
    # 配置
    MEMORY_SIZE = 3000
    ACTION_SPACE = 25

    # 创建 TensorFlow 会话
    sess = tf.Session()

    # 创建环境
    env = create_environment()

    # 初始化 DQN 算法
    natural_dqn, dueling_dqn = initialize_dqn(sess, ACTION_SPACE, MEMORY_SIZE)

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 训练模型
    cost_natural, reward_natural = train_dqn(natural_dqn, env, MEMORY_SIZE)
    cost_dueling, reward_dueling = train_dqn(dueling_dqn, env, MEMORY_SIZE)

    # 绘制训练结果
    plot_results(cost_natural, cost_dueling, reward_natural, reward_dueling)


# 运行主函数
if __name__ == '__main__':
    main()
