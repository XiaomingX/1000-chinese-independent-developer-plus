"""
强化学习迷宫示例。

红色矩形：探险者。
黑色矩形：地狱（奖励 = -1）。
黄色圆形：天堂（奖励 = +1）。
其他所有状态：地面（奖励 = 0）。

此脚本控制了更新方法的主要部分。
强化学习的实现位于 RL_brain.py。

更多内容可以参考我的教程页面：https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable

def run_episode(env, RL):
    """
    执行单个训练回合
    """
    # 初始化观察
    observation = env.reset()

    # 每一步执行直到本回合结束
    while True:
        # 渲染当前环境状态
        env.render()

        # 根据当前观察选择行动
        action = RL.choose_action(str(observation))

        # 执行动作，得到下一个状态和奖励
        observation_, reward, done = env.step(action)

        # 强化学习算法根据当前状态、动作、奖励和下一个状态进行学习
        RL.learn(str(observation), action, reward, str(observation_))

        # 更新当前观察
        observation = observation_

        # 如果回合结束，跳出循环
        if done:
            break

def main():
    """
    主函数，初始化环境和强化学习模型，开始训练过程
    """
    # 创建迷宫环境
    env = Maze()

    # 创建强化学习模型（Q学习）
    RL = QLearningTable(actions=list(range(env.n_actions)))

    # 执行多次训练回合
    for episode in range(100):
        print(f"Episode {episode + 1} starts:")
        run_episode(env, RL)

    # 游戏结束
    print('游戏结束！')
    env.destroy()

if __name__ == "__main__":
    # 启动主函数
    main()
