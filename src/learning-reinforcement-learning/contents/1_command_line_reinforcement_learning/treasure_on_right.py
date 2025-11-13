import numpy as np
import pandas as pd
import time

# 设置随机种子以保证结果可复现
np.random.seed(2)

# 常量定义
N_STATES = 6  # 一维世界的长度
ACTIONS = ['left', 'right']  # 可用动作：向左或向右
EPSILON = 0.9  # ε-贪婪策略中的探索率（90%概率选择最优动作，10%概率随机探索）
ALPHA = 0.1  # 学习率（更新Q值的幅度）
GAMMA = 0.9  # 折扣因子（未来奖励的衰减系数）
MAX_EPISODES = 13  # 最大训练回合数
FRESH_TIME = 0.3  # 每步动作的延迟时间，用于可视化过程

# 初始化Q表
def build_q_table(n_states, actions):
    """
    初始化Q表，所有Q值初始化为0
    
    参数:
        n_states: 状态数量
        actions: 可用动作列表
        
    返回:
        初始化后的Q表（DataFrame格式）
    """
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # Q值初始化为0
        columns=actions  # 列名为动作名称
    )
    return table

# 根据ε-贪婪策略选择动作
def choose_action(state, q_table):
    """
    根据当前状态和Q表，使用ε-贪婪策略选择动作
    
    参数:
        state: 当前状态
        q_table: Q表
        
    返回:
        选择的动作名称
    """
    state_actions = q_table.iloc[state, :]  # 获取当前状态下所有动作的Q值
    
    # 两种情况选择随机动作：
    # 1. 以(1-EPSILON)的概率进行探索
    # 2. 当前状态下所有动作的Q值都为0（初始状态）
    if np.random.uniform() > EPSILON or (state_actions == 0).all():
        action_name = np.random.choice(ACTIONS)
    else:  # 选择当前状态下Q值最大的动作（利用已知信息）
        action_name = state_actions.idxmax()  # idxmax返回最大值对应的列名（动作）
    return action_name

# 与环境交互，获取反馈
def get_env_feedback(state, action):
    """
    模拟环境对智能体动作的反馈（新状态和奖励）
    
    参数:
        state: 当前状态
        action: 执行的动作
        
    返回:
        next_state: 执行动作后的新状态
        reward: 获得的奖励
    """
    if action == 'right':  # 向右移动
        # 如果当前状态是倒数第二个位置，向右移动就到达目标（宝藏）
        if state == N_STATES - 2:
            return 'terminal', 1  # 返回终止状态和奖励1（到达目标）
        else:
            return state + 1, 0  # 否则向右移动一步，无奖励
    else:  # 向左移动
        if state == 0:  # 已经在最左端，无法向左移动（撞墙）
            return state, 0  # 状态不变，无奖励
        else:
            return state - 1, 0  # 向左移动一步，无奖励

# 更新并打印环境状态（可视化）
def update_env(state, episode, step_counter):
    """
    更新环境的可视化状态，展示智能体当前位置
    
    参数:
        state: 当前状态
        episode: 当前回合数
        step_counter: 当前回合的步数
    """
    # 环境初始设置：N_STATES-1个 '-' 表示路径，最后一个位置 'T' 表示宝藏（目标）
    env_list = ['-'] * (N_STATES - 1) + ['T']
    
    if state == 'terminal':  # 智能体到达目标，打印回合结果
        print(f'\r回合 {episode + 1}: 总步数 = {step_counter}', end='')
        time.sleep(2)  # 暂停2秒展示结果
        print('\r' + ' ' * 30, end='')  # 清空当前行
    else:
        env_list[state] = 'o'  # 在当前位置放置智能体标识 'o'
        interaction = ''.join(env_list)  # 将列表转换为字符串表示环境
        print(f'\r{interaction}', end='')  # 在同一行打印环境状态
        time.sleep(FRESH_TIME)  # 暂停一段时间，展示智能体移动过程

# 主Q-learning算法实现
def rl():
    """
    执行Q-learning算法，训练智能体
    
    返回:
        训练完成后的Q表
    """
    # 初始化Q表
    q_table = build_q_table(N_STATES, ACTIONS)
    
    # 循环训练回合
    for episode in range(MAX_EPISODES):
        state = 0  # 初始状态：从最左端开始
        step_counter = 0  # 记录当前回合的步数
        is_terminated = False  # 标记回合是否结束
        update_env(state, episode, step_counter)  # 初始化环境显示
        
        # 循环执行步骤，直到回合结束
        while not is_terminated:
            action = choose_action(state, q_table)  # 选择动作
            next_state, reward = get_env_feedback(state, action)  # 获取环境反馈
            
            # 预测当前状态-动作对的Q值
            q_predict = q_table.loc[state, action]
            
            if next_state != 'terminal':  # 若下一状态不是终止状态，计算目标Q值
                # 目标Q值 = 即时奖励 + 折扣因子 * 下一状态的最大Q值
                q_target = reward + GAMMA * q_table.iloc[next_state, :].max()
            else:  # 若下一状态是终止状态，目标Q值就是即时奖励
                q_target = reward
                is_terminated = True  # 标记回合结束
            
            # 更新Q值：新Q值 = 旧Q值 + 学习率 * (目标Q值 - 预测Q值)
            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            
            state = next_state  # 移动到下一状态
            update_env(state, episode, step_counter + 1)  # 更新环境显示
            step_counter += 1  # 步数加1
    
    return q_table  # 返回训练好的Q表

# 主函数，整合所有流程
def main():
    """
    主函数：执行Q-learning训练过程并打印结果
    """
    print("开始Q-learning训练过程...")
    q_table = rl()  # 运行Q-learning算法
    print('\n训练完成后的Q表：\n')
    print(q_table)  # 显示最终的Q表

if __name__ == "__main__":
    main()  # 启动程序
