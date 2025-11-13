import numpy as np
import random
import time
import os

class Maze:
    def __init__(self, size=10):
        self.size = size
        self.maze = self.generate_maze()
        self.start = (0, 0)
        self.end = (size-1, size-1)
        # 确保起点和终点不是障碍物
        self.maze[self.start[0]][self.start[1]] = 0
        self.maze[self.end[0]][self.end[1]] = 0
        
    def generate_maze(self):
        """生成一个随机的10x10迷宫，0表示通路，1表示障碍物"""
        maze = np.zeros((self.size, self.size), dtype=int)
        
        # 随机生成障碍物，障碍物比例约为30%
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < 0.3:
                    maze[i][j] = 1
                    
        return maze
    
    def is_valid_move(self, x, y):
        """检查移动是否有效（在边界内且不是障碍物）"""
        return 0 <= x < self.size and 0 <= y < self.size and self.maze[x][y] == 0
    
    def print_maze(self, agent1_pos, agent2_pos):
        """打印迷宫，显示两个智能体的位置"""
        # 清屏（兼容macOS）
        os.system('clear')
        
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if (i, j) == self.start:
                    row.append('S')  # 起点
                elif (i, j) == self.end:
                    row.append('G')  # 目标
                elif (i, j) == agent1_pos:
                    row.append('A')  # 训练过的智能体
                elif (i, j) == agent2_pos:
                    row.append('B')  # 随机智能体
                elif self.maze[i][j] == 1:
                    row.append('#')  # 障碍物
                else:
                    row.append(' ')  # 通路
            print('|'.join(row))
            print('-' * (self.size * 2 - 1))


class QLearningAgent:
    def __init__(self, maze, epsilon=0.9, alpha=0.1, gamma=0.9):
        self.maze = maze
        self.epsilon = epsilon  # 探索率
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子
        
        # 初始化Q表，状态是位置(x,y)，动作是上下左右
        self.q_table = np.zeros((maze.size, maze.size, 4))  # 4个方向
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # 上、下、左、右
        
    def choose_action(self, state, is_training=True):
        """选择动作，训练时使用ε-贪婪策略，测试时使用最优策略"""
        if is_training and random.random() < self.epsilon:
            # 探索：随机选择一个有效动作
            valid_actions = []
            for i, action in enumerate(self.actions):
                x, y = state
                nx, ny = x + action[0], y + action[1]
                if self.maze.is_valid_move(nx, ny):
                    valid_actions.append(i)
            return random.choice(valid_actions) if valid_actions else None
        else:
            # 利用：选择Q值最大的有效动作
            x, y = state
            q_values = self.q_table[x, y, :]
            
            # 先过滤掉无效动作
            valid_actions = []
            for i, action in enumerate(self.actions):
                nx, ny = x + action[0], y + action[1]
                if self.maze.is_valid_move(nx, ny):
                    valid_actions.append(i)
                    
            if not valid_actions:
                return None
                
            # 从有效动作中选择Q值最大的
            max_q = max(q_values[valid_actions])
            best_actions = [i for i in valid_actions if q_values[i] == max_q]
            return random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state):
        """更新Q表"""
        x, y = state
        nx, ny = next_state
        
        # 当前Q值
        current_q = self.q_table[x, y, action]
        
        # 下一个状态的最大Q值
        next_max_q = np.max(self.q_table[nx, ny, :])
        
        # 更新Q值
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[x, y, action] = new_q
    
    def train(self, episodes=1000):
        """训练智能体"""
        print(f"Training Q-learning agent for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.maze.start
            steps = 0
            
            while state != self.maze.end:
                action = self.choose_action(state)
                if action is None:
                    break  # 陷入死胡同，重新开始
                
                # 执行动作
                x, y = state
                dx, dy = self.actions[action]
                next_state = (x + dx, y + dy)
                steps += 1
                
                # 给予奖励
                if next_state == self.maze.end:
                    reward = 100  # 到达目标
                else:
                    reward = -1   # 每走一步都有小惩罚，鼓励尽快到达
                
                # 学习
                self.learn(state, action, reward, next_state)
                state = next_state
            
            # 每100回合打印一次进度
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}, Steps taken: {steps}")


class RandomAgent:
    def __init__(self, maze):
        self.maze = maze
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        
    def choose_action(self, state):
        """随机选择一个有效动作"""
        x, y = state
        valid_actions = []
        
        for action in self.actions:
            nx, ny = x + action[0], y + action[1]
            if self.maze.is_valid_move(nx, ny):
                valid_actions.append(action)
                
        return random.choice(valid_actions) if valid_actions else None


def simulate(maze, q_agent, random_agent):
    """模拟两个智能体同时走迷宫"""
    print("Starting simulation...")
    print("Legend: S=Start, G=Goal, A=Q-Learning Agent, B=Random Agent, #=Obstacle")
    time.sleep(2)
    
    # 两个智能体的初始位置
    agent1_pos = maze.start
    agent2_pos = maze.start
    
    # 步数计数
    agent1_steps = 0
    agent2_steps = 0
    
    # 标记是否到达终点
    agent1_done = False
    agent2_done = False
    
    while not (agent1_done and agent2_done):
        # 打印当前迷宫状态
        maze.print_maze(agent1_pos, agent2_pos)
        print(f"Steps - Q-Learning Agent: {agent1_steps}, Random Agent: {agent2_steps}")
        
        # Q-learning智能体移动（如果还没到达终点）
        if not agent1_done:
            action_idx = q_agent.choose_action(agent1_pos, is_training=False)
            if action_idx is not None:
                dx, dy = q_agent.actions[action_idx]
                agent1_pos = (agent1_pos[0] + dx, agent1_pos[1] + dy)
                agent1_steps += 1
                agent1_done = (agent1_pos == maze.end)
        
        # 随机智能体移动（如果还没到达终点）
        if not agent2_done:
            action = random_agent.choose_action(agent2_pos)
            if action is not None:
                agent2_pos = (agent2_pos[0] + action[0], agent2_pos[1] + action[1])
                agent2_steps += 1
                agent2_done = (agent2_pos == maze.end)
        
        # 等待1秒，方便观察
        time.sleep(0.5)
    
    # 最终状态
    maze.print_maze(agent1_pos, agent2_pos)
    print(f"Simulation complete!")
    print(f"Q-Learning Agent reached goal in {agent1_steps} steps")
    print(f"Random Agent reached goal in {agent2_steps} steps")


if __name__ == "__main__":
    # 创建迷宫
    maze = Maze(size=10)
    print("Generated Maze:")
    maze.print_maze(maze.start, maze.start)  # 初始时两个智能体都在起点
    time.sleep(2)
    
    # 创建并训练Q-learning智能体
    q_agent = QLearningAgent(maze)
    q_agent.train(episodes=1000)
    
    # 创建随机智能体
    random_agent = RandomAgent(maze)
    
    # 模拟对比
    simulate(maze, q_agent, random_agent)
