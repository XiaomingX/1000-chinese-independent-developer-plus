import numpy as np
import tensorflow as tf

# 固定随机种子以保证结果可重复
np.random.seed(1)
tf.set_random_seed(1)

class DoubleDQN:
    def __init__(self, n_actions, n_features, learning_rate=0.005, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=3000, batch_size=32, e_greedy_increment=None,
                 output_graph=False, double_q=True, sess=None):
        """
        初始化 Double DQN 算法的参数和网络。
        """
        self.n_actions = n_actions  # 动作数
        self.n_features = n_features  # 特征数
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励衰减因子
        self.epsilon_max = e_greedy  # 最大探索概率
        self.replace_target_iter = replace_target_iter  # 更新目标网络的频率
        self.memory_size = memory_size  # 经验回放池大小
        self.batch_size = batch_size  # 批处理大小
        self.epsilon_increment = e_greedy_increment  # epsilon增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 初始epsilon

        self.double_q = double_q  # 是否使用 Double DQN
        self.learn_step_counter = 0  # 学习步数计数器

        # 初始化经验回放池
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        
        # 构建神经网络
        self._build_net()

        # 构建目标网络替换操作
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # 如果没有传入sess，则创建新的session
        self.sess = sess if sess else tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # 如果output_graph为True，则输出TensorBoard日志
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        
        # 存储损失值历史
        self.cost_his = []

    def _build_net(self):
        """
        构建评估网络和目标网络。
        """
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            """
            构建神经网络层
            """
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)  # 第一层使用ReLU激活

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2  # 输出层
            return out

        # ------------------ 构建评估网络 ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 当前状态
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 目标Q值

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))  # 计算损失函数

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)  # 使用RMSProp优化器

        # ------------------ 构建目标网络 ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # 下一个状态
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        """
        存储转移到经验回放池。
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))  # 合并当前状态、动作、奖励和下一个状态
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # 更新经验池
        self.memory_counter += 1

    def choose_action(self, observation):
        """
        选择一个动作，使用epsilon-greedy策略。
        """
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)  # 选择Q值最大的动作

        if not hasattr(self, 'q'):  # 初始化Q值记录
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # 探索
            action = np.random.randint(0, self.n_actions)  # 随机选择动作
        return action

    def learn(self):
        """
        学习过程：更新目标网络，采样经验，更新Q值，优化网络。
        """
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 从经验回放池中随机采样
        sample_index = np.random.choice(self.memory_size, size=self.batch_size) if self.memory_counter > self.memory_size else \
            np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 计算Q值
        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
                                            feed_dict={self.s_: batch_memory[:, -self.n_features:],  # 下一个状态
                                                       self.s: batch_memory[:, -self.n_features:]})  # 当前状态
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # 动作的索引
        reward = batch_memory[:, self.n_features + 1]  # 奖励

        # Double DQN更新规则
        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)  # 选择最大Q值的动作
            selected_q_next = q_next[batch_index, max_act4next]  # 使用Q评估网络的动作选择Q值
        else:
            selected_q_next = np.max(q_next, axis=1)  # 传统DQN，选择最大Q值

        # 更新目标Q值
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # 训练评估网络
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # 更新epsilon值
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

# 主函数，整合所有步骤
def main():
    # 参数设置
    n_actions = 4  # 动作数（例如，四个方向）
    n_features = 4  # 状态空间的维度（例如，机器人位置的x, y坐标）
    agent = DoubleDQN(n_actions, n_features)

    # 假设我们有一个环境（此处使用伪代码示例）
    for episode in range(1000):
        observation = np.random.rand(n_features)  # 伪造的初始状态
        done = False

        while not done:
            action = agent.choose_action(observation)  # 选择动作
            next_observation = np.random.rand(n_features)  # 伪造的下一个状态
            reward = np.random.rand()  # 伪造的奖励
            done = np.random.rand() < 0.1  # 假设随机结束

            agent.store_transition(observation, action, reward, next_observation)  # 存储经验
            agent.learn()  # 进行学习

            observation = next_observation  # 更新状态

if __name__ == '__main__':
    main()
