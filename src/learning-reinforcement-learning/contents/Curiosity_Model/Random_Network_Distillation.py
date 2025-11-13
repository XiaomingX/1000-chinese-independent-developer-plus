import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt


class CuriosityNet:
    def __init__(self, n_actions, n_states, learning_rate=0.01, gamma=0.95, epsilon=1.0,
                 target_replace_interval=300, memory_size=10000, batch_size=128, output_graph=False):
        """
        初始化CuriosityNet类，设置各项超参数并构建神经网络。
        """
        self.n_actions = n_actions  # 动作空间的大小
        self.n_states = n_states    # 状态空间的大小
        self.learning_rate = learning_rate
        self.gamma = gamma          # 奖励折扣因子
        self.epsilon = epsilon      # 探索概率
        self.target_replace_interval = target_replace_interval  # 目标网络更新间隔
        self.memory_size = memory_size  # 记忆库大小
        self.batch_size = batch_size    # 批量大小
        self.state_encoding_size = 1000  # 给预测器学习的一个硬任务

        self.memory_counter = 0  # 记忆库的当前索引
        self.learn_step_counter = 0  # 学习步骤计数器

        # 初始化记忆库，用于存储每个状态转换
        self.memory = np.zeros((self.memory_size, n_states * 2 + 2))

        # 创建神经网络和训练操作
        self.tfs, self.tfa, self.tfr, self.tfs_, self.pred_train, self.dqn_train, self.q = self._build_nets()

        # 获取目标网络和评估网络的变量
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        # 目标网络和评估网络的参数替换操作
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # 创建 TensorFlow 会话
        self.sess = tf.Session()

        # 如果需要，输出 TensorFlow 图
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        # 初始化变量
        self.sess.run(tf.global_variables_initializer())

    def _build_nets(self):
        """
        构建随机网络、预测器网络和强化学习的Q网络。
        """
        # 定义占位符
        tfs = tf.placeholder(tf.float32, [None, self.n_states], name="s")    # 输入状态
        tfa = tf.placeholder(tf.int32, [None, ], name="a")                    # 输入动作
        tfr = tf.placeholder(tf.float32, [None, ], name="ext_r")              # 外部奖励
        tfs_ = tf.placeholder(tf.float32, [None, self.n_states], name="s_")  # 输入下一状态

        # 构建固定的随机网络
        with tf.variable_scope("random_net"):
            rand_encode_s_ = tf.layers.dense(tfs_, self.state_encoding_size)

        # 构建预测器
        ri, pred_train = self._build_predictor(tfs_, rand_encode_s_)

        # 构建 DQN 网络
        q, dqn_loss, dqn_train = self._build_dqn(tfs, tfa, ri, tfr, tfs_)
        return tfs, tfa, tfr, tfs_, pred_train, dqn_train, q

    def _build_predictor(self, s_, rand_encode_s_):
        """
        构建预测器，用于计算内部奖励。
        """
        with tf.variable_scope("predictor"):
            net = tf.layers.dense(s_, 128, tf.nn.relu)
            out = tf.layers.dense(net, self.state_encoding_size)

        with tf.name_scope("int_r"):
            ri = tf.reduce_sum(tf.square(rand_encode_s_ - out), axis=1)  # 内部奖励计算

        # 定义优化器，用于最小化内部奖励
        train_op = tf.train.RMSPropOptimizer(self.learning_rate, name="predictor_opt").minimize(
            tf.reduce_mean(ri), var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "predictor"))

        return ri, train_op

    def _build_dqn(self, s, a, ri, re, s_):
        """
        构建 DQN 网络。
        """
        # 评估网络
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(s, 128, tf.nn.relu)
            q = tf.layers.dense(e1, self.n_actions, name="q")

        # 目标网络
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(s_, 128, tf.nn.relu)
            q_ = tf.layers.dense(t1, self.n_actions, name="q_")

        # Q目标计算
        with tf.variable_scope('q_target'):
            q_target = re + ri + self.gamma * tf.reduce_max(q_, axis=1, name="Qmax_s_")

        # 根据动作选择Q值
        with tf.variable_scope('q_wrt_a'):
            a_indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
            q_wrt_a = tf.gather_nd(params=q, indices=a_indices)

        # 损失函数：均方误差
        loss = tf.losses.mean_squared_error(labels=q_target, predictions=q_wrt_a)
        train_op = tf.train.RMSPropOptimizer(self.learning_rate, name="dqn_opt").minimize(
            loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "eval_net"))

        return q, loss, train_op

    def store_transition(self, state, action, reward, next_state):
        """
        存储状态转移。
        """
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        """
        选择动作，基于ε-greedy策略。
        """
        state = state[np.newaxis, :]  # 增加批次维度

        if np.random.uniform() < self.epsilon:
            # 选择最优动作
            actions_value = self.sess.run(self.q, feed_dict={self.tfs: state})
            action = np.argmax(actions_value)
        else:
            # 随机选择动作
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        """
        执行学习步骤，更新网络权重。
        """
        if self.learn_step_counter % self.target_replace_interval == 0:
            # 替换目标网络的参数
            self.sess.run(self.target_replace_op)

        # 从记忆库中随机采样
        top = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
        sample_index = np.random.choice(top, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 分割批次数据
        states, actions, rewards, next_states = batch_memory[:, :self.n_states], batch_memory[:, self.n_states], \
            batch_memory[:, self.n_states + 1], batch_memory[:, -self.n_states:]

        # 执行 DQN 网络训练
        self.sess.run(self.dqn_train, feed_dict={self.tfs: states, self.tfa: actions, self.tfr: rewards, self.tfs_: next_states})

        if self.learn_step_counter % 100 == 0:
            # 延迟训练预测器，以保持探索心态
            self.sess.run(self.pred_train, feed_dict={self.tfs_: next_states})

        self.learn_step_counter += 1


def main():
    """
    主函数：训练和评估模型。
    """
    # 创建MountainCar环境
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    # 初始化CuriosityNet模型
    curiosity_net = CuriosityNet(n_actions=3, n_states=2, learning_rate=0.01, output_graph=False)

    ep_steps = []  # 记录每个episode的步数

    for episode in range(200):
        state = env.reset()  # 重置环境
        steps = 0
        while True:
            action = curiosity_net.choose_action(state)
            next_state, reward, done, _ = env.step(action)  # 执行动作
            curiosity_net.store_transition(state, action, reward, next_state)  # 存储状态转移
            curiosity_net.learn()  # 学习

            if done:
                print(f'Episode: {episode}, Steps: {steps}')
                ep_steps.append(steps)
                break

            state = next_state  # 更新状态
            steps += 1

    # 绘制学习曲线
    plt.plot(ep_steps)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.show()


if __name__ == "__main__":
    main()
