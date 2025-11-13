import numpy as np
import pyglet


# 设置帧率限制
pyglet.clock.set_fps_limit(10000)

class RobotArmEnvironment:
    """模拟一个机器人臂的环境，其中包含操作和渲染机器人臂的功能。"""
    
    ACTION_BOUNDS = [-1, 1]  # 动作范围
    ACTION_DIM = 2           # 动作维度
    STATE_DIM = 7            # 状态维度
    REFRESH_RATE = 0.1       # 刷新率
    ARM1_LENGTH = 100        # 第一段臂的长度
    ARM2_LENGTH = 100        # 第二段臂的长度
    GRAB_THRESHOLD = 50      # 抓取物体的阈值
    POINT_SIZE = 15          # 目标点的大小

    def __init__(self, mode='easy'):
        """
        初始化机器人臂环境，设置模式（easy 或 hard）以及机器人臂和目标点的初始状态。
        """
        self.mode = mode
        self.arm_info = np.zeros((2, 4))  # 机器人臂信息：长度、角度、坐标
        self.arm_info[0, 0] = self.ARM1_LENGTH
        self.arm_info[1, 0] = self.ARM2_LENGTH
        self.target_point = np.array([250, 303])  # 目标点位置
        self.initial_target_point = self.target_point.copy()  # 目标点的初始位置
        self.center_coord = np.array([400, 400]) / 2  # 中心坐标
        self.is_target_grabbed = False  # 是否已抓取目标点
        self.grab_counter = 0  # 记录抓取次数
        self.viewer = None

    def step(self, action):
        """
        执行动作，更新机器人臂的位置，并计算奖励。
        :param action: 动作，表示机器臂关节的角速度
        :return: 状态、奖励、是否抓取到目标
        """
        action = np.clip(action, *self.ACTION_BOUNDS)  # 限制动作范围
        self.arm_info[:, 1] += action * self.REFRESH_RATE  # 更新臂的角度
        self.arm_info[:, 1] %= np.pi * 2  # 确保角度在0到2π之间

        # 计算机器人臂末端的坐标
        arm1_angle = self.arm_info[0, 1]
        arm2_angle = self.arm_info[1, 1]
        arm1_dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1_angle), self.arm_info[0, 0] * np.sin(arm1_angle)])
        arm2_dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2_angle), self.arm_info[1, 0] * np.sin(arm2_angle)])
        
        self.arm_info[0, 2:4] = self.center_coord + arm1_dx_dy  # 第一段臂末端坐标
        self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2_dx_dy  # 第二段臂末端坐标

        # 获取状态和奖励
        state, arm2_distance = self._get_state()
        reward = self._calculate_reward(arm2_distance)

        return state, reward, self.is_target_grabbed

    def reset(self):
        """重置环境到初始状态。"""
        self.is_target_grabbed = False
        self.grab_counter = 0

        if self.mode == 'hard':
            self.target_point = np.clip(np.random.rand(2) * 400, 100, 300)
        else:
            # 随机化机器人臂的角度
            arm1_angle, arm2_angle = np.random.rand(2) * np.pi * 2
            self.arm_info[0, 1] = arm1_angle
            self.arm_info[1, 1] = arm2_angle
            self._update_arm_positions()

            self.target_point = self.initial_target_point

        return self._get_state()[0]

    def render(self):
        """渲染环境（显示机器人臂和目标点）。"""
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.target_point, self.POINT_SIZE)
        self.viewer.render()

    def sample_action(self):
        """随机采样一个动作。"""
        return np.random.uniform(*self.ACTION_BOUNDS, size=self.ACTION_DIM)

    def set_fps(self, fps=30):
        """设置帧率。"""
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self):
        """
        获取当前状态，包括机器人臂的位置、与目标点的距离等信息。
        :return: 状态信息，距离信息
        """
        arm_end_coords = self.arm_info[:, 2:4]
        distance_to_target = np.ravel(arm_end_coords - self.target_point)
        normalized_center_distance = (self.center_coord - self.target_point) / 200
        target_grabbed = 1 if self.grab_counter > 0 else 0

        state = np.hstack([target_grabbed, distance_to_target / 200, normalized_center_distance])
        return state, distance_to_target[-2:]

    def _calculate_reward(self, distance):
        """
        计算奖励，基于与目标点的距离。
        :param distance: 与目标点的距离
        :return: 奖励值
        """
        distance_magnitude = np.linalg.norm(distance)
        reward = -distance_magnitude / 200  # 目标越远，奖励越低

        if distance_magnitude < self.POINT_SIZE and not self.is_target_grabbed:
            reward += 1.0
            self.grab_counter += 1
            if self.grab_counter > self.GRAB_THRESHOLD:
                reward += 10.0
                self.is_target_grabbed = True
        elif distance_magnitude > self.POINT_SIZE:
            self.grab_counter = 0
            self.is_target_grabbed = False

        return reward

    def _update_arm_positions(self):
        """根据当前角度更新机器人臂的位置。"""
        arm1_angle = self.arm_info[0, 1]
        arm2_angle = self.arm_info[1, 1]
        arm1_dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1_angle), self.arm_info[0, 0] * np.sin(arm1_angle)])
        arm2_dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2_angle), self.arm_info[1, 0] * np.sin(arm2_angle)])

        self.arm_info[0, 2:4] = self.center_coord + arm1_dx_dy
        self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2_dx_dy


class Viewer(pyglet.window.Window):
    """用于渲染机器人臂和目标点的Viewer类。"""
    
    def __init__(self, arm_info, target_point, point_size):
        super().__init__(width=800, height=800, caption="Robot Arm", resizable=True)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.target_point = target_point
        self.point_size = point_size
        self.center_coord = np.array([400, 400])

    def render(self):
        """渲染窗口内容。"""
        self.clear()
        self.update_arm_positions()
        self.batch.draw()

    def update_arm_positions(self):
        """更新机器臂和目标点的绘制位置。"""
        arm1_coords = self.arm_info[0, 2:4]
        arm2_coords = self.arm_info[1, 2:4]
        # 这里添加代码更新绘制内容，例如绘制机器人臂、目标点等
        pass


def main():
    env = RobotArmEnvironment(mode='easy')
    env.reset()

    # 模拟一个简单的动作序列
    for _ in range(100):
        action = env.sample_action()
        state, reward, target_grabbed = env.step(action)
        env.render()
        if target_grabbed:
            print("Target grabbed!")
            break


if __name__ == "__main__":
    main()
