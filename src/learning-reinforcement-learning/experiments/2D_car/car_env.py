import numpy as np
import pyglet


pyglet.clock.set_fps_limit(10000)


class CarEnv:
    n_sensor = 5  # Number of sensors
    action_dim = 1  # Action dimension
    state_dim = n_sensor  # State dimension matches number of sensors
    sensor_max = 150.  # Max sensor range
    start_point = [450, 300]  # Starting position of the car
    speed = 50.  # Speed of the car
    dt = 0.1  # Time step
    viewer = None  # Viewer object for rendering
    viewer_xy = (500, 500)  # Window size

    def __init__(self, discrete_action=False):
        """ Initialize the environment. """
        self.is_discrete_action = discrete_action
        self.actions = [-1, 0, 1] if discrete_action else None
        self.action_bound = [-1, 1] if not discrete_action else None
        self.terminal = False
        self.car_info = np.array([0, 0, 0, 20, 40], dtype=np.float64)  # (x, y, rotation, width, length)
        self.obstacle_coords = np.array([
            [120, 120],
            [380, 120],
            [380, 380],
            [120, 380],
        ])
        self.sensor_info = np.full((self.n_sensor, 3), self.sensor_max)  # Sensors: (distance, end_x, end_y)

    def step(self, action):
        """ Take a step in the environment given an action. """
        action = self._get_action_value(action)
        self._update_car_position(action)
        self._update_sensor_info()
        state = self._get_state()
        reward = -1 if self.terminal else 0
        return state, reward, self.terminal

    def reset(self):
        """ Reset the environment to the initial state. """
        self.terminal = False
        self.car_info[:3] = [*self.start_point, -np.pi / 2]
        self._update_sensor_info()
        return self._get_state()

    def render(self):
        """ Render the environment (car and obstacles) using pyglet. """
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.car_info, self.sensor_info, self.obstacle_coords)
        self.viewer.render()

    def sample_action(self):
        """ Sample a random action (discrete or continuous). """
        if self.is_discrete_action:
            return np.random.choice(self.actions)
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def set_fps(self, fps=30):
        """ Set the frames per second for the simulation. """
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self):
        """ Get the current state of the environment. """
        return self.sensor_info[:, 0] / self.sensor_max

    def _get_action_value(self, action):
        """ Convert action into the appropriate value based on action type. """
        if self.is_discrete_action:
            return self.actions[action]
        return np.clip(action, *self.action_bound)[0]

    def _update_car_position(self, action):
        """ Update the car's position and rotation. """
        self.car_info[2] += action * np.pi / 30  # Update rotation (max 6 degrees)
        direction = np.array([np.cos(self.car_info[2]), np.sin(self.car_info[2])])
        self.car_info[:2] += self.speed * self.dt * direction

    def _update_sensor_info(self):
        """ Update the sensor information (distances and intersections). """
        cx, cy, rotation = self.car_info[:3]
        sensor_angles = np.linspace(-np.pi / 2, np.pi / 2, self.n_sensor)
        sensor_positions = self._get_sensor_end_positions(cx, cy, sensor_angles)

        self.sensor_info[:, -2:] = self._apply_rotation(sensor_positions, rotation, cx, cy)
        self._check_sensor_collisions(cx, cy)

    def _get_sensor_end_positions(self, cx, cy, angles):
        """ Calculate the end positions of the sensors. """
        xs = cx + self.sensor_max * np.cos(angles)
        ys = cy + self.sensor_max * np.sin(angles)
        return np.column_stack((xs, ys))

    def _apply_rotation(self, positions, rotation, cx, cy):
        """ Apply rotation to the sensor positions. """
        dx = positions[:, 0] - cx
        dy = positions[:, 1] - cy
        rotated_x = dx * np.cos(rotation) - dy * np.sin(rotation)
        rotated_y = dx * np.sin(rotation) + dy * np.cos(rotation)
        return np.column_stack((rotated_x + cx, rotated_y + cy))

    def _check_sensor_collisions(self, cx, cy):
        """ Check for sensor collisions with obstacles or window boundaries. """
        for si, sensor in enumerate(self.sensor_info):
            s = sensor[-2:] - np.array([cx, cy])
            distances = [self.sensor_max]  # List to store possible intersection distances
            intersections = [sensor[-2:]]  # List to store intersections with obstacles

            # Check for obstacle intersections
            for obstacle in self.obstacle_coords:
                self._check_obstacle_intersection(obstacle, s, distances, intersections)

            # Check for window boundary intersections
            self._check_window_intersection(s, distances, intersections)

            # Update sensor distance
            min_distance = np.min(distances)
            self.sensor_info[si, 0] = min_distance

            if min_distance < self.car_info[-1] / 2:  # If too close to an obstacle, terminate
                self.terminal = True

    def _check_obstacle_intersection(self, obstacle, s, distances, intersections):
        """ Check if a sensor intersects an obstacle. """
        p = obstacle
        r = self.obstacle_coords[(i + 1) % len(self.obstacle_coords)] - obstacle
        cross_prod = np.cross(r, s)
        if cross_prod != 0:
            t = np.cross((q - p), s) / cross_prod
            u = np.cross((q - p), r) / cross_prod
            if 0 <= t <= 1 and 0 <= u <= 1:
                intersection = q + u * s
                intersections.append(intersection)
                distances.append(np.linalg.norm(u * s))

    def _check_window_intersection(self, s, distances, intersections):
        """ Check if a sensor intersects with the window boundaries. """
        window = np.array([[0, 0], [self.viewer_xy[0], 0], [self.viewer_xy[0], self.viewer_xy[1]], [0, self.viewer_xy[1]], [0, 0]])
        for oi in range(4):
            p = window[oi]
            r = window[(oi + 1) % len(window)] - window[oi]
            cross_prod = np.cross(r, s)
            if cross_prod != 0:
                t = np.cross((q - p), s) / cross_prod
                u = np.cross((q - p), r) / cross_prod
                if 0 <= t <= 1 and 0 <= u <= 1:
                    intersection = p + t * r
                    intersections.append(intersection)
                    distances.append(np.linalg.norm(intersection - q))


class Viewer(pyglet.window.Window):
    def __init__(self, width, height, car_info, sensor_info, obstacle_coords):
        super().__init__(width, height, resizable=False, caption='2D Car Simulation')
        pyglet.gl.glClearColor(1, 1, 1, 1)  # Set background color

        self.car_info = car_info
        self.sensor_info = sensor_info
        self.obstacle_coords = obstacle_coords

        # Initialize graphics batch
        self.batch = pyglet.graphics.Batch()
        self.sensors = self._create_sensors()
        self.car = self._create_car()
        self.obstacle = self._create_obstacles()

    def _create_sensors(self):
        """ Create sensor lines for rendering. """
        sensor_lines = []
        for _ in range(len(self.sensor_info)):
            sensor_lines.append(self.batch.add(2, pyglet.gl.GL_LINES, None, ('v2f', [0, 0, 0, 0]), ('c3B', [73, 73, 73] * 2)))
        return sensor_lines

    def _create_car(self):
        """ Create the car rectangle for rendering. """
        return self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [0, 0] * 4), ('c3B', [249, 86, 86] * 4))

    def _create_obstacles(self):
        """ Create the obstacles for rendering. """
        return self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', self.obstacle_coords.flatten()), ('c3B', [134, 181, 244] * 4))

    def render(self):
        """ Render the environment with updated car and sensor positions. """
        pyglet.clock.tick()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        """ Draw everything on the screen. """
        self.clear()
        self.batch.draw()

    def _update(self):
        """ Update the car and sensor positions for rendering. """
        cx, cy, r, w, l = self.car_info
        # Update sensor positions
        for i, sensor in enumerate(self.sensors):
            sensor.vertices = [cx, cy, *self.sensor_info[i, -2:]]
        # Update car position
        car_coords = [
            [cx + l / 2, cy + w / 2],
            [cx - l / 2, cy + w / 2],
            [cx - l / 2, cy - w / 2],
            [cx + l / 2, cy - w / 2]
        ]
        rotated_coords = self._rotate_car(car_coords, r, cx, cy)
        self.car.vertices = np.array(rotated_coords).flatten()

    def _rotate_car(self, coords, rotation, cx, cy):
        """ Rotate car coordinates around the center. """
        rotated_coords = []
        for x, y in coords:
            dx, dy = x - cx, y - cy
            rotated_x = dx * np.cos(rotation) - dy * np.sin(rotation)
            rotated_y = dx * np.sin(rotation) + dy * np.cos(rotation)
            rotated_coords.extend([rotated_x + cx, rotated_y + cy])
        return rotated_coords


def main():
    np.random.seed(1)  # For reproducibility
    env = CarEnv()
    env.set_fps(30)

    # Run the environment for 20 episodes
    for ep in range(20):
        state = env.reset()
        while not env.terminal:
            env.render()
            action = env.sample_action()
            state, reward, done = env.step(action)
            if done:
                break


if __name__ == '__main__':
    main()
