from collections import deque
from d4rl.pointmaze import MazeEnv, U_MAZE, LARGE_MAZE, MEDIUM_MAZE
from gym.envs.mujoco import mujoco_env
import mujoco_py
import numpy as np
from gym import spaces
from gym.wrappers import LazyFrames
import cv2

from d4rl.pointmaze.maze_model import EMPTY, parse_maze, point_maze


def generate_maze_str(height, width, n_obs):
    ret = []
    obstacles = set(np.random.choice(height * width, size=n_obs, replace=False))
    for i in range(height + 2):
        for j in range(width + 2):
            if (
                i == 0
                or i == height + 1
                or j == 0
                or j == width + 1
                or (i - 1) * width + (j - 1) in obstacles
            ):
                ret.append("#")
            else:
                ret.append("O")
        ret.append("\\")
    return "".join(ret[:-1])


class VisualMazeEnv(MazeEnv):
    ACTIONS = np.array(
        [[-1, 0], [0, -1], [0, 1], [1, 0], [1, 1], [-1, -1], [-1, 1], [1, -1]]
    )
    RENDER_SIZE = 40

    def __init__(
        self,
        maze_spec=LARGE_MAZE,
        num_obstacles=0,
        reward_type="sparse",
        reset_target=True,
        max_timesteps=120,
        frame_skip=5,
        egocentric=False,
        **kwargs
    ):
        self.step_count = 0
        self.max_timesteps = max_timesteps
        self.initialized = False
        self.num_obstacles = num_obstacles
        super().__init__(maze_spec, reward_type, reset_target, **kwargs)
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.RENDER_SIZE, self.RENDER_SIZE),
            dtype=np.uint8,
        )
        self.frame_skip = frame_skip
        self.egocentric = egocentric
        self.position_buffer = []
        self.action_buffer = []
        self._get_viewer("rgb_array")

    def reset_model(self):
        self.step_count = 0
        self.initialized = True
        self.position_buffer = []
        self.action_buffer = []
        if self.num_obstacles:
            self.str_maze_spec = generate_maze_str(7, 10, self.num_obstacles)
            self.maze_arr = parse_maze(self.str_maze_spec)
            self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
            self.reset_locations.sort()
            self.empty_and_goal_locations = self.reset_locations

            model = point_maze(self.str_maze_spec)
            with model.asfile() as f:
                self.model = mujoco_py.load_model_from_path(f.name)
            self.sim = mujoco_py.MjSim(self.model)
            self.sim.forward()
            self.data = self.sim.data
            self.viewer = None
            self._viewers = {}
            self._get_viewer("rgb_array")

        super().reset_model()
        self.set_marker()  # required to update goal pos visually
        self.sim.model.vis.global_.fovy = 5
        self.sim.model.vis.map.zfar = 2000
        return self._get_obs()

    def step(self, action):
        if not self.initialized:
            return (
                np.zeros((self.RENDER_SIZE, self.RENDER_SIZE, 3), dtype=np.uint8),
                0,
                False,
                {},
            )

        self.position_buffer.append(self.sim.data.qpos.ravel().copy())
        self.action_buffer.append(action)
        self.step_count += 1
        self.clip_velocity()
        self.do_simulation(self.ACTIONS[action], self.frame_skip)
        self.set_marker()
        ob = self._get_obs()
        reward = self._compute_reward(self.sim.data.qpos.ravel(), self._target)
        done = reward == 0 or self.step_count >= self.max_timesteps

        return (
            ob,
            reward,
            done,
            {"dist": np.linalg.norm(self.sim.data.qpos.ravel() - self._target)},
        )

    def _compute_reward(self, pos, target):
        if self.reward_type == "sparse":
            reward = 0.0 if np.linalg.norm(pos - target) <= 0.5 else -1.0
        elif self.reward_type == "dense":
            reward = np.exp(-np.linalg.norm(pos - target))
        return reward

    def _get_obs(self):
        if self.initialized and self.egocentric:
            self.viewer.cam.lookat[:2] = self.sim.data.qpos.ravel() + 1.2
        obs = np.array(self.render(
            mode="rgb_array", width=self.RENDER_SIZE, height=self.RENDER_SIZE
        ))
        if self.initialized and self.egocentric:
            vec = self.sim.data.qpos.ravel() - self._target + 1e-8
            vec /= np.linalg.norm(vec)
            start = vec * (self.RENDER_SIZE // 8) + (self.RENDER_SIZE // 2)
            end = start + vec * (self.RENDER_SIZE // 8)
            cv2.line(obs, tuple(start[::-1].astype(int)), tuple(end[::-1].astype(int)), (0, 0, 0), 1)
        return np.ascontiguousarray(obs.transpose(2, 0, 1))

    def viewer_setup(self):
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0
        self.viewer.cam.distance = 60 if self.egocentric else 125
        if not self.egocentric:
            self.viewer.cam.lookat[:] = [5, 6.5, 0]

    def set_marker(self):
        # the coordinates are offset by 1.2 for some reason...
        self.data.site_xpos[self.model.site_name2id("target_site")] = np.array(
            [self._target[0] + 1.2, self._target[1] + 1.2, 0.0]
        )

    def her(self, frame_stack=1):
        fake_target = self.sim.data.qpos.ravel().copy()
        self.set_target(fake_target)
        self.position_buffer.append(fake_target)

        self.set_state(self.position_buffer[0], np.zeros(2))
        self.set_marker()
        first_obs = self._get_obs()
        frames = deque([first_obs for _ in range(frame_stack)], maxlen=frame_stack)

        obs = LazyFrames(list(frames), lz4_compress=True)
        ret = []
        for pos, action in zip(self.position_buffer[1:], self.action_buffer):
            self.set_state(pos, np.zeros(2))
            self.set_marker()
            frames.append(self._get_obs())
            obs_next = LazyFrames(list(frames), lz4_compress=True)
            reward = self._compute_reward(pos, fake_target)
            ret.append((obs, action, reward, obs_next))
            obs = obs_next

        return ret
