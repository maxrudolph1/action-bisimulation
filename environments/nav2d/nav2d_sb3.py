import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.utils import seeding
import heapq
import numpy as np
import yaml
# from copy import copy


class Navigate2D(gym.Env):
    actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=int)
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        num_obstacles,
        env_config=None,
        maze=False,
        grid_size=20,
        obstacle_diameter=2,
        min_goal_dist=1,
        max_timesteps=50,
        hard_coded_obs=False,
        obstacle_distance_metric=False,
        static_goal=False,
    ):

        self.n_obs = num_obstacles
        self.size = grid_size
        self.r_obs = np.max([obstacle_diameter // 2, 1])
        self.min_goal_dist = min_goal_dist
        self.max_timesteps = max_timesteps
        self.hard_coded_obs = hard_coded_obs
        self.obstacle_distance_metric = obstacle_distance_metric
        self.static_goal = static_goal
        self.observation_space = spaces.Box(
            0, 255, [3, self.size, self.size], np.uint8
        )

        self.action_space = spaces.Discrete(4)

        self.obstacles = None
        self.pos = None
        self.goal = None
        self.grid = None
        self.dist = None
        self.np_random = None
        self.step_count = 0
        self.cumulative_reward = 0
        self.optimal_path_length = -1
        if env_config is not None and env_config != -1:
            with open(env_config, 'r') as file:
                self.config = yaml.load(file)
        else:
            self.config = -1
        self.render_mode = 'rgb_array'
        # print(grid_size)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self, seed=0, options=None):
        while True:
            self.step_count = 0
            self.cumulative_reward = 0
            grid = np.zeros((3, self.size, self.size), dtype=np.float32)
            obs = np.zeros((self.n_obs, 2), dtype=np.uint8)
            for i in range(self.n_obs):
                center = self.np_random.integers(0, self.size, 2)
                minX = np.maximum(center[0] - self.r_obs, 0)
                minY = np.maximum(center[1] - self.r_obs, 0)
                maxX = np.minimum(center[0] + self.r_obs, self.size)
                maxY = np.minimum(center[1] + self.r_obs, self.size)
                grid[0, minX:maxX, minY:maxY] = 1.0
                obs[i] = center

            min_center = self.size // 2 - self.r_obs
            max_center = self.size // 2 + self.r_obs
            grid[0, min_center:max_center, min_center:max_center] = 0.0

            free_idx = np.argwhere(grid[0, :, :] == 0)
            start = free_idx[
                self.np_random.integers(0, free_idx.shape[0], 1), :
            ].squeeze()
            free_idx = free_idx[
                np.linalg.norm(free_idx - start, ord=1, axis=-1) >= self.min_goal_dist
            ]

            grid[1, start[0], start[1]] = 1.0

            if self.static_goal:
                goal = np.array([self.size // 2, self.size // 2])
                grid[2, goal[0], goal[1]] = 1.0
                min_center = self.size // 2 - self.r_obs
                max_center = self.size // 2 + self.r_obs
                grid[0, min_center:max_center, min_center:max_center] = 0.0
            else:
                goal = free_idx[
                    self.np_random.integers(0, free_idx.shape[0], 1), :
                ].squeeze()
                grid[2, goal[0], goal[1]] = 1.0
                goal_x = goal[0] - self.r_obs
                goal_y = goal[1] - self.r_obs
                grid[0, goal_x:(goal_x+self.r_obs), goal_y:(goal_y+self.r_obs)] = 0.0

            self.obstacles = obs
            self.pos = start
            self.goal = goal
            self.grid = grid
            self.dist = np.linalg.norm(start - goal, ord=1)

            optimal_path = self.find_path()

            if optimal_path is not None:
                self.optimal_path_length = len(optimal_path)
                break

        return self._get_obs(self.grid, self.pos, self.goal), {}

    def step(self, action):
        self.step_count += 1
        # old_grid = self.grid.copy()
        old_pos = self.pos.copy()
        try:
            new_pos = old_pos + self.actions[action]
        except:
            import pdb
            pdb.set_trace()
        reward = -1
        if (
            np.all(new_pos >= 0)
            and np.all(new_pos < self.size)
            and not self.grid[0, new_pos[0], new_pos[1]]
        ):
            self.dist = np.linalg.norm(new_pos - self.goal, ord=1)
            self.grid[1, old_pos[0], old_pos[1]] = 0
            self.grid[1, new_pos[0], new_pos[1]] = 1.0
            np.copyto(self.pos, new_pos)
            if np.all(new_pos == self.goal):
                reward = 0
        self.cumulative_reward += reward

        terminated = (reward == 0)

        truncated = (self.step_count >= self.max_timesteps)

        info = {}
        info["pos"] = self.pos.copy()
        info["goal"] = self.goal.copy()
        info["dist"] = self.dist.copy()
        info["grid"] = self.grid.copy()
        info["optimal_path_length"] = self.optimal_path_length
        info["steps_taken"] = int(self.step_count)
        info["cumulative_reward"] = int(self.cumulative_reward)
        info["success"] = terminated
        if terminated:
            info["terminal_observation"] = self._get_obs(self.grid, self.pos, self.goal)
        if truncated:
            info["final_observation"] = self._get_obs(self.grid, self.pos, self.goal)
        return self._get_obs(self.grid, self.pos, self.goal), reward, terminated, truncated, info

    def _get_obs(self, grid, pos=None, goal=None):
        return (grid * 255).astype(np.uint8)

    def find_path(self):
        # each element is (total_cost, forward_cost, pos, action path)
        queue = [(np.linalg.norm(self.pos - self.goal, ord=1), 0, tuple(self.pos), [])]
        visited = set()
        while queue:
            _, fcost, pos, actions = heapq.heappop(queue)
            if pos in visited:
                continue
            if np.all(np.array(pos) == self.goal):
                return actions
            visited.add(pos)
            for i, action in enumerate(self.actions):
                new_pos = np.array(pos) + action
                if (np.all(new_pos >= 0) and np.all(new_pos < self.size) and not self.grid[0, new_pos[0], new_pos[1]]):
                    heapq.heappush(
                        queue,
                        (
                            np.linalg.norm(new_pos - self.goal, ord=1) + fcost + 1,
                            fcost + 1,
                            tuple(new_pos),
                            actions + [i],
                        ),
                    )
        return None

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return self.grid.transpose(1, 2, 0)
        else:
            super().render(mode=mode)
