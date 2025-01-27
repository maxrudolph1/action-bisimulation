import gymnasium as gym
# from gym import spaces
# from gym.utils import seeding
import gymnasium.utils.seeding as seeding
import gymnasium.spaces as spaces
import heapq
import numpy as np
import yaml


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
            -1.0, 1.0, [3, self.size, self.size], np.float32
        )

        self.action_space = spaces.Discrete(4)

        self.obstacles = None
        self.pos = None
        self.goal = None
        self.grid = None
        self.dist = None
        self.buffer = None
        self.np_random = None
        self.step_count = 0
        if env_config is not None and env_config != -1:
            with open(env_config, 'r') as file:
                self.config = yaml.load(file)
        else:
            self.config = -1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self, seed=0, options=None):
        while True:
            self.step_count = 0
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
            goal = free_idx[
                self.np_random.integers(0, free_idx.shape[0], 1), :
            ].squeeze()
            grid[1, start[0], start[1]] = 1.0
            
            if self.static_goal:
                grid[2, self.size // 2, self.size // 2] = 1.0
                min_center = self.size // 2 - self.r_obs
                max_center = self.size // 2 + self.r_obs
                grid[0, min_center:max_center, min_center:max_center] = 0.0
            else:
                grid[2, goal[0], goal[1]] = 1.0
                goal_x = goal[0] - self.r_obs
                goal_y = goal[1] - self.r_obs
                grid[0, goal_x:(goal_x+self.r_obs), goal_y:(goal_y+self.r_obs)] = 0.0
            

            self.obstacles = obs
            self.pos = start
            self.goal = goal
            self.grid = grid
            self.dist = np.linalg.norm(start - goal, ord=1)
            self.buffer = []
        
            if self.find_path() is not None:
                break

        return self._get_obs(self.grid, self.pos, self.goal), {}

    def step(self, action):
        self.step_count += 1
        old_grid = self.grid.copy()
        old_pos = self.pos.copy()
        new_pos = old_pos + self.actions[action]
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
        done = reward == 0 or self.step_count >= self.max_timesteps

        self.buffer.append((old_grid, old_pos, action))
        info = {}
        info["pos"] = self.pos.copy()
        info["goal"] = self.goal.copy()
        info["dist"] = self.dist.copy()
        return self._get_obs(self.grid, self.pos, self.goal), reward, done, done, info

    def forward_oracle(self, state):
        # state: shape (n, c, h, w)
        # return: shape (4, n, c, h, w)
        grid = state[..., :: self.scale, :: self.scale]
        grid = (grid + 1) / 2
        old_pos = np.argwhere(grid[:, 1] != 0)  # shape (n, 3)
        assert np.all(old_pos[:, 0] == np.arange(state.shape[0]))
        old_pos = old_pos[:, 1:]  # shape (n, 2)

        new_pos = old_pos + self.actions[:, None, :]  # shape (4, n, 2)
        mask = np.all(
            [
                np.all(new_pos >= 0, axis=-1),  # shape (4, n)
                np.all(new_pos < self.size, axis=-1),  # shape (4, n)
                np.logical_not(
                    grid[
                        np.arange(grid.shape[0]),
                        0,
                        new_pos[:, :, 0] % self.size,
                        new_pos[:, :, 1] % self.size,
                    ]
                ),
            ],
            axis=0,
        )
        old_grid = grid.copy()
        grid = np.broadcast_to(grid, (4,) + grid.shape).copy()
        grid[:, np.arange(state.shape[0]), 1, old_pos[:, 0], old_pos[:, 1]] = 0
        grid[
            np.arange(4)[:, None],
            np.arange(state.shape[0]),
            1,
            new_pos[:, :, 0] % self.size,
            new_pos[:, :, 1] % self.size,
        ] = 1.0
        grid = np.where(mask[..., None, None, None], grid, old_grid)
        normed = grid * 2 - 1
        return normed.repeat(self.scale, axis=-2).repeat(self.scale, axis=-1)

    def _get_obs(self, grid, pos=None, goal=None):
        return grid * 2 - 1

    def her(self):
        fake_goal_grid = self.grid[1, :, :]
        fake_goal_pos = self.pos

        # buffer contains (grid, pos, action)
        buffer = self.buffer + [(self.grid.copy(), self.pos.copy(), None)]
        for grid, _, _ in buffer:
            grid[2, :, :] = fake_goal_grid

        ret = []
        for (grid, pos, action), (grid_next, pos_next, _) in zip(buffer, buffer[1:]):
            if np.all(pos == fake_goal_pos):
                break
            reward = 0 if np.all(pos_next == fake_goal_pos) else -1
            obs = self._get_obs(grid, pos, fake_goal_pos)
            obs_next = self._get_obs(grid_next, pos_next, fake_goal_pos)
            ret.append((obs, action, reward, obs_next))
            if reward == 0:
                break
        return ret

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
                if (
                    np.all(new_pos >= 0)
                    and np.all(new_pos < self.size)
                    and not self.grid[0, new_pos[0], new_pos[1]]
                ):
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
