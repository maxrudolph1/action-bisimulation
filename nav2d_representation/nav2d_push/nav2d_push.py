import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import tqdm
from numba import njit


class Navigate2DPush(gym.Env):
    actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=int)
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, h):
        self.size = h["grid_size"]
        self.d_block = h["block_diameter"]
        self.goal_dist_min = h["min_goal_dist"]
        self.factorized = h["use_factorized_state"]
        self.max_step_count = h["max_episode_length"]

        if self.factorized:
            self.observation_space = spaces.Box(
                -1.0, 1.0, [self.n_obs + 2, 2], np.float32
            )
        else:
            self.observation_space = spaces.Box(
                -1.0, 1.0, [3, self.size, self.size], np.float32
            )

        self.action_space = spaces.Discrete(4)

        self.pos = None
        self.block_pos = None
        self.goal = None
        self.grid = None
        self.dist = None
        self.buffer = None
        self.np_random = None
        self.step_count = 0

        # self.grid_mean = np.zeros((3, self.size, self.size), dtype=np.float32)
        # if not self.factorized:
        #     self.grid_mean = self.calc_grid_norm(1000)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self):
        self.step_count = 0
        grid = np.zeros((3, self.size, self.size), dtype=np.float32)
        block_pos = self.np_random.randint(0, self.size, 2)
        grid[0][self._block_ind(block_pos)] = 1
        free_idx = np.argwhere(grid[0] == 0)
        start = free_idx[self.np_random.randint(0, free_idx.shape[0], 1), :].squeeze()
        # while True:
        #     finish = free_idx[
        #         self.np_random.randint(0, free_idx.shape[0], 1), :
        #     ].squeeze()
        #     if np.linalg.norm(start - finish) >= self.goal_dist_min:
        #         break
        grid[1, start[0], start[1]] = 1.0
        # grid[2, finish[0], finish[1]] = 1.0

        self.pos = start
        self.block_pos = block_pos
        self.grid = grid
        self.buffer = []

        return self.state(self.grid, self.pos, self.goal)

    def step(self, action):
        self.step_count += 1
        old_grid = self.grid.copy()
        old_pos = self.pos.copy()
        old_block_pos = self.block_pos.copy()
        new_pos = (old_pos + self.actions[action]) % self.size
        reward = -1

        block_ind = np.array(np.broadcast_arrays(*self._block_ind(old_block_pos)))
        if (new_pos[:, None, None] == block_ind).all(axis=0).any():
            new_block_pos = (old_block_pos + self.actions[action]) % self.size
            self.grid[0][self._block_ind(old_block_pos)] = 0
            self.grid[0][self._block_ind(new_block_pos)] = 1
            np.copyto(self.block_pos, new_block_pos)
        self.grid[1, old_pos[0], old_pos[1]] = 0
        self.grid[1, new_pos[0], new_pos[1]] = 1
        np.copyto(self.pos, new_pos)

        done = self.step_count >= self.max_step_count

        self.buffer.append([old_grid, old_pos, action, reward, done])
        info = {
            "image": self.grid.copy(),
            "position": self.pos.copy(),
        }
        return self.state(self.grid, self.pos, self.goal), reward, done, info

    def forward_oracle(self, state):
        # state: shape (n, c, h, w)
        # return: shape (4, n, c, h, w)
        assert not self.factorized
        return self._forward_oracle(state, self.size, self.actions)

    @staticmethod
    @njit
    def _forward_oracle(states, size, actions):
        result = np.stack((states, states, states, states))
        for i, state in enumerate(states):
            old_pos = np.argwhere(state[1] == 1)[0]  # shape (2,)
            old_block_ind = np.argwhere(state[0] == 1)  # shape (block_diam**2, 2)
            for j, action in enumerate(actions):
                new_pos = (old_pos + action) % size
                mask = new_pos == old_block_ind  # shape (block_diam**2, 2)
                if np.any(np.logical_and(mask[:, 0], mask[:, 1])):
                    new_block_ind = (old_block_ind + action) % size
                    for ind in old_block_ind:
                        result[j, i, 0, ind[0], ind[1]] = -1
                    for ind in new_block_ind:
                        result[j, i, 0, ind[0], ind[1]] = 1
                result[j, i, 1, old_pos[0], old_pos[1]] = -1
                result[j, i, 1, new_pos[0], new_pos[1]] = 1
        return result

    def state(self, grid, pos, goal):
        if self.factorized:
            state = np.concatenate(
                [pos[None, ...], self.obstacles], axis=0
            )
            half_size = (self.size - 1) / 2
            return (state - half_size) / half_size
        else:
            normed = grid * 2 - 1
            return normed

    def calc_grid_norm(self, num_episodes):
        states = []
        for _ in tqdm.tqdm(range(num_episodes)):
            self.reset()
            states.append(self.grid.copy())
            done = False
            while not done:
                _, _, done, _ = self.step(self.action_space.sample())
                states.append(self.grid.copy())

        states = np.array(states)
        return states.mean(axis=0)
        # self.image_std = states.std(dim=0)

    def her(self):
        goal_grid = self.grid[1, :, :]
        goal_pos = self.pos
        for i in range(len(self.buffer)):
            self.buffer[-1 - i][0][2, :, :] = goal_grid
            if i == 0 or np.all(self.buffer[-i][1] == goal_pos):
                self.buffer[-1 - i][3] = 0
                self.buffer[-1 - i][4] = True
            else:
                self.buffer[-1 - i][3] = -1
                self.buffer[-1 - i][4] = False

        ret = [[self.state(g, p, goal_pos), a, r, d] for g, p, a, r, d in self.buffer]
        final_grid = self.grid.copy()
        final_grid[2, :, :] = goal_grid
        ret.append([self.state(final_grid, self.pos, goal_pos), None, None, None])
        self.buffer = None
        return ret
    
    def _block_ind(self, pos):
        y_ind = (
            np.arange(pos[0] - self.d_block, pos[0] + self.d_block + 1) % self.size
        )
        x_ind = (
            np.arange(pos[1] - self.d_block, pos[1] + self.d_block + 1) % self.size
        )
        return np.ix_(y_ind, x_ind)

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            img[:, :, 0][self.grid[0] != 0] = 255
            img[:, :, 1][self.grid[1] != 0] = 255
            img[:, :, 2][self.grid[2] != 0] = 255
            img = img.repeat(10, axis=0).repeat(10, axis=1)
            return img
        else:
            super().render(mode=mode)
