import glob
import multiprocessing
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from nav2d_representation import utils
from nav2d_representation.nav2d.nav2d import Navigate2D

NUM_TRIALS = 1000


def eval(h, model_path, epsilon):
    # pid = int(multiprocessing.current_process().name.split("-")[-1])
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(pid % 4)

    # model = torch.load(model_path)

    env = Navigate2D(h)
    env.seed(0)
    min_dists = []
    for i in tqdm.tqdm(range(NUM_TRIALS)):
        obs = env.reset()
        done = False
        min_dist = env.dist
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    pass
                    # Q = model(
                    #     torch.tensor(
                    #         obs, dtype=torch.float, device="cuda",
                    #     ).unsqueeze(0)
                    # ).squeeze(0)
                    # action = torch.argmax(Q).item()
                rel = env.goal - env.pos
                if rel[0] > 0:
                    action = 0
                elif rel[1] > 0:
                    action = 1
                elif rel[0] < 0:
                    action = 2
                else:
                    action = 3
            obs, reward, done, info = env.step(action)
            min_dist = min(min_dist, env.dist)
        min_dists.append(min_dist)
    return np.mean(np.array(min_dists) == 0)


if __name__ == "__main__":
    HYPERPARAMETERS = {
        "grid_size": 20,
        "obstacle_diameter": 1,
        "scale": 1,
        "min_goal_dist": 10,
        "num_obstacles": 0,
        "use_factorized_state": False,
        "max_episode_length": 50,
    }

    parser = ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    args = parser.parse_args()

    print(eval(HYPERPARAMETERS, None, 0.0))

    seed_paths = glob.glob(os.path.join(args.model_dir, "*"))
    model_paths = [glob.glob(os.path.join(seed_path, "*.pt")) for seed_path in seed_paths]
    model_paths = [[path for path in seed if "final" not in path] for seed in model_paths]
    model_paths = np.array(model_paths)

    def fn(p):
        return eval(HYPERPARAMETERS, p, 0.2)

    shape = model_paths.shape
    with Pool(16) as p:
        success_rates = p.map(fn, model_paths.flatten())

    success_rates = np.array(success_rates).reshape(shape)

