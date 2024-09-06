from argparse import ArgumentParser
from collections import defaultdict
from d4rl.pointmaze import waypotroller
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, Manager
import h5py
from d4rl.pointmaze import maze_model
from nav2d_representation.pointmass.d4rl_maze2d import VisualMazeEnv
from nav2d_representation.pointmass.utils import create_video
from gym.wrappers import FrameStack


def collect(num, seed, epsilon, frame_stack, maze_spec, egocentric, n_obs, idx, queue):
    env = FrameStack(
        VisualMazeEnv(maze_spec=maze_spec, egocentric=egocentric, num_obstacles=n_obs),
        frame_stack,
    )
    env.seed(seed + idx)
    np.random.seed(seed + idx)


    global_step = 0
    while global_step < num:
        obs = env.reset()
        controller = waypoint_controller.WaypointController(env.str_maze_spec)
        done = False
        while not done:
            position = env.sim.data.qpos.ravel()
            velocity = env.sim.data.qvel.ravel()
            if np.random.rand() < epsilon:
                act = env.action_space.sample()
            else:
                act, done = controller.get_action(position, velocity, env.get_target())
                act = np.argmax(
                    (
                        VisualMazeEnv.ACTIONS
                        / np.linalg.norm(VisualMazeEnv.ACTIONS, axis=1, keepdims=True)
                    )
                    @ act
                )

            obs_next, rew, done, info = env.step(act)

            queue.put(
                {
                    "obs": obs,
                    "obs_next": obs_next,
                    "action": act,
                    "reward": rew,
                    "done": done,
                    "info/goal": env.get_target(),
                    "info/qpos": position,
                    "info/qvel": velocity,
                }
            )

            obs = obs_next
            global_step += 1
            if global_step >= num:
                done = True


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--frame_stack", type=int, default=2)
    parser.add_argument("--maze_spec", type=str, default="LARGE_MAZE")
    parser.add_argument("--egocentric", action="store_true")
    parser.add_argument("--n_obs", default=0, type=int)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    def _raise(e):
        raise e

    data = defaultdict(list)
    with Pool(args.num_workers) as p, Manager() as m:
        queue = m.Queue()
        result = p.starmap_async(
            collect,
            [
                (
                    args.size // args.num_workers,
                    args.seed,
                    args.epsilon,
                    args.frame_stack,
                    getattr(maze_model, args.maze_spec),
                    args.egocentric,
                    args.n_obs,
                    i,
                    queue,
                )
                for i in range(args.num_workers)
            ],
            callback=lambda _: queue.put_nowait(None),
            error_callback=lambda e: _raise(e),
        )
        pbar = tqdm(total=(args.size // args.num_workers) * args.num_workers)
        while not result.ready() or not queue.empty():
            d = queue.get(timeout=60)
            if d is not None:
                for k, v in d.items():
                    data[k].append(v)
                pbar.update(1)

    with h5py.File(args.name) as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="lzf")

        f.attrs["env"] = "pointmass"
        f.attrs["maze_spec"] = getattr(maze_model, args.maze_spec)
        f.attrs["egocentric"] = args.egocentric
        f.attrs["num_obstacles"] = args.n_obs

    with h5py.File(args.name) as f:
        create_video([o[0] for o in f["obs"][:500]], "video.avi", 10)


if __name__ == "__main__":
    main()
