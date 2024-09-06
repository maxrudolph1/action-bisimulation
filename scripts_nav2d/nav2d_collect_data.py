from argparse import ArgumentParser
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, RLock
import h5py
import os

from nav2d_representation.nav2d.nav2d import Navigate2D
from nav2d_representation.nav2d.nav2d_po import Navigate2DPO
# import gymnasium as gym
# import nav2d


def collect(num, idx, seed, epsilon, num_obstacles, args):

    if args.env == "nav2d":
        env = Navigate2D(num_obstacles, grid_size=args.grid_size, 
                                    static_goal=not args.random_goal,
                                    obstacle_diameter=args.obstacle_size,
                                    env_config=args.env_config)
    elif args.env == "nav2dpo":
        env = Navigate2DPO(num_obstacles, static_goal=(not args.random_goal), env_config=args.env_config)

    env.seed(seed + idx)
    np.random.seed(seed + idx)

    buffer = []

    global_step = 0
    pbar = tqdm(total=num, position=idx)
    while global_step < num:
        obs= env.reset()
        # if not args.keep_goal_channel and args.env == "nav2d":
        #     obs[2] = -1  # clear goal channel
        done = False
        recompute = True
        kaction_buffer = list()
        actions = list()
        while not done and global_step < num:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                recompute = True
            else:
                if recompute:
                    optimal_actions = env.find_path()
                action = optimal_actions.pop(0)
                recompute = False
            obs_next, rew, done, info = env.step(action)
            # if not args.keep_goal_channel and args.env == "nav2d":
            #     obs_next[2] = -1  # clear goal channel
            kaction_buffer.append((obs, action, rew, obs_next, done, info["pos"], info["goal"]))
            actions.append(action)
            obs = obs_next
            global_step += 1
            pbar.update(1)
            
        # create length k seqeuences of actions
        action_sequences = list()
        kvalid_values = list()
        k_obs = list()

        for i in range(len(kaction_buffer)):
            action_sequences.append([kaction_buffer[kidx][1] for kidx in np.pad(list(range(i,min(i+args.k_step_action, len(kaction_buffer)))), 
                                                                                (0, max(0, i+args.k_step_action - len(kaction_buffer))))])
            kvalid_values.append(len(kaction_buffer) - (i+args.k_step_action) > 0) # not equal because we don't have obs_next for the final state
            # for idx in range(len(args.k_step_action)):
            if args.k_step_action > 0:
                k_obs_vals = list()
                for k in range(2, args.k_step_action + 1):
                    k_obs_vals.append(kaction_buffer[min(i+args.k_step_action, len(kaction_buffer)-1)][0])
                    # k2_obs.append(kaction_buffer[min(i+args.k_step_action, len(kaction_buffer)-1)][0])
                k_obs_vals = np.stack(k_obs_vals, axis=0)
                k_obs.append(k_obs_vals)
            else:
                k_obs.append(np.zeros(1)) # unused

        for kact, kvalid, kobs, (obs, action, rew, obs_next, done, pos, goal) in zip(action_sequences, kvalid_values, k_obs,  kaction_buffer):
            buffer.append((obs, action, rew, obs_next, done, pos, goal, kact, kvalid, kobs))

        # create length k seqeuences of actions
        # action_sequences = list()
        # kvalid_values = list()
        # k_obs = list()
        # for i in range(len(kaction_buffer)):
        #     action_sequences.append([kaction_buffer[kidx][1] for kidx in np.pad(list(range(i,min(i+args.k_step_action, len(kaction_buffer)))), (0, max(0, i+args.k_step_action -len(kaction_buffer))))])
        #     kvalid_values.append(len(kaction_buffer) - (i+args.k_step_action) > 0) # not equal because we don't have obs_next for the final state
        #     k_obs.append(kaction_buffer[min(i+args.k_step_action, len(kaction_buffer)-1)][0])
        # for kact, kvalid, kobs, (obs, action, rew, obs_next, done, pos, goal) in zip(action_sequences, kvalid_values, k_obs, kaction_buffer):
        #     buffer.append((obs, action, rew, obs_next, done, pos, goal, kact, kvalid, kobs))

    return buffer


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-workers", default=32, type=int)
    parser.add_argument("--size", type=int, default=1000000)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--num-obstacles", type=int, default=10)
    parser.add_argument("--obstacle-size", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--k-step-action", type=int, default=0) # number of lookahead steps for the "single step" model
    parser.add_argument("--maze",default=False, action="store_true")
    parser.add_argument("--env-config", default=None)
    parser.add_argument("--env", default="nav2d")
    parser.add_argument("--random-goal", default=False, action="store_true")
    parser.add_argument("--keep-goal-channel", default=False, action="store_true")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--save-path", type=str, default=".")
    args = parser.parse_args()

    
    with Pool(args.num_workers, initargs=(RLock(),), initializer=tqdm.set_lock) as p:
        buffers = p.starmap(
            collect,
            [
                (
                    args.size // args.num_workers,
                    i,
                    args.seed,
                    args.epsilon,
                    args.num_obstacles,
                    args,
                )
                for i in range(args.num_workers)
            ],
        )
    buffer = [x for b in buffers for x in b]
    full_path = args.save_path + f"/datasets/nav2d_dataset_s{args.seed}_e{args.epsilon}_size{args.size}_{args.name}_k_steps_{args.k_step_action}.hdf5"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with h5py.File(
        args.save_path + f"/datasets/nav2d_dataset_s{args.seed}_e{args.epsilon}_size{args.size}_{args.name}_k_steps_{args.k_step_action}.hdf5", "w"
    ) as f:
        f["obs"] = np.array([x[0] for x in buffer])
        f["action"] = np.array([x[1] for x in buffer])
        f["reward"] = np.array([x[2] for x in buffer])
        f["obs_next"] = np.array([x[3] for x in buffer])
        f["done"] = np.array([x[4] for x in buffer])
        f["info/pos"] = np.array([x[5] for x in buffer])
        f["info/goal"] = np.array([x[6] for x in buffer])
        f["kaction"] = np.array([x[7] for x in buffer])
        f["kvalid"] = np.array([x[8] for x in buffer])
        f["kobs"] = np.array([x[9] for x in buffer]) # stores dataset_size, k-1 (skips obs next), 3, grid, grid, or data_size, 1 if not used
        f.attrs["k_step_action"] = args.k_step_action
        f.attrs["num_obstacles"] = args.num_obstacles
        f.attrs["static_goal"] = not args.random_goal
        f.attrs["cleared_goal_channel"] = not args.keep_goal_channel
        f.attrs["env"] = args.env
        f.attrs["env_config"] = -1 if args.env_config is None else args.env_config
if __name__ == "__main__":
    main()
