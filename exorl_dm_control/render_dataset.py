import os, sys, glob
from multiprocessing import Pool, cpu_count
import numpy as np
import h5py
from tqdm import tqdm

# make sure your ExoRL clone is on the path
sys.path.insert(0, os.path.expanduser('~/bisim/exorl'))
import dmc

os.environ['MUJOCO_GL'] = 'egl'


TASK = 'point_mass_maze_reach_top_left'
DISCRETE = True
BUFFER_DIR = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/buffer')
OUT_PATH = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/all_eps.hdf5')
IMG_H, IMG_W = 64, 64


def make_discrete_mappings():
    # 9 actions: no-op, N, S, E, W, NE, NW, SE, SW
    vecs = {
      0: np.array([0.0, 0.0]),
      1: np.array([1.0, 0.0]),   # East
      2: np.array([-1.0, 0.0]),  # West
      3: np.array([0.0, 1.0]),   # North
      4: np.array([0.0, -1.0]),  # South
      5: np.array([1.0, 1.0]),   # NE
      6: np.array([-1.0, 1.0]),  # NW
      7: np.array([1.0, -1.0]),  # SE
      8: np.array([-1.0, -1.0]), # SW
    }
    # normalize diagonals (so all actions have magnitude 1)
    for k in (5,6,7,8):
        vecs[k] /= np.linalg.norm(vecs[k])

    def cont2disc(a2):
        # find the index whose vec is closest in L2 to a2
        dists = [(k, np.linalg.norm(a2 - v)) for k, v in vecs.items()]
        return min(dists, key=lambda x: x[1])[0]

    return cont2disc, vecs


def process_chunk(args):
    task_name, file_list, seed = args

    pixel_env = dmc.make(
        task_name,
        obs_type='pixels',
        frame_stack=1,
        action_repeat=1,
        seed=seed,
    )
    state_env = dmc.make(
        task_name,
        obs_type='states',
        frame_stack=1,
        action_repeat=1,
        seed=seed+999,
    )

    images, physics, actions, rewards, discounts = [], [], [], [], []
    cont2disc, disc2cont = make_discrete_mappings()

    for ep in tqdm(file_list, desc=f"Worker {seed}", position=seed, leave=False):
        data = np.load(ep)
        cont_action = data['action']
        if not DISCRETE:
            phys = data['physics']
            reward = data['reward']
            disc = data['discount']

            frames = []
            for state in phys:
                with pixel_env.physics.reset_context():
                    pixel_env.physics.set_state(state)
                frames.append(
                    pixel_env.physics.render(width=IMG_W, height=IMG_H, camera_id=0)
                )
            images.append(np.stack(frames, axis=0))
            physics.append(phys)
            actions.append(np.array(cont_action, dtype=np.float32))
            rewards.append(np.array(reward)[...,None])
            discounts.append(np.array(disc)[...,None])
        else:
            phys_init = data['physics'][0]

            # start from initial state
            with pixel_env.physics.reset_context():
                pixel_env.physics.set_state(phys_init)
            state_env.reset()
            state_env.physics.set_state(phys_init)

            frames, new_phys, new_acts, new_rews, new_discs = [], [], [], [], []

            for a_cont in cont_action:
                ts = state_env.step(a_cont)
                r = ts.reward if ts.reward is not None else 0.0
                new_rews.append(float(r))
                # new_rews.append(float(ts.reward))
                d = ts.discount if ts.discount is not None else 1.0
                new_discs.append(float(d))
                # new_discs.append(float(ts.discount))

                s = state_env.physics.get_state()
                new_phys.append(s)

                idx = cont2disc(a_cont)
                new_acts.append([idx])
                a_use = disc2cont[idx]

                with pixel_env.physics.reset_context():
                    pixel_env.physics.set_state(s)
                frames.append(
                    pixel_env.physics.render(width=IMG_W, height=IMG_H, camera_id=0)
                )

            images.append(np.stack(frames, axis=0))
            physics.append(np.stack(new_phys, axis=0))
            actions.append(np.array(new_acts, dtype=np.int32))
            rewards.append(np.array(new_rews) [...,None])
            discounts.append(np.array(new_discs) [...,None])

    return (
        np.concatenate(images, axis=0),
        np.concatenate(physics, axis=0),
        np.concatenate(actions, axis=0),
        np.concatenate(rewards, axis=0),
        np.concatenate(discounts, axis=0),
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    all_eps = sorted(glob.glob(f"{BUFFER_DIR}/*.npz"))
    N = min(max(cpu_count()-2, 1), len(all_eps))
    chunks = [all_eps[i::N] for i in range(N)]
    args = [(TASK, chunks[i], i) for i in range(N)]

    with h5py.File(OUT_PATH, 'w') as hf:
        hf.create_dataset('images',
            shape=(0,IMG_H,IMG_W,3), maxshape=(None,IMG_H,IMG_W,3),
            dtype='uint8', chunks=(1,IMG_H,IMG_W,3), compression='gzip')
        hf.create_dataset('physics',
            shape=(0,4), maxshape=(None,4),
            dtype='float64', chunks=(1024,4), compression='gzip')
        hf.create_dataset('action',
            shape=(0,1) if DISCRETE else (0,2),
            maxshape=(None,1) if DISCRETE else (None,2),
            dtype='int32'   if DISCRETE else 'float32',
            chunks=(1024,1) if DISCRETE else (1024,2),
            compression='gzip')
        hf.create_dataset('reward',
            shape=(0,4), maxshape=(None,4),
            dtype='float32', chunks=(1024,4), compression='gzip')
        hf.create_dataset('discount',
            shape=(0,1), maxshape=(None,1),
            dtype='float32', chunks=(1024,1), compression='gzip')
    
        with Pool(N) as pool:
            for imgs, phys, acts, rews, discs in tqdm(
                    pool.imap_unordered(process_chunk, args),
                    total=N, desc="Appending chunks"):

                T = imgs.shape[0]   # frames in this chunk

                # resize & write the new slice
                for name, arr in [("images",imgs),
                                  ("physics",phys),
                                  ("action",acts),
                                  ("reward",rews),
                                  ("discount",discs)]:
                    ds = hf[name]
                    old = ds.shape[0]
                    ds.resize(old + T, axis=0)
                    ds[old:old+T] = arr

                # free the chunk from memory
                del imgs, phys, acts, rews, discs

        hf.attrs['total_steps'] = hf['images'].shape[0]
