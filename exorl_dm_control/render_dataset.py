import os
import sys, glob
from multiprocessing import Pool, cpu_count
import numpy as np
import h5py
from tqdm import tqdm

# make sure your ExoRL clone is on the path
sys.path.insert(0, os.path.expanduser('~/bisim/exorl'))
import dmc

os.environ['MUJOCO_GL'] = 'egl'

TASK = 'point_mass_maze_reach_top_left'
# DISCRETE = True
DISCRETE = False
BUFFER_DIR = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/buffer')
# OUT_PATH = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5')
OUT_PATH = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_continuous.hdf5')
IMG_H, IMG_W = 64, 64

MAX_STEPS = 2_000_000
RNG_SEED = 0

def make_discrete_mappings():
    # 9 actions: no-op, N, S, E, W, NE, NW, SE, SW
    vecs = {
      0: np.array([0.0, 0.0]),
      1: np.array([1.0, 0.0]),    # East
      2: np.array([-1.0, 0.0]),   # West
      3: np.array([0.0, 1.0]),    # North
      4: np.array([0.0, -1.0]),   # South
      5: np.array([1.0, 1.0]),    # NE
      6: np.array([-1.0, 1.0]),   # NW
      7: np.array([1.0, -1.0]),   # SE
      8: np.array([-1.0, -1.0]),  # SW
    }
    # normalize diagonals (so all actions have magnitude 1)
    for k in (5, 6, 7, 8):
        vecs[k] /= np.linalg.norm(vecs[k])

    def cont2disc(a2):
        # find the index whose vec is closest in L2 to a2
        dists = [(k, np.linalg.norm(a2 - v)) for k, v in vecs.items()]
        return min(dists, key=lambda x: x[1])[0]

    return cont2disc, vecs


def process_chunk(args):
    task_name, file_list, seed = args

    env = dmc.make(
        task_name,
        obs_type='pixels',
        frame_stack=1,
        action_repeat=1,
        seed=seed,
    )

    images, physics, actions = [], [], []
    ep_lens = []
    cont2disc, disc2cont = make_discrete_mappings()

    for ep in tqdm(file_list, desc=f"Worker {seed}", position=seed, leave=False):
        # data = np.load(ep)
        data = np.load(ep, allow_pickle=False)
        cont_action = data['action']

        phys_orig = data['physics']
        if not DISCRETE:
            frames = []
            for state in phys_orig:
                with env.physics.reset_context():
                    env.physics.set_state(state)
                frames.append(
                    env.physics.render(width=IMG_W, height=IMG_H, camera_id=0)
                )
            images.append(np.stack(frames, axis=0))
            physics.append(phys_orig)
            actions.append(np.array(cont_action, dtype=np.float32))

        else:
            phys_init = phys_orig[0]
            with env.physics.reset_context():
                env.physics.set_state(phys_init)

            frames, new_phys, new_acts = [], [], []

            for a_cont in cont_action:
                idx = cont2disc(a_cont)
                a_use = disc2cont[idx]

                ts = env.step(a_use)
                if ts.last():
                    break

                s = env.physics.get_state()
                new_phys.append(s)
                new_acts.append([idx])
                frames.append(env.physics.render(width=IMG_W, height=IMG_H, camera_id=0))

            if len(frames) == 0:
                continue

            images.append(np.stack(frames, axis=0))
            physics.append(np.stack(new_phys, axis=0))
            actions.append(np.array(new_acts, dtype=np.int32))
        ep_lens.append(images[-1].shape[0])

    return (
        np.concatenate(images, axis=0),
        np.concatenate(physics, axis=0),
        np.concatenate(actions, axis=0),
        np.array(ep_lens, dtype=np.int32),
    )

def choose_episodes_up_to_cap(all_eps, max_steps, rng):
    """Shuffle episodes, then take whole episodes until sum(lengths) <= max_steps."""
    # Randomize episode order
    all_eps = list(all_eps)
    rng.shuffle(all_eps)

    selected = []
    total = 0
    for p in all_eps:
        with np.load(p, allow_pickle=False) as f:
            # Use physics len (== action len) as per-step count
            L = f['physics'].shape[0]
        if L <= 1:
            continue  # skip trivially short episodes
        if total + L > max_steps:
            # stop BEFORE exceeding the cap (no partial episode)
            break
        selected.append(p)
        total += L
        if total == max_steps:
            break
    return selected, total

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    all_eps = sorted(glob.glob(f"{BUFFER_DIR}/*.npz"))
    rng = np.random.default_rng(RNG_SEED)

    # If we're in continuous mode, preselect a random set of whole episodes up to the cap.
    if not DISCRETE:
        selected_eps, planned_steps = choose_episodes_up_to_cap(all_eps, MAX_STEPS, rng)
        if len(selected_eps) == 0:
            raise RuntimeError("No episodes selected; check buffer path or episode lengths.")
        print(f"[info] selected {len(selected_eps)} episodes totaling {planned_steps} steps (cap={MAX_STEPS})")
        eps_for_work = selected_eps
    else:
        # No cap applied to discrete path unless you want it—then just reuse the same selection code.
        eps_for_work = all_eps

    N = min(max(cpu_count() - 2, 1), len(eps_for_work))
    chunks = [eps_for_work[i::N] for i in range(N)]
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
        hf.create_dataset('episode_lengths',
            shape=(0,), maxshape=(None,),
            dtype='int32', chunks=(1024,), compression='gzip')

        hf.attrs['max_steps_cap'] = MAX_STEPS
        hf.attrs['rng_seed'] = RNG_SEED
        hf.attrs['mode'] = 'discrete' if DISCRETE else 'continuous'

        with Pool(N) as pool:
            for imgs, phys, acts, ep_len in tqdm(
                    pool.imap_unordered(process_chunk, args),
                    total=N, desc="Appending chunks"):

                T = imgs.shape[0]   # frames in this chunk

                # resize & write the new slice
                for name, arr in [("images",imgs),
                                  ("physics",phys),
                                  ("action",acts)]:
                    if T == 0:
                        continue
                    ds = hf[name]
                    old = ds.shape[0]
                    ds.resize(old + T, axis=0)
                    ds[old:old+T] = arr

                M = ep_len.shape[0]
                ds = hf["episode_lengths"]
                old = ds.shape[0]
                ds.resize(old + M, axis=0)
                ds[old:old + M] = ep_len

                # free the chunk from memory
                del imgs, phys, acts, ep_len

        hf.attrs['total_steps'] = hf['images'].shape[0]
        print(f"[info] wrote {hf.attrs['total_steps']} steps to {OUT_PATH}")
