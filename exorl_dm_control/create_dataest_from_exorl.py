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
DISCRETE = True  # we produce displacement-discretized actions
BUFFER_DIR = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/buffer')
OUT_PATH = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/displacement_discretized.hdf5')
ACTION_SIZE = 9
IMG_H, IMG_W = 64, 64

# 9-cell mapping consistent with your previous action ids
GRID2ID = {
    (0, 0): 0,   # center / "no-op"
    (1, 0): 1,   # East
    (-1, 0): 2,  # West
    (0, 1): 3,   # North
    (0, -1): 4,  # South
    (1, 1): 5,   # NE
    (-1, 1): 6,  # NW
    (1, -1): 7,  # SE
    (-1, -1): 8  # SW
}

# Options: 'first' (default), 'sum', 'mean', 'norm'
REWARD_REDUCE = 'first'
DISCOUNT_REDUCE = 'first'


# Reduce potentially vector rewards/discounts to a scalar
def reduce_scalar_at(arr, t, how='first'):
    v = np.asarray(arr)[t]
    v = np.asarray(v).squeeze()
    if v.ndim == 0:
        return float(v)
    if how == 'sum':
        return float(v.sum())
    if how == 'mean':
        return float(v.mean())
    if how == 'norm':
        return float(np.linalg.norm(v))
    # default: first component
    return float(v[..., 0])

def delta_to_action_id(dx, dy, thrx, thry):
    # guard tiny thresholds
    thrx = max(thrx, 1e-12)
    thry = max(thry, 1e-12)
    gx = 1 if dx >  thrx else (-1 if dx < -thrx else 0)
    gy = 1 if dy >  thry else (-1 if dy < -thry else 0)
    return GRID2ID[(gx, gy)]


def delta_to_action_id_signed(dx, dy, x_low, x_high, y_low, y_high):
    gx = -1 if dx < x_low  else (1 if dx > x_high else 0)
    gy = -1 if dy < y_low  else (1 if dy > y_high else 0)
    return GRID2ID[(gx, gy)]


def compute_quantile_thresholds(all_eps, center_mass=1.0/3.0, bins=16384):
    """
    Compute thresholds thrx, thry such that P(|dx| <= thrx) ≈ center_mass
    and P(|dy| <= thry) ≈ center_mass, using a two-pass histogram over |Δpos|.
    """
    # ---- pass 1: find global max |dx|, |dy| for histogram ranges
    max_ax, max_ay = 0.0, 0.0
    for ep in tqdm(all_eps, desc="Pass 1: scanning extrema"):
        data = np.load(ep)
        phys = data['physics']
        if phys.shape[0] < 2:
            continue
        d = phys[1:, :2] - phys[:-1, :2]
        ax = np.abs(d[:, 0])
        ay = np.abs(d[:, 1])
        if ax.size:
            max_ax = max(max_ax, float(ax.max()))
        if ay.size:
            max_ay = max(max_ay, float(ay.max()))

    # guard against degenerate cases
    max_ax = max(max_ax, 1e-12)
    max_ay = max(max_ay, 1e-12)

    # ---- pass 2: build histograms
    hx = np.zeros(bins, dtype=np.int64)
    hy = np.zeros(bins, dtype=np.int64)
    for ep in tqdm(all_eps, desc="Pass 2: building histograms"):
        data = np.load(ep)
        phys = data['physics']
        if phys.shape[0] < 2:
            continue
        d = phys[1:, :2] - phys[:-1, :2]
        ax = np.abs(d[:, 0])
        ay = np.abs(d[:, 1])
        if ax.size:
            hx += np.histogram(ax, bins=bins, range=(0.0, max_ax))[0]
        if ay.size:
            hy += np.histogram(ay, bins=bins, range=(0.0, max_ay))[0]

    # ---- convert histograms to quantile thresholds
    def hist_quantile(h, vmax, q):
        cdf = np.cumsum(h)
        if cdf[-1] == 0:
            return 0.0
        idx = int(np.searchsorted(cdf, q * cdf[-1], side="left"))
        idx = min(max(idx, 0), len(h) - 1)
        # map bin index to value (bin center)
        bin_width = vmax / len(h)
        return (idx + 0.5) * bin_width

    thrx = hist_quantile(hx, max_ax, center_mass)
    thry = hist_quantile(hy, max_ay, center_mass)
    return thrx, thry

def compute_signed_split_quantiles(all_eps, q_low=1.0/3.0, q_high=2.0/3.0, bins=16384):
    """
    Find per-axis signed cutpoints (x_low, x_high, y_low, y_high) such that
    P(Δx < x_low) ≈ 1/3, P(Δx > x_high) ≈ 1/3, same for Δy.
    Two-pass streaming histogram: memory-safe and fast.
    """
    # Pass 1: get symmetric histogram ranges from max |Δ|
    max_ax, max_ay = 0.0, 0.0
    for ep in tqdm(all_eps, desc="Pass 1: extrema (signed)"):
        data = np.load(ep)
        phys = data['physics']
        if phys.shape[0] < 2:
            continue
        d = phys[1:, :2] - phys[:-1, :2]
        ax = np.abs(d[:, 0]); ay = np.abs(d[:, 1])
        if ax.size: max_ax = max(max_ax, float(ax.max()))
        if ay.size: max_ay = max(max_ay, float(ay.max()))
    max_ax = max(max_ax, 1e-12)
    max_ay = max(max_ay, 1e-12)

    # Pass 2: build signed histograms over [-max, max]
    hx, edges_x = np.zeros(bins, dtype=np.int64), None
    hy, edges_y = np.zeros(bins, dtype=np.int64), None
    for ep in tqdm(all_eps, desc="Pass 2: histograms (signed)"):
        data = np.load(ep)
        phys = data['physics']
        if phys.shape[0] < 2:
            continue
        d = phys[1:, :2] - phys[:-1, :2]
        dx = d[:, 0]; dy = d[:, 1]
        cx, ex = np.histogram(dx, bins=bins, range=(-max_ax, max_ax))
        cy, ey = np.histogram(dy, bins=bins, range=(-max_ay, max_ay))
        hx += cx; hy += cy
        edges_x = ex; edges_y = ey  # same each time

    def hist_q(counts, edges, q):
        cdf = np.cumsum(counts)
        if cdf[-1] == 0:
            return 0.0
        k = int(np.searchsorted(cdf, q * cdf[-1], side="left"))
        k = min(max(k, 0), len(counts)-1)
        # choose the bin edge at that cumulative cut
        return float(edges[k])

    x_low  = hist_q(hx, edges_x, q_low)
    x_high = hist_q(hx, edges_x, q_high)
    y_low  = hist_q(hy, edges_y, q_low)
    y_high = hist_q(hy, edges_y, q_high)
    return x_low, x_high, y_low, y_high


def compute_global_thresholds(all_eps, percentile=100.0):
    """
    Scan once to compute global per-axis displacement maxima (or percentile).
    Returns thrx, thry where thresholds are +/- (Mx/3) and +/- (My/3)
    to create 3 equal-width bins along each axis.
    """
    max_dx, max_dy = 0.0, 0.0
    use_pct = percentile < 100.0
    for ep in tqdm(all_eps, desc="Scanning deltas for thresholds"):
        data = np.load(ep)
        phys = data['physics']
        if phys.shape[0] < 2:
            continue
        d = phys[1:, :2] - phys[:-1, :2]
        if use_pct:
            max_dx = max(max_dx, float(np.percentile(np.abs(d[:, 0]), percentile)))
            max_dy = max(max_dy, float(np.percentile(np.abs(d[:, 1]), percentile)))
        else:
            max_dx = max(max_dx, float(np.max(np.abs(d[:, 0]))))
            max_dy = max(max_dy, float(np.max(np.abs(d[:, 1]))))
    thrx = max_dx / 3.0
    thry = max_dy / 3.0
    return thrx, thry

def process_chunk(args):
    # task_name, file_list, seed, thrx, thry = args
    task_name, file_list, seed, x_low, x_high, y_low, y_high = args


    env = dmc.make(
        task_name,
        obs_type='pixels',
        frame_stack=1,
        action_repeat=1,
        seed=seed,
    )

    images, physics, actions, rewards, discounts = [], [], [], [], []
    ep_lens = []

    for ep in tqdm(file_list, desc=f"Worker {seed}", position=seed, leave=False):
        data = np.load(ep)
        phys_orig = data['physics']          # shape (L, 4): [x, y, vx, vy]
        rew_orig  = data['reward']           # shape (L,) or (L-1,) depending on source
        disc_orig = data['discount']         # shape like reward

        L = phys_orig.shape[0]
        if L < 2:
            ep_lens.append(0)
            continue

        # guard lengths (some buffers store L or L-1 rewards/discounts)
        Ttrans = min(L - 1, len(rew_orig), len(disc_orig))

        # We build transitions for t=0..L-2 so that next obs always exists.
        frames, new_phys, new_acts, new_rews, new_discs = [], [], [], [], []

        # render and record at state s_t, map displacement s_{t+1}-s_t
        for t in range(Ttrans):
            s_t = phys_orig[t]
            s_tp1 = phys_orig[t + 1]
            dx, dy = float(s_tp1[0] - s_t[0]), float(s_tp1[1] - s_t[1])
            # a_idx = delta_to_action_id(dx, dy, thrx, thry)
            a_idx = delta_to_action_id_signed(dx, dy, x_low, x_high, y_low, y_high)

            # set state and render obs for s_t
            with env.physics.reset_context():
                env.physics.set_state(s_t)
            frm = env.physics.render(width=IMG_W, height=IMG_H, camera_id=0)

            frames.append(frm)
            new_phys.append(s_t)
            new_acts.append([a_idx])

            # align reward/discount with transition t->t+1
            # if source arrays are length L, we take element t; if L-1, also fine
            new_rews.append(reduce_scalar_at(rew_orig,  t, how=REWARD_REDUCE))
            new_discs.append(reduce_scalar_at(disc_orig, t, how=DISCOUNT_REDUCE))


        if len(new_acts) == 0:
            ep_lens.append(0)
            continue

        images.append(np.stack(frames, axis=0))                 # (L-1, H, W, 3)
        physics.append(np.stack(new_phys, axis=0))              # (L-1, 4)
        actions.append(np.asarray(new_acts, dtype=np.int32))    # (L-1, 1)
        rewards.append(np.asarray(new_rews, dtype=np.float32)[..., None])   # (L-1, 1)
        discounts.append(np.asarray(new_discs, dtype=np.float32)[..., None])# (L-1, 1)
        ep_lens.append(len(new_acts))

    if len(ep_lens) == 0:
        # Return empty chunks if nothing processed in this worker
        return (
            np.zeros((0, IMG_H, IMG_W, 3), dtype=np.uint8),
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0, 1), dtype=np.int32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.array([], dtype=np.int32),
        )

    return (
        np.concatenate(images, axis=0),
        np.concatenate(physics, axis=0),
        np.concatenate(actions, axis=0),
        np.concatenate(rewards, axis=0),
        np.concatenate(discounts, axis=0),
        np.array(ep_lens, dtype=np.int32),
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    all_eps = sorted(glob.glob(f"{BUFFER_DIR}/*.npz"))

    # 1) First pass: compute global thresholds from displacement
    #    (use percentile=99.5 to trim outliers if you like)
    # THRX, THRY = compute_global_thresholds(all_eps, percentile=99.5)
    # THRX, THRY = compute_quantile_thresholds(all_eps, center_mass=1.0/3.0)
    X_LOW, X_HIGH, Y_LOW, Y_HIGH = compute_signed_split_quantiles(all_eps, q_low=1/3, q_high=2/3, bins=8192)
    print(f"[info] x_low={X_LOW:.6g}, x_high={X_HIGH:.6g}, y_low={Y_LOW:.6g}, y_high={Y_HIGH:.6g}")


    # print(f"[info] thresholds: thrx={THRX:.6g}, thry={THRY:.6g}")

    # 2) Multiprocessing rendering/writing
    N = min(max(cpu_count() - 2, 1), len(all_eps))
    chunks = [all_eps[i::N] for i in range(N)]
    # args = [(TASK, chunks[i], i, THRX, THRY) for i in range(N)]
    args = [(TASK, chunks[i], i, X_LOW, X_HIGH, Y_LOW, Y_HIGH) for i in range(N)]


    with h5py.File(OUT_PATH, 'w') as hf:
        hf.create_dataset('images',
            shape=(0, IMG_H, IMG_W, 3), maxshape=(None, IMG_H, IMG_W, 3),
            dtype='uint8', chunks=(1, IMG_H, IMG_W, 3), compression='gzip')
        hf.create_dataset('physics',
            shape=(0, 4), maxshape=(None, 4),
            dtype='float64', chunks=(1024, 4), compression='gzip')
        hf.create_dataset('action',
            shape=(0, 1) if DISCRETE else (0, 2),
            maxshape=(None, 1) if DISCRETE else (None, 2),
            dtype='int32' if DISCRETE else 'float32',
            chunks=(1024, 1) if DISCRETE else (1024, 2),
            compression='gzip')
        hf.create_dataset('reward',
            shape=(0, 1), maxshape=(None, 1),
            dtype='float32', chunks=(1024, 1), compression='gzip')  # FIXED
        hf.create_dataset('discount',
            shape=(0, 1), maxshape=(None, 1),
            dtype='float32', chunks=(1024, 1), compression='gzip')
        hf.create_dataset('episode_lengths',
            shape=(0,), maxshape=(None,),
            dtype='int32', chunks=(1024,), compression='gzip')

        # Optional convenience: precompute episode starts (doesn't change old readers)
        hf.create_dataset('episode_starts',
            shape=(0,), maxshape=(None,),
            dtype='int64', chunks=(1024,), compression='gzip')

        action_counts = np.zeros(ACTION_SIZE, dtype=np.int64)

        with Pool(N) as pool:
            cum_steps = 0
            for imgs, phys, acts, rews, discs, ep_len in tqdm(
                    pool.imap_unordered(process_chunk, args),
                    total=N, desc="Appending chunks"):
                action_counts += np.bincount(acts[:, 0], minlength=ACTION_SIZE)

                T = imgs.shape[0]
                for name, arr in [("images", imgs),
                                  ("physics", phys),
                                  ("action",  acts),
                                  ("reward",  rews),
                                  ("discount", discs)]:
                    ds = hf[name]
                    old = ds.shape[0]
                    ds.resize(old + T, axis=0)
                    ds[old:old+T] = arr

                # episode_lengths
                M = ep_len.shape[0]
                if M > 0:
                    ds_len = hf["episode_lengths"]
                    old_m = ds_len.shape[0]
                    ds_len.resize(old_m + M, axis=0)
                    ds_len[old_m:old_m + M] = ep_len

                    # episode_starts for convenience (no compatibility impact)
                    starts = np.empty_like(ep_len, dtype=np.int64)
                    s = cum_steps
                    for i, L in enumerate(ep_len):
                        starts[i] = s
                        s += int(L)
                    cum_steps = int(s)

                    ds_st = hf["episode_starts"]
                    old_s = ds_st.shape[0]
                    ds_st.resize(old_s + M, axis=0)
                    ds_st[old_s:old_s + M] = starts

                # free memory
                del imgs, phys, acts, rews, discs, ep_len

        hf.attrs['total_steps'] = hf['images'].shape[0]
        hf.attrs['grid_size'] = 3
        # hf.attrs['thrx'] = THRX
        # hf.attrs['thry'] = THRY
        hf.attrs['x_low']  = X_LOW
        hf.attrs['x_high'] = X_HIGH
        hf.attrs['y_low']  = Y_LOW
        hf.attrs['y_high'] = Y_HIGH
        hf.attrs['thresholding'] = "signed_quantiles_1/3,2/3"
        hf.attrs['action_mapping'] = "center=0, E=1, W=2, N=3, S=4, NE=5, NW=6, SE=7, SW=8"
        hf.attrs['reward_reduce'] = REWARD_REDUCE
        hf.attrs['discount_reduce'] = DISCOUNT_REDUCE

        total_actions = int(action_counts.sum())
        action_dist = (action_counts / max(total_actions, 1)).astype(np.float64)

        print("[info] action counts per id 0..8:", action_counts.tolist())
        print("[info] action distribution:", np.round(action_dist, 6).tolist())

        # store for later inspection (non-breaking new datasets)
        hf.create_dataset('action_counts', data=action_counts, dtype='int64')
        hf.create_dataset('action_distribution', data=action_dist, dtype='float64')

        # convenient attrs too
        hf.attrs['action_counts'] = action_counts
        hf.attrs['action_distribution'] = action_dist
