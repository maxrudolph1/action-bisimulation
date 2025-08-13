#!/usr/bin/env python3
import numpy as np
import os
import glob
import h5py


# --------------------
# utilities
# --------------------
def _valid_time_indices(ep_lengths, K):
    """Return global time indices t where (t-K+1..t) and (t+1) exist in the same episode."""
    starts = np.empty_like(ep_lengths, dtype=np.int64)
    starts[0] = 0
    if len(ep_lengths) > 1:
        starts[1:] = np.cumsum(ep_lengths[:-1])
    valid_t = []
    for s, L in zip(starts, ep_lengths):
        # require t >= K-1 and t+1 <= s+L-1  ->  t <= s+L-2
        for t_in_ep in range(K - 1, L - 1):
            valid_t.append(s + t_in_ep)
    return np.asarray(valid_t, dtype=np.int64)


def _bincount_stream_action(ds_action, idx, num_actions=9, chunk=200_000):
    """Stream counts for ds_action[idx] in chunks to avoid loading everything."""
    counts = np.zeros(num_actions, dtype=np.int64)
    n = len(idx)
    for i in range(0, n, chunk):
        sl = idx[i:i+chunk]
        arr = ds_action[sl].reshape(-1)      # (chunk,)
        counts += np.bincount(arr, minlength=num_actions)
    return counts


def _bincount_whole_action(ds_action, num_actions=9, chunk=1_000_000):
    """Stream counts over the whole 'action' dataset (ignores K/episode validity)."""
    T = ds_action.shape[0]
    counts = np.zeros(num_actions, dtype=np.int64)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        arr = ds_action[start:end].reshape(-1)
        counts += np.bincount(arr, minlength=num_actions)
    return counts

def _apply_stall_filter(ds_physics, valid_t, eps=1e-3):
    """
    Keep transitions whose position delta magnitude >= eps.
    Assumes physics columns [:2] are x,y (as in your dataset).
    """
    p = ds_physics
    # vectorized displacement on the selected t’s
    dxy = p[valid_t + 1, :2] - p[valid_t, :2]
    disp = np.linalg.norm(dxy, axis=1)
    mask = disp >= eps
    return valid_t[mask], disp[mask].mean() if mask.any() else 0.0

def _print_counts(name, counts):
    total = counts.sum()
    fracs = counts / max(total, 1)
    print(f"\n{name}")
    print(f"  total samples: {int(total)}")
    for a, (c, f) in enumerate(zip(counts, fracs)):
        print(f"  action {a}: count={int(c):>8}  frac={f:0.6f}")

# --------------------
# your original helpers (kept)
# --------------------

def examine_raw_episode(episode_path):
    data = np.load(episode_path)
    print(f"Episode: {episode_path}")
    print(f"Keys: {list(data.keys())}")
    for key in data.keys():
        print(f"\n  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        print(f" ---Sample {key}: {data[key][10]}")
    print()

def examine_hdf5(ds_path, K=4, stall_eps=None, num_actions=9, max_preview=1003):
    with h5py.File(ds_path, "r") as ds:
        # lightweight preview (first ~1k) just for sanity; skip heavy image loads
        phys = ds["physics"][:max_preview]
        acts = ds["action"][:max_preview]
        rews = ds["reward"][:max_preview]
        discs = ds["discount"][:max_preview]

        def print_data(key, data_item):
            n = data_item.shape[0]
            print(f"\n  {key}: shape={data_item.shape}, dtype={data_item.dtype}")
            for j in range(n-4, n):
                if j >= 0:
                    print(f" ---Sample {key}[{j}]: {data_item[j]}")
        print_data('phys', phys)
        print_data('acts', acts)
        print_data('rews', rews)
        print_data('discs', discs)

        # full-file actionable stats (no big arrays loaded)
        ep_lens = ds["episode_lengths"][:]
        ds_action = ds["action"]            # (T,1)
        ds_physics = ds["physics"]          # (T,4)

        # 1) Whole-file action counts (ignores K/episode validity)
        counts_all = _bincount_whole_action(ds_action, num_actions=num_actions)
        _print_counts("Whole-file action counts (no K constraint)", counts_all)

        # 2) Counts over valid transitions used by training (respects K and t+1)
        valid_t = _valid_time_indices(ep_lens, K)
        counts_valid = _bincount_stream_action(ds_action, valid_t, num_actions=num_actions)
        _print_counts(f"Valid transition counts (K={K}, same-episode t..t+1)", counts_valid)

        # 3) Optional: stall filter (removes tiny motion deltas)
        if stall_eps is not None:
            valid_t_sf, mean_disp = _apply_stall_filter(ds_physics, valid_t, eps=stall_eps)
            counts_valid_sf = _bincount_stream_action(ds_action, valid_t_sf, num_actions=num_actions)
            _print_counts(f"Valid counts with stall filter (K={K}, |Δpos| ≥ {stall_eps})", counts_valid_sf)
            print(f"\n  Kept {len(valid_t_sf)} / {len(valid_t)} valid transitions "
                  f"({100.0*len(valid_t_sf)/max(len(valid_t), 1):.2f}%). "
                  f"Mean |Δpos| among kept: {mean_disp:.6f}")

        # 4) Max strictly-balanced unique subset size (from the valid pool you’d use)
        base_counts = counts_valid if stall_eps is None else counts_valid_sf
        per_class_min = int(base_counts.min())
        max_unique_balanced = per_class_min * num_actions
        print(f"\nMax strictly-balanced unique subset possible (given K{'' if stall_eps is None else f' and stall_eps={stall_eps}'}):")
        print(f"  per-class = {per_class_min}, total = {max_unique_balanced}")


# --------------------
# main
# --------------------
def main():
    # Raw Episodes
    buffer_dir = "~/bisim/exorl/datasets/point_mass_maze/rnd/buffer"
    episodes = glob.glob(os.path.expanduser(f"{buffer_dir}/*.npz"))
    for ep in episodes[:1]:
        examine_raw_episode(ep)

    # HDF5
    ds_path = "/home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5"

    # Configure these two as you like:
    K = 4                 # obs_buffer_size you use in training
    stall_eps = None      # e.g., 1e-3 to filter very small motion; or None to disable

    examine_hdf5(ds_path, K=K, stall_eps=stall_eps, num_actions=9, max_preview=1003)


if __name__ == "__main__":
    main()
