#!/usr/bin/env python3
"""
python build_balanced_subset.py \
  --dataset /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/displacement_discretized.hdf5 \
  --out /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/balanced_subset_displacement_discretized.npy
"""
import argparse, os, numpy as np, h5py

def valid_time_indices(ep_lengths: np.ndarray, K: int) -> np.ndarray:
    starts = np.empty_like(ep_lengths, dtype=np.int64)
    starts[0] = 0
    if len(ep_lengths) > 1:
        starts[1:] = np.cumsum(ep_lengths[:-1])
    out = []
    for s, L in zip(starts, ep_lengths):
        if L >= K+1:
            out.append(np.arange(s + (K-1), s + (L-1), dtype=np.int64))  # [K-1 .. L-2]
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--obs-buffer-size", type=int, default=1)
    ap.add_argument("--num-actions", type=int, default=9)
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.dataset, "r") as ds:
        ep = ds["episode_lengths"][:]
        valid_t = valid_time_indices(ep, args.obs_buffer_size)
        acts_all = ds["action"][:].reshape(-1)
        labels = acts_all[valid_t]

    pools = [np.where(labels == a)[0] for a in range(args.num_actions)]
    per_class_min = min(len(p) for p in pools)
    per_class = per_class_min if args.per_class is None else min(args.per_class, per_class_min)
    if per_class == 0:
        raise RuntimeError("A class has zero examples in the pool.")

    chosen_local = np.concatenate([rng.choice(p, size=per_class, replace=False) for p in pools])
    rng.shuffle(chosen_local)
    balanced_valid_t = valid_t[chosen_local].astype(np.int64)
    np.save(args.out, balanced_valid_t)
    print(f"saved {args.out}  (total {balanced_valid_t.size})")

if __name__ == "__main__":
    main()
