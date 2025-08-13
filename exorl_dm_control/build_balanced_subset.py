#!/usr/bin/env python3

"""RUN COMMAND
python build_balanced_subset.py \
  --dataset /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5 \
  --obs-buffer-size 4 \
  --num-actions 9 \
  --seed 0 \
  --out /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/balanced_valid_t_all_eps_eplen.npy

# optional:
#   --stall-eps 1e-3
#   --per-class 50000          # if you want smaller than the min across classes
"""

import argparse, os, time
import numpy as np
import h5py
from typing import Optional
from tqdm import tqdm


def load_action_and_disp(ds):
    # Load contiguously once (fast)
    acts_all = np.asarray(ds["action"][:]).reshape(-1)        # int32
    pos = np.asarray(ds["physics"][:, :2])                    # float64
    disp_all = np.linalg.norm(pos[1:] - pos[:-1], axis=1)     # (T-1,)
    return acts_all, disp_all


def fast_counts_for_indices(acts_all, valid_t, num_actions):
    labels = acts_all[valid_t]
    return np.bincount(labels, minlength=num_actions)


def class_aware_stall_filter_fast(acts_all, disp_all, valid_t,
                                  eps_move, eps_noop, noop_cls=0):
    labels = acts_all[valid_t]
    disp_v = disp_all[valid_t]
    keep = ((labels == noop_cls) & (disp_v <= eps_noop)) | ((labels != noop_cls) & (disp_v >= eps_move))
    return valid_t[keep]


def valid_time_indices(ep_lengths: np.ndarray, K: int) -> np.ndarray:
    """Global t where (t-K+1..t) and (t+1) exist within the same episode."""
    starts = np.empty_like(ep_lengths, dtype=np.int64)
    starts[0] = 0
    if len(ep_lengths) > 1:
        starts[1:] = np.cumsum(ep_lengths[:-1])
    out = []
    for s, L in zip(starts, ep_lengths):
        # t ranges so that K frames ending at t exist, and t+1 exists: t in [K-1, L-2]
        if L >= K+1:
            rng = np.arange(s + (K-1), s + (L-1))  # inclusive K-1 .. L-2
            out.append(rng)
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(out, axis=0)


def stream_stall_filter(ds_physics, valid_t: np.ndarray, eps: float, chunk: int = 500_000):
    """Return filtered valid_t keeping |Δpos|>=eps, computed in chunks to save RAM."""
    keep = np.zeros(valid_t.shape[0], dtype=bool)
    mean_acc = 0.0
    n_kept = 0
    n = len(valid_t)
    for i in range(0, n, chunk):
        sl = slice(i, min(i+chunk, n))
        tblock = valid_t[sl]
        # physics is (T, 4); we use [:2] as x,y
        dxy = ds_physics[tblock + 1, :2] - ds_physics[tblock, :2]  # (B, 2)
        disp = np.linalg.norm(dxy, axis=1)
        m = disp >= eps
        keep[sl] = m
        n_kept += int(m.sum())
        mean_acc += float(disp[m].sum())
    mean_disp = (mean_acc / max(n_kept, 1)) if n_kept > 0 else 0.0
    return valid_t[keep], mean_disp


def compute_counts_for_indices(ds_action, idx: np.ndarray, num_actions: int, chunk: int = 500_000):
    counts = np.zeros(num_actions, dtype=np.int64)
    n = len(idx)
    for i in tqdm(range(0, n, chunk), desc="computing indices counts"):
        sl = idx[i:i+chunk]
        a = ds_action[sl].reshape(-1)
        counts += np.bincount(a, minlength=num_actions)
    return counts


def build_balanced_unique_indices(
    ds_action,
    valid_t: np.ndarray,
    num_actions: int,
    per_class: Optional[int],
    seed: int,
):
    """
    Returns a shuffled array of valid_t indices, selecting exactly per_class from each class
    without replacement. If per_class=None, uses the minimum available across classes.
    """
    rng = np.random.default_rng(seed)

    # Read all labels for valid_t in one shot (int32 -> small)
    # labels = ds_action[valid_t].reshape(-1)  # (N,)
    acts_all = ds_action[:].reshape(-1)
    labels   = acts_all[valid_t]
    A = num_actions

    # Build pools per class (positions within valid_t)
    idxs_by_a = [np.where(labels == a)[0] for a in range(A)]
    per_class_min = min(len(p) for p in idxs_by_a)
    if per_class is None:
        per_class = per_class_min
    else:
        per_class = min(per_class, per_class_min)
    if per_class == 0:
        raise ValueError("At least one class has zero valid examples; cannot build a balanced unique subset.")

    chosen_local = []
    for a in range(A):
        pool = idxs_by_a[a]
        sel = rng.choice(pool, size=per_class, replace=False)
        chosen_local.append(sel)
    chosen_local = np.concatenate(chosen_local, axis=0)
    rng.shuffle(chosen_local)

    # Map back to global time indices
    balanced_valid_t = valid_t[chosen_local]
    # Sanity: counts of selected
    sel_counts = np.bincount(labels[chosen_local], minlength=A)
    return balanced_valid_t, sel_counts


def scan_disp_quantiles_by_class_fast(acts_all, disp_all, valid_t,
                                      noop_cls=0, sample=2_000_000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    v = valid_t if len(valid_t) <= sample else valid_t[rng.choice(len(valid_t), sample, replace=False)]
    labels = acts_all[v]
    disp_v  = disp_all[v]   # valid_t always refers to t where t+1 exists

    d_noop = disp_v[labels == noop_cls]
    d_move = disp_v[labels != noop_cls]

    def dump(name, arr):
        qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\n[|Δpos| quantiles: {name}] n={arr.size}")
        for q in qs:
            print(f"  q{q:>3}: {np.percentile(arr, q):.6g}")

    dump("noop", d_noop)
    dump("move", d_move)

    if d_noop.size and d_move.size:
        q95_noop = float(np.percentile(d_noop, 95))
        q10_move = float(np.percentile(d_move, 10))
        eps_noop_suggest = q95_noop
        eps_move_suggest = max(q95_noop, q10_move)
        print("\n[threshold suggestions]")
        print(f"  eps_noop ≈ q95(noop) = {eps_noop_suggest:.6g}")
        print(f"  eps_move ≈ max(q95(noop), q10(move)) = {eps_move_suggest:.6g}")


def main():
    ap = argparse.ArgumentParser(description="Build a strictly-balanced UNIQUE subset of PointMaze transitions.")
    ap.add_argument("--dataset", required=True, type=str, help="Path to HDF5 (images, physics, action, episode_lengths)")
    ap.add_argument("--obs-buffer-size", type=int, default=4, help="K frames stacked (must match training)")
    ap.add_argument("--num-actions", type=int, default=9)
    ap.add_argument("--per-class", type=int, default=None, help="Optional cap per class; default uses min across classes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="Output .npy for balanced valid_t")

    ap.add_argument("--stall-eps-move", type=float, default=None,
                    help="Keep moving actions only if |Δpos| >= this.")
    ap.add_argument("--stall-eps-noop", type=float, default=None,
                    help="Keep no-op (class 0) only if |Δpos| <= this.")
    ap.add_argument("--scan-quantiles", action="store_true",
                    help="Print |Δpos| quantiles by class and exit (to help pick thresholds).")

    args = ap.parse_args()

    t0 = time.time()
    with h5py.File(args.dataset, "r") as ds:
        ep = ds["episode_lengths"][:]
        print(f"[info] episodes: {len(ep)}")

        valid_t = valid_time_indices(ep, args.obs_buffer_size)
        print(f"[info] valid_t (K={args.obs_buffer_size}): {len(valid_t):,}")

        print("[phase] loading action & physics as arrays ...")
        acts_all, disp_all = load_action_and_disp(ds)
        print("[phase] arrays loaded")

        # Optional: scan and exit
        if args.scan_quantiles:
            print("[phase] scanning quantiles (fast path) ...")
            scan_disp_quantiles_by_class_fast(acts_all, disp_all, valid_t, noop_cls=0)
            print("[info] scan complete; exiting.")
            return

        # Optional: class-aware stall filter
        if (args.stall_eps_move is not None) or (args.stall_eps_noop is not None):
            em = args.stall_eps_move if args.stall_eps_move is not None else 0.0
            en = args.stall_eps_noop if args.stall_eps_noop is not None else float("inf")
            print(f"[phase] class-aware stall filter: move >= {em}, noop <= {en}")
            valid_t = class_aware_stall_filter_fast(acts_all, disp_all, valid_t, em, en, noop_cls=0)
            print(f"[info] kept after stall filter: {len(valid_t):,}")

        # Counts over the (possibly filtered) pool
        print("[phase] computing pool counts ...")
        counts_pool = fast_counts_for_indices(acts_all, valid_t, args.num_actions)

        # if args.stall_eps is not None:
        #     print(f"[info] applying stall filter |Δpos| >= {args.stall_eps} ...")
        #     valid_t, mean_disp = stream_stall_filter(ds["physics"], valid_t, eps=args.stall_eps)
        #     print(f"[info]   kept: {len(valid_t):,}  mean |Δpos|: {mean_disp:.6f}")

        # Show counts over the pool we're sampling from
        # counts_pool = compute_counts_for_indices(ds["action"], valid_t, args.num_actions)
        total_pool = counts_pool.sum()
        print("\n[pool counts]")
        for a, c in enumerate(counts_pool):
            print(f"  action {a}: {int(c):>10}  frac={c/total_pool:0.6f}")
        per_class_min = int(counts_pool.min())
        print(f"[pool min per-class] {per_class_min}  => max unique balanced total = {per_class_min*args.num_actions:,}")

        # Build balanced unique subset
        balanced_valid_t, sel_counts = build_balanced_unique_indices(
            ds["action"], valid_t, args.num_actions, per_class=args.per_class, seed=args.seed
        )

    print("\n[selected balanced counts]")
    for a, c in enumerate(sel_counts):
        print(f"  action {a}: {int(c):>10}")
    print(f"[selected total] {int(sel_counts.sum()):,}")

    # Save
    out_path = args.out or (os.path.splitext(args.dataset)[0] + f"_balancedK{args.obs_buffer_size}.npy")
    np.save(out_path, balanced_valid_t.astype(np.int64))
    print(f"[saved] {out_path}")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
