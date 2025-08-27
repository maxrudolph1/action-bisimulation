#!/usr/bin/env python3
"""
Plot ALL PointMaze (x,y) positions from EXORL RND without subsampling.

Usage examples
--------------
# From episodes directory of .npz files:
python exorl_dm_control/plot_dataset.py \
  --buffer-dir ~/bisim/exorl/datasets/point_mass_maze/rnd/buffer \
  --output ~/bisim/exorl/datasets/point_mass_maze/rnd/pointmaze_all_points.png

# From a single HDF5 with a 'physics' dataset:
python exorl_dm_control/plot_dataset.py \
  --hdf5 /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_continuous.hdf5 \
  --output /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/pointmaze_all_points_from_hdf5.png
"""
import argparse
import glob
import os
import sys
from typing import Tuple, List

import numpy as np
import h5py  # only needed if --hdf5 is used

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt


def load_points_from_buffer(buffer_dir: str) -> Tuple[np.ndarray, int]:
    """Load ALL (x, y) points from every *.npz in buffer_dir (no subsampling)."""
    ep_paths = sorted(glob.glob(os.path.expanduser(os.path.join(buffer_dir, "*.npz"))))
    if not ep_paths:
        raise FileNotFoundError(f"No .npz episodes found under: {buffer_dir}")

    chunks: List[np.ndarray] = []
    used = 0
    for p in ep_paths:
        try:
            with np.load(p) as data:
                if "physics" in data:
                    phys = data["physics"]
                else:
                    raise KeyError("'physics' not in episode.")

                if phys.ndim != 2 or phys.shape[1] < 2:
                    raise ValueError(f"Bad shape for physics in {p}: {phys.shape}")

                xy = np.asarray(phys[:, :2], dtype=np.float64, order="C")
                chunks.append(xy)
                used += 1
        except Exception as e:
            print(f"[warn] Skipping {p}: {e}", file=sys.stderr)

    if not chunks:
        raise RuntimeError("Found episodes but none produced valid (x,y) points.")
    return np.concatenate(chunks, axis=0), used


def load_points_from_hdf5(hdf5_path: str) -> np.ndarray:
    """Load ALL (x, y) points from HDF5 /physics (no subsampling)."""
    if h5py is None:
        raise ImportError("h5py not installed; required for --hdf5.")
    hdf5_path = os.path.expanduser(hdf5_path)
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        if "physics" not in f:
            raise KeyError("HDF5 missing 'physics' dataset.")
        d = f["physics"]
        if d.ndim != 2 or d.shape[1] < 2:
            raise ValueError(f"Expected physics shape (N, >=2), got {d.shape}")
        return np.array(d[:, :2], dtype=np.float64)


def plot_points(xy: np.ndarray, output_path: str, title: str) -> None:
    """Render scatter of ALL points and save to file."""
    if xy.size == 0:
        raise ValueError("No points to plot.")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    # # Plot every point; rasterized for speed/size with large N
    # sc = ax.scatter(xy[:, 0], xy[:, 1], s=0.5, alpha=0.2, linewidths=0, rasterized=True)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_title(title)

    plt.tight_layout()
    output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot ALL (x,y) physics points from EXORL PointMaze.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--buffer-dir", type=str, help="Directory with episode .npz files (e.g., .../rnd/buffer)")
    src.add_argument("--hdf5", type=str, help="Path to HDF5 with 'physics' dataset")

    ap.add_argument("--output", type=str, default="pointmaze_all_points.png",
                    help="Output image path (e.g., PNG/PDF/SVG)")

    args = ap.parse_args()

    if args.buffer_dir:
        xy, n_eps = load_points_from_buffer(args.buffer_dir)
        title = f"EXORL PointMaze (ALL points; episodes={n_eps}, N={len(xy)})"
    else:
        xy = load_points_from_hdf5(args.hdf5)
        title = f"EXORL PointMaze (ALL points; N={len(xy)})"

    print(f"[info] Loaded {len(xy)} points.")
    plot_points(xy, args.output, title=title)
    print(f"[ok] Saved: {os.path.expanduser(args.output)}")


if __name__ == "__main__":
    main()
