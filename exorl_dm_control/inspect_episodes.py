import os
import h5py
import imageio
import numpy as np


class PointMazeH5Reader:
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.f = h5py.File(h5_path, 'r')
        self.images    = self.f['images']      # (total_steps, H, W, 3)
        self.actions   = self.f['action']      # (total_steps, 1) int32
        self.rewards   = self.f['reward']      # (total_steps, 1) float32
        self.discounts = self.f['discount']    # (total_steps, 1) float32

    def num_steps(self) -> int:
        return self.images.shape[0]

    def get_frames(self, start: int, length: int) -> np.ndarray:
        """Return a (length × H × W × 3) array of uint8."""
        return self.images[start : start + length]

    def save_gif(self, out_path: str, start: int, length: int, fps: int = 20):
        """Slice out [start : start+length] frames and write to `out_path`."""
        frames = self.get_frames(start, length)
        imageio.mimsave(out_path, list(frames), fps=fps, loop=0)

    def close(self):
        self.f.close()


def load_episode_boundaries(h5_path):
    """
    Prefer 'episode_starts' if present; else compute from 'episode_lengths'.
    Returns (starts, ends) as 1D int arrays.
    """
    with h5py.File(h5_path, 'r') as f:
        if 'episode_starts' in f:
            starts = f['episode_starts'][:].astype(np.int64)
            lengths = f['episode_lengths'][:].astype(np.int64)
            ends = starts + lengths - 1
        else:
            lengths = f['episode_lengths'][:].astype(np.int64)
            starts = np.cumsum(np.concatenate([[0], lengths[:-1]])).astype(np.int64)
            ends   = starts + lengths - 1
    return starts, ends


if __name__ == "__main__":
    # Path of the dataset you created in the converter script
    dataset_path = os.path.expanduser(
        "~/bisim/exorl/datasets/point_mass_maze/rnd/displacement_discretized.hdf5"
    )

    starts, ends = load_episode_boundaries(dataset_path)
    print("Found", len(starts), "episodes.")
    # for i, (s, e) in enumerate(zip(starts, ends)):
    #     print(f"  Episode {i:03d}: frames {s}..{e} (len {e - s + 1})")

    reader = PointMazeH5Reader(dataset_path)
    try:
        for i, (s, e) in enumerate(zip(starts, ends)):
            if i >= 5:  # save first 5 episodes as GIFs
                break
            length = int(e - s + 1)
            out_name = f"episode_{i:03d}.gif"
            print(f"Saving {out_name} (len={length})...")
            reader.save_gif(out_name, start=int(s), length=length, fps=20)
    finally:
        reader.close()

