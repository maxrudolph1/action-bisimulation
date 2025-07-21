import h5py
import imageio
import numpy as np


class PointMazeH5Reader:
    def __init__(self, h5_path: str):
        self.f = h5py.File(h5_path, 'r')
        self.images   = self.f['images']    # (total_steps, H, W, 3)
        self.actions  = self.f['action']    # (total_steps, 1) or (total_steps, 2)
        self.rewards  = self.f['reward']    # (total_steps, 1) or (total_steps, N)
        self.discounts= self.f['discount']  # (total_steps, 1)

    def num_steps(self) -> int:
        return self.images.shape[0]

    def get_frames(self, start: int, length: int) -> np.ndarray:
        """
        Return a (length × H × W × 3) array of uint8.
        """
        return self.images[start : start + length]

    def save_gif(
        self,
        out_path: str,
        start:    int,
        length:   int,
        fps:      int = 20,
    ):
        """
        Slice out [start : start+length] frames and write to `out_path`.
        """
        frames = self.get_frames(start, length)
        # imageio wants a list of H×W×3 arrays:
        imageio.mimsave(out_path, list(frames), fps=fps)

    def close(self):
        self.f.close()


def load_episode_boundaries(ds_path, ep_len=1000):
    with h5py.File(ds_path, 'r') as f:
        discounts = f['discount'][:].squeeze()
    total = discounts.shape[0]
    n_eps = total // ep_len
    starts = np.arange(0, n_eps * ep_len, ep_len)
    ends = starts + ep_len - 1
    return starts, ends

# Using the episode boundaries
# def load_episode_boundaries(h5_path):
#     with h5py.File(h5_path, 'r') as f:
#         lengths = f['episode_lengths'][:]   # shape = (num_episodes,)
#     starts = np.cumsum(np.concatenate([[0], lengths[:-1]]))
#     ends   = starts + lengths - 1
#     return starts, ends

if __name__ == "__main__":
    dataset_path = "/home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_0721.hdf5"
    starts, ends = load_episode_boundaries(dataset_path)
    print("Found", len(starts), "episode starts and", len(ends), "ends")
    # for i, (s, e) in enumerate(zip(starts, ends)):
    #     print(f"  Episode {i:03d}: frames {s} .. {e} (length {e-s+1})")

    reader = PointMazeH5Reader(dataset_path)
    for i, (s, e) in enumerate(zip(starts, ends)):
        if i >= 5:
            break  # stop after 5 episodes for now
        length = e - s + 1
        out_name = f"episode_{i:03d}.gif"
        print(f"Saving {out_name} (len={length})...")
        reader.save_gif(out_name, start=s, length=length, fps=20)
    reader.close()
