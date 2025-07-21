import numpy as np
import os
import glob

def examine_raw_episode(episode_path):
    data = np.load(episode_path)
    print(f"Episode: {episode_path}")
    print(f"Keys: {list(data.keys())}")

    for key in data.keys():
        print(f"\n  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        print(f" ---Sample {key}: {data[key][10]}")
    print()


def main():
    # Raw Episodes
    buffer_dir = "~/bisim/exorl/datasets/point_mass_maze/rnd/buffer"
    episodes = glob.glob(os.path.expanduser(f"{buffer_dir}/*.npz"))

    for ep in episodes[:2]:
        examine_raw_episode(ep)


if __name__ == "__main__":
    main()
