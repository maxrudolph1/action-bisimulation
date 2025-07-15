import numpy as np
import os
import glob

def examine_episode(episode_path):
    data = np.load(episode_path)
    print(f"Episode: {episode_path}")
    print(f"Keys: {list(data.keys())}")

    for key in data.keys():
        print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        if key == 'observation':
            print(f"    Sample observation: {data[key][0]}")
    print()

# Examine first few episodes
buffer_dir = "~/bisim/exorl/datasets/point_mass_maze/rnd/buffer"
episodes = sorted(glob.glob(os.path.expanduser(f"{buffer_dir}/*.npz")))

for ep in episodes[:2]:
    examine_episode(ep)
