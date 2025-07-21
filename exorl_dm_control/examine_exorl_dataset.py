import numpy as np
import os
import glob

import h5py

def examine_raw_episode(episode_path):
    data = np.load(episode_path)
    print(f"Episode: {episode_path}")
    print(f"Keys: {list(data.keys())}")

    for key in data.keys():
        print(f"\n  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        print(f" ---Sample {key}: {data[key][10]}")
    print()


def examine_hdf5(ds_path):
    with h5py.File(ds_path, "r") as ds:
        imgs = ds["images"][:1003]
        phys = ds["physics"][:1003]
        acts = ds["action"][:1003]
        rews = ds["reward"][:1003]
        discs = ds["discount"][:1003]

    def print_data(key, data_item):
        print(f"\n  {key}: shape={data_item.shape}, dtype={data_item.dtype}")
        print(f" ---Sample {key}: {data_item[999]}")
        print(f" ---Sample {key}: {data_item[1000]}")
        print(f" ---Sample {key}: {data_item[1001]}")
        print(f" ---Sample {key}: {data_item[1002]}")

    print_data('imgs', imgs)
    print_data('phys', phys)
    print_data('acts', acts)
    print_data('rews', rews)
    print_data('discs', discs)


def main():
    # Raw Episodes
    buffer_dir = "~/bisim/exorl/datasets/point_mass_maze/rnd/buffer"
    episodes = glob.glob(os.path.expanduser(f"{buffer_dir}/*.npz"))

    for ep in episodes[:1]:
        examine_raw_episode(ep)

    # HDF5
    ds_path = "/home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_0721.hdf5"
    examine_hdf5(ds_path)


if __name__ == "__main__":
    main()
