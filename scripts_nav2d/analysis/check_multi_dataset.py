import h5py
import matplotlib.pyplot as plt
import numpy as np
# file = '/home/mrudolph/documents/actbisim/scripts_nav2d/datasets/nav2d_dataset_s0_e0.5_size60_patooty_2.hdf5'
# file = '/home/mrudolph/documents/actbisim/scripts_nav2d/datasets/nav2d_dataset_s0_e0.5_size100_test_1.hdf5'
# file = '/home/mrudolph/documents/actbisim/scripts_nav2d/datasets/nav2d_dataset_s0_e0.5_size100_bloop_1.hdf5'
file = '/home/mrudolph/documents/actbisim/scripts_nav2d/datasets/nav2d_dataset_s0_e0.5_size1000_bloop_test_2_3.hdf5'
dataset = h5py.File(file, "r")

dataset_keys = []
dataset.visit(
    lambda key: dataset_keys.append(key)
    if isinstance(dataset[key], h5py.Dataset)
    else None
)

mem_dataset = {}
for key in dataset_keys:
    mem_dataset[key] = dataset[key][:]
dataset = mem_dataset


obs = (dataset['obs'][0] + 1)/2

plt.imshow(obs.transpose(1, 2, 0))
plt.savefig('obstacle.png')
