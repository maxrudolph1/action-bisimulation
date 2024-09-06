import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import deque

import h5py
import tqdm
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter

import nets
import torch
import numpy as np

import utils
from models import SingleStep, SingleStepVAE, Autoencoder

np.set_printoptions(threshold=10000)
import torch.nn.functional as F
import random
from nav2d.nav2d import Navigate2D


def train(h, args):
    log_path = args.logdir
    if os.path.exists(log_path):
        response = input("Overwrite [y/n]? ")
        if response == "n":
            sys.exit(1)
        elif response == "y":
            shutil.rmtree(log_path)
        else:
            raise RuntimeError()
    os.makedirs(log_path)

    with open(os.path.join(log_path, "hyperparameters.json"), "w") as f:
        json.dump(dict(vars(args), **h), f)

    tensorboard = SummaryWriter(log_path)

    random.seed(h["seed"])
    torch.manual_seed(h["seed"])

    dataset = h5py.File(args.dataset, "r")
    dataset_keys = []
    dataset.visit(lambda key: dataset_keys.append(key) if isinstance(dataset[key], h5py.Dataset) else None)
    env = Navigate2D(dataset.attrs)

    model = Autoencoder(env.observation_space.shape, env.action_space.n, h).cuda()

    global_step = 0
    for epoch in range(h["n_epochs"]):
        sample_ind_all = np.random.permutation(len(dataset["obs"]))
        for i in tqdm.tqdm(range(-(len(sample_ind_all) // -h["batch_size"])), desc=f"Epoch #{epoch}"):
            start = i * h["batch_size"]
            end = min(len(sample_ind_all), (i + 1) * h["batch_size"])
            sample_ind = np.sort(sample_ind_all[start:end])
            samples = {key: dataset[key][sample_ind] for key in dataset_keys}

            losses, recon = model.train_step(samples)
            tensorboard.add_scalars("auto", losses, global_step)

            if global_step > 0 and global_step % 1000 == 0:
                obs = (samples["obs"][0] + 1) / 2
                recon = (recon[0].cpu().numpy() + 1) / 2
                obs[:, :, -1] = [[0], [1], [0]]
                recon[:, :, 0] = [[0], [1], [0]]
                tensorboard.add_images("auto", np.stack([obs, recon]), global_step)

                torch.save(model, os.path.join(log_path, "model.pt"))

            global_step += 1

    torch.save(model, os.path.join(log_path, "model.pt"))


if __name__ == "__main__":
    HYPERPARAMETERS = {
        "batch_size": 128,
        "seed": 1,
        "n_epochs": 40,
        "learning_rate": 0.0004,
        "latent_dim": 256,
        "beta": 0.1,
    }

    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    train(HYPERPARAMETERS, args)
