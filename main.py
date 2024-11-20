import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import deque
import time
import h5py
import tqdm
from matplotlib import cm
import torch
import numpy as np
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra

import torch.nn.functional as F
import random
from environments.nav2d.utils import perturb_heatmap
import datetime
from representations.acro import Acro
from representations.single_step import SingleStep
from representations.multi_step import MultiStep
from representations.bvae import BetaVariationalAutoencoder
import os


MODEL_DICT = {'single_step': SingleStep, 'acro': Acro, 'multi_step': MultiStep, 'bvae': BetaVariationalAutoencoder}

def load_dataset(dataset_path):
    with h5py.File(dataset_path, "r") as dataset:
        dataset_keys = []
        dataset.visit(lambda key: dataset_keys.append(key) if isinstance(dataset[key], h5py.Dataset) else None)

        mem_dataset = {}
        for key in dataset_keys:
            mem_dataset[key] = dataset[key][:]
        dataset = mem_dataset

    obs_shape = dataset["obs"][0].shape
    act_shape = dataset["action"].max() + 1
    return dataset, obs_shape, act_shape

def create_models(cfg: DictConfig, obs_shape, act_shape):
    algo_cfgs = cfg.algos
    model_names = list(algo_cfgs.keys())
    models = {}

    for model_name in model_names:
        model_cfg = algo_cfgs[model_name]
        model = MODEL_DICT[model_name](obs_shape=obs_shape, act_shape=act_shape, cfg=cfg)
        models[model_name] = model
    return models

def initialize_dependant_models(models):
    for model_name, model in models.items():
        model.share_dependant_models(models)
    return models

def log_to_wandb(cfg, models, logs, samples, train_step):
    if train_step % cfg.met_log_freq == 0:
        labeled_logs = {f"{algo_name}/{key}": value for algo_name, algo_log in logs.items() for key, value in algo_log.items()}
        wandb.log(labeled_logs, step=train_step)
    if train_step % cfg.img_log_freq == 0:
        for model_name, model in models.items():
            obs = samples["obs"][0]
            obs[1, :, :] = -1
            obs[1, obs.shape[1] // 2, obs.shape[2] // 2] = 1
            img = wandb.Image(np.swapaxes(perturb_heatmap(obs, model.encoder)[1], 0,2))
            wandb.log({f"{model_name}/heatmap": img}, step=train_step)

            if model_name == "bvae":
                obs = torch.tensor(samples["obs"][0])
                if (model.decode_forward):
                    act = torch.tensor(samples["action"][0])
                    obs_recon = model.decoder(model.forward_model(model.encoder(obs[None].cuda()), act[None].cuda())).squeeze().detach().cpu().numpy()
                    disp_obs = np.swapaxes(samples["obs_next"][0], 0, 2)
                else:
                    obs_recon = model.decoder(model.encoder(obs[None].cuda())).squeeze().detach().cpu().numpy()
                    disp_obs = np.swapaxes(samples["obs"][0], 0, 2)
                img = wandb.Image(np.concatenate([np.swapaxes(obs_recon, 0,2), disp_obs], axis=1))
                wandb.log({f"{model_name}/reconstruction": img}, step=train_step)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log_path = cfg.logdir + ("_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    if cfg.wandb:
        wandb.init(entity='evan-kuo-edu', project="nav2d", config=OmegaConf.to_container(cfg),)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # dataset, obs_shape, act_shape = load_dataset(cfg.dataset)
    dataset_paths = [
        '/nfs/homes/bisim/ekuo/action-bisimulation/datasets/nav2d_dataset_s0_e0.5_size200000_p1_k_steps_10.hdf5',
        '/nfs/homes/bisim/ekuo/action-bisimulation/datasets/nav2d_dataset_s0_e0.5_size200000_p2_k_steps_10.hdf5',
        '/nfs/homes/bisim/ekuo/action-bisimulation/datasets/nav2d_dataset_s0_e0.5_size200000_p3_k_steps_10.hdf5',
        '/nfs/homes/bisim/ekuo/action-bisimulation/datasets/nav2d_dataset_s0_e0.5_size200000_p4_k_steps_10.hdf5',
        '/nfs/homes/bisim/ekuo/action-bisimulation/datasets/nav2d_dataset_s0_e0.5_size200000_p5_k_steps_10.hdf5',
    ]
    obs_shape = (3, 15, 15)
    act_shape = 4

    models = create_models(cfg, obs_shape, act_shape)
    models = initialize_dependant_models(models)

    ts = 0
    for d in dataset_paths:
        dataset, obs_shape, act_shape = load_dataset(d)
        ts = train(cfg, dataset, models, ts)
        dataset = None

    # train(cfg, dataset_paths, models)


def train(cfg: DictConfig, dataset, models, ts):
    dataset_keys = list(dataset.keys())
    wandb_logs = {key: {} for key in models.keys()}
    # train_step = 0
    train_step = ts

    for epoch in range(cfg.n_epochs):
        sample_ind_all = np.random.permutation(len(dataset["obs"]))
        sample_ind_next = np.random.permutation(len(dataset["obs"]))
        steps_per_epoch = -(len(sample_ind_all) // -cfg.batch_size)
        for i in tqdm.tqdm(range(steps_per_epoch), desc=f"Epoch #{epoch}"):
            start = i * cfg.batch_size
            end = min(len(sample_ind_all), (i + 1) * cfg.batch_size)
            sample_ind = np.sort(sample_ind_all[start:end]) # TODO: WHY DO WE SORT?
            samples = {key: dataset[key][sample_ind] for key in dataset_keys}
            for model_name, model in models.items():
                log = model.train_step(samples, epoch, train_step)
                wandb_logs[model_name].update(log)

            if cfg.wandb:
                log_to_wandb(cfg, models, wandb_logs, samples, train_step)

            train_step += 1

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logdir = os.path.join(cfg.logdir, time_str)
    os.makedirs(logdir)
    for model_name, model in models.items():
        model.save(logdir + f"/{model_name}.pt")

    return train_step

# NEW REVISION
# def train(cfg: DictConfig, dataset_paths, models):
#     # Load dataset keys and shapes from the first dataset
#     with h5py.File(dataset_paths[0], "r") as file:
#         dataset_keys = []
#         file.visit(
#             lambda key: dataset_keys.append(key)
#             if isinstance(file[key], h5py.Dataset)
#             else None
#         )
#         total_samples_per_file = len(file["obs"])

#     wandb_logs = {key: {} for key in models.keys()}
#     train_step = 0
#     num_files = len(dataset_paths)
#     total_samples = num_files * total_samples_per_file


#     for epoch in range(cfg.n_epochs):
#         # Create a global permutation of indices across all files
#         global_indices = np.random.permutation(total_samples)
#         steps_per_epoch = -(len(global_indices) // -cfg.batch_size)

#         for i in tqdm.tqdm(range(steps_per_epoch), desc=f"Epoch #{epoch}"):
#             start = i * cfg.batch_size
#             end = min(len(global_indices), (i + 1) * cfg.batch_size)
#             batch_indices = global_indices[start:end]

#             # Group indices by file
#             file_to_indices = {file_idx: [] for file_idx in range(num_files)}
#             for global_index in batch_indices:
#                 file_index = global_index // total_samples_per_file
#                 local_index = global_index % total_samples_per_file
#                 file_to_indices[file_index].append(local_index)

#             # Initialize batch samples as a dictionary like in load_dataset
#             samples = {key: [] for key in dataset_keys}

#             # Load lazily from each file and aggregate into the batch
#             for file_index, local_indices in file_to_indices.items():
#                 if not local_indices:  # Skip files without indices
#                     continue

#                 # Sort indices for HDF5 access
#                 sorted_indices = np.sort(local_indices)

#                 with h5py.File(dataset_paths[file_index], "r") as file:
#                     for key in dataset_keys:
#                         # Fetch data in sorted order
#                         sorted_data = file[key][sorted_indices]

#                         # Map back to the original order using argsort
#                         original_order = np.argsort(np.argsort(local_indices))
#                         reordered_data = sorted_data[original_order]

#                         # Append to samples
#                         samples[key].append(reordered_data)

#             # Convert lists into NumPy arrays (similar to original load_dataset output)
#             samples = {key: np.concatenate(value, axis=0) for key, value in samples.items()}

#             # Convert samples to tensors for training (if needed by the models)
#             samples = {key: torch.as_tensor(value, device="cuda") for key, value in samples.items()}

#             # Train step for each model
#             for model_name, model in models.items():
#                 log = model.train_step(samples, epoch, train_step)
#                 wandb_logs[model_name].update(log)

#             # Log to WandB if enabled
#             if cfg.wandb:
#                 log_to_wandb(cfg, models, wandb_logs, samples, train_step)

#             train_step += 1

#     # Save models
#     time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#     logdir = os.path.join(cfg.logdir, time_str)
#     os.makedirs(logdir)
#     for model_name, model in models.items():
#         model.save(logdir + f"/{model_name}.pt")

if __name__=="__main__":
    main()
