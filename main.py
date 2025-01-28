import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import deque
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
# from representations.info_nce import KNCEStep
from representations.single_step import SingleStep
from representations.multi_step import MultiStep
from representations.bvae import BetaVariationalAutoencoder
import os

import pdb


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
    # print("===== Finished loading dataset", dataset_path)
    return dataset, obs_shape, act_shape


def create_models(cfg: DictConfig, obs_shape, act_shape):
    algo_cfgs = cfg.algos
    model_names = list(algo_cfgs.keys())
    models = {}

    for model_name in model_names:
        # model_cfg = algo_cfgs[model_name]
        model = MODEL_DICT[model_name](obs_shape=obs_shape, act_shape=act_shape, cfg=cfg)
        models[model_name] = model
    return models


def initialize_dependant_models(models):
    for model_name, model in models.items():
        model.share_dependant_models(models)
    return models


# NOTE: THERE IS A DIFFERENCE ON THIS FUNCTION
def log_to_wandb(cfg, models, logs, samples, train_step):
    if train_step % cfg.met_log_freq == 0:
        labeled_logs = {f"{algo_name}/{key}": value for algo_name, algo_log in logs.items() for key, value in algo_log.items()}
        wandb.log(labeled_logs, step=train_step)
    if train_step % cfg.img_log_freq == 0:
        for model_name, model in models.items():
            obs = samples["obs"][0]
            obs[1, :, :] = -1
            obs[1, obs.shape[1] // 2, obs.shape[2] // 2] = 1
            img = wandb.Image(np.swapaxes(perturb_heatmap(obs, model.encoder)[1], 0, 2))
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


def printMergedObs(obs):
    # ANSI color codes
    COLORS = {
        'O': "\033[32m",   # Green for O (1 from obstacles)
        'X': "\033[31m",   # Red for X (1 from agent)
        '-': "\033[33m",   # Yellow for - (-1 from either grid)
        '0': "\033[37m",   # White for 0
    }
    RESET = "\033[0m"

    # Function to merge grids and format for printing
    def merge_and_format_grids(grid1, grid2):
        merged_grid = np.full(grid1.shape, '0', dtype=object)

        # Merging logic
        for i in range(grid1.shape[0]):
            for j in range(grid1.shape[1]):
                if grid1[i, j] == 1:  # grid1's 'O' takes precedence
                    merged_grid[i, j] = 'O'
                elif grid2[i, j] == 1:  # grid2's 'X' comes next
                    merged_grid[i, j] = 'X'
                elif grid1[i, j] == -1 or grid2[i, j] == -1:  # Any -1 becomes '-'
                    merged_grid[i, j] = '-'
                else:  # Otherwise, it's 0
                    merged_grid[i, j] = '0'

        # Create a formatted grid with colors
        formatted_grid = "\n".join([
            " ".join(f"{COLORS[cell]}{cell}{RESET}" for cell in row)
            for row in merged_grid
        ])
        return formatted_grid

    print(merge_and_format_grids(obs[0], obs[1]))


def validateObs(dataset):
    print("dataset size = ", len(dataset["obs"]))
    idx = 0
    samples = {key: dataset[key][idx] for key in dataset.keys()}

    print("IDX", idx)
    print("kvalid", samples["kvalid"])

    print("-"*15, "obs", "-"*15)
    printMergedObs(samples["obs"])

    print(">"*5, "action:", samples["action"])
    print(">"*5, "kaction buffer:", samples["kaction"])

    print("-"*15, "obs_next", "-"*15)
    printMergedObs(samples["obs_next"])

    print("-"*15, "kobs_0", "-"*15)
    printMergedObs(samples["kobs"][0])

    print("-"*15, "kobs_1", "-"*15)
    printMergedObs(samples["kobs"][1])

    print("-"*15, "kobs_2", "-"*15)
    printMergedObs(samples["kobs"][2])

    # print("-"*15, "kobs_3", "-"*15)
    # printMergedObs(samples["kobs"][3])

    # print("-"*15, "kobs_4", "-"*15)
    # printMergedObs(samples["kobs"][4])

    # print("-"*15, "kobs_5", "-"*15)
    # printMergedObs(samples["kobs"][5])

    # print("obs", samples["obs"].shape)
    # print("obs_next", samples["obs_next"].shape)
    print("kobs", samples["kobs"].shape)
    print("kaction", samples["kaction"])
    print("kvalid", samples["kvalid"])


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    cur_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wandb_name = None
    if cfg.wandb:
        # name = f"acro_k_{cfg.algos.acro.k_steps}_epochs_{cfg.n_epochs}_{cur_date_time}"
        name = f"acro_sweeps_k{cfg.algos.acro.k_steps}_l1_{cfg.algos.acro.l1_penalty}_{cur_date_time}"
        # name = f"gamma_sweeps_bisim_gamma{cfg.algos.multi_step.gamma}"

        # wandb.init(entity='evan-kuo-edu', project="nav2d", config=OmegaConf.to_container(cfg),)
        wandb.init(entity='evan-kuo-edu', project="nav2d", name=name, config=OmegaConf.to_container(cfg),)

        wandb_name = wandb.run.name
        print("NOW RUNNING:", wandb_name)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # A bit hard coded at the moment. Future fix
    obs_shape = (3, 15, 15)
    act_shape = 4

    models = create_models(cfg, obs_shape, act_shape)
    models = initialize_dependant_models(models)

    train_step = 0
    for dataset_file in cfg.datasets:
        # DEBUG STUFF
        # validateObs(load_dataset(dataset_file)[0])
        # exit()

        # CORRECT STUFF
        dataset, obs_shape, act_shape = load_dataset(dataset_file)
        print(f"FINISHED LOADING {dataset_file}")

        train_step = train(cfg, dataset, models, train_step, wandb_name, cur_date_time)
        dataset = None


def train(cfg: DictConfig, dataset, models, train_step, wandb_name, cur_date_time):
    dataset_keys = list(dataset.keys())
    wandb_logs = {key: {} for key in models.keys()}

    for epoch in range(cfg.n_epochs):
        sample_ind_all = np.random.permutation(len(dataset["obs"]))
        sample_ind_next = np.random.permutation(len(dataset["obs"]))
        steps_per_epoch = -(len(sample_ind_all) // -cfg.batch_size)
        for i in tqdm.tqdm(range(steps_per_epoch), desc=f"Epoch #{epoch}"):
            start = i * cfg.batch_size
            end = min(len(sample_ind_all), (i + 1) * cfg.batch_size)
            sample_ind = np.sort(sample_ind_all[start:end])
            samples = {key: dataset[key][sample_ind] for key in dataset_keys}
            for model_name, model in models.items():
                log = model.train_step(samples, epoch, train_step)
                wandb_logs[model_name].update(log)

            if cfg.wandb:
                log_to_wandb(cfg, models, wandb_logs, samples, train_step)

            train_step += 1

    log_name = ((wandb_name + "_") if wandb_name is not None else cur_date_time) + ("ts_" + str(train_step))
    logdir = os.path.join(cfg.logdir, log_name)
    os.makedirs(logdir)
    for model_name, model in models.items():
        model.save(logdir + f"/{model_name}.pt")

    return train_step


if __name__ == "__main__":
    main()
