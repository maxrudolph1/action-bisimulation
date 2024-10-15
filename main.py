# TEMP
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
import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra

import torch.nn.functional as F
import random
from environments.nav2d.utils import perturb_heatmap
import datetime
from representations.single_step import SingleStep
from representations.multi_step import MultiStep


from omegaconf import DictConfig, OmegaConf
import hydra

MODEL_DICT = {'single_step': SingleStep, 'multi_step': MultiStep}

def load_dataset(dataset_path):
    dataset = h5py.File(dataset_path, "r")
    dataset_keys = []
    dataset.visit(lambda key: dataset_keys.append(key) if isinstance(dataset[key], h5py.Dataset) else None)

    mem_dataset = {}
    for key in dataset_keys:
        mem_dataset[key] = dataset[key][:]
    dataset = mem_dataset

    obs_shape = dataset["obs"][0].shape
    act_shape = dataset["action"].max() + 1
    return dataset, obs_shape, act_shape

def create_models(cfg: DictConfig, obs_shape, act_shape, single_step_path=None):
    algo_cfgs = cfg.algos
    model_names = list(algo_cfgs.keys())
    models = {}

    for model_name in model_names:
        model_cfg = algo_cfgs[model_name]
        model = MODEL_DICT[model_name](obs_shape=obs_shape, act_shape=act_shape, encoder_cfg=cfg.encoder, forward_cfg=cfg.forward, inverse_cfg=cfg.inverse, **model_cfg,)

        if model_name == 'single_step' and single_step_path:
            print(f"Loading pretrained single_step model from {single_step_path}")
            model.load_state_dict(torch.load(single_step_path))
            model.eval()
        models[model_name] = model
    return models

def initialize_dependant_models(models):
    for model_name, model in models.items():
        model.share_dependant_models(models)
    return models

def save_heatmap(cfg: DictConfig, model_encoder, obs, save_file: str):
    img, heatmap = perturb_heatmap(obs, model_encoder)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(heatmap[0], cmap='gray')
    ax.axis('off')
    os.makedirs("maps", exist_ok=True)
    fig_path = os.path.join("maps", save_file)
    plt.savefig(fig_path)
    plt.close()

    # if (cfg.wandb):
    #     wandb.log({f"perturbation_maps/{save_file}": wandb.Image(fig_path)})

def log_to_wandb(cfg, models, logs, samples, train_step):
    if train_step % cfg.met_log_freq == 0:
        labeled_logs = {f"{algo_name}/{key}": value for algo_name, algo_log in logs.items() for key, value in algo_log.items()}
        wandb.log(labeled_logs, step=train_step)

    if train_step % cfg.img_log_freq == 0:
        for model_name, model in models.items():
            obs = samples["obs"][0]
            img = wandb.Image(np.swapaxes(perturb_heatmap(obs, model.encoder)[1], 0,2))
            wandb.log({f"{model_name}/heatmap": img}, step=train_step)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log_path = cfg.logdir + ("_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    if cfg.wandb:
        wandb.init(entity='rhearai-university-of-texas-at-austin', project="nav2d", config={})

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset, obs_shape, act_shape = load_dataset(cfg.dataset)
    models = create_models(cfg, obs_shape, act_shape, single_step_path='/nfs/homes/bisim/rrai/action-bisimulation/trained_models/single_step.pt')

    models = initialize_dependant_models(models)

    # train(cfg, dataset, models) # should only train singlestep
    train(cfg, dataset, models, train_multi_step=True)

    # torch.save(models['single_step'].state_dict(), os.path.join("trained_models", "single_step.pt"))
    # torch.save(models['single_step'].state_dict(), os.path.join(cfg.trained_model_dir, "single_step.pt"))
    torch.save(models['multi_step'].state_dict(), os.path.join("trained_models", "multi_step.pt"))

    obs = dataset["obs"][np.random.randint(len(dataset["obs"]))]

    """ Plot the heatmap's first layer single_step """
    
    # save_heatmap(cfg, models["single_step"].encoder, obs, "single_step.png")
    save_heatmap(cfg, models["multi_step"].encoder, obs, "multi_step.png")

def train(cfg: DictConfig, dataset, models, train_multi_step=False):
    dataset_keys = list(dataset.keys())
    wandb_logs = {key: {} for key in models.keys()}
    train_step = 0

    forward_model_next_loss_list = []

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
                if (train_multi_step and model_name == "single_step"):
                    continue
                if (not train_multi_step and model_name == "multi_step"):
                    continue

                log = model.train_step(samples, epoch)
                wandb_logs[model_name].update(log)
                
                if 'fw_model_step_loss' in log:
                    forward_model_next_loss_list.append(log['fw_model_step_loss']) 

            if cfg.wandb:
                log_to_wandb(cfg, models, wandb_logs, samples, train_step)

            
            train_step += 1



if __name__=="__main__":
    main()