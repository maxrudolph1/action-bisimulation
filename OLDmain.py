import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import deque
import time
import h5py
import tqdm
from matplotlib import cm, pyplot as plt
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


def create_models(cfg: DictConfig, obs_shape, act_shape, ss_path=None):
    algo_cfgs = cfg.algos
    model_names = list(algo_cfgs.keys())
    models = {}
    # breakpoint()

    for model_name in model_names:
        model_cfg = algo_cfgs[model_name]
        model = MODEL_DICT[model_name](obs_shape=obs_shape, act_shape=act_shape, encoder_cfg=cfg.encoder, forward_cfg=cfg.forward, inverse_cfg=cfg.inverse, **model_cfg,)
        
        
        if (model_name == 'single_step'):
            # if ss already exists (bc training multistep), just load it in via path
            if (ss_path != None):
                model.load_state_dict(torch.load(ss_path))
                model.eval()
            
    
        #save model regardless
        models[model_name] = model
            
    return models


def initialize_dependant_models(models):
    for model_name, model in models.items():
        model.share_dependant_models(models)
    return models


def save_heatmap_locally(cfg, obs, encoder, file_name):
    img, heatmap = perturb_heatmap(obs, encoder)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(heatmap[0], cmap='gray')
    ax.axis('off')
    
    os.makedirs(cfg.logdir, exist_ok=True)
    file_path = os.path.join(cfg.logdir, file_name)
    plt.savefig(file_path)
    plt.close()
    


def log_to_wandb(cfg, models, logs, samples, train_step):
    if train_step % cfg.met_log_freq == 0:
        labeled_logs = {f"{algo_name}/{key}": value for algo_name,
                        algo_log in logs.items() for key, value in algo_log.items()}
        wandb.log(labeled_logs, step=train_step)
    if train_step % cfg.img_log_freq == 0:
        for model_name, model in models.items():
            obs = samples["obs"][0]
            img = wandb.Image(np.swapaxes(
                perturb_heatmap(obs, model.encoder)[1], 0, 2))
            wandb.log({f"{model_name}/heatmap": img}, step=train_step)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # breakpoint()
    print(f"entering main\n ")
    log_path = cfg.logdir + ("_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    # breakpoint()
    
    if cfg.wandb:
        # breakpoint()
        wandb.init(entity='rhearai', project="nav2d", config={})
    # breakpoint()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset, obs_shape, act_shape = load_dataset(cfg.dataset)
    # breakpoint()
    print(f"about to check single step\n ")
    if cfg.train_single_step==True:
        print("enterting single step, creating models...")
        # if true, train single step model as normal
        models = create_models(cfg, obs_shape, act_shape)
        models = initialize_dependant_models(models)
        print("about to train single step....") 
        train(cfg, dataset, models, model_to_train='single_step')
        torch.save(models['single_step'].state_dict(), os.path.join(cfg.trained_model_dir, "single_step.pt"))

        if cfg.wandb==False:
            # breakpoint()
            save_heatmap_locally(cfg, dataset["obs"][np.random.randint(len(dataset["obs"]))], models["single_step"].encoder, "single_step_heatmap.png")
    else:
        print("entering multistep")
        # if false, train multi step and don't save the new ss
        models = create_models(cfg, obs_shape, act_shape, cfg.single_step_path)
        models = initialize_dependant_models(models)

        # ss_model = models["single_step"]
    
        train(cfg, dataset, models, model_to_train='multi_step')
        # torch.save(models['single_step'].state_dict(), cfg.trained_model_dir)
    # else:
    #     # if false, train multi step and don't save the new ss

    

# model_to_train should indciiate which model names: single_step or multi_step
def train(cfg: DictConfig, dataset, models, model_to_train=None):
    print(f"enter training ")

    # breakpoint()
    dataset_keys = list(dataset.keys())
    wandb_logs = {key: {} for key in models.keys()}
    train_step = 0

    print(f"starting to iterate over this many epochs: {cfg.n_epochs}")
    for epoch in range(cfg.n_epochs):
        # breakpoint()
        sample_ind_all = np.random.permutation(len(dataset["obs"]))
        sample_ind_next = np.random.permutation(len(dataset["obs"]))
        steps_per_epoch = -(len(sample_ind_all) // -cfg.batch_size)
        print(f"Steps per epoch: {steps_per_epoch}")

        for i in tqdm.tqdm(range(steps_per_epoch), desc=f"Epoch #{epoch}"):
            start = i * cfg.batch_size
            end = min(len(sample_ind_all), (i + 1) * cfg.batch_size)
            sample_ind = np.sort(sample_ind_all[start:end])
            samples = {key: dataset[key][sample_ind] for key in dataset_keys}
            for model_name, model in models.items():
                # check
                # breakpoint()
                if(model_to_train!=None and model_name == model_to_train):
                    # breakpoint()
                    log = model.train_step(samples, epoch)
                    wandb_logs[model_name].update(log)

            if cfg.wandb==True:
                # breakpoint()
                log_to_wandb(cfg, models, wandb_logs, samples, train_step)

            train_step += 1

        print(f"Epoch {epoch}, Loss: {log['loss']}")


 

if __name__ == "__main__":
    main()
