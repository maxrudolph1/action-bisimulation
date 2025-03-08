# Unused imports were commented out
# import json
import os
# import shutil
# import sys
# from argparse import ArgumentParser
# from collections import deque
import h5py
import tqdm
# from matplotlib import cm
import torch
import numpy as np
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra

# import torch.nn.functional as F
import random
# from environments.nav2d.utils import perturb_heatmap
import datetime
from representations.acro import Acro
from representations.single_step import SingleStep
from representations.multi_step import MultiStep
from representations.bvae import BetaVariationalAutoencoder
from representations.evaluators import Evaluators

from call_rl_main import call_rl


MODEL_DICT = {'single_step': SingleStep,
              'multi_step': MultiStep,
              'bvae': BetaVariationalAutoencoder,
              'evaluators': Evaluators,
              'acro': Acro}


def load_dataset(dataset_path):
    with h5py.File(dataset_path, "r") as dataset:
        dataset_keys = []
        dataset.visit(lambda key: dataset_keys.append(key)
                      if isinstance(dataset[key], h5py.Dataset)
                      else None)

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
    evaluators = {}

    for model_name in model_names:
        model = MODEL_DICT[model_name](
            obs_shape=obs_shape,
            act_shape=act_shape,
            cfg=cfg
        )
        models[model_name] = model
        evaluators[model_name] = Evaluators(
            obs_shape=obs_shape,
            act_shape=act_shape,
            cfg=cfg.evaluators,
            model=model
        )
    return models, evaluators


def initialize_dependant_models(models):
    for model_name, model in models.items():
        model.share_dependant_models(models)
    return models


def log_to_wandb(cfg, evaluators, logs, samples, train_step):
    if train_step % cfg.met_log_freq == 0:
        labeled_logs = {
            f"{algo_name}/{key}": value
            for algo_name, algo_log in logs.items()
            for key, value in algo_log.items()
        }
        wandb.log(labeled_logs, step=train_step)
    if train_step % cfg.img_log_freq == 0:
        for model_name, evaluator in evaluators.items():
            imgs = evaluator.eval_imgs(samples)
            wandb_imgs_log = {
                f"{model_name}/{key}": img
                for key, img in imgs.items()
            }
            wandb.log(wandb_imgs_log, step=train_step)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    cur_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wandb_name = None
    if cfg.wandb:
        # name = f"{cfg.name}_gamma_{cfg.algos.multi_step.gamma}_{cur_date_time}"
        # name = f"{cfg.name}_{cur_date_time}"
        # name = f"acro_sweeps_k{cfg.algos.acro.k_steps}_l1_{cfg.algos.acro.l1_penalty}_grd_30_obstcls_100_smpls_1250000_{cur_date_time}"
        name = f"{cfg.name}_gamma_{cfg.algos.multi_step.gamma}_grd_30_obstcls_100_smpls_1250000_{cur_date_time}"
        wandb.init(
            entity=cfg.wandb_entity,
            project="nav2d",
            group="ms_acro_grd_30_obstcls_100",
            name=name,
            config=OmegaConf.to_container(cfg)
        )

        wandb_name = wandb.run.name
        print("NOW RUNNING:", wandb_name)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    models, evaluators = None, None

    train_step = 0
    first_dataset = True
    for dataset_file in cfg.datasets:
        dataset, obs_shape, act_shape = load_dataset(dataset_file)
        print(f"FINISHED LOADING {dataset_file}")

        if first_dataset:
            models, evaluators = create_models(cfg, obs_shape, act_shape)
            print("Evaluators", evaluators.keys())
            models = initialize_dependant_models(models)
            first_dataset = False

        train_step, save_paths, log_name = train(
            cfg,
            dataset,
            models,
            evaluators,
            train_step,
            wandb_name,
            cur_date_time
        )
        dataset = None

    if (len(cfg.eval_encoder) > 0) and (cfg.eval_encoder in save_paths):
        wandb.finish()
        call_rl(
            name=("dqn_" + log_name),
            grid_size=30,
            num_obstacles=100,
            latent_encoder_path=save_paths[cfg.eval_encoder],
        )



def train(
    cfg: DictConfig,
    dataset,
    models,
    evaluators,
    train_step,
    wandb_name,
    cur_date_time
):
    dataset_keys = list(dataset.keys())
    wandb_logs = {key: {} for key in models.keys()}

    for epoch in range(cfg.n_epochs):
        sample_ind_all = np.random.permutation(len(dataset["obs"]))
        # sample_ind_next = np.random.permutation(len(dataset["obs"]))
        steps_per_epoch = -(len(sample_ind_all) // -cfg.batch_size)

        for i in tqdm.tqdm(range(steps_per_epoch), desc=f"Epoch #{epoch}"):
            start = i * cfg.batch_size
            end = min(len(sample_ind_all), (i + 1) * cfg.batch_size)
            sample_ind = np.sort(sample_ind_all[start:end])
            samples = {key: dataset[key][sample_ind] for key in dataset_keys}

            # train the representation models
            for model_name, model in models.items():
                log = model.train_step(samples, epoch, train_step)
                wandb_logs[model_name].update(log)

            # train the evaluator models if needed
            if cfg.train_evaluators:
                for model_name, evaluator in evaluators.items():
                    log = evaluator.train_step(samples, epoch, train_step)
                    wandb_logs[model_name].update(log)

            if cfg.wandb:
                log_to_wandb(cfg, evaluators, wandb_logs, samples, train_step)

            train_step += 1

    # time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # logdir = os.path.join(cfg.logdir, time_str)
    #
    # os.makedirs(logdir)
    # for model_name, model in models.items():
    #     model.save(logdir + f"/{model_name}.pt")

    log_name = ((wandb_name + "_") if wandb_name is not None else cur_date_time) + ("ts_" + str(train_step))
    logdir = os.path.join(cfg.logdir, log_name)
    os.makedirs(logdir)

    save_paths = {}
    for model_name, model in models.items():
        path = logdir + f"/{model_name}.pt"
        model.save(path)
        save_paths[model_name] = path
        print(f"Saved {model_name} to {path}")

    return train_step, save_paths, log_name


if __name__ == "__main__":
    main()
