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

from torch.utils.data import Dataset, DataLoader

# import torch.nn.functional as F
import random
# from environments.nav2d.utils import perturb_heatmap
import datetime
from representations.acro import Acro
from representations.single_step import SingleStep
from representations.multi_step import MultiStep
from representations.bvae import BetaVariationalAutoencoder
from representations.evaluators import Evaluators
from representations.info_nce import NCE

from call_rl_main import call_rl


MODEL_DICT = {'single_step': SingleStep,
              'multi_step': MultiStep,
              'bvae': BetaVariationalAutoencoder,
              'evaluators': Evaluators,
              'acro': Acro,
              'nce': NCE}


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


class PointMazeDataset(Dataset):
    def __init__(self, wrappers):
        # wrappers is the dict {"obs": obs_wrap, "obs_next": next_wrap, "action": act_wrap}
        self.obs = wrappers["obs"]
        self.obs_next = wrappers["obs_next"]
        self.act = wrappers["action"]
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        # h5 wrappers return numpy -> torch will auto‑convert if collate_fn is default
        return ( self.obs[idx], 
                 self.obs_next[idx], 
                 self.act[idx].squeeze()  # to get shape (,) instead of (1,)
               )


class H5SliceWrapper:
    """Wrap a h5py Dataset + a valid‐index array so that
       wrapper[idx] → ds[valid_indices[idx]] without preloading."""
    def __init__(self, ds: h5py.Dataset, valid_idx: np.ndarray):
        print("Creating wrapper...")
        self.ds = ds
        self.valid = valid_idx
        print("Finished wrapper")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        # allow integer or array indexing
        return self.ds[self.valid[idx]]


def load_pointmaze_dataset(dataset_path, boundary=1000, max_transitions=None):
    """
    Lazily open the PointMaze HDF5 and return three H5SliceWrapper objects
    for obs, obs_next, action so that data[k] only pulls those frames.
    """
    print(f"[PointMaze] opening {dataset_path!r}", flush=True)
    f = h5py.File(dataset_path, 'r')
    imgs = f['images'] # shape (T, H, W, 3)
    acts = f['action'] # shape (T, 1)
    # ep_lens = f['episode_lengths'][:]

    T = imgs.shape[0]
    print(f"[PointMaze] dataset has {T} frames, boundary={boundary}", flush=True)

    all_steps = np.arange(T - 1, dtype=np.int64)
    invalid = (np.arange(boundary - 1, T, boundary, dtype=np.int64))
    valid   = np.setdiff1d(all_steps, invalid, assume_unique=True)

    if max_transitions is not None:
        valid = valid[:max_transitions]
        print(f"[PointMaze] truncating to first {len(valid)} transitions", flush=True)
    else:
        print(f"[PointMaze] keeping {len(valid)}/{T-1} transitions", flush=True)

    # build three wrappers
    obs = H5SliceWrapper(imgs, valid)
    obs_next = H5SliceWrapper(imgs, valid + 1)
    action = H5SliceWrapper(acts, valid)
    print("[PointMaze] built wrappers")

    obs_shape = imgs.shape[1:] # (H, W, 3)
    print("Obs shape:", obs_shape)
    DISCRETE = True
    if DISCRETE:
        act_shape = 9 # FIXME: Make sure this is the correct formatting
    else:
        act_shape = acts.shape[1]
    print("Action shape:", act_shape)

    return {"obs": obs, "obs_next": obs_next, "action": action}, obs_shape, act_shape


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
        continue # HACK: Fix this....
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
        name = f"{cfg.name}_{cur_date_time}"
        # name = f"{cfg.name}_grd_15_obstcls_20_smpls_1250000_{cur_date_time}"
        # name = f"acro_sweeps_k{cfg.algos.acro.k_steps}_l1_{cfg.algos.acro.l1_penalty}_grd_15_obstcls_20_smpls_1250000_{cur_date_time}"
        # name = f"{cfg.name}_gamma_{cfg.algos.multi_step.gamma}_grd_15_obstcls_20_smpls_1250000_{cur_date_time}"
        wandb.init(
            entity=cfg.wandb_entity,
            project="nav2d",
            # group="ms_acro_grd_30_obstcls_100",
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
        print(f"LOADING {dataset_file}...")
        # dataset, obs_shape, act_shape = load_dataset(dataset_file)
        wrappers, obs_shape, act_shape = load_pointmaze_dataset(dataset_file, max_transitions=1500000)
        dataset = PointMazeDataset(wrappers)
        print(f"FINISHED LOADING {dataset_file}")

        if first_dataset:
            models, evaluators = create_models(cfg, obs_shape, act_shape)
            print("Evaluators", evaluators.keys())
            models = initialize_dependant_models(models)
            first_dataset = False

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=8,       # adjust to your machine
            pin_memory=True,
        )

        train_step, save_paths, log_name = train(
            cfg,
            loader,
            models,
            evaluators,
            train_step,
            wandb_name,
            cur_date_time
        )
        dataset = None

    if (len(cfg.eval_encoder) > 0) and (cfg.eval_encoder in save_paths):
        wandb.finish()
        # grid = 30
        # num_obs = 100
        grid = 15
        num_obs = 20
        total_timesteps = 600000  # default is 1 mil
        # seeds = list(range(2))
        seeds = []

        if (cfg.eval_encoder == "single_step") or (cfg.eval_encoder == "acro"):
            penalty = cfg.algos.acro.l1_penalty if (cfg.eval_encoder == "acro") else cfg.algos.single_step.l1_penalty

            # base case with l1_penalty
            for seed in seeds:
                call_rl(
                    name=("dqn_" + log_name),
                    grid_size=grid,
                    num_obstacles=num_obs,
                    seed=seed,
                    latent_encoder_path=save_paths[cfg.eval_encoder],
                    l1_penalty=penalty,
                    total_timesteps=total_timesteps,
                )
        elif (cfg.eval_encoder == "multi_step"):
            # multi-step with gamma
            for seed in seeds:
                call_rl(
                    name=("dqn_" + log_name),
                    grid_size=grid,
                    num_obstacles=num_obs,
                    seed=seed,
                    latent_encoder_path=save_paths[cfg.eval_encoder],
                    gamma=cfg.algos.multi_step.gamma,
                    total_timesteps=total_timesteps,
                )
        else:
            # other?
            for seed in seeds:
                call_rl(
                    name=("dqn_" + log_name),
                    grid_size=grid,
                    num_obstacles=num_obs,
                    seed=seed,
                    latent_encoder_path=save_paths[cfg.eval_encoder],
                    total_timesteps=total_timesteps,
                )


def train(
    cfg: DictConfig,
    loader: DataLoader,
    models,
    evaluators,
    train_step,
    wandb_name,
    cur_date_time
):
    # dataset_keys = list(dataset.keys())
    wandb_logs = {key: {} for key in models.keys()}

    for epoch in range(cfg.n_epochs):
        for batch in tqdm.tqdm(loader, desc=f"Epoch #{epoch}"):
            # batch is a tuple: (obs_np, obs_next_np, act_np)
            obs_np, obs_next_np, act_np = batch

            # 2) Transfer to GPU (non_blocking because pin_memory=True)
            obs      = obs_np.cuda(non_blocking=True)
            obs_next = obs_next_np.cuda(non_blocking=True)
            action   = act_np.cuda(non_blocking=True).long()

            samples = {
                "obs":      obs,
                "obs_next": obs_next,
                "action":   action,
            }

            # 3) Run your step
            for name, model in models.items():
                logs = model.train_step(samples, epoch, train_step)
                wandb_logs[name].update(logs)

            if cfg.wandb:
                log_to_wandb(cfg, evaluators, wandb_logs, samples, train_step)

            train_step += 1

    # for epoch in range(cfg.n_epochs):
    #     sample_ind_all = np.random.permutation(len(dataset["obs"]))
    #     # sample_ind_next = np.random.permutation(len(dataset["obs"]))
    #     steps_per_epoch = -(len(sample_ind_all) // -cfg.batch_size)
    #
    #     for i in tqdm.tqdm(range(steps_per_epoch), desc=f"Epoch #{epoch}"):
    #         start = i * cfg.batch_size
    #         end = min(len(sample_ind_all), (i + 1) * cfg.batch_size)
    #         sample_ind = np.sort(sample_ind_all[start:end])
    #         samples = {key: dataset[key][sample_ind] for key in dataset_keys}
    #
    #         # train the representation models
    #         for model_name, model in models.items():
    #             log = model.train_step(samples, epoch, train_step)
    #             wandb_logs[model_name].update(log)
    #
    #         # train the evaluator models if needed
    #         if cfg.train_evaluators:
    #             for model_name, evaluator in evaluators.items():
    #                 log = evaluator.train_step(samples, epoch, train_step)
    #                 wandb_logs[model_name].update(log)
    #
    #         if cfg.wandb:
    #             log_to_wandb(cfg, evaluators, wandb_logs, samples, train_step)
    #
    #         train_step += 1

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
