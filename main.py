import os
import h5py
import tqdm
import torch
import numpy as np
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra

from torch.utils.data import Dataset, DataLoader

import random
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
        self.obs = wrappers["obs"]
        self.obs_next = wrappers["obs_next"]
        self.act = wrappers["action"]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.obs_next[idx],
            self.act[idx].squeeze(),
        )  # squeeze to get shape (,) instead of (1,)


class H5StackedObsWrapper:
    """Given a base image dataset and precomputed per-sample K indices,
       returns stacked frames [t-K+1 ... t] concatenated along channel."""

    def __init__(self, ds: h5py.Dataset, stack_indices: np.ndarray):
        # stack_indices: shape (N, K) of integer indices into ds
        self.ds = ds
        self.stack_indices = stack_indices  # dtype should be integer

    def __len__(self):
        return len(self.stack_indices)

    def __getitem__(self, idx):
        frames = self.ds[self.stack_indices[idx]]  # (K, H, W, C)
        return np.concatenate(list(frames), axis=2)  # (H, W, C * K)


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


def load_pointmaze_dataset(
    dataset_path,
    obs_buffer_size=3,
    max_transitions=None,
):
    """
    Lazily open the PointMaze HDF5 and return three H5SliceWrapper objects
    for obs, obs_next, action so that data[k] only pulls those frames.
    """

    print(f"[PointMaze] opening {dataset_path!r}", flush=True)
    f = h5py.File(dataset_path, 'r')
    imgs = f['images'] # shape (T, H, W, 3)
    acts = f['action'] # shape (T, 1)
    ep_lens = f['episode_lengths'][:]

    T = imgs.shape[0]
    print(f"[PointMaze] dataset has {T} frames, using obs buffer size of {obs_buffer_size}", flush=True)

    starts = np.empty_like(ep_lens, dtype=np.int64)
    starts[0] = 0
    if len(ep_lens) > 1:
        starts[1:] = np.cumsum(ep_lens[:-1])

    valid_t = []
    for start, length in zip(starts, ep_lens):
        for t_in_ep in range(obs_buffer_size - 1, length - 1):
            global_t = start + t_in_ep
            valid_t.append(global_t)
    valid_t = np.array(valid_t, dtype=np.int64)

    if max_transitions is not None:
        valid_t = valid_t[:max_transitions]
        print(f"[PointMaze] truncating to first {len(valid_t)} transitions", flush=True)
    else:
        print(f"[PointMaze] keeping {len(valid_t)}/{T-1} transitions", flush=True)

    # Build stacked obs indices: for each valid_t = v, we need [v - (K-1), ..., v]
    K = obs_buffer_size
    if K < 1:
        raise ValueError("obs_buffer_size must be >=1")
    offsets = np.arange(K)[::-1]  # e.g. K=3 -> [2,1,0]; v - offsets = [v-2, v-1, v]
    obs_stack_indices = valid_t[:, None] - offsets[None, :]  # shape (N, K)

    # stacked next observations (slid forward by one)
    next_center = valid_t + 1  # t+1
    obs_next_stack_indices = next_center[:, None] - offsets[None, :]  # shape (N, K)

    obs = H5StackedObsWrapper(imgs, obs_stack_indices)
    obs_next = H5StackedObsWrapper(imgs, obs_next_stack_indices)
    action = H5SliceWrapper(acts, valid_t)

    print("[PointMaze] built wrappers")

    # compute obs_shape after stacking: original imgs shape is (T, H, W, C)
    h, w, c = imgs.shape[1:]
    stacked_obs_shape = (h, w, c * obs_buffer_size)
    print("Obs shape (stacked):", stacked_obs_shape)

    DISCRETE = True
    if DISCRETE:
        act_shape = 9 # FIXME: Make sure this is the correct formatting
    else:
        act_shape = acts.shape[1]
    print("Action shape:", act_shape)

    return {"obs": obs, "obs_next": obs_next, "action": action}, stacked_obs_shape, act_shape


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
            if (cfg.env=='nav2d'):
                imgs = evaluator.eval_imgs(samples)
            elif (cfg.env=='pointmaze'):
                continue
            else:
                continue
            wandb_imgs_log = {
                f"{model_name}/{key}": img
                for key, img in imgs.items()
            }
            wandb.log(wandb_imgs_log, step=train_step)


def run_downstream_rl(cfg: DictConfig, save_paths, log_name, n_seeds=2):
    # DOWNSTREAM RL
    if (len(cfg.eval_encoder) > 0) and (cfg.eval_encoder in save_paths):
        wandb.finish()
        grid = 15
        num_obs = 20
        total_timesteps = 600000  # default is 1 mil
        seeds = list(range(n_seeds))

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
        # LOAD DATSET AND DATALOADER
        print(f"LOADING {dataset_file}...")
        wrappers, obs_shape, act_shape = load_pointmaze_dataset(
            dataset_file,
            obs_buffer_size=cfg.obs_buffer_size,
            max_transitions=1_500_000
        )

        dataset = PointMazeDataset(wrappers)

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        print(f"FINISHED LOADING {dataset_file}")

        if first_dataset:
            # CREATE MODELS AND EVALUATORS
            models, evaluators = create_models(cfg, obs_shape, act_shape)
            print("Evaluators", evaluators.keys())
            models = initialize_dependant_models(models)
            first_dataset = False

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

    run_downstream_rl(cfg, save_paths, log_name)


def train(
    cfg: DictConfig,
    loader: DataLoader,
    models,
    evaluators,
    train_step,
    wandb_name,
    cur_date_time
):
    wandb_logs = {key: {} for key in models.keys()}

    for epoch in range(cfg.n_epochs):
        for batch in tqdm.tqdm(loader, desc=f"Epoch #{epoch}"):
            obs_np, obs_next_np, act_np = batch

            # Transfer to GPU (non_blocking because pin_memory=True)
            obs = obs_np.cuda(non_blocking=True)
            obs_next = obs_next_np.cuda(non_blocking=True)
            action = act_np.cuda(non_blocking=True).long()

            samples = {"obs": obs, "obs_next": obs_next, "action": action}

            for name, model in models.items():
                logs = model.train_step(samples, epoch, train_step)
                wandb_logs[name].update(logs)

            # train the evaluators if needed
            if cfg.train_evaluators:
                for model_name, evaluator in evaluators.items():
                    log = evaluator.train_step(samples, epoch, train_step)
                    wandb_logs[model_name].update(log)

            if cfg.wandb:
                log_to_wandb(cfg, evaluators, wandb_logs, samples, train_step)

            train_step += 1

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
