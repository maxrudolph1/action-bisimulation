import os
import h5py
import tqdm
import torch
import numpy as np
import wandb
import random
import datetime

from torch.utils.data import Dataset, DataLoader

from omegaconf import DictConfig, OmegaConf
import hydra

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


class PointMazeDataset(Dataset):
    def __init__(self, wrappers):
        self.obs = wrappers["obs"]
        self.obs_next = wrappers["obs_next"]
        self.action = wrappers["action"]
        self.physics = wrappers["physics"]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return (
            self.obs[i],
            self.obs_next[i],
            self.action[i].squeeze(),
            self.physics[i],
        )


class H5StackedObsWrapper:
    """Given base image dataset and precomputed per-sample K indices,
       returns stacked frames [t-K+1..t] concatenated on channel (H,W, C*K)."""
    def __init__(self, ds: h5py.Dataset, stack_indices: np.ndarray):
        self.ds = ds
        self.stack_indices = stack_indices  # (N,K) int64

    def __len__(self):
        return len(self.stack_indices)

    def __getitem__(self, idx):
        frames = self.ds[self.stack_indices[idx]]  # (K,H,W,3), HDF5 fancy read
        return np.concatenate(list(frames), axis=2)  # (H,W,3K), uint8


class H5SliceWrapper:
    """Wrap an HDF5 dataset with a valid-index array. wrapper[i] -> ds[valid[i]]."""
    def __init__(self, ds: h5py.Dataset, valid_idx: np.ndarray):
        self.ds = ds
        self.valid = valid_idx

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        return self.ds[self.valid[idx]]


def _filter_valid_t_for_K(valid_t: np.ndarray, starts: np.ndarray, lens: np.ndarray, K: int, T: int) -> np.ndarray:
    """Ensure valid_t respects both t+1 in-bounds and K-stack within episode."""
    # Build a map from time->episode (fast + one pass)
    ep_of_t = np.full(T, -1, dtype=np.int32)
    e = 0
    for s, L in zip(starts, lens):
        ep_of_t[s:s+L] = e
        e += 1
    ep_id = ep_of_t[valid_t]
    ok = (ep_id >= 0)

    # episode-local boundaries for K stack
    s_for = starts[ep_id[ok]]
    L_for = lens[ep_id[ok]]
    t_ok  = valid_t[ok]
    cond = (t_ok >= s_for + (K-1)) & (t_ok <= s_for + L_for - 2)  # t+1 exists & K in-episode
    keep = t_ok[cond]
    # also clip to [0, T-2]
    keep = keep[(keep >= 0) & (keep < T-1)]
    return keep


def load_pointmaze_dataset(
    dataset_path,
    obs_buffer_size=1,        # K; set >1 if you want stacked frames
    max_transitions=None,     # optional head cap
    valid_t_override=None, # e.g., your balanced indices
):
    """
    Lazily open HDF5 and return wrappers for obs/obs_next/action/physics.
    """
    f = h5py.File(dataset_path, 'r')
    imgs = f['images']     # (T,H,W,3) uint8, gzip-chunked
    acts= f['action']     # (T,1) int32
    physics = f['physics']    # (T,4) float64
    ep_len = f['episode_lengths'][:].astype(np.int64)

    starts = np.empty_like(ep_len, dtype=np.int64)
    starts[0] = 0
    if len(ep_len) > 1:
        starts[1:] = np.cumsum(ep_len[:-1])

    T, H, W, C = imgs.shape

    K = int(obs_buffer_size)
    if valid_t_override is None:
        # TODO: CHECK THIS
        starts = np.empty_like(ep_len, dtype=np.int64)
        starts[0] = 0
        if len(ep_len) > 1:
            starts[1:] = np.cumsum(ep_len[:-1])

        out = []
        for start, length in zip(starts, ep_len):
            for t_in_ep in range(obs_buffer_size - 1, length - 1):
                out.append(start + t_in_ep)
            # TODO: FLAGGED
            # if length >= K + 1:
            #     out.append(np.arange(start + (K-1), start + (length-1), dtype=np.int64))
        # valid_t = np.concatenate(out, axis=0)
        valid_t = np.array(out, dtype=np.int64)
    else:
        # TODO: CHECK THIS
        # ensure the provided valid_t still respects K stacking and t+1 in-bounds
        v = np.asarray(valid_t_override, dtype=np.int64)
        valid_t = _filter_valid_t_for_K(v, starts, ep_len, K, T)

    if max_transitions is not None and valid_t_override is None:
        valid_t = valid_t[:max_transitions]

    # Build stacked indices for obs and obs_next just like before
    offsets = np.arange(K)[::-1]  # e.g., K=3 -> [2,1,0]
    obs_stack_idx      = valid_t[:, None] - offsets[None, :]
    next_center        = valid_t + 1
    obs_next_stack_idx = next_center[:, None] - offsets[None, :]

    wrappers = {
        "obs":      H5StackedObsWrapper(imgs, obs_stack_idx),
        "obs_next": H5StackedObsWrapper(imgs, obs_next_stack_idx),
        "action":   H5SliceWrapper(acts, valid_t),
        "physics":  H5SliceWrapper(physics, valid_t),
    }

    # TODO: FLAGGED
    stacked_obs_shape = (C * K, H, W)
    # stacked_obs_shape = (H, W, C * K)
    print("Obs shape (stacked):", stacked_obs_shape)

    # discrete actions: 9
    act_shape = 9

    return wrappers, stacked_obs_shape, act_shape


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
    for _, model in models.items():
        model.share_dependant_models(models)
    return models


def log_to_wandb(cfg, evaluators, logs, batch, train_step):
    if train_step % cfg.met_log_freq == 0:
        labeled_logs = {
            f"{algo_name}/{key}": value
            for algo_name, algo_log in logs.items()
            for key, value in algo_log.items()
        }
        wandb.log(labeled_logs, step=train_step)

    if train_step % cfg.img_log_freq == 0:
        for model_name, evaluator in evaluators.items():
            if cfg.env == "nav2d":
                imgs = evaluator.eval_imgs(batch)
            elif cfg.env == "pointmaze":
                # imgs = evaluator.eval_plan2vec_figure5(batch)
                pass

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
        name = f"{cfg.name}_{cur_date_time}"
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
        print(f"LOADING {dataset_file} ...")
        balanced_idx_path = getattr(cfg, "balanced_idx_path", None),
        if (balanced_idx_path is not None and len(balanced_idx_path) <= 1):
            balanced_idx_path = None

        valid_t_override = None
        if balanced_idx_path and os.path.exists(balanced_idx_path):
            valid_t_override = np.load(balanced_idx_path)

        wrappers, obs_shape, act_shape = load_pointmaze_dataset(
            dataset_file,
            obs_buffer_size=getattr(cfg, "obs_buffer_size", 1),   # K (set to 1 if you don't want stacking)            max_transitions=max_transitions,
            max_transitions=getattr(cfg, "max_transitions", None),
            valid_t_override=valid_t_override,
        )
        dataset = PointMazeDataset(wrappers)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )


        if first_dataset:
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

    if (len(cfg.eval_encoder) > 0) and (cfg.eval_encoder in save_paths):
        wandb.finish()
        grid = 15
        num_obs = 20
        total_timesteps = 600000  # default is 1 mil
        seeds = list(range(2))

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
    cfg,
    loader,
    models,
    evaluators,
    train_step,
    wandb_name,
    cur_date_time,
):
    wandb_logs = {key: {} for key in models.keys()}

    for epoch in range(cfg.n_epochs):
        for batch in tqdm.tqdm(loader, desc=f"Epoch #{epoch}"):
            if cfg.env == 'pointmaze':
                obs, obs_next, action, physics = batch
            else:
                obs, obs_next, action = batch

            # GPU (non_blocking since pin_memory=True)
            obs = obs.cuda(non_blocking=True)
            obs_next = obs_next.cuda(non_blocking=True)
            action = action.cuda(non_blocking=True).long()

            # TODO: FLAGGED
            obs = obs.permute(0, 3, 1, 2).contiguous()
            obs_next = obs_next.permute(0, 3, 1, 2).contiguous()

            # keep normalization in main (works with current SingleStep)
            if cfg.env == "pointmaze":
                obs      = obs.float().mul_(1/127.5).add_(-1.0)
                obs_next = obs_next.float().mul_(1/127.5).add_(-1.0)
                samples = {"obs": obs, "obs_next": obs_next, "action": action, "physics": physics}
            else:
                samples = {"obs": obs, "obs_next": obs_next, "action": action}

            for name, model in models.items():
                logs = model.train_step(samples, epoch, train_step)
                wandb_logs[name].update(logs)

            if cfg.train_evaluators:
                for model_name, evaluator in evaluators.items():
                    log = evaluator.train_step(samples, epoch, train_step)
                    wandb_logs[model_name].update(log)

            if cfg.wandb:
                log_to_wandb(cfg, evaluators, wandb_logs, samples, train_step)

            train_step += 1

    log_name = ((wandb_name + "_") if wandb_name is not None else cur_date_time) + ("ts_" + str(train_step))
    logdir = os.path.join(cfg.logdir, log_name)
    os.makedirs(logdir, exist_ok=True)

    save_paths = {}
    for model_name, model in models.items():
        path = os.path.join(logdir, f"{model_name}.pt")
        model.save(path)
        save_paths[model_name] = path
        print(f"Saved {model_name} to {path}")

    return train_step, save_paths, log_name


if __name__ == "__main__":
    main()
