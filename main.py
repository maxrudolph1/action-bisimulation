import os
import h5py
import tqdm
import torch
import numpy as np
import wandb
import random
import datetime

from torch.utils.data import Dataset, DataLoader, Sampler

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


class BlockRandomSampler(Sampler):
    def __init__(self, valid_idx, block_size=4096, drop_last=True, seed=0):
        self.N = len(valid_idx)
        positions = np.arange(self.N, dtype=np.int64)  # positions, not values
        blocks = [positions[i:i+block_size] for i in range(0, self.N, block_size)]
        if drop_last and len(blocks) and len(blocks[-1]) < block_size:
            blocks.pop()
        self.blocks = blocks
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = np.arange(len(self.blocks))
        self.rng.shuffle(order)
        for b in order:
            block = self.blocks[b].copy()
            # optional tiny within-block shuffle:
            # self.rng.shuffle(block)
            for pos in block:
                yield int(pos)  # position into dataset

    def __len__(self):
        return sum(len(b) for b in self.blocks)


class H5TransitionsDataset(Dataset):
    """
    Returns dict with:
      obs       : (H,W,3) uint8
      obs_next  : (H,W,3) uint8
      action    : (1,)    int32/int64 (discrete id)
      reward    : (1,)    float32     (optional)
      discount  : (1,)    float32     (optional)
    We exclude the last step of each episode so obs_next = obs[i+1] is in-bounds.
    """
    def __init__(
        self,
        h5_path: str,
        return_reward_discount: bool = True,
        max_transitions: int = None,
        max_transitions_mode: str = "head",  # "head" | "random"
        valid_t_path: str = None,
        seed: int = 0,
    ):
        self.h5_path = h5_path
        self._f = None
        self.return_reward_discount = return_reward_discount
        self._rng = np.random.default_rng(seed)

        with h5py.File(h5_path, 'r') as f:
            H, W, C = f['images'].shape[1:]
            self.obs_shape = (C, H, W)
            T = f['images'].shape[0]

            if valid_t_path is not None:
                valid_t = np.load(valid_t_path).astype(np.int64)
            else:
                # fallback: compute from episode_lengths
                lens = f['episode_lengths'][:].astype(np.int64)
                starts = np.cumsum(np.concatenate([[0], lens[:-1]])).astype(np.int64) if len(lens) > 1 else np.array([0], np.int64)
                valid = []
                for s, L in zip(starts, lens):
                    if L >= 2:
                        valid.append(np.arange(s, s + L - 1, dtype=np.int64))  # [s .. s+L-2]
                valid_t = np.concatenate(valid) if len(valid) else np.zeros((0,), dtype=np.int64)

            # safety: clip to [0, T-2]
            valid_t = valid_t[(valid_t >= 0) & (valid_t < T-1)]
            self.valid_idx = valid_t

            # optional cap
            if (max_transitions is not None) and (max_transitions < self.valid_idx.shape[0]):
                if max_transitions_mode == "random":
                    self.valid_idx = np.sort(self._rng.choice(self.valid_idx, size=max_transitions, replace=False))
                else:
                    self.valid_idx = self.valid_idx[:max_transitions]

            # robust max(action) without loading all
            a_ds = f['action']
            step = max(1, min(1_000_000, a_ds.shape[0]))
            a_max = -1
            for s in range(0, a_ds.shape[0], step):
                a_max = max(a_max, int(np.asarray(a_ds[s:s+step]).max()))
            self.act_shape = a_max + 1

    def _ensure_open(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, 'r',
                rdcc_nbytes=512*1024*1024,   # 512MB cache
                rdcc_nslots=1<<20,           # many hash slots
                rdcc_w0=0.75)
            self.images    = self._f['images']
            self.actions   = self._f['action']
            self.rewards   = self._f['reward']
            self.discounts = self._f['discount']

    def __len__(self):
        return int(self.valid_idx.shape[0])

    def __getitem__(self, i):
        self._ensure_open()
        idx = int(self.valid_idx[i])
        sample = {
            "obs":       np.asarray(self.images[idx]),
            "obs_next":  np.asarray(self.images[idx + 1]),
            "action":    np.asarray(self.actions[idx]),   # (1,)
        }
        if self.return_reward_discount:
            sample["reward"]   = np.asarray(self.rewards[idx])
            sample["discount"] = np.asarray(self.discounts[idx])
        return sample

    def close(self):
        try:
            if self._f is not None:
                self._f.close()
        except Exception:
            pass


def collate_nchw_uint8(batch):
    # batch is a list of dicts
    obs      = torch.from_numpy(np.stack([b["obs"] for b in batch], axis=0))       # (B,H,W,3) uint8
    obs_next = torch.from_numpy(np.stack([b["obs_next"] for b in batch], axis=0))  # (B,H,W,3) uint8
    action   = torch.from_numpy(np.stack([b["action"] for b in batch], axis=0))    # (B,1)

    # NHWC -> NCHW (still uint8); pin for faster H2D
    obs      = obs.permute(0, 3, 1, 2).contiguous()
    obs_next = obs_next.permute(0, 3, 1, 2).contiguous()
    batch_out = {"obs": obs, "obs_next": obs_next, "action": action}

    if "reward" in batch[0]:
        batch_out["reward"] = torch.from_numpy(np.stack([b["reward"] for b in batch], axis=0))     # (B,1)
        batch_out["discount"] = torch.from_numpy(np.stack([b["discount"] for b in batch], axis=0))   # (B,1)
    return batch_out


def make_loader(
    h5_path,
    batch_size,
    num_workers=8,
    shuffle=True,
    return_reward_discount=True,
    max_transitions=None,
    max_transitions_mode="head",
    valid_t_path=None,
    seed=0,
):
    ds = H5TransitionsDataset(
        h5_path,
        return_reward_discount=return_reward_discount,
        max_transitions=max_transitions,
        max_transitions_mode=max_transitions_mode,
        seed=seed,
        valid_t_path=valid_t_path,
    )
    order_for_locality = np.sort(ds.valid_idx)
    sampler = BlockRandomSampler(order_for_locality, block_size=4096, seed=0)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_nchw_uint8
    )
    return ds, loader


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
        if cfg.env == "nav2d":
            for model_name, evaluator in evaluators.items():
                imgs = evaluator.eval_imgs(batch)
                wandb_imgs_log = {
                    f"{model_name}/{key}": img
                    for key, img in imgs.items()
                }
                wandb.log(wandb_imgs_log, step=train_step)
        elif cfg.env == "pointmaze":
            # TODO: include tsne stuff here for evals
            pass


def warm_cache(h5_path, stride=1<<20):
    with h5py.File(h5_path,'r') as f:
        imgs = f['images']
        T = imgs.shape[0]
        # read large slabs sequentially; discard result
        slab = 131072
        for i in range(0, T, slab):
            _ = imgs[i:i+slab]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    cur_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wandb_name = None
    if cfg.wandb:
        name = f"{cfg.name}_{cur_date_time}"
        # name = f"{cfg.name}_gamma_{cfg.algos.multi_step.gamma}_{cur_date_time}"
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
        # warm_cache(dataset_file)
        ds, loader = make_loader(
            dataset_file,
            batch_size=cfg.batch_size,
            num_workers=8,
            return_reward_discount=True,
            max_transitions=getattr(cfg, "max_transitions", None),
            max_transitions_mode=getattr(cfg, "max_transitions_mode", "head"),
            valid_t_path=cfg.balanced_idx_path,
            seed=cfg.seed,

        )
        print(f"START TRAINING ON: {dataset_file}")

        if first_dataset:
            obs_shape, act_shape = ds.obs_shape, ds.act_shape
            models, evaluators = create_models(cfg, obs_shape, act_shape)
            print("Evaluators", evaluators.keys())
            models = initialize_dependant_models(models)
            first_dataset = False

        train_step, save_paths, log_name = train_with_loader(
            cfg, loader, models, evaluators, train_step, wandb_name, cur_date_time
        )

        # tidy
        try: ds.close()
        except Exception: pass
        del ds, loader

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


def train_with_loader(
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

            # # ---- NHWC -> NCHW so Conv2d/Decoder see channels-first
            # if batch["obs"].ndim == 4 and batch["obs"].shape[-1] == 3:
            #     batch["obs"]      = batch["obs"].permute(0, 3, 1, 2).contiguous()
            #     batch["obs_next"] = batch["obs_next"].permute(0, 3, 1, 2).contiguous()

            # ensure CE-compatible labels (B,)
            if "action" in batch:
                batch["action"] = batch["action"].squeeze(-1).long()

            batch["obs"]      = batch["obs"].to("cuda", non_blocking=True)
            batch["obs_next"] = batch["obs_next"].to("cuda", non_blocking=True)
            batch["action"]   = batch["action"].to("cuda", non_blocking=True)

            # normalize for pointmaze
            if getattr(cfg, "env", None) == "pointmaze":
                # batch["obs"]      = (batch["obs"].float()      / 127.5 - 1.0)
                # batch["obs_next"] = (batch["obs_next"].float() / 127.5 - 1.0)
                batch["obs"] = batch["obs"].float().mul_(1/127.5).add_(-1.0)
                batch["obs_next"] = batch["obs_next"].float().mul_(1/127.5).add_(-1.0)

            # train representation models
            for model_name, model in models.items():
                log = model.train_step(batch, epoch, train_step)
                if isinstance(log, dict):
                    wandb_logs[model_name].update(log)

            # optional evaluators
            if cfg.train_evaluators:
                for model_name, evaluator in evaluators.items():
                    log = evaluator.train_step(batch, epoch, train_step)
                    wandb_logs[model_name].update(log)

            if cfg.wandb:
                log_to_wandb(cfg, evaluators, wandb_logs, batch, train_step)

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
