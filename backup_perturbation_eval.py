import random
import datetime

import h5py
import numpy as np
import torch
import wandb
import hydra
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

# your encoders
from representations.acro import Acro
from representations.single_step import SingleStep
from representations.multi_step import MultiStep
from representations.info_nce import NCE


MODEL_DICT = {
    "acro":        Acro,
    "nce":         NCE,
    "single_step": SingleStep,
    "multi_step":  MultiStep,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# from environments/nav2d/utils.py
def render(obs):
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    obs = (obs + 1) / 2
    _, h, w = obs.shape
    img = np.zeros((3, h, w))
    img[0][obs[0] != 0] = 1
    img[1][obs[1] != 0] = 1
    if obs.shape[0] > 2:
        img[2][obs[2] != 0] = 1
    return img


def load_dataset(path):
    """Load every dataset in the HDF5 file into memory."""
    with h5py.File(path, "r") as f:
        keys = []
        f.visit(lambda k: keys.append(k) if isinstance(f[k], h5py.Dataset) else None)
        return {k: f[k][:] for k in keys}


def compute_distances(obs: np.ndarray, encoder: torch.nn.Module) -> np.ndarray:
    c, h, w = obs.shape
    inp = torch.as_tensor(obs, device=DEVICE).unsqueeze(0)  # [1,C,H,W]
    with torch.no_grad():
        z = encoder(inp).squeeze(0).cpu().numpy()            # [D]

    # build a big batch of perturbed obs: [h*w, C, H, W]
    obs_pert = np.broadcast_to(obs, (h*w, c, h, w)).copy()
    mask     = (-np.eye(h*w) * 2 + 1).reshape(h*w, h, w)
    obs_pert[:, 0] *= mask  # only perturb channel 0

    with torch.no_grad():
        z_pert = encoder(torch.as_tensor(obs_pert, device=DEVICE)).cpu().numpy()  # [h*w, D]

    dists = np.linalg.norm(z - z_pert, ord=1, axis=-1).reshape(h, w)

    player_pos = np.argwhere(obs[1] == 1)[0]
    dists[player_pos[0], player_pos[1]] = 0
    if obs.shape[0] > 2:
        goal = np.argwhere(obs[2] == 1)
        if len(goal) > 0:
            g0, g1 = goal[0]
            dists[g0, g1] = 0

    distances = dists / np.max(dists)
    img = render(obs)

    return img, distances


@hydra.main(version_base=None, config_path="configs/evals", config_name="distance")
def main(cfg: DictConfig):
    # ── seeds ───────────────────────────────────────────
    seed = cfg.evals.distance.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── load + sample one subset of obs ───────────────────
    ds_path = to_absolute_path(cfg.evals.distance.datasets[0])
    data    = load_dataset(ds_path)
    all_obs = data["obs"]               # [N, C, H, W]
    N       = len(all_obs)
    S       = cfg.evals.distance.num_samples
    if N < S:
        raise RuntimeError(f"Need ≥{S} obs, found {N}")

    idxs       = np.random.choice(N, S, replace=False)
    obs_sub_cpu = torch.from_numpy(all_obs[idxs]).float()  # keep on CPU

    # ── load all encoders ────────────────────────────────
    encoders = {}
    for name in cfg.evals.distance.reprs:
        ckpt = torch.load(to_absolute_path(cfg.evals.distance.ckpts[name]), map_location="cpu")
        enc = ckpt["encoder"].cuda().eval()  # unorthodox load
        encoders[name] = enc

    # ── setup split threshold & repeats ───────────────────
    threshold = cfg.evals.distance.perturb_threshold
    repeats = cfg.evals.distance.perturb_repeats

    # ── precompute manhattan distances for the grid ────────
    _, C, H, W = obs_sub_cpu.shape
    ys, xs = np.ogrid[:H, :W]
    cy, cx = H//2, W//2
    dist_map = (np.abs(ys - cy) + np.abs(xs - cx))  # [H, W]

    # ── init W&B ──────────────────────────────────────────
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"perturb_box_{seed}_{now}"
    wandb.init(
        project=cfg.evals.distance.wandb.project,
        entity=cfg.evals.distance.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name
    )

    # define this once, above the loops
    def remove_outliers_iqr(x, k=1.5):
        q1, q3 = np.percentile(x, [25, 75])
        iqr    = q3 - q1
        lower, upper = q1 - k*iqr, q3 + k*iqr
        return x[(x >= lower) & (x <= upper)]

    q = random.randrange(S)
    rand_obs = obs_sub_cpu[q].numpy()  # [C, H, W]

    for name, encoder in encoders.items():
        obs = rand_obs.copy()
        obs[1, :, :] = -1
        obs[1, obs.shape[1] // 2, obs.shape[2] // 2] = 1

        # compute your rendered image + distance map
        img, d2d = compute_distances(obs, encoder)  # img:[3,H,W], d2d:[H,W]

        # NOTE: L1 Distance
        # near = d2d[dist_map <= threshold].ravel()
        # far  = d2d[dist_map > threshold].ravel()

        # NOTE: SQUARE
        mask_sq = (np.abs(ys - cy) <= threshold) & (np.abs(xs - cx) <= threshold)
        near    = d2d[mask_sq].ravel()
        far     = d2d[~mask_sq].ravel()

        # remove outliers via IQR
        near_clean = remove_outliers_iqr(near)
        far_clean  = remove_outliers_iqr(far)

        # ─── box & whisker ────────────────────────────────────
        fig, ax = plt.subplots()
        ax.boxplot(
            [near_clean, far_clean],
            tick_labels=["near", "far"],
            showfliers=False
        )

        # overlay the inlier points as a scatter
        x_near = np.random.normal(1, 0.04, size=len(near_clean))
        x_far  = np.random.normal(2, 0.04, size=len(far_clean))
        ax.scatter(x_near, near_clean, alpha=0.3)
        ax.scatter(x_far,  far_clean,  alpha=0.3)

        ax.set_title(f"{name}")
        ax.set_ylabel("Sensitivity of representation")
        plt.tight_layout()

        # ─── log raw rendered obs ─────────────────────────────
        # convert img [3,H,W] → [H,W,3] for wandb
        img_hwc = img.transpose(1, 2, 0)
        raw_img = wandb.Image(
            img_hwc,
            caption=f"{name}"
        )

        # ─── send to W&B ──────────────────────────────────────
        wandb.log({
            f"{name}/boxplot": wandb.Image(fig),
            f"{name}/raw_obs": raw_img
        })
        plt.close(fig)

    wandb.finish()


if __name__ == "__main__":
    main()
