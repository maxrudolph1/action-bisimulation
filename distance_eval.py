import random
import datetime
import h5py
import numpy as np
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

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


def load_dataset(path):
    with h5py.File(path, "r") as f:
        keys = []
        f.visit(lambda k: keys.append(k) if isinstance(f[k], h5py.Dataset) else None)
        data = {k: f[k][:] for k in keys}
    return data


def encode_in_batches(encoder, obs_cpu: torch.Tensor, batch_size: int):
    """
    encoder: your .cuda().eval() module
    obs_cpu: all your obs_sub on CPU, shape [S, C, H, W]
    returns: numpy array [S, D] on CPU
    """
    zs = []
    S = obs_cpu.size(0)
    for i in range(0, S, batch_size):
        batch = obs_cpu[i : i + batch_size].to(DEVICE)
        with torch.no_grad():
            z_batch = encoder(batch)        # [B, D] on GPU
        zs.append(z_batch.cpu().numpy())   # move to CPU immediately
        del batch, z_batch
        torch.cuda.empty_cache()
    return np.vstack(zs)


@hydra.main(version_base=None, config_path="configs/evals", config_name="distance")
def main(cfg: DictConfig):
    random.seed(cfg.evals.distance.seed)
    np.random.seed(cfg.evals.distance.seed)
    torch.manual_seed(cfg.evals.distance.seed)

    data = load_dataset(cfg.evals.distance.datasets[0])
    all_obs = data["obs"]                   # (N, C, H, W)
    N = len(all_obs)
    print(f"There are {N} available obs")
    if N < cfg.evals.distance.num_samples:
        raise RuntimeError(f"Need ≥{cfg.evals.distance.num_samples} obs, found {N}")

    idxs    = np.random.choice(N, cfg.evals.distance.num_samples, replace=False)
    # obs_sub = torch.from_numpy(all_obs[idxs]).float().to(DEVICE)  # [S,C,H,W]
    obs_sub = torch.from_numpy(all_obs[idxs]).float()  # keep on CPU

    # instantiate & load each model, encode all S obs → Z[name]: [S, D]
    Z = {}
    for name in cfg.evals.distance.reprs:
        print(f"Encoding with {name}...\n\n")
        ckpt = torch.load(cfg.evals.distance.ckpts[name], map_location="cpu")
        if "encoder" not in ckpt:
            raise KeyError(f"Checkpoint for {name} has no 'encoder' key")
        encoder = ckpt["encoder"].cuda().eval()    # unorthodox load

        Z[name] = encode_in_batches(
            encoder,
            obs_sub,
            batch_size=10_000,
        )

        # with torch.no_grad():
        #     z = encoder(obs_sub)
        # Z[name] = z.cpu().numpy()

    name = ""
    cur_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if (len(cfg.evals.distance.wandb.name) <= 0):
        name = f"distance_analysis_{cfg.evals.distance.seed}_{cur_date_time}"
    else:
        name = f"{cfg.evals.distance.wandb.name}_{cur_date_time}"

    wandb.init(
        entity=cfg.evals.distance.wandb.entity,
        project="nav2d",
        config=OmegaConf.to_container(cfg),
        name=name
    )

    # for each representation: pick a random query, L1 dists, top-K, log table + images
    for name, z in Z.items():
        S, D = z.shape
        q = random.randrange(S)           # query index in [0..S)
        z0 = z[q : q + 1]                 # [1, D]
        dists = np.abs(z - z0).sum(axis=1)  # [S,]

        # sort and take top K (excluding the query itself)
        order     = np.argsort(dists)
        neighbors = [i for i in order if i != q][: cfg.evals.distance.num_neighbors]

        # build a W&B table of (repr, query_idx, neighbor_idx, distance)
        table = wandb.Table(columns=["repr", "query_idx", "neighbor_idx", "distance"])
        for nbr in neighbors:
            table.add_data(name, int(q), int(nbr), float(dists[nbr]))

        # log the table + a histogram of distances
        wandb.log({
            f"{name}/top{cfg.evals.distance.num_neighbors}_table": table,
            f"{name}/distance_hist": wandb.Histogram(dists)
        })

        def captioned_image(t: torch.Tensor, caption: str):
            img = t.cpu().numpy()
            img = np.swapaxes(img, 0, 2)      # C,H,W → H,W,C
            return wandb.Image(img, caption=caption)

        # caption the query
        orig_q = idxs[q]  # original dataset index of this sub-sample
        query_cap = f"sub_idx={q}, orig_idx={orig_q}"
        query_img = captioned_image(obs_sub[q], query_cap)

        # caption each neighbor with its sub-sample idx, original idx, and L1 distance
        neighbor_imgs = []
        for n in neighbors[: cfg.evals.distance.raw_obs_count]:
            orig_n = idxs[n]
            dist_n = dists[n]
            cap = f"sub_idx={n}, orig_idx={orig_n}, L1={dist_n:.3f}"
            neighbor_imgs.append(captioned_image(obs_sub[n], cap))

        wandb.log({
            f"{name}/query_raw":     [query_img],
            f"{name}/neighbors_raw": neighbor_imgs,
        })
        print(f"Logged {name}.\n\n")

    wandb.finish()


if __name__ == "__main__":
    main()
