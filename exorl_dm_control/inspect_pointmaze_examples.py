#!/usr/bin/env python3

import os
import argparse
import numpy as np
import h5py
import wandb
import torch

# ---------- Minimal wrappers (lazy, no preload) ----------
class H5StackedObsWrapper:
    def __init__(self, ds: h5py.Dataset, stack_indices: np.ndarray):
        self.ds = ds
        self.stack_indices = stack_indices

    def __len__(self): return len(self.stack_indices)

    def __getitem__(self, idx):
        frames = self.ds[self.stack_indices[idx]]  # (K,H,W,3) uint8
        return np.concatenate(list(frames), axis=2)  # (H,W,3K) uint8


class H5SliceWrapper:
    def __init__(self, ds: h5py.Dataset, valid_idx: np.ndarray):
        self.ds = ds
        self.valid = valid_idx

    def __len__(self): return len(self.valid)

    def __getitem__(self, idx): return self.ds[self.valid[idx]]


def load_pointmaze_dataset_lazy(
    dataset_path,
    obs_buffer_size=3,
    max_transitions=None,
):
    f = h5py.File(dataset_path, "r")
    imgs = f["images"]           # (T,H,W,3) uint8
    acts = f["action"]           # (T,1) int32
    physics = f["physics"]       # (T,4) float64
    ep_lens = f["episode_lengths"][:]  # (M,)

    starts = np.empty_like(ep_lens, dtype=np.int64)
    starts[0] = 0
    if len(ep_lens) > 1:
        starts[1:] = np.cumsum(ep_lens[:-1])

    K = int(obs_buffer_size)
    valid_t = []
    for start, length in zip(starts, ep_lens):
        for t_in_ep in range(K - 1, length - 1):
            valid_t.append(start + t_in_ep)
    valid_t = np.array(valid_t, dtype=np.int64)
    if max_transitions is not None:
        valid_t = valid_t[:max_transitions]

    offsets = np.arange(K)[::-1]                 # e.g., [2,1,0]
    obs_idx = valid_t[:, None] - offsets[None]   # (N,K)
    next_center = valid_t + 1
    obsn_idx = next_center[:, None] - offsets[None]

    wrappers = {
        "obs": H5StackedObsWrapper(imgs, obs_idx),
        "obs_next": H5StackedObsWrapper(imgs, obsn_idx),
        "action": H5SliceWrapper(acts, valid_t),
        "physics": H5SliceWrapper(physics, valid_t),
    }
    h, w, c = imgs.shape[1:]
    stacked_obs_shape = (h, w, c * K)
    act_shape = 9
    return wrappers, stacked_obs_shape, act_shape, os.path.basename(dataset_path)


# ---------- Stats + visualization helpers ----------
def compute_action_hist_h5(action_wrapper, num_actions, chunk=200_000):
    valid = action_wrapper.valid
    ds = action_wrapper.ds
    counts = np.zeros(num_actions, dtype=np.int64)
    n = len(valid)
    for i in range(0, n, chunk):
        idx = valid[i:i+chunk]
        arr = ds[idx].reshape(-1)
        counts += np.bincount(arr, minlength=num_actions)
    return counts


def _unstack_frames(stacked_hwck):
    H, W, CK = stacked_hwck.shape
    assert CK % 3 == 0, f"Expected RGB multiples, got {CK}"
    K = CK // 3
    return [stacked_hwck[..., 3*i:3*(i+1)] for i in range(K)]


def _hstack(frames, gap=2, bg=255):
    H, W, _ = frames[0].shape
    canvas = np.full((H, W*len(frames) + gap*(len(frames)-1), 3), bg, dtype=np.uint8)
    x = 0
    for i, f in enumerate(frames):
        canvas[:, x:x+W] = f
        x += W
        if i != len(frames)-1:
            canvas[:, x:x+gap] = bg
            x += gap
    return canvas


def _vstack(rows, gap=2, bg=255):
    H, W, _ = rows[0].shape
    canvas = np.full((H*len(rows) + gap*(len(rows)-1), W, 3), bg, dtype=np.uint8)
    y = 0
    for i, r in enumerate(rows):
        canvas[y:y+H] = r
        y += H
        if i != len(rows)-1:
            canvas[y:y+gap] = bg
            y += gap
    return canvas


def build_panel(stacked_obs, stacked_next):
    top = _hstack(_unstack_frames(stacked_obs), gap=2, bg=255)
    bot = _hstack(_unstack_frames(stacked_next), gap=2, bg=255)
    return _vstack([top, bot], gap=3, bg=255)  # (2H+gaps, K*W+gaps, 3)


def log_gt_distribution(ds_name, counts, step):
    fracs = counts / max(counts.sum(), 1)
    rows = [[str(i), int(counts[i]), float(fracs[i])] for i in range(len(counts))]
    table = wandb.Table(data=rows, columns=["action", "count", "fraction"])

    wandb.log({f"dataset/{ds_name}/gt_action_dist_table": table}, step=step, commit=False)
    bar_counts = wandb.plot.bar(table, "action", "count", title=f"{ds_name}: GT action counts")
    bar_fracs = wandb.plot.bar(table, "action", "fraction", title=f"{ds_name}: GT action fractions")
    wandb.log({
        f"dataset/{ds_name}/gt_action_counts_bar": bar_counts,
        f"dataset/{ds_name}/gt_action_fractions_bar": bar_fracs,
    }, step=step)


# ---------- Model loading & prediction ----------
def load_models_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    encoder = ckpt["encoder"].to(device).eval()
    inverse = ckpt["inverse_model"].to(device).eval()
    return encoder, inverse

@torch.no_grad()
def predict_action(encoder, inverse_model, stacked_obs, stacked_next, device):
    # (H,W,3K) uint8 -> (1,H,W,3K) float in [-1,1]
    obs = torch.from_numpy(stacked_obs).unsqueeze(0).to(device).float() / 127.5 - 1.0
    obsn = torch.from_numpy(stacked_next).unsqueeze(0).to(device).float() / 127.5 - 1.0
    z = encoder(obs)
    zn = encoder(obsn)
    logits = inverse_model(z, zn)
    probs = torch.softmax(logits, dim=1).squeeze(0)   # (A,)
    conf, pred = probs.max(dim=0)
    top3_conf, top3_idx = probs.topk(k=min(3, probs.numel()))
    top3 = [(int(top3_idx[i]), float(top3_conf[i])) for i in range(top3_idx.numel())]
    return int(pred), float(conf), top3


def log_action_examples(ds_name, wrappers, act_shape, n_per_action, seed, step,
                        encoder=None, inverse_model=None, device="cpu"):
    rng = np.random.default_rng(seed)
    labels = wrappers["action"].ds[wrappers["action"].valid].reshape(-1)  # (N,)

    cols = ["action", "index", "panel"]
    if encoder is not None and inverse_model is not None:
        cols += ["gt_action", "pred_action", "confidence", "top3"]
    table = wandb.Table(columns=cols)

    montage_logs = {}

    for a in range(act_shape):
        where = np.where(labels == a)[0]
        if where.size == 0:
            montage_logs[f"dataset/{ds_name}/action_{a:02d}_examples"] = wandb.Image(
                np.zeros((32, 32, 3), dtype=np.uint8), caption=f"action={a} (no samples)"
            )
            continue

        pick = rng.choice(where.size, size=min(n_per_action, where.size), replace=False)
        idxs = where[pick]

        panels = []
        for i in idxs:
            stacked_obs  = wrappers["obs"][i]
            stacked_next = wrappers["obs_next"][i]
            panel = build_panel(stacked_obs, stacked_next)
            panels.append(panel)

            if encoder is None:
                table.add_data(int(a), int(i), wandb.Image(panel, caption=f"a={a}, idx={i}"))
            else:
                pred, conf, top3 = predict_action(encoder, inverse_model, stacked_obs, stacked_next, device)
                table.add_data(
                    int(a), int(i), wandb.Image(panel, caption=f"a={a}, idx={i}"),
                    int(a), int(pred), float(conf), str(top3)
                )

        cap = f"action={a} ({len(panels)} examples)"
        if encoder is not None:
            # add GT/PRED of the last sample to the caption for quick glance
            cap = f"{cap} | e.g., GT={a}, Pred={pred} ({conf:.2f})"
        montage = _vstack(panels, gap=6, bg=255)
        montage_logs[f"dataset/{ds_name}/action_{a:02d}_examples"] = wandb.Image(montage, caption=cap)

    wandb.log({f"dataset/{ds_name}/action_examples_table": table, **montage_logs}, step=step)


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Inspect PointMaze dataset: label dist + per-action examples (+optional preds)")
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--obs-buffer-size", type=int, default=4)
    ap.add_argument("--max-transitions", type=int, default=1_500_000)
    ap.add_argument("--n-per-action", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder-checkpoint", type=str, default=None,
                    help="Path to saved single_step .pt to show model predictions")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-project", type=str, default="nav2d")
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    wrappers, obs_shape, act_shape, ds_name = load_pointmaze_dataset_lazy(
        args.dataset,
        obs_buffer_size=args.obs_buffer_size,
        max_transitions=args.max_transitions
    )

    encoder = inverse = None
    if args.encoder_checkpoint:
        encoder, inverse = load_models_from_checkpoint(args.encoder_checkpoint, args.device)

        idx = np.random.choice(len(wrappers["obs"]), size=512, replace=False)
        X = []
        for i in idx:
            x  = torch.from_numpy(wrappers["obs"][i]).unsqueeze(0).float()/127.5-1.0
            X.append(x)
        X = torch.cat(X,0).to(args.device)
        with torch.no_grad():
            Z = encoder(X)  # (B, D)
        print("enc mean/std across batch:", Z.mean().item(), Z.std().item())

        with torch.no_grad():
            avg = torch.zeros(9, device=args.device)
            for i in idx:
                o  = torch.from_numpy(wrappers["obs"][i]).unsqueeze(0).float()/127.5-1.0
                on = torch.from_numpy(wrappers["obs_next"][i]).unsqueeze(0).float()/127.5-1.0
                z, zn = encoder(o), encoder(on)
                avg += inverse(z, zn).squeeze(0)
            avg /= len(idx)
        print("avg logits:", avg.cpu().numpy(), "argmax:", int(avg.argmax()))

    if not args.no_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name or f"inspect_{ds_name}",
            config={
                "dataset": args.dataset,
                "obs_buffer_size": args.obs_buffer_size,
                "max_transitions": args.max_transitions,
                "n_per_action": args.n_per_action,
                "encoder_checkpoint": args.encoder_checkpoint,
            },
        )

    counts = compute_action_hist_h5(wrappers["action"], act_shape)
    print(f"[GT actions] {ds_name} counts: {counts.tolist()}")
    print(f"[GT actions] {ds_name} fracs : {(counts / max(counts.sum(), 1)).tolist()}")

    if not args.no_wandb:
        log_gt_distribution(ds_name, counts, step=0)
        log_action_examples(
            ds_name, wrappers, act_shape, n_per_action=args.n_per_action, seed=args.seed, step=0,
            encoder=encoder, inverse_model=inverse, device=args.device
        )
        print("Logged distribution + examples to W&B.")
    else:
        print("W&B disabled; nothing logged.")


if __name__ == "__main__":
    main()
