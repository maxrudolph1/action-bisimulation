# from copy import deepcopy
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch.nn
# from matplotlib import cm
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# from models import nets
from models import gen_model_nets
import torch.nn.functional as F
import torch
import wandb
from environments.nav2d.utils import perturb_heatmap
# from . import utils

import pdb

from itertools import product
from tqdm import tqdm


class Evaluators():
    def __init__(
        self, obs_shape, act_shape,
        cfg,
        model,
        **kwargs,
    ):
        super().__init__()

        self.cfg = cfg
        self.evaluators = list(self.cfg.keys())
        self.model = model
        self.obs_shape = obs_shape
        self.act_shape = act_shape

        if self.cfg.reconstruction:
            self.init_reconstruction()

        if self.cfg.action_prediction:
            self.init_action_prediction()

    def init_reconstruction(self):
        self.decoder = gen_model_nets.GenDecoder2D(self.model.embed_dim, self.obs_shape, self.cfg.reconstruction.decoder).cuda()

        self.decoder_optimizer = torch.optim.Adam(
            list(self.decoder.parameters()),
            lr=0.001,
        )

    def init_action_prediction(self):
        self.inverse_model = gen_model_nets.GenInverseDynamics(self.model.embed_dim, self.act_shape, self.cfg.action_prediction.inverse).cuda()
        self.inverse_model_optimizer = torch.optim.Adam(
            list(self.inverse_model.parameters()),
            lr=0.001
        )

    def decoder_train_step(self, batch, epoch, train_step):
        obs_x = torch.as_tensor(batch["obs"], device="cuda")
        z = self.model.encoder(obs_x)
        obs_x_recon = self.decoder(z)
        recon_loss = F.mse_loss(obs_x_recon, obs_x)
        self.decoder_optimizer.zero_grad()
        recon_loss.backward()
        self.decoder_optimizer.step()
        return {"recon_loss": recon_loss.item()}

    def action_prediction_train_step(self, batch, epoch, train_step):
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda")
        o_encoded = self.model.encoder(obs)
        on_encoded = self.model.encoder(obs_next)
        act_pred = self.inverse_model(o_encoded, on_encoded)  # logits

        action_loss = F.cross_entropy(act_pred, act)
        self.inverse_model_optimizer.zero_grad()
        action_loss.backward()
        self.inverse_model_optimizer.step()
        return {"action_loss": action_loss.item()}

    def train_step(self, batch, epoch, train_step):
        if train_step % self.cfg.reset_freq == 0:
            self.init_action_prediction()
            self.init_reconstruction()

        evaluator_losses = {}
        if self.cfg.reconstruction:
            recon_loss = self.decoder_train_step(batch, epoch, train_step)
            evaluator_losses.update(recon_loss)
        if self.cfg.action_prediction:
            action_loss = self.action_prediction_train_step(batch, epoch, train_step)
            evaluator_losses.update(action_loss)

        return evaluator_losses

    def ascii_obs(self, obs):
        """
        Just a debugging function that prints the obstacle layer of the obs
        """
        obs = obs.copy()

        obs[0] = np.where(obs[0] == -1, 0, obs[0])

        print('-' * 80)
        for row in obs[0]:
            print(" ".join(str(int(v)) for v in row))
        print('-' * 80)

    def eval_imgs_single(self, samples,):  # old code. doesn't average, probably don't need anymore
        obs = samples["obs"][0]  # gets a random observation
        obs[1, :, :] = -1
        obs[1, obs.shape[1] // 2, obs.shape[2] // 2] = 1
        heatmap = wandb.Image(np.swapaxes(perturb_heatmap(obs, self.model.encoder)[1], 0, 2))

        obs = torch.tensor(samples["obs"][0])
        obs_recon = self.decoder(self.model.encoder(obs[None].cuda())).squeeze().detach().cpu().numpy()
        disp_obs = np.swapaxes(samples["obs"][0], 0, 2)
        reconstruction = wandb.Image(np.concatenate([np.swapaxes(obs_recon, 0, 2), disp_obs], axis=1))
        return {"reconstruction": reconstruction, "heatmap": heatmap}

    def eval_imgs(self, samples):
        raw_obs = samples["obs"][:10]

        heatmaps = []
        for obs in raw_obs:
            obs = obs.copy()
            obs[1, :, :] = -1
            obs[1, obs.shape[1] // 2, obs.shape[2] // 2] = 1
            heatmap = perturb_heatmap(obs, self.model.encoder)[1]
            if isinstance(heatmap, torch.Tensor):
                heatmap = heatmap.detach().cpu().numpy()
            heatmaps.append(heatmap)

        avg_hm = np.mean(np.stack(heatmaps, axis=0), axis=0)
        avg_hm_img = np.swapaxes(avg_hm, 0, 2)
        heatmap = wandb.Image(avg_hm_img)

        obs = torch.tensor(samples["obs"][0])
        obs_recon = self.decoder(self.model.encoder(obs[None].cuda())).squeeze().detach().cpu().numpy()
        disp_obs = np.swapaxes(samples["obs"][0], 0, 2)
        reconstruction = wandb.Image(np.concatenate([np.swapaxes(obs_recon, 0, 2), disp_obs], axis=1))
        return {"reconstruction": reconstruction, "heatmap": heatmap}

    # def eval_plan2vec_figure5(self, samples):
    #     B = samples["obs"].shape[0]
    #     N = max(1024, B)
    #     idxs = np.random.choice(B, size=N, replace=False)
    #     obs_batch  = samples["obs"][idxs].detach().cpu()
    #     phys_batch = samples["physics"][idxs].detach().cpu().numpy()
    #
    #     obs_norm = (obs_batch.float() / 127.5) - 1.0
    #
    #     with torch.no_grad():
    #         Z = self.model.encoder(obs_norm.cuda()).cpu().numpy()  # (N, D)
    #
    #     print("Min/max/mean of Z:", Z.min(), Z.max(), Z.mean())
    #
    #     Z2 = TSNE(n_components=2, init="pca", perplexity=30, max_iter=1000,
    #               random_state=0).fit_transform(Z)
    #
    #     # Normalize true positions to [0,1]
    #     xy = phys_batch[:, :2].copy()
    #     min_xy = xy.min(axis=0, keepdims=True)
    #     max_xy = xy.max(axis=0, keepdims=True)
    #     xy = (xy - min_xy) / (max_xy - min_xy + 1e-8)
    #
    #     # Build RGB
    #     rgb = np.zeros((N, 3), dtype=np.float32)
    #     rgb[:, 0] = xy[:, 0]
    #     rgb[:, 1] = xy[:, 1]
    #
    #     fig, (ax_tsne, ax_gt) = plt.subplots(1, 2, figsize=(8,4), dpi=100)
    #     ax_tsne.scatter(Z2[:,0], Z2[:,1], c=rgb, s=6, alpha=0.8)
    #     ax_tsne.set_title("TSNE: Encoded Latents")
    #     ax_tsne.set_xticks([]); ax_tsne.set_yticks([])
    #
    #     ax_gt.scatter(xy[:,0], xy[:,1], c=rgb, s=6, alpha=0.8)
    #     ax_gt.set_title("Ground Truth (x,y)")
    #     ax_gt.set_xticks([]); ax_gt.set_yticks([])
    #
    #     plt.tight_layout()
    #     return {"plan2vec_tsne_gt": wandb.Image(fig)}

    # def eval_plan2vec_figure5(self, samples):
    #     """
    #     Plan2Vec-style TSNE visualization over a dense grid of (x,y) positions in
    #     the ExoRL point-mass maze, using *rendered* frames as inputs to the encoder.
    #
    #     - We infer an XY bounding box from the current batch physics and expand it a bit.
    #     - We create a grid of points over that box, render each state, and (if K>1)
    #       tile the single frame K times along channels to match the stacked input.
    #     - We run the encoder, TSNE the latents, and color by true (x,y).
    #     - If the env import/render fails, we gracefully fall back to the *batch-only*
    #       variant (but using the full batch, not a random subset).
    #
    #     You can control grid density with `cfg.evaluators.tsne_grid` (default 33).
    #     """
    #     import numpy as np
    #     import torch
    #     import matplotlib.pyplot as plt
    #     from sklearn.manifold import TSNE
    #     import wandb
    #     import os, sys, importlib
    #
    #     # ---- helper: robust to torch/numpy input for physics ----
    #     def to_numpy(a):
    #         if isinstance(a, np.ndarray):
    #             return a
    #         if hasattr(a, "detach"):
    #             return a.detach().cpu().numpy()
    #         return np.asarray(a)
    #
    #     # Physics from the batch (B, 4) → use to estimate bounds for XY sweep
    #     phys_batch = to_numpy(samples["physics"])
    #     xy_batch = phys_batch[:, :2]
    #     span = np.ptp(xy_batch, axis=0) + 1e-6
    #     lo = xy_batch.min(axis=0) - 0.15 * span
    #     hi = xy_batch.max(axis=0) + 0.15 * span
    #
    #     # How dense a grid to render? (default 33×33 ≈ 1.1k points)
    #     grid = 70
    #     grid_x = np.linspace(lo[0], hi[0], grid)
    #     grid_y = np.linspace(lo[1], hi[1], grid)
    #
    #     # Figure out (H, W, C_total) and stack factor K from your stacked obs
    #     H, W, C_total = self.obs_shape
    #     K = max(1, C_total // 3)  # number of frames stacked along channels
    #
    #     Z = None
    #     xy_used = None
    #     used_rendered = False
    #
    #     try:
    #         load_dmc = True
    #         if ("dm_control" in sys.modules) or ("dmc" in sys.modules):
    #             load_dmc = False
    #             # raise RuntimeError("dm_control already imported; backend is fixed in this process")
    #
    #         # Pick backend before importing any dm_control modules
    #         mujoco_gl = str(getattr(self.cfg, "mujoco_gl", "egl")).lower()
    #         if mujoco_gl not in ("egl", "osmesa"):
    #             mujoco_gl = "egl"
    #         os.environ.setdefault("MUJOCO_GL", mujoco_gl)
    #
    #         # Ensure ExoRL repo is importable, then import dmc
    #         if load_dmc:
    #             exorl_root = os.path.expanduser("~/bisim/exorl")
    #             if exorl_root not in sys.path:
    #                 sys.path.insert(0, exorl_root)
    #             dmc = importlib.import_module("dmc")
    #
    #         task_name = "point_mass_maze_reach_top_left"
    #         env = dmc.make(
    #             task_name,
    #             obs_type='pixels',
    #             frame_stack=1,
    #             action_repeat=1,
    #             seed=0,
    #         )
    #
    #         frames = []
    #         xy_points = []
    #
    #         # Render a dense sweep of XY; zero out velocities
    #         total_pts = len(grid_y) * len(grid_x)
    #         for yy, xx in tqdm(
    #             product(grid_y, grid_x),
    #             total=total_pts,
    #             desc="Rendering env grid",
    #             leave=False,
    #             mininterval=0.1,
    #             dynamic_ncols=True,
    #         ):
    #             with env.physics.reset_context():
    #                 state = env.physics.get_state().copy()
    #                 # Expect (x, y, vx, vy) for this task:
    #                 state[0] = float(xx)
    #                 state[1] = float(yy)
    #                 if state.shape[0] >= 4:
    #                     state[2] = 0.0
    #                     state[3] = 0.0
    #                 env.physics.set_state(state)
    #
    #             img = env.physics.render(width=W, height=H, camera_id=0)  # (H, W, 3), uint8
    #             # Tile to match stacked channels expected by the encoder
    #             if K > 1:
    #                 img = np.concatenate([img] * K, axis=2)  # (H, W, 3*K)
    #             frames.append(img)
    #             xy_points.append((xx, yy))
    #
    #         frames = np.stack(frames, axis=0)         # (G, H, W, 3*K)
    #         xy_points = np.asarray(xy_points, dtype=np.float32)  # (G, 2)
    #
    #         # Normalize exactly like training
    #         obs = torch.as_tensor(frames, device="cuda", dtype=torch.float32)
    #         obs = (obs / 127.5) - 1.0
    #
    #         with torch.no_grad():
    #             Z = self.model.encoder(obs).detach().cpu().numpy()  # (G, D)
    #
    #         xy_used = xy_points
    #         used_rendered = True
    #
    #     except Exception as e:
    #         print("[plan2vec-tsne] Env render unavailable, falling back to batch-only:", repr(e))
    #
    #     # ---- Fallback: use the *entire* batch (no sub-sampling) ----
    #     if Z is None:
    #         obs_batch = samples["obs"]
    #         if hasattr(obs_batch, "detach"):
    #             obs_batch = obs_batch.detach().cpu().float()
    #         else:
    #             obs_batch = torch.as_tensor(obs_batch, dtype=torch.float32)
    #
    #         obs_norm = (obs_batch / 127.5) - 1.0
    #         with torch.no_grad():
    #             Z = self.model.encoder(obs_norm.cuda()).cpu().numpy()
    #         xy_used = xy_batch  # color by true (x,y) from the batch
    #
    #     # ---- Diagnostics ----
    #     print("Min/max/mean of Z:", float(Z.min()), float(Z.max()), float(Z.mean()))
    #
    #     # ---- TSNE (pick a safe perplexity) ----
    #     n_pts = Z.shape[0]
    #     perplexity = max(5, min(30, n_pts // 3))
    #     Z2 = TSNE(
    #         n_components=2,
    #         init="pca",
    #         perplexity=perplexity,
    #         max_iter=1000,
    #         random_state=0
    #     ).fit_transform(Z)
    #
    #     # Normalize XY to [0,1] for nice RGB coloring
    #     xy_min = xy_used.min(axis=0, keepdims=True)
    #     xy_max = xy_used.max(axis=0, keepdims=True)
    #     xy01 = (xy_used - xy_min) / (xy_max - xy_min + 1e-8)
    #
    #     rgb = np.zeros((n_pts, 3), dtype=np.float32)
    #     rgb[:, 0] = xy01[:, 0]
    #     rgb[:, 1] = xy01[:, 1]
    #
    #     # ---- Plot ----
    #     fig, (ax_tsne, ax_gt) = plt.subplots(1, 2, figsize=(9, 4.5), dpi=110)
    #
    #     ax_tsne.scatter(Z2[:, 0], Z2[:, 1], c=rgb, s=6, alpha=0.85)
    #     ax_tsne.set_title(f"t-SNE of encoder latents\n({'env renders' if used_rendered else 'batch frames'})")
    #     ax_tsne.set_xticks([]); ax_tsne.set_yticks([])
    #
    #     ax_gt.scatter(xy01[:, 0], xy01[:, 1], c=rgb, s=6, alpha=0.85)
    #     ax_gt.set_title("Ground-truth (x,y) normalized")
    #     ax_gt.set_xticks([]); ax_gt.set_yticks([])
    #
    #     plt.tight_layout()
    #     img = wandb.Image(fig)
    #     plt.close(fig)
    #     return {"plan2vec_tsne_gt": img}

    def eval_plan2vec_figure5(self, samples):
        """
        t-SNE over encoder latents using env renders on a dense XY grid.
        - Uses environment geoms to get global XY bounds (so walls are visible).
        - Adds tqdm progress over the render grid.
        - Encodes all points but subsamples for t-SNE to keep it fast.

        Config knobs under self.cfg:
          - mujoco_gl: 'egl' (default) or 'osmesa'
          - tsne_task: DMC task (default 'point_mass_maze_reach_top_left')
          - tsne_camera_id: int camera id (default 0)
          - tsne_grid: grid per axis (default 101 -> ~10k renders)
          - tsne_max_points: cap points fed to t-SNE (default 5000)
          - tsne_expand: margin on env bounds (default 0.02)
          - tsne_xlim/tsne_ylim: optional explicit bounds; if set, override env/batch
          - tsne_progress: bool to show/hide tqdm (default True)
        """
        import os, sys, importlib
        import numpy as np, torch, matplotlib.pyplot as plt, wandb
        from sklearn.manifold import TSNE
        from itertools import product
        try:
            from tqdm import tqdm
        except Exception:
            def tqdm(x, **k): return x  # fallback if tqdm not available

        # ---- helper ----
        def to_numpy(a):
            return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)

        # Shapes: stacked obs = (H, W, 3*K)
        H, W, Ctot = self.obs_shape
        K = max(1, Ctot // 3)
        cam_id      = int(getattr(self.cfg, "tsne_camera_id", 0))
        grid        = int(getattr(self.cfg, "tsne_grid", 101))
        max_tsne    = int(getattr(self.cfg, "tsne_max_points", 5000))
        expand      = float(getattr(self.cfg, "tsne_expand", 0.02))
        show_bar    = bool(getattr(self.cfg, "tsne_progress", True))
        task_name   = getattr(self.cfg, "tsne_task", "point_mass_maze_reach_top_left")

        # Optional explicit bounds via config
        xlim = getattr(self.cfg, "tsne_xlim", None)
        ylim = getattr(self.cfg, "tsne_ylim", None)

        Z = None
        xy_used = None
        used_rendered = False
        env = None

        # ---- Try env renders w/ robust import and backend selection ----
        try:
            # Ensure GL backend before any dm_control import
            mujoco_gl = str(getattr(self.cfg, "mujoco_gl", "egl")).lower()
            if mujoco_gl not in ("egl", "osmesa"):
                mujoco_gl = "egl"
            os.environ.setdefault("MUJOCO_GL", mujoco_gl)

            # Ensure ExoRL repo on path, then import dmc (works even if already in sys.modules)
            exorl_root = os.path.expanduser("~/bisim/exorl")
            if exorl_root not in sys.path:
                sys.path.insert(0, exorl_root)
            dmc = sys.modules.get("dmc") or importlib.import_module("dmc")

            env = dmc.make(task_name, obs_type='pixels', frame_stack=1, action_repeat=1, seed=0)

            # ---- Bounds from env geoms (so walls appear) ----
            phys = env.physics
            if xlim is not None and ylim is not None:
                lo = np.array([float(xlim[0]), float(ylim[0])], dtype=np.float32)
                hi = np.array([float(xlim[1]), float(ylim[1])], dtype=np.float32)
            else:
                geom_pos  = phys.model.geom_pos      # (ngeom, 3)
                geom_size = phys.model.geom_size     # (ngeom, 3)
                names = [phys.model.id2name(i, 'geom') for i in range(phys.model.ngeom)]
                keep = [i for i,n in enumerate(names)
                        if n and any(t in n.lower() for t in ("wall", "maze", "boundary", "bound", "floor", "arena"))]
                if not keep:
                    keep = list(range(phys.model.ngeom))  # fallback: all geoms

                xs, ys = [], []
                for i in keep:
                    px, py = float(geom_pos[i, 0]), float(geom_pos[i, 1])
                    sx, sy = float(geom_size[i, 0]), float(geom_size[i, 1])
                    xs.extend([px - sx, px + sx])
                    ys.extend([py - sy, py + sy])

                lo = np.array([min(xs), min(ys)], dtype=np.float32)
                hi = np.array([max(xs), max(ys)], dtype=np.float32)
                span = hi - lo
                lo -= expand * span
                hi += expand * span

            gx = np.linspace(lo[0], hi[0], grid)
            gy = np.linspace(lo[1], hi[1], grid)
            total_pts = len(gx) * len(gy)

            # ---- Render grid with progress ----
            frames = []
            xy_points = np.empty((total_pts, 2), dtype=np.float32)
            idx = 0
            for yy, xx in tqdm(
                product(gy, gx),
                total=total_pts,
                desc="Rendering env grid for t-SNE",
                leave=False,
                mininterval=0.1,
                dynamic_ncols=True,
                disable=not show_bar,
            ):
                with phys.reset_context():
                    s = phys.get_state().copy()
                    s[0], s[1] = float(xx), float(yy)
                    if s.shape[0] >= 4:
                        s[2], s[3] = 0.0, 0.0
                    phys.set_state(s)
                img = phys.render(width=W, height=H, camera_id=cam_id)  # (H,W,3) uint8
                if K > 1:
                    img = np.concatenate([img] * K, axis=2)
                frames.append(img)
                xy_points[idx] = (xx, yy)
                idx += 1

            frames = np.stack(frames, axis=0)                # (G,H,W,3K)
            obs = torch.as_tensor(frames, device="cuda", dtype=torch.float32)
            obs = (obs / 127.5) - 1.0

            with torch.no_grad():
                Z = self.model.encoder(obs).detach().cpu().numpy()
            xy_used = xy_points
            used_rendered = True

        except Exception as e:
            print("[plan2vec-tsne] render unavailable, fallback to batch:", repr(e))
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

        # ---- Fallback: use the entire batch (no subsampling) ----
        if Z is None:
            obs_b = samples["obs"]
            obs_b = obs_b.detach().cpu().float() if hasattr(obs_b, "detach") else torch.as_tensor(obs_b, dtype=torch.float32)
            obs_b = (obs_b / 127.5) - 1.0
            with torch.no_grad():
                Z = self.model.encoder(obs_b.cuda()).cpu().numpy()
            phys_batch = to_numpy(samples["physics"])
            xy_used = phys_batch[:, :2]

        # ---- t-SNE (subsample if huge to keep it snappy) ----
        n_all = Z.shape[0]
        if n_all > max_tsne:
            idx = np.linspace(0, n_all - 1, max_tsne, dtype=np.int64)
        else:
            idx = np.arange(n_all, dtype=np.int64)

        Z_tsne = Z[idx]
        xy_tsne = xy_used[idx]
        print(f"[plan2vec-tsne] encoded={n_all} | tsne_points={len(idx)}")

        perplexity = max(5, min(30, len(idx) // 3 if len(idx) >= 6 else 5))
        Z2 = TSNE(n_components=2, init="pca", perplexity=perplexity, max_iter=1000, random_state=0)\
            .fit_transform(Z_tsne)

        # color by normalized xy
        xy_min = xy_tsne.min(axis=0, keepdims=True)
        xy_max = xy_tsne.max(axis=0, keepdims=True)
        xy01 = (xy_tsne - xy_min) / (xy_max - xy_min + 1e-8)
        rgb = np.zeros((len(idx), 3), dtype=np.float32)
        rgb[:, 0], rgb[:, 1] = xy01[:, 0], xy01[:, 1]

        fig, (ax_tsne, ax_gt) = plt.subplots(1, 2, figsize=(9, 4.5), dpi=110)
        ax_tsne.scatter(Z2[:, 0], Z2[:, 1], c=rgb, s=6, alpha=0.85)
        ax_tsne.set_title(f"t-SNE of encoder latents\n({'env renders' if used_rendered else 'batch frames'})")
        ax_tsne.set_xticks([]); ax_tsne.set_yticks([])

        ax_gt.scatter(xy01[:, 0], xy01[:, 1], c=rgb, s=6, alpha=0.85)
        ax_gt.set_title("Ground-truth (x,y) normalized")
        ax_gt.set_xticks([]); ax_gt.set_yticks([])

        plt.tight_layout()
        img = wandb.Image(fig); plt.close(fig)
        return {"plan2vec_tsne_gt": img}


    def save(self, path):
        pass
