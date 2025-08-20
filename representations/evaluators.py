# from copy import deepcopy

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

    def eval_plan2vec_figure5(self, samples):
        # Expects a stack of obs. NOTE: Needs to be edited before use
        B = samples["obs"].shape[0]
        N = min(1024, B)
        idxs = np.random.choice(B, size=N, replace=False)
        obs_batch  = samples["obs"][idxs].detach().cpu()
        phys_batch = samples["physics"][idxs].detach().cpu().numpy()

        obs_norm = (obs_batch.float() / 127.5) - 1.0

        with torch.no_grad():
            Z = self.model.encoder(obs_norm.cuda()).cpu().numpy()  # (N, D)

        print("Min/max/mean of Z:", Z.min(), Z.max(), Z.mean())

        Z2 = TSNE(n_components=2, init="pca", perplexity=30, max_iter=1000,
                  random_state=0).fit_transform(Z)

        # Normalize true positions to [0,1]
        xy = phys_batch[:, :2].copy()
        min_xy = xy.min(axis=0, keepdims=True)
        max_xy = xy.max(axis=0, keepdims=True)
        xy = (xy - min_xy) / (max_xy - min_xy + 1e-8)

        # Build RGB
        rgb = np.zeros((N, 3), dtype=np.float32)
        rgb[:, 0] = xy[:, 0]
        rgb[:, 1] = xy[:, 1]

        fig, (ax_tsne, ax_gt) = plt.subplots(1, 2, figsize=(8,4), dpi=100)
        ax_tsne.scatter(Z2[:,0], Z2[:,1], c=rgb, s=6, alpha=0.8)
        ax_tsne.set_title("TSNE: Encoded Latents")
        ax_tsne.set_xticks([]); ax_tsne.set_yticks([])

        ax_gt.scatter(xy[:,0], xy[:,1], c=rgb, s=6, alpha=0.8)
        ax_gt.set_title("Ground Truth (x,y)")
        ax_gt.set_xticks([]); ax_gt.set_yticks([])

        plt.tight_layout()
        return {"plan2vec_tsne_gt": wandb.Image(fig)}

    def save(self, path):
        pass
