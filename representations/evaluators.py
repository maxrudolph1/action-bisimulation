# from copy import deepcopy

import numpy as np
import torch.nn
# from matplotlib import cm

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
        # pdb.set_trace()
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

    def save(self, path):
        pass
