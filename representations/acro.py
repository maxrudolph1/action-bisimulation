from copy import deepcopy

import random
import numpy as np
import torch.nn
from matplotlib import cm

from models import gen_model_nets
import torch.nn.functional as F
import torch

from . import utils


class Acro(torch.nn.Module):
    def __init__(
        self, obs_shape, act_shape, cfg,
    ):
        super().__init__()
        encoder_cfg = cfg.algos.acro.encoder
        forward_cfg = cfg.algos.acro.forward
        inverse_cfg = cfg.algos.acro.inverse

        self.encoder = gen_model_nets.GenEncoder(obs_shape, cfg=encoder_cfg).cuda()
        self.embed_dim = self.encoder.output_dim
        self.forward_model = gen_model_nets.GenForwardDynamics(self.embed_dim, act_shape, forward_cfg).cuda()
        self.inverse_model = gen_model_nets.GenInverseDynamics(self.embed_dim, act_shape, inverse_cfg).cuda()

        self.learning_rate = cfg.algos.acro.learning_rate
        self.weight_decay = cfg.algos.acro.weight_decay
        self.forward_weight = cfg.algos.acro.forward_weight
        self.l1_penalty = cfg.algos.acro.l1_penalty
        self.dynamic_l1_penalty = cfg.algos.acro.dynamic_l1_penalty
        self.train_stop_epochs = cfg.algos.acro.train_stop_epochs

        self.k_steps = cfg.algos.acro.k_steps


        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.forward_model.parameters())
            + list(self.inverse_model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def share_dependant_models(self, models):
        pass

    def train_step(self, batch, epoch, train_step):
        if epoch >= self.train_stop_epochs:
            return {}
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda")

        # 0 = next_obs, 1 = next_next_obs
        random_step = random.randint(0, self.k_steps - 1)  # randint is inclusive
        obs_next = torch.as_tensor(batch["kobs"][:, random_step], device="cuda")

        # remove all obs, action, and obs_next entries that are invalid
        kvalid = batch["kvalid"].squeeze()
        obs = obs[kvalid]
        act = act[kvalid]
        obs_next = obs_next[kvalid]

        # NOTE:
        # - this is wrong.... should not be selecting a random next observation
        #   to use as the obs_next value
        # - additionally, we should be using kvalid to make sure that the
        #   sequence of observations/actions are valid and would work in the
        #   environment

        o_encoded = self.encoder(obs)
        on_encoded = self.encoder(obs_next)

        if self.forward_weight > 0:
            forward_model_loss = F.mse_loss(
                self.forward_model(o_encoded, act),
                on_encoded,
            )
        else:
            forward_model_loss = 0

        if self.l1_penalty > 0:
            l1_loss = (
                torch.linalg.vector_norm(o_encoded, ord=1, dim=1).mean()
                + torch.linalg.vector_norm(on_encoded, ord=1, dim=1).mean()
            ) / 2
        else:
            l1_loss = torch.zeros(1, device="cuda")

        inverse_model_pred = self.inverse_model(o_encoded, on_encoded)
        inverse_model_loss = F.cross_entropy(
            inverse_model_pred,
            act,
        )

        accuracy =  torch.mean(
                (torch.argmax(inverse_model_pred, dim=1) == act).float()
            )

        if self.dynamic_l1_penalty:
            gain = 5
            cur_l1_penalty = self.l1_penalty * np.exp(- gain * (accuracy.detach().item() - 1) ** 2)
        else:
            cur_l1_penalty = self.l1_penalty

        l1_loss = cur_l1_penalty * l1_loss
        forward_model_loss = self.forward_weight * forward_model_loss
        total_loss = (
            forward_model_loss
            + l1_loss
            + inverse_model_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        mean_element_magnitude = torch.abs(o_encoded).float().mean().detach().item()
        ret = {
            "inverse_loss": inverse_model_loss.detach().item(),
            "l1_loss": l1_loss.detach().item(),
            "loss": total_loss.detach().item(),
            "accuracy": accuracy.detach().item(),
            "cur_l1_pentaly": cur_l1_penalty,
            "mean_element_magnitude": mean_element_magnitude,
            "mean_representation_magnitude": torch.linalg.vector_norm(o_encoded, ord=1, dim=1).mean().detach().item(),
        }
        self.last_ret = ret
        return ret

    def save(self, path):
        torch.save(
            {
                "encoder": self.encoder,
                "forward_model": self.forward_model,
                "inverse_model": self.inverse_model,
                "optimizer": self.optimizer,
            },
            path,
        )
