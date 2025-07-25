# from copy import deepcopy

import numpy as np
import torch.nn
# from matplotlib import cm

from models import gen_model_nets
import torch.nn.functional as F
import torch

from . import utils

class SingleStep(torch.nn.Module):
    def __init__(
        self, obs_shape, act_shape, cfg,
    ):
        super().__init__()
        encoder_cfg = cfg.algos.single_step.encoder
        forward_cfg = cfg.algos.single_step.forward
        inverse_cfg = cfg.algos.single_step.inverse

        if encoder_cfg.normalization == 'l1':
            assert cfg.algos.single_step.l1_penalty == 0, "l1 normalization and l1 penalty must not be used together"
        if encoder_cfg.normalization == 'l2':
            assert cfg.algos.single_step.l2_penalty == 0, "l2 normalization and l2 penalty must not be used together"
        if encoder_cfg.normalization == 'softmax':
            assert cfg.algos.single_step.l1_penalty == 0, "softmax normalization and l1 penalty must not be used together"

        self.l2_penalty = cfg.algos.single_step.l2_penalty
        self.l1_penalty = cfg.algos.single_step.l1_penalty

        self.dynamic_regularization_penalty = cfg.algos.single_step.dynamic_regularization_penalty

        self.encoder = gen_model_nets.GenEncoder(obs_shape, cfg=encoder_cfg).cuda()
        self.embed_dim = self.encoder.output_dim

        self.forward_model = gen_model_nets.GenForwardDynamics(self.embed_dim, act_shape, forward_cfg).cuda()
        self.inverse_model = gen_model_nets.GenInverseDynamics(self.embed_dim, act_shape, inverse_cfg).cuda()

        self.learning_rate = cfg.algos.single_step.learning_rate
        self.weight_decay = cfg.algos.single_step.weight_decay
        self.forward_weight = cfg.algos.single_step.forward_weight

        self.train_stop_epochs = cfg.algos.single_step.train_stop_epochs

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.forward_model.parameters()) + list(self.inverse_model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def share_dependant_models(self, models):
        pass

    def train_step(self, batch, epoch, train_step):
        if epoch >= self.train_stop_epochs:
            return {}
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda")
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
        elif self.l2_penalty > 0:
            l2_loss = (
                torch.linalg.norm(o_encoded, ord=2, dim=1).mean()
                + torch.linalg.norm(on_encoded, ord=2, dim=1).mean()
            ) / 2

        inverse_model_pred = self.inverse_model(o_encoded, on_encoded)
        inverse_model_loss = F.cross_entropy(
            inverse_model_pred,
            act,
        )

        accuracy = torch.mean(
            (torch.argmax(inverse_model_pred, dim=1) == act).float()
        )

        if self.dynamic_regularization_penalty:
            gain = 5
            multiplier =  np.exp(- gain * (accuracy.detach().item() - 1) ** 2)

        forward_model_loss = self.forward_weight * forward_model_loss

        # regularization loss stuff
        if self.l2_penalty > 0:
            total_loss = (forward_model_loss + multiplier * self.l2_penalty * l2_loss + inverse_model_loss)
        elif self.l1_penalty > 0:
            total_loss = (forward_model_loss + multiplier * self.l1_penalty * l1_loss + inverse_model_loss)
        else:
            total_loss = (forward_model_loss + inverse_model_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # mean_element_magnitude = torch.abs(o_encoded).float().mean().detach().item()
        ret = {
            "inverse_loss": inverse_model_loss.detach().item(),
            "l1_loss": l1_loss.detach().item() if self.l1_penalty > 0 else 0,
            "l2_loss": l2_loss.detach().item() if self.l2_penalty > 0 else 0,
            # "mean_encoded_magnitude": pre_penalized_l1_loss,  # purely for debugging and logging
            # for pre-penalized loss, expect lower l1 values to result in higher pre-penalty
            # aka: 0.01 should have a LOWER pre-penalty
            # aka: 0.0001 should have a HIGHER pre-penalty
            "loss": total_loss.detach().item(),
            "accuracy": accuracy.detach().item(),
            "cur_regularization_penalty": multiplier,

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
