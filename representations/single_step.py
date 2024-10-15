from copy import deepcopy

import numpy as np
import torch.nn
from matplotlib import cm

from models import gen_model_nets
import torch.nn.functional as F
import torch

from . import utils


class SingleStep(torch.nn.Module):
    def __init__(
        self, obs_shape, act_shape, encoder_cfg, forward_cfg, inverse_cfg, learning_rate=0.01, forward_weight=0.01, l1_penalty=0.0, weight_decay=1e-5, **kwargs,
    ):
        super().__init__()
        encoder_type = list(encoder_cfg.keys())[0]
        forward_type = list(forward_cfg.keys())[0]
        inverse_type = list(inverse_cfg.keys())[0]

        self.encoder = gen_model_nets.GenEncoder(obs_shape, cfg=encoder_cfg[encoder_type]).cuda()
        self.embed_dim = self.encoder.output_dim
        self.forward_model = gen_model_nets.GenForwardDynamics(self.embed_dim, act_shape, **forward_cfg[forward_type]).cuda()
        self.inverse_model = gen_model_nets.GenInverseDynamics(self.embed_dim, act_shape, **inverse_cfg[inverse_type]).cuda()

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.forward_model.parameters())
            + list(self.inverse_model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.forward_model_weight = forward_weight
        self.l1_penalty = l1_penalty
        self.dynamic_l1_penalty = kwargs.get("dynamic_l1_penalty", False)
        self.train_stop_epochs = kwargs.get("train_stop_epochs", 1e6)

    def share_dependant_models(self, models):
        pass
    
    def train_step(self, batch, epoch):
        if epoch >= self.train_stop_epochs:
            return {}
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda")
        o_encoded = self.encoder(obs)
        on_encoded = self.encoder(obs_next)

        if self.forward_model_weight > 0:
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
        forward_model_loss = self.forward_model_weight * forward_model_loss
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