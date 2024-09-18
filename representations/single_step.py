from copy import deepcopy

import numpy as np
import torch.nn
from matplotlib import cm

from . import nets
from . import gen_model_nets
import torch.nn.functional as F
import torch

from . import utils


class SingleStep(torch.nn.Module):
    def __init__(
        self, obs_shape, action_dim, learning_rate, forward_model_weight, args, l1_penalty,weight_decay=1e-5
    ):
        super().__init__()

        if args.use_gen_nets:
            self.encoder = gen_model_nets.GenEncoder(obs_shape,
                                                  hidden_layers=args.encode_hidden_layers,
                                                  num_pooling=args.encode_num_pooling,
                                                  acti=args.encode_activation,
                                                  use_layer_norm=args.encode_layer_norm).cuda()
            self.embed_dim = self.encoder.output_dim
            self.forward_model = gen_model_nets.GenForwardDynamics(self.embed_dim, action_dim,
                                                                hidden_layers=args.post_hidden_layers,
                                                                acti=args.post_activation,
                                                                use_layer_norm=args.post_layer_norm).cuda()
            self.inverse_model = gen_model_nets.GenInverseDynamics(self.embed_dim, action_dim,
                                                                hidden_layers=args.post_hidden_layers,
                                                                acti=args.post_activation,
                                                                use_layer_norm=args.post_layer_norm).cuda()
        else:
            self.encoder = nets.Encoder(obs_shape).cuda()
            self.embed_dim = self.encoder.output_dim
            self.forward_model = nets.ForwardDynamics(self.embed_dim, action_dim).cuda()
            self.inverse_model = nets.InverseDynamics(self.embed_dim, action_dim).cuda()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.forward_model.parameters())
            + list(self.inverse_model.parameters()),
            lr=learning_rate,
            # weight_decay=h["weight_decay"],
            weight_decay=weight_decay,
        )

        self.forward_model_weight = forward_model_weight
        self.l1_penalty = l1_penalty

    def train_step(self, batch):
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
            l1_loss = 0

        inverse_model_pred = self.inverse_model(o_encoded, on_encoded)
        inverse_model_loss = F.cross_entropy(
            inverse_model_pred,
            act,
        )

        total_loss = (
            self.forward_model_weight * forward_model_loss
            + self.l1_penalty * l1_loss
            + inverse_model_loss
        )
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        
        

        ret = {
            "inverse": inverse_model_loss.detach().item(),
            "total": total_loss.detach().item(),
            "accuracy": torch.mean(
                (torch.argmax(inverse_model_pred, dim=1) == act).float()
            )
            .detach()
            .item(),
            # "rep_size": o_encoded.float().mean().detach().item(),
        }
        if self.forward_model_weight > 0:
            ret["forward"] = forward_model_loss.detach().item()
        if self.l1_penalty > 0:
            ret["l1_penalty"] = l1_loss.detach().item()
        
        return ret