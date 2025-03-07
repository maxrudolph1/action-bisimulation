from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from representations import utils
from models import gen_model_nets


class StochasticEncoder(torch.nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.encoder = Encoder(obs_dim)
        assert self.encoder.output_dim % 2 == 0
        self.output_dim = self.encoder.output_dim // 2

    def forward(self, x):
        # return self.encoder(x)[:, : self.output_dim]
        mu, log_var = self.mu_log_var(x)
        return utils.reparameterize(mu, log_var)
        # return self.fc_mu(self.encoder(x))

    def mu_log_var(self, x):
        encoded = self.encoder(x)
        return encoded[:, : self.output_dim], encoded[:, self.output_dim :]


class InverseDynamics(torch.nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        )

    def forward(self, embed, embed_next):
        return self.fc(torch.cat([embed, embed_next], dim=-1))

class ActionSetPredictor(torch.nn.Module):
    def __init__(self, embed_dim, action_dim, ksteps=1):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 16), #action_dim * ksteps),
        )

    def forward(self, embed):
        return torch.sigmoid(self.fc(embed))

    def pred_action_set(self, embed):
        return self.forward(embed).reshape(embed.shape[0], -1, self.action_dim)

class StochasticForwardDynamics(torch.nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.fc_mu = torch.nn.Sequential(
            torch.nn.Linear(embed_dim + action_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, embed_dim),
        )
        self.fc_log_var = torch.nn.Sequential(
            torch.nn.Linear(embed_dim + action_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, embed_dim),
        )

    def forward(self, embed, action):
        x = torch.cat([embed, F.one_hot(action, num_classes=self.action_dim)], dim=-1)
        return self.fc_mu(x), self.fc_log_var(x)


class ForwardDynamics(torch.nn.Module):
    def __init__(self, embed_dim, action_dim, hidden_size=256, hidden_layers=1):
        super().__init__()
        self.action_dim = action_dim
        layers = []
        top_layers = [torch.nn.Linear(embed_dim + action_dim, hidden_size), torch.nn.ReLU()]
        middle_layers = [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()] * (hidden_layers-1)
        last_layers = [torch.nn.Linear(hidden_size, embed_dim)]

        layers.extend(top_layers)
        layers.extend(middle_layers)
        layers.extend(last_layers)

        self.fc = torch.nn.Sequential(
            *layers
        )

    def forward(self, embed, action):
        x = torch.cat([embed, F.one_hot(action, num_classes=self.action_dim)], dim=-1)
        return self.fc(x)


class DQN(torch.nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        )

    def forward(self, embed):
        return self.fc(embed)

class DQNHER(torch.nn.Module):
    def __init__(self, state_shape, action_dim=4, args=None, atoms=1, split_obs=False, device='cpu', encoder_path=''):
        super().__init__()

        # self.encoder = GoallessEncoder(state_shape)
        if not (encoder_path is None) and (len(encoder_path) > 0):
            self.encoder=torch.load(encoder_path).encoder
        else:

            self.encoder = gen_model_nets.Encoder(state_shape,
                                                hidden_layers=args.encode_hidden_layers,
                                                num_pooling=args.encode_num_pooling,
                                                acti=args.encode_activation,
                                                use_layer_norm=args.encode_layer_norm).cuda()

        if args.use_gen_nets:
            self.dqn = gen_model_nets.GenDQN(self.encoder.output_dim,
                                                4 * atoms,
                                                hidden_layers=args.DQN_hidden_layers,
                                                acti=args.DQN_activation,
                                                use_layer_norm=args.DQN_layer_norm).cuda()
        else:
            self.dqn = DQN(self.encoder.output_dim, 4 * atoms)
        self.atoms = atoms
        self.action_dim = action_dim
        self.device = device
        self.split_obs = split_obs

    def forward(self, obs, state=None, info={}):

        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        embed = self.encoder(obs)
        if self.atoms == 1:
            logits = self.dqn(embed).view(-1,self.action_dim)
        else:
            logits = self.dqn(embed).view(-1,self.action_dim, self.atoms)
        return logits, state

class DQNFull(torch.nn.Module):
    def __init__(self, state_shape, action_dim=4, atoms=1, device='cpu'):
        super().__init__()
        self.encoder = Encoder(state_shape)
        self.dqn = DQN(self.encoder.output_dim, 4 * atoms)
        self.atoms = atoms
        self.action_dim = action_dim
        self.device = device

    def forward(self, obs, state=None, info={}):

        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        embed = self.encoder(obs)
        if self.atoms == 1:
            logits = self.dqn(embed).view(-1,self.action_dim)
        else:
            logits = self.dqn(embed).view(-1,self.action_dim, self.atoms)
        return logits, state

class SideTuner(torch.nn.Module):
    def __init__(self, encoder, obs_dim):
        super().__init__()
        c, h, w = obs_dim
        self.encoder = encoder
        # self.side_encoder = deepcopy(encoder)
        self.side_encoder = Encoder((1, h, w))
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))
        # self.alpha = 0

    def forward(self, x):
        a = torch.sigmoid(self.alpha)
        return a * self.encoder(x[:, :2]).detach() + (1 - a) * self.side_encoder(
            x[:, 2].unsqueeze(1)
        )
        # return torch.cat([self.encoder(x[:, :2]).detach(), self.side_encoder(x[:, 2].unsqueeze(1))], dim=1)


class Decoder(torch.nn.Module):
    def __init__(self, embed_dim, obs_dim, **kwargs):
        super().__init__()
        c, h, w = obs_dim
        self.kwargs = kwargs
        self.use_grid = 'use_grid' in kwargs and kwargs['use_grid']
        self.grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, h) / (h - 1), torch.arange(0, w) / (w - 1)
                ),
                dim=0,
            )
            * 2
            - 1
        ).cuda()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=embed_dim + 2 if 'use_grid' in kwargs and kwargs['use_grid'] else embed_dim,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        if self.use_grid:
            grid_expand = self.grid.expand(x.shape[0], -1, -1, -1)
        x_expand = (
            x.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 15, 15)
        )

        if self.use_grid:
            # combined = torch.cat([obs, grid_expand], dim=1)
            combined = torch.cat([x_expand, grid_expand], dim=1)
        else:
            combined = x_expand

        return self.conv(combined)
