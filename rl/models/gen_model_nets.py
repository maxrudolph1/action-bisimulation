from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from models import gen_nets

class GenEncoder(torch.nn.Module):
    def __init__(self, obs_dim, cfg):
        super().__init__()
        self.cnn_encoder = gen_nets.GeneralConv2DEncoder(obs_dim, normalized=False, **cfg)
        if 'output_dim' in cfg:
            print(f"*** gen_model_nets.py - use output layer with output dim {cfg['output_dim']}")
            self.use_output_layer = True
            self.output_dim = cfg['output_dim']
            self.last_layer = nn.Sequential(nn.ReLU(), nn.Linear(self.cnn_encoder.output_dim, self.output_dim))
        else:    
            self.output_dim = self.encoder.output_dim
            self.use_output_layer = False
        self.cfg = cfg
        self.obs_dim = obs_dim
        
    def forward(self, obs):
        z = self.cnn_encoder(obs)
        if self.use_output_layer:
            z = self.last_layer(z)
        return z
    
    def save(self, path):
        torch.save(dict(obs_dim=self.obs_dim, cfg=self.cfg, state_dict=self.state_dict()), path)

class GenInverseDynamics(torch.nn.Module):
    def __init__(self, embed_dim, action_dim, cfg):
        super().__init__()
        self.fc = gen_nets.LinearNetwork(
            embed_dim * 2, action_dim, **cfg
        )

    def forward(self, embed, embed_next):
        return self.fc(torch.cat([embed, embed_next], dim=-1))
    
class GenActionSetPredictor(torch.nn.Module):
    def __init__(self, embed_dim, action_dim, ksteps=1, **kwargs):
        super().__init__()
        self.fc = gen_nets.LinearNetwork(
            embed_dim, action_dim * ksteps, **kwargs
        )

    def forward(self, embed):
        return torch.sigmoid(self.fc(embed))

    def pred_action_set(self, embed):
        return self.forward(embed).reshape(embed.shape[0], -1, self.action_dim)

class GenStochasticForwardDynamics(torch.nn.Module):
    def __init__(self, embed_dim, action_dim, **kwargs):
        super().__init__()
        self.action_dim = action_dim
        self.fc = gen_nets.LinearNetwork(
            embed_dim + action_dim, embed_dim, **kwargs
        )

        self.fc = gen_nets.LinearNetwork(
            embed_dim + action_dim, embed_dim, **kwargs
        )


    def forward(self, embed, action):
        x = torch.cat([embed, F.one_hot(action, num_classes=self.action_dim)], dim=-1)
        return self.fc_mu(x), self.fc_log_var(x)


class GenForwardDynamics(torch.nn.Module):
    def __init__(self, embed_dim, action_dim, cfg):
        super().__init__()
        self.action_dim = action_dim
        layers = []
        self.activation=cfg['post_activation']

        self.fc = gen_nets.LinearNetwork(
            embed_dim + action_dim, embed_dim, **cfg
        )
        
    def forward(self, embed, action):
        x = torch.cat([embed, F.one_hot(action, num_classes=self.action_dim)], dim=-1)
        return self.fc(x)
    
    def reset_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity=self.activation)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)


class GenDQN(torch.nn.Module):
    def __init__(self, embed_dim, action_dim, **kwargs):
        super().__init__()
        self.fc = gen_nets.LinearNetwork(
            embed_dim, action_dim, **kwargs
        )

    def forward(self, embed):
        return self.fc(embed)

class GenDQNHER(torch.nn.Module):
    def __init__(self, state_shape, action_dim=4, atoms=1, split_obs=False, device='cpu', encoder_path='', **kwargs):
        super().__init__()        
        if not (encoder_path is None) and (len(encoder_path) > 0):
            self.encoder=torch.load(encoder_path).encoder
        else:
            self.encoder = gen_nets.GenEncoder2D(state_shape, **kwargs)
        self.dqn = GenDQN(self.encoder.output_dim, 4 * atoms, **kwargs["DQN_args"])
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

class GenDQNFull(torch.nn.Module):
    def __init__(self, state_shape, action_dim, cfg):
        super().__init__()
        self.encoder = GenEncoder(state_shape, cfg['encoder_args'])
        self.dqn = GenDQN(embed_dim=self.encoder.output_dim, action_dim=action_dim * cfg['dqn_args']['atoms'], **cfg['dqn_args'])
        self.atoms = cfg['dqn_args']['atoms']
        self.action_dim = action_dim
        self.device = cfg['device']
        
    def forward(self, obs, state=None, info={}):
        
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        
        embed = self.encoder(obs)
        if self.atoms == 1:
            logits = self.dqn(embed).view(-1,self.action_dim)
        else:
            logits = self.dqn(embed).view(-1,self.action_dim, self.atoms)
        return logits, state

class GenSideTuner(torch.nn.Module):
    def __init__(self, encoder, obs_dim, **kwargs):
        super().__init__()
        c, h, w = obs_dim
        self.encoder = encoder
        # self.side_encoder = deepcopy(encoder)
        self.side_encoder = GenEncoder((1, h, w), **kwargs)
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))
        # self.alpha = 0

    def forward(self, x):
        a = torch.sigmoid(self.alpha)
        return a * self.encoder(x[:, :2]).detach() + (1 - a) * self.side_encoder(
            x[:, 2].unsqueeze(1)
        )
        # return torch.cat([self.encoder(x[:, :2]).detach(), self.side_encoder(x[:, 2].unsqueeze(1))], dim=1)


class GenDecoder2D(torch.nn.Module):
    def __init__(self, embed_dim, obs_dim, cfg):
        super().__init__()
        c, h, w = obs_dim
        self.cfg = cfg
        self.use_grid = True #'use_grid' in cfg and cfg['use_grid']
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
        self.h, self.w = h,w

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=embed_dim + 2 if self.use_grid else embed_dim,
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
            .expand(-1, -1, self.h, self.w)
        )
            
        if self.use_grid:
            combined = torch.cat([x_expand, grid_expand], dim=1)
        else:
            combined = x_expand
        
        return self.conv(combined)
