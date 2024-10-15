from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np


class InplaceOperator(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x)


def crelu(x, dim=1):
    return torch.cat((F.relu(x), F.relu(-x)), dim)

class CReLU(nn.Module):
    def __init__(self, dim=1):
        super(CReLU, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), self.dim)

def get_inplace_acti(acti, param=0.1):
    if acti == "relu": return nn.ReLU(inplace=True)
    elif acti == "leakyrelu": return nn.LeakyReLU(negative_slope=param, inplace=True)
    elif acti == "sin": return InplaceOperator(torch.sin)
    elif acti == "sinc": return InplaceOperator(torch.sinc)
    elif acti == "sigmoid": return nn.Sigmoid()
    elif acti == "tanh": return InplaceOperator(torch.tanh)
    elif acti == "softmax": return nn.SoftMax(-1)
    elif acti == "cos": return InplaceOperator(torch.cos)
    elif acti == "none": return nn.Identity()
    elif acti == "prelu": return nn.PReLU()
    elif acti == "crelu": return CReLU()

class GeneralConv2DEncoder(torch.nn.Module):
    '''
    Generalized 2d encoder, with variable hidden layers, pooling layers, activations and layer norm
    '''
    def __init__(self, obs_dim, normalized=False, **kwargs):
        super().__init__()
        c, h, w = obs_dim
        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else [64,64,64,96,96,128]
        self.num_pooling = kwargs["num_pooling"] if "num_pooling" in kwargs else 2
        self.normalized = normalized
        self.acti = kwargs["acti"] if "acti" in kwargs else "relu"
        self.use_layer_norm = kwargs["use_layer_norm"] if "use_layer_norm" in kwargs else False

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
        
        in_channels = c + 2 if 'use_grid' in kwargs and kwargs['use_grid'] else c
        num_non_pool = len(self.hidden_layers) - self.num_pooling - 1 # subtract number of pooling layers and last layer which has no pooling
        self.conv = torch.nn.Sequential( * # separate logic for the first layer
            (([nn.LayerNorm((in_channels, h, w))] if self.use_layer_norm else list()) + 
            [torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.hidden_layers[0],
                kernel_size=1,
                stride=1,
                padding="same",
            ), get_inplace_acti(self.acti)
            ] + sum([ # add in the non-pooled layers
            ([nn.LayerNorm((self.hidden_layers[i-1], h, w))] if self.use_layer_norm else list()) +
            [torch.nn.Conv2d(
                in_channels=self.hidden_layers[i-1],
                out_channels=self.hidden_layers[i],
                kernel_size=3,
                stride=1,
                padding="same",
            ), get_inplace_acti(self.acti)] for i in range(1, num_non_pool)], start=list())
            + (sum([ # add in the pooled layers 
            ([nn.LayerNorm((self.hidden_layers[i-1], h / (2 ** (i-num_non_pool)), w / (2 ** (i-num_non_pool))))] if self.use_layer_norm else list()) +
            [torch.nn.Conv2d(
                in_channels=self.hidden_layers[i-1],
                out_channels=self.hidden_layers[i],
                kernel_size=3,
                stride=1,
                padding="same",
            ), get_inplace_acti(self.acti), torch.nn.MaxPool2d((2, 2))] for i in range(num_non_pool, len(self.hidden_layers) - 1)], start=list())) +
            [torch.nn.Conv2d( # the last layer
                in_channels=self.hidden_layers[len(self.hidden_layers) - 2],
                out_channels=self.hidden_layers[len(self.hidden_layers) - 1],
                kernel_size=3,
                stride=1,
                padding="same",
            )]) # no activation, no pooling after final layer
            # torch.nn.Tanh(),
            # torch.nn.MaxPool2d((2, 2)), #torch.nn.MaxPool2d((5, 5) if h // 4 > 2 else (2, 2)),
        )
        self.kwargs = kwargs

        self.output_dim = self.conv(torch.zeros([1, in_channels, h, w])).flatten().shape[0]
        
        

    def forward(self, obs):
        obs = torch.as_tensor(obs, device="cuda")

        if len(obs.shape) == 5:
            obs = obs.flatten(1, 2)
        if obs.dtype == torch.uint8:
            obs = obs / 127.5 - 1
        grid_expand = self.grid.expand(obs.shape[0], -1, -1, -1)
        
        # try:
        #     a = self.kwargs['use_grid']
        # except:
        #     self.kwargs = {'use_grid': True}
        if 'use_grid' in self.kwargs and self.kwargs['use_grid']:
            combined = torch.cat([obs, grid_expand], dim=1)
        else:
            combined = obs
        # print(self.conv)
        # print(combined.shape)
        z = self.conv(combined).flatten(start_dim=1)
        return  z

class LinearNetwork(torch.nn.Module): # TODO: replace with general linear networks for tuning reasons
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else [256]
        self.num_pooling = kwargs["num_pooling"] if "num_pooling" in kwargs else 2
        self.acti = kwargs["acti"] if "acti" in kwargs else "relu"
        self.use_layer_norm = kwargs["use_layer_norm"] if "use_layer_norm" in kwargs else False
        layers = [input_dim] + self.hidden_layers
        self.fc = torch.nn.Sequential( *sum([
            ([nn.LayerNorm(self.num_inputs)] if self.use_layer_norm else list()) + 
            [torch.nn.Linear(layers[i-1], layers[i]),
            get_inplace_acti(self.acti)] for i in range(1,len(layers))], start=list()) +
            [torch.nn.Linear(layers[-1], output_dim)]
        )

    def forward(self, embed):
        return self.fc(embed)


class GeneralConv2DDecoder(torch.nn.Module):
    def __init__(self, embed_dim, obs_dim, **kwargs):
        super().__init__()
        super().__init__()
        c, h, w = obs_dim
        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else [64,64,64,96,96,128]
        self.num_pooling = kwargs["num_pooling"] if "num_pooling" in kwargs else 2
        self.acti = kwargs["acti"] if "acti" in kwargs else "relu"
        self.use_layer_norm = kwargs["use_layer_norm"] if "use_layer_norm" in kwargs else False

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
        
        in_channels = c + 2 if 'use_grid' in kwargs and kwargs['use_grid'] else c
        num_expand = len(self.hidden_layers) - self.num_pooling
        self.conv = torch.nn.Sequential( *[ # separate logic for the first layer
            ([nn.LayerNorm((in_channels, h, w))] if self.use_layer_norm else list()) + 
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.hidden_layers[0],
                kernel_size=1,
                stride=1,
                padding="same",
            ), get_inplace_acti(self.acti)
            ] + sum([ # add in the non-pooled layers
            ([nn.LayerNorm((self.hidden_layers[i-1], h, w))] if self.use_layer_norm else list()) +
            [torch.nn.Conv2d(
                in_channels=self.hidden_layers[i-1],
                out_channels=self.hidden_layers[i],
                kernel_size=1,
                stride=1,
                padding="same",
            ), get_inplace_acti(self.acti)] for i in range(1, len(self.hidden_layers))], start=list())
            [torch.nn.Conv2d( # the last layer
                in_channels=self.hidden_layers[len(self.hidden_layers) - 1],
                out_channels=3, # 3 output channels
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.Tanh()] # no activation, no pooling after final layer
            # torch.nn.Tanh(),
            # torch.nn.MaxPool2d((2, 2)), #torch.nn.MaxPool2d((5, 5) if h // 4 > 2 else (2, 2)),
        )
        self.kwargs = kwargs

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
