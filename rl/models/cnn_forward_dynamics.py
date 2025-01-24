from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from . import utils
from . import gen_model_nets

class ForwardNet(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, **kwargs):
        super().__init__()
        c, h, w = obs_dim
        self.kwargs = kwargs

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=c + action_dim,
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
        )

    def forward(self, x, act):
        bs, c, h, w = x.shape
        action_channel = torch.zeros([bs, 4, h, w], device=x.device)
        action_channel[torch.arange(bs), act, :, :] = 1
        combined = torch.cat([x, action_channel], dim=1)
        return self.conv(combined)