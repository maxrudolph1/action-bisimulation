import torch as th
from gymnasium import spaces
from torch import nn
from models import gen_model_nets
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)



class BisimCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, **kwargs):
        super(BisimCNN, self).__init__(observation_space, 1)
        feature_extractor_kwargs = kwargs
        self.encoder = gen_model_nets.GenEncoder(observation_space.shape, cfg=feature_extractor_kwargs).cuda() 
        self._features_dim = self.encoder.output_dim

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.encoder(x)
