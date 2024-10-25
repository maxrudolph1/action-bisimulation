from copy import deepcopy

import numpy as np
import torch.nn
from matplotlib import cm

from models import nets
from models import gen_model_nets
import torch.nn.functional as F
import torch

from . import utils



class BetaVariationalAutoencoder(torch.nn.Module):
    def __init__(
        self, obs_shape, act_shape,
        cfg=None, 
        **kwargs,
    ):
        super().__init__()

        self.algo_config = cfg.algos.bvae
        encoder_cfg = self.algo_config.encoder
        self.learning_rate = self.algo_config.learning_rate

        decoder_cfg = self.algo_config.decoder
        
        self.obs_shape = obs_shape
        self.encoder = gen_model_nets.GenEncoder(obs_shape, cfg=encoder_cfg).cuda() 
        self.decoder = gen_model_nets.GenDecoder2D(self.encoder.output_dim, obs_shape,cfg=decoder_cfg).cuda()
        self.embed_dim = self.encoder.output_dim
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())+ list(self.decoder.parameters()),
            lr=self.learning_rate,
        )
        

    def share_dependant_models(self, models):
        pass
            
        
    def train_step(self, batch, epoch, train_step):
        obs_x = torch.as_tensor(batch["obs"], device="cuda")  # observation ( x )
        z = self.encoder(obs_x) # encoder z = phi(x)

        obs_x_recon = self.decoder(z) # decoder x = phi^-1((phi(x))_

        recon_loss = F.mse_loss(obs_x_recon, obs_x) 

        self.optimizer.zero_grad()
        recon_loss.backward()
        self.optimizer.step()

        return {"recon_loss": recon_loss.item()}


    def save(self, path):
        torch.save(
            {
                "encoder": self.encoder,
                "decoder": self.decoder,
            },
            path,
        )
