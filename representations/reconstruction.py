from copy import deepcopy

import torch
import torch.nn.functional as F
from models import gen_model_nets


class EncoderReconstruction(torch.nn.Module):
    def __init__(
        self,
        obs_shape,
        act_shape=None,
        encoder_cfg=None,
        forward_cfg=None,
        inverse_cfg=None,
        learning_rate=0.01,
        weight_decay=1e-5,
        tau=0.95,
        sync_freq=1,
        **kwargs
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.tau = tau
        self.steps_until_sync = 0
        self.sync_freq = sync_freq

        # self.decoder_model = gen_model_nets.GenDecoder2D(obs_shape[0], obs_shape).cuda()
        self.encoder = gen_model_nets.GenEncoder(obs_shape).cuda()
        self.decoder_model = gen_model_nets.GenDecoder2D(1152, obs_shape).cuda()

        # self.decoder_optimizer = torch.optim.Adam(
        #     list(self.decoder_model.parameters()),
        #     lr=learning_rate,
        #     weight_decay=weight_decay,
        #     # weight_decay=0,
        # )

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder_model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, factor=0.9, patience=12)

        # self.encoder = None

    def share_dependant_models(self, model):
        # self.encoder = model.encoder
        pass

    def train_step(self, batch, epoch):
        if self.encoder is None:
            raise ValueError("Encoder not shared. Call share_dependant_models() before training.")

        obs_x = torch.as_tensor(batch["obs"], device="cuda")

        ox_encoded_online = self.encoder(obs_x).detach()
        obs_x_reconstructed = self.decoder_model(ox_encoded_online)
        decoder_loss = F.mse_loss(obs_x_reconstructed, obs_x) # change so proportion from all channels
        # TODO: NOTE TO LOOK AT FOR THIS
        # check if the loss function works by doing autoencoder with loss function
        # likely neeed normalization on the mse loss
        # - suspicion is normalizaing across all the pixels
        # - might be normalizing over every pixel across the channels
        # - if autoencoder works, then might be issue with tuning single step and the normalization for that

        # self.decoder_optimizer.zero_grad()
        # decoder_loss.backward()
        # self.decoder_optimizer.step()

        self.optimizer.zero_grad()
        decoder_loss.backward()
        self.optimizer.step()

        log = {
            "decoder_loss": decoder_loss.detach().item(),
            # "learning_rate": learning_rate,
        }
        return log
